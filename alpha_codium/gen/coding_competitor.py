import asyncio
import functools
import logging
import os
import re

import numpy as np
import yaml
from jinja2 import Environment, StrictUndefined

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.code_test_runners import eval_solution
from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_handler import AiHandler
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)

class CodeContestsCompetitor:
    def __init__(self, test_flavor='local'):
        self.prompt = {}
        for set in get_settings():
            if 'code_contests_prompt' in set.lower():
                self.prompt[set.lower()] = get_settings()[set]
        self.ai_handler = AiHandler()
        # self.prompt = get_settings().code_contests_prompt_baseline
        # self.token_handler = TokenHandler(
        #     None, None, self.prompt.system, self.prompt.user
        # )

    def render(self, problem_json, prompt: str):
        environment = Environment(undefined=StrictUndefined)
        environment.globals["zip"] = zip
        environment.globals["enumerate"] = enumerate
        sys_prompt = environment.from_string(self.prompt[prompt].system).render(problem_json)
        usr_prompt = environment.from_string(self.prompt[prompt].user).render(problem_json)
        return sys_prompt, usr_prompt

    async def _run(self, model, problem, prompt:str = "code_contests_prompt_reflect"):
        system_prompt, user_prompt = self.render(problem, prompt)

        response, finish_reason = await self.ai_handler.chat_completion(
            model=model, system=system_prompt, user=user_prompt
        )
        return response, finish_reason

    def postprocess_response(self, response):
        response = str(response)
        if response.endswith("stop"):
            response = response[:-4]
        pattern = r'```\w*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]
        return response

    async def run(self, problem):
        logger.info(f"Running code contests competitor, model {get_settings().config['model']}")

        # configurations
        problem = {k: problem.get(k) for k in ["name", "description", "public_tests"]}
        use_baseline = get_settings().get("solve.use_baseline", False)
        do_recording = get_settings().get("solve.do_recording", False)
        use_recording = get_settings().get("solve.use_recording", False)
        if use_recording or do_recording:
            recording_path = f"./code_contests/{problem['name']}/{get_settings().config['model']}/"
            logger.info(f"recording_path: {recording_path}\ndo_record: {do_recording}\nuse_record: {use_recording}")
            if do_recording:
                os.makedirs(recording_path, exist_ok=True)

        if use_baseline:
            logging.info("Using baseline prompt")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_baseline")
            response_baseline, _ = await retry_with_fallback_models(f)
            if response_baseline:
                recent_solution = self.postprocess_response(response_baseline)
        else:
            # reflect
            logger.info("--reflection stage--")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_reflect")
            if use_recording:
                response_reflect = np.load(recording_path + 'reflect.npy', allow_pickle=True).tolist()
                logger.info("Using recording")
                logger.debug(f"response_reflect:\n{response_reflect}")
            else:
                response_reflect, _ = await retry_with_fallback_models(f)
                if do_recording:
                    np.save(recording_path + 'reflect.npy', response_reflect)
            response_reflect = response_reflect.rstrip("` \n")
            try:
                response_reflect_yaml = yaml.safe_load(response_reflect)
            except yaml.YAMLError:
                response_reflect = self.postprocess_response(response_reflect) # try to include only the yaml part
                response_reflect_yaml = yaml.safe_load(response_reflect)
            problem['response_reflect'] = response_reflect
            problem['self_description'] = response_reflect_yaml['self_description']
            problem['possible_solutions'] = response_reflect_yaml['possible_solutions']

            # choose best solution
            logger.info("--choose best solution stage--")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_choose_best_solution")
            if use_recording:
                response_best_solution = np.load(recording_path + 'best_solution.npy', allow_pickle=True)\
                                                .tolist()
                logger.info("Using recording")
                logger.debug(f"response_best_solution:\n{response_best_solution}")
            else:
                response_best_solution, _ = await retry_with_fallback_models(f)
                if do_recording:
                    np.save(recording_path + 'best_solution.npy', response_best_solution)
            response_best_solution = response_best_solution.rstrip("` \n")
            problem['response_best_solution'] = response_best_solution
            response_best_solution_yaml = yaml.safe_load(response_best_solution) # noqa

            # solve
            logger.info("--solve stage--")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_solve")
            if use_recording:
                response_solve = np.load(recording_path + 'solve.npy', allow_pickle=True).tolist()
                logger.info("Using recording")
                logger.debug(f"response_solve:\n{response_solve}")
            else:
                response_solve, _ = await retry_with_fallback_models(f)
                if do_recording:
                    np.save(recording_path + 'solve.npy', response_solve)
            response_solve = response_solve.rstrip("` \n")
            problem['best_solution_code'] = response_solve
            recent_solution = response_solve

            # evaluate public tests
            logger.info("--iterate on public tests stage--")
            is_all_passed_public = False
            counter = 0
            max_allowed_counter = 5
            problem['recent_solution'] = problem['last_solution_code'] = problem['best_solution_code']

            while not is_all_passed_public:

                # run the solution on the public tests
                logging.info(f"evaluating public tests. attempt {counter}")
                test_inputs, results = eval_solution(example=problem,
                                                     prediction=recent_solution,
                                                     test_inputs=problem['public_tests']['input'],
                                                     test_outputs=problem['public_tests']['output'], )

                # analyze the tests results
                if str(results.test_results[0].program_status) == 'ProgramStatus.kTimeout':
                    logger.error("timeout. reverting to last solution")
                    counter += 1
                    recent_solution = problem['last_solution_code']
                    if counter > max_allowed_counter:
                        logger.error(f"Failed to pass public tests after {max_allowed_counter} attempts")
                        break
                    continue
                elif str(results.test_results[0].program_status) == 'ProgramStatus.kFailed':
                    logger.error("failed to run solution")
                    error_str = results.test_results[0].sandbox_result
                    trace_str = f"trace information:\n{self.render_trace(results.test_results[0].trace)}\n\n"
                    is_all_passed_public = False
                    is_valid_output = True
                else:
                    # build the error string
                    error_str = ""
                    trace_str = ""
                    is_all_passed_public = True
                    is_valid_output = True
                    for i, t in enumerate(results.test_results):
                        error_str += f"test input:\n{test_inputs[i]}\n" \
                                     f"expected output:\n{t.expected_output}\n" \
                                     f"code output:\n{t.actual_output}\n" \
                                     f"====================\n====================\n"

                        trace_str += f"trace:\n{self.render_trace(t.trace)}\n" \
                                     f"====================\n====================\n"

                        if get_settings().code_tester.calc_trace:
                            logger.debug(f"trace_str:\n{trace_str}")

                        # is_all_passed_public = actual_output == expected_output
                        is_all_passed_public = is_all_passed_public and t.passed
                        is_valid_output = is_valid_output and t.actual_output

                if is_all_passed_public:
                    logger.info(f"Passed public tests after {counter+1} attempts")
                    break

                counter += 1
                if counter > max_allowed_counter:
                    logger.error(f"Failed to pass public tests after {max_allowed_counter} attempts")
                    break

                if not is_valid_output:
                    logging.info("Failed to pass public tests. actual_output is empty")
                    recent_solution = problem['last_solution_code']
                    counter += 1
                    continue
                else:
                    # tests run. save the last solution
                    problem['last_solution_code'] = recent_solution

                # try to fix the solution
                problem['error_str'] = error_str
                if error_str:
                    logger.debug (f"error string:\n{error_str}")
                if get_settings().code_tester.use_trace:
                    problem['trace_str'] = trace_str
                else:
                    problem['trace_str'] = ''
                problem['possible_test_error'] = ''
                f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_fix_solution")
                response_fixed_code, _ = await retry_with_fallback_models(f)
                try:
                    response_fixed_code_yaml = yaml.safe_load(response_fixed_code)
                    recent_solution = response_fixed_code_yaml['new_solution_code']
                    problem['recent_solution'] = recent_solution
                    # result = remove_if_main(result)
                except yaml.YAMLError:
                    print(f"Failed to parse yaml: {response_fixed_code}")
                    # result = response_fixed_code

            if not is_all_passed_public:
                logger.error(f"Failed to pass public tests after {max_allowed_counter} attempts. exiting")
                if get_settings().get("solve.terminate_on_failure", False):
                    exit(-1)

            # # evaluate AI-generated tests
            # max_ai_tests = 10
            # for i, test_case in enumerate(problem['more_test_cases']):
            #     if i > max_ai_tests:
            #         break
            #     print(f"evaluating AI tests, test case remaining {len(problem['more_test_cases'])-i}")
            #     test_inputs, results = eval_solution(example=problem,
            #                                  prediction= remove_if_main(result),
            #                                  test_inputs=[test_case['input']],
            #                                  test_outputs=[test_case['output']],)
            #     if str(results.compilation_result.program_status) == 'ProgramStatus.kTimeout':
            #         print(f"timeout - took more than 3 seconds to run")
            #         print(f"test input: {test_case['input']}")
            #         print(f"expected output: {test_case['output']}")
            #         actual_output = 'timeout - took more than 3 seconds to run'
            #         expected_output = test_case['output']
            #         # is_passed_AI = False
            #         is_passed_AI = True # For now, we don't care about runtime
            #         break
            #     else:
            #         actual_output = results.test_results[0].actual_output
            #         expected_output = results.test_results[0].expected_output
            #         is_passed_AI = results.test_results[0].passed
            #
            #     if not is_passed_AI:
            #         problem['test_inputs'] = test_inputs
            #         problem['expected_output'] = expected_output
            #         problem['actual_output'] = actual_output
            #         problem['possible_test_error'] = 'true'
            #         if not actual_output:
            #             logging.info(f"Failed to generate. actual_output is empty")
            #             break
            #         f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_fix_solution")
            #         response_fixed_code, _ = await retry_with_fallback_models(f)
            #         try:
            #             response_fixed_code_yaml = yaml.safe_load(response_fixed_code)
            #             result = response_fixed_code_yaml['improved_code']
            #             # result = remove_if_main(result)
            #         except:
            #             logging.info(f"Failed to parse yaml: {response_fixed_code}")
            #             # result = response_fixed_code

        return recent_solution, is_all_passed_public

    def render_trace(self, trace_data):
        if not trace_data:
            return ''

        max_trace_lines = get_settings().code_tester.get("max_trace_lines")
        trace_lines = trace_data.split("\n")
        if max_trace_lines is not None and 0 < max_trace_lines < len(trace_lines):
            logger.debug(f"clipping trace from {len(trace_lines)} to {max_trace_lines}")
            half_lines = int(max_trace_lines / 2)
            trace_lines = (
                    trace_lines[:half_lines] +
                    [f".... {len(trace_lines) - max_trace_lines} omitted lines ...."] +
                    trace_lines[-half_lines:]
            )
        joined_lines = "\n".join(trace_lines)
        return joined_lines

    def solve_problem(self, example):
        problem = {k: example.get(k) for k in ["name", "description", 'public_tests']}
        prediction, passed_all_public = asyncio.run(self.run(problem=problem))
        logger.info("testing solution on private tests")
        logger.info(f"prediction:\n{prediction}")
        return prediction, passed_all_public


def solve_and_test(dataset_name, split_name=None, problem_name=None, evaluation_test_type=None, problem_number=None):
    # logger.info('solve_and_test')

    # load dataset
    data_provider = CodeContestDataProvider(dataset_location=dataset_name)
    if not problem_name and problem_number:
        problem_name = data_provider.dataset[split_name][int(problem_number)]['name']
        logger.info(f"problem_name: {problem_name}")
    problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name,
                                         evaluation_test_type=evaluation_test_type)

    # solve problem
    solver = CodeContestsCompetitor()
    solution, passed_all_public = solver.solve_problem(problem)

    # test solution
    test_results = None
    if evaluation_test_type and passed_all_public:
        test_results = eval_solution(evaluation_test_type=evaluation_test_type, example=problem, prediction=solution)

    return solution, test_results


if __name__ == "__main__":
        solve_and_test(dataset_name="deepmind/code_contests", split_name="valid",
                       #problem_name="1560_F1. Nearest Beautiful Number (easy version)",
                       problem_name="1548_D1. Gregor and the Odd Cows (Easy)",
                       evaluation_test_type="public_tests")


