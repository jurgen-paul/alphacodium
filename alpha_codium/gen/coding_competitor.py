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
from alpha_codium.gen.stages.run_analyze_tests_failure import run_analyze_test_failure
from alpha_codium.gen.stages.run_baseline import run_baseline
from alpha_codium.gen.stages.run_choose_best_solution import run_choose_best_solution
from alpha_codium.gen.stages.run_fix_code_from_tests_failure import run_fix_code_from_tests_failure
from alpha_codium.gen.stages.run_initial_solve import run_initial_solve
from alpha_codium.gen.stages.run_self_reflect import run_self_reflect
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.gen.stages.utils import set_configurations
from alpha_codium.llm.ai_handler import AiHandler
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
        problem = set_configurations(problem)


        if get_settings().get("solve.use_baseline", False):
            problem['code_recent_solution'] = await run_baseline(self, problem)
        else:
            # self-reflect
            problem = await run_self_reflect(self, problem)

            # choose best solution
            problem = await run_choose_best_solution(self, problem)

            # initial solve
            problem = await run_initial_solve(self, problem)

            # evaluate public tests
            logger.info("--iterate on public tests stage--")
            test_inputs_all = problem['public_tests']['input']
            test_outputs_all = problem['public_tests']['output']
            load_final_code_solution = problem.get('load_final_code_solution', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')
            if load_final_code_solution and not do_recording:
                problem['code_recent_solution'] = np.load(recording_path + 'code_recent_solution.npy', allow_pickle=True) \
                    .tolist()
                logger.info("Using recording")
                logger.debug(f"code_recent_solution:\n{problem['code_recent_solution']}")

            for test_inputs, test_outputs in zip(test_inputs_all, test_outputs_all):
                logger.info(f"test_inputs:\n{test_inputs}")
                if not isinstance(test_inputs, list):
                    test_inputs = [test_inputs]
                    test_outputs = [test_outputs]
                counter = 0
                max_allowed_counter = 4
                all_passed = False
                last_error_str = None
                problem['diff_that_didnt_help'] = ''
                while not all_passed:
                    # run the solution on the tests
                    problem, all_passed, non_empty_output, error_str, trace_str, tests_timeout \
                        = run_tests(self, problem, counter, test_inputs, test_outputs)

                    # if last fix didn't change anything, revert to last solution, and add the patch to prompt
                    if last_error_str == error_str:
                        logger.error("error string did not change.")
                        # problem['code_recent_solution'] = problem['code_prev_solution']
                        problem['diff_that_didnt_help'] = self.clip_string(problem['diff_patch'], get_settings().code_tester.get("max_trace_lines"))
                    last_error_str = error_str

                    # analyze the tests results
                    counter += 1
                    logger.info(f"counter: {counter}")
                    if all_passed:
                        logger.info(f"Passed public tests after {counter} attempts")
                        break
                    elif tests_timeout:
                        logger.error("timeout (no output). reverting to last solution")
                        problem['code_recent_solution'] = problem['code_last_solution']
                        continue
                    elif counter > max_allowed_counter:
                        logger.error(f"Failed to pass public tests after {max_allowed_counter} attempts")
                        break
                    elif not non_empty_output:
                        logging.info("Failed to pass public tests. actual_output is empty")
                        problem['recent_solution'] = problem['last_solution_code']
                        continue
                    else:
                        # tests run. save the last solution
                        problem['code_prev_solution'] = problem['code_recent_solution']

                    # run 'fix code from tests failure' stage

                    problem = await run_analyze_test_failure(self, problem, error_str, trace_str, counter)

                    problem = await run_fix_code_from_tests_failure(self, problem, error_str, trace_str)

                if not all_passed:
                    logger.error(f"Failed to pass public tests after {max_allowed_counter} attempts. exiting")
                    exit(-1)

            # if we reached here, we passed the public tests
            logger.info("Passed public tests")
            if do_recording:
                np.save(recording_path + 'code_recent_solution.npy', problem['code_recent_solution'])

        return problem['code_recent_solution']

    def clip_string(self, s: str, max_lines: int = None):
        lines = s.split("\n")
        if max_lines is not None and 0 < max_lines < len(lines):
            logger.debug(f"clipping string from {len(lines)} to {max_lines}")
            half_lines = int(max_lines / 2)
            lines = (
                    lines[:half_lines] +
                    [f"\n.... {len(lines) - max_lines} omitted lines ....\n"] +
                    lines[-half_lines:]
            )
            return "\n".join(lines)
        else:
            return s
    def render_trace(self, trace_data):
        if not trace_data:
            return ''

        max_trace_lines = get_settings().code_tester.get("max_trace_lines")
        trace_data = self.clip_string(trace_data, max_trace_lines)
        return trace_data

    def solve_problem(self, example):
        problem = {k: example.get(k) for k in ["name", "description", 'public_tests']}
        prediction = asyncio.run(self.run(problem=problem))
        logger.info(f"testing solution on private tests with prediction:\n{prediction}")
        return prediction


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
    solution = solver.solve_problem(problem)

    # test solution
    test_results = None
    if evaluation_test_type:
        test_results = eval_solution(evaluation_test_type=evaluation_test_type, example=problem, prediction=solution)

    test_passed = 0
    test_failed = 0
    for test in test_results[1].test_results:
        if not test.passed:
            test_failed += 1
        else:
            test_passed += 1
    logger.info("=====================================")
    logger.info(f"test_passed: {test_passed}, test_failed: {test_failed}")
    logger.info("=====================================")
    return solution, test_results


if __name__ == "__main__":
        solve_and_test(dataset_name="deepmind/code_contests", split_name="valid",
                       #problem_name="1560_F1. Nearest Beautiful Number (easy version)",
                       problem_name="1548_D1. Gregor and the Odd Cows (Easy)",
                       evaluation_test_type="public_tests")


