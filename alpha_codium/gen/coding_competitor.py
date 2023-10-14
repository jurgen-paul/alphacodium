import asyncio
import functools
import logging

import yaml
from jinja2 import Environment, StrictUndefined

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.code_test_runners import eval_solution

from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_handler import AiHandler
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.llm.token_handler import TokenHandler

import re


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
        result = None
        problem = {k: problem.get(k) for k in ["name", "description", "public_tests"]}
        use_baseline = False
        if use_baseline:
            logging.info("Using baseline")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_baseline")
            response_baseline, _ = await retry_with_fallback_models(f)
            if response_baseline:
                result = self.postprocess_response(response_baseline)
        else:

            # reflect
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_reflect")
            response_reflect, _ = await retry_with_fallback_models(f)
            response_reflect = response_reflect.rstrip("` \n")
            try:
                response_reflect_yaml = yaml.safe_load(response_reflect)
            except:
                try:
                    response_reflect = self.postprocess_response(response_reflect) # try to include only the yaml part
                    response_reflect_yaml = yaml.safe_load(response_reflect)
                except:
                    logging.info(f"Failed to parse yaml: {response_reflect}")
                    response_reflect_yaml = {'self_description': response_reflect}
            problem['response_reflect'] = response_reflect
            problem['self_description'] = response_reflect_yaml['self_description']

            # generate more test cases
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_more_test_cases")
            response_more_cases, _ = await retry_with_fallback_models(f)
            response_more_cases = response_more_cases.rstrip("` \n")
            response_more_cases_yaml = yaml.safe_load(response_more_cases)
            problem['response_more_cases'] = response_more_cases
            problem['more_test_cases'] = response_more_cases_yaml['test_cases']


            # possible solutions
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_possible_solutions")
            response_possible_solutions, _ = await retry_with_fallback_models(f)
            response_possible_solutions = response_possible_solutions.rstrip("` \n")
            problem['response_possible_solutions'] = response_possible_solutions
            response_possible_solutions_yaml = yaml.safe_load(response_possible_solutions)
            best_solution_name = response_possible_solutions_yaml['best_solution']['name']
            for solution in response_possible_solutions_yaml['possible_solutions']:
                if solution['name'] == best_solution_name:
                    problem['best_solution'] = solution
                    break

            # solve
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_solve")
            response_solve = await retry_with_fallback_models(f)
            if isinstance(response_solve, tuple): # (code, 'stop')
                response_solve, _ = response_solve
            response_solve = response_solve.rstrip("` \n")
            problem['response_solve'] = response_solve
            result = response_solve


        # remove the if __name__ == '__main__' part. python eval fails to generate output with it
        result = remove_if_main(result)
        return result

    def solve_problem(self, example):
        problem = {k: example.get(k) for k in ["name", "description", 'public_tests']}
        prediction = asyncio.run(self.run(problem=problem))
        return prediction


def solve_and_test(dataset_name, split_name=None, problem_name=None, evaluation_test_type=None):
    data_provider = CodeContestDataProvider(dataset_location=dataset_name)
    problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name,
                                         evaluation_test_type=evaluation_test_type)
    solver = CodeContestsCompetitor()
    solution = solver.solve_problem(problem)
    test_results = None
    if evaluation_test_type:
        test_results = eval_solution(evaluation_test_type=evaluation_test_type, example=problem, prediction=solution)
    return solution, test_results

def remove_if_main(result: str):
    if 'if __name__ ==' in result:
        result_lines = result.split('\n')
        start_dedent = False
        for i, line in enumerate(result_lines):
            if 'if __name__ ==' in line:
                start_dedent = True
                result_lines[i] = ''
            if start_dedent:
                result_lines[i] = result_lines[i][4:]
        result = '\n'.join(result_lines)
    return result

if __name__ == "__main__":
    solve_and_test("assaf_test", "train")
