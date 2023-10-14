import asyncio
import functools

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
        self.prompt = get_settings().code_contests_prompt_baseline
        self.ai_handler = AiHandler()
        self.token_handler = TokenHandler(
            None, None, self.prompt.system, self.prompt.user
        )

    def render(self, problem_json):
        environment = Environment(undefined=StrictUndefined)
        environment.globals["zip"] = zip
        environment.globals["enumerate"] = enumerate
        sys_prompt = environment.from_string(self.prompt.system).render(problem_json)
        usr_prompt = environment.from_string(self.prompt.user).render(problem_json)
        return sys_prompt, usr_prompt

    async def _run(self, model, problem):
        system_prompt, user_prompt = self.render(problem)

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
        f = functools.partial(self._run, problem=problem)
        response, _ = await retry_with_fallback_models(f)
        if response:
            result = self.postprocess_response(response)
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


if __name__ == "__main__":
    solve_and_test("assaf_test", "train")
