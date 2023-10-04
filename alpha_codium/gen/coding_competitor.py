import asyncio
import functools

from jinja2 import Environment, StrictUndefined
import json
from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.code_contests_eval import TestsRunner
from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_handler import AiHandler
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.llm.token_handler import TokenHandler


class CodeContestsCompetitor:
    def __init__(self):
        self.prompt = get_settings().code_contests_prompt
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

    async def run(self, problem):
        problem = {k: problem.get(k) for k in ["name", "description", "public_tests"]}
        f = functools.partial(self._run, problem=problem)
        response, _ = await retry_with_fallback_models(f)
        if response:
            response = str(response)
            if response.endswith("stop"):
                return response[:-4]
        return response


def solve_and_test_problem(dataset_name, split_name='train', problem_name=None, evaluation_test_type='private_tests'):
    p = CodeContestsCompetitor()
    cc = CodeContestDataProvider(dataset_location=dataset_name)
    ds = cc.dataset[split_name]
    example = None
    if not problem_name:
        for e in ds:
            tests = e.get(evaluation_test_type)
            if tests and tests.get("input"):
                example = e
                break
    else:
        problems = ds.filter(lambda example: example['name'] == problem_name)
        if problems:
            example = problems[0]
        else:
            raise ValueError(
                f"problem with name {problem_name} doesn't exist in dataset {dataset_name} in split {split_name}")

    problem = {k: example.get(k) for k in ["name", "description", evaluation_test_type]}
    prediction = asyncio.run(p.run(problem=problem))
    test_inputs = example.get(evaluation_test_type).get("input") if example.get(evaluation_test_type) else None
    test_outputs = example.get(evaluation_test_type).get("output") if example.get(evaluation_test_type) else None
    if test_inputs and test_outputs:
        test_runner = TestsRunner(
            path_to_python_bin="/usr/bin/python3.11",
            path_to_python_lib=["/usr/lib64", "/usr/lib64/python3.11"],
            num_threads=4,
            stop_on_first_failure=True,
        )
        _, _, results = test_runner.run_test(
            test_id=example["name"],
            candidate_id="id",
            candidate=prediction,
            test_inputs=test_inputs,
            tests_outputs=test_outputs,
        )
        test_case_results = [test_result.passed for test_result in results.test_results]
        candidate_pass_fail = all(test_case_results)

        print(f"Competitor solution:\n {prediction}")
        print("=====================================")
        print(f"Final pass/fail: {candidate_pass_fail}")
        print(f"Individual test results: {test_case_results}")

    else:
        print(f"The problem didn't have tests of type {evaluation_test_type}")



if __name__ == "__main__":
    solve_and_test_problem("assaf_test")
