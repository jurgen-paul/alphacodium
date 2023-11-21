import asyncio
import copy
import functools
import logging
import os
import re
import copy
import numpy as np
import yaml
from jinja2 import Environment, StrictUndefined

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.code_test_runners import eval_solution
from alpha_codium.config_loader import get_settings
from alpha_codium.gen.stages.run_analyze_tests_failure import run_analyze_test_failure
from alpha_codium.gen.stages.run_baseline import run_baseline
from alpha_codium.gen.stages.run_choose_best_solution import run_choose_best_solution
from alpha_codium.gen.stages.run_evaluate_all_ai_tests import run_evaluate_all_ai_tests
from alpha_codium.gen.stages.run_evaluate_public_tests import run_evaluate_public_tests
from alpha_codium.gen.stages.run_fix_code_from_tests_failure import run_fix_code_from_tests_failure
from alpha_codium.gen.stages.run_generate_ai_test import run_generate_ai_tests
from alpha_codium.gen.stages.run_initial_solve import run_initial_solve
from alpha_codium.gen.stages.run_self_reflect import run_self_reflect
from alpha_codium.gen.stages.run_evaluate_a_simple_test import run_evaluate_a_simple_test
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.gen.stages.utils import set_configurations
from alpha_codium.gen.utils import evaluate_solution_on_subset
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
        temperature = self.prompt[prompt].temperature
        return sys_prompt, usr_prompt, temperature

    async def _run(self, model, problem, prompt:str = "code_contests_prompt_reflect"):
        system_prompt, user_prompt, temperature = self.render(problem, prompt)

        response, finish_reason = await self.ai_handler.chat_completion(
            model=model, system=system_prompt, user=user_prompt, temperature=temperature
        )
        return response, finish_reason

    async def run(self, problem, iteration=0):
        logger.info(f"Running code contests competitor, model {get_settings().config['model']}")

        try:
            # configurations
            problem = set_configurations(problem, iteration)

            # self-reflect
            problem = await run_self_reflect(self, problem)

            # choose best solution
            problem = await run_choose_best_solution(self, problem)

            # initial solve
            problem = await run_initial_solve(self, problem)

            # generate ai tests
            problem = await run_generate_ai_tests(self, problem)

            # run a simple test first
            problem = await run_evaluate_a_simple_test(self, problem)

            # evaluate on public tests
            problem = await run_evaluate_public_tests(self, problem)

            # evaluate on ai tests
            problem = await run_evaluate_all_ai_tests(self, problem)

            return problem['code_recent_solution']
        except Exception as e:
            logging.error(f"Error: {e}")
            return ""

    def solve_problem(self, example, iteration=0):
        problem = {k: example.get(k) for k in ["name", "description", 'public_tests']}
        prediction = asyncio.run(self.run(problem=problem, iteration=iteration))
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
    logger.info(f"problem['cf_tags']: {problem['cf_tags']}")

    # evaluate prev solutions
    evaluate_prev_solutions = get_settings().get("dataset.evaluate_prev_solutions", False)
    if evaluate_prev_solutions:
        try:
            if not problem['solutions']['solution']:
                logger.info("No public solutions for this problem")
            found_solution = False
            for index_published, sol_published in enumerate(problem['solutions']['solution']):
                logger.info(f"evaluating public solution {index_published} on private tests...")
                test_results, test_passed_private, test_failed_private, test_timeout_private\
                    = evaluate_solution_on_subset('private_tests', problem, sol_published, silent=True)
                logger.info(f"evaluating public solution {index_published} on generated tests...")
                test_results, test_passed_generate, test_failed_generate, test_timeout_generate = (
                    evaluate_solution_on_subset('generated_tests', problem, sol_published, silent=True))


                if (test_failed_private == test_failed_generate ==test_timeout_private == test_timeout_generate == 0) \
                        and test_passed_private + test_passed_generate> 0:
                    logger.info(f"sol_published index {index_published} passed all tests:\n{sol_published}")
                    found_solution = True
                    break

            if not found_solution:
                logger.info(f"None of the public solutions passed all tests")
        except Exception as e:
            logger.error(f"Error evaluating public solutions: {e}")
            pass

    # solve problem
    if not problem['private_tests']['input']:
        logger.info("No private tests for this problem")
        # return None, None

    solver = CodeContestsCompetitor()
    iteration = 0
    solution = solver.solve_problem(problem, iteration)
    logger.info(f"evaluating solution on private tests...")
    test_results, test_passed,test_failed_private, test_timeout_private  = evaluate_on_private_tests('private_tests', problem, solution, silent=True)

    logger.info(f"evaluating solution on generated tests...")
    test_results, test_passed, test_failed_generate , test_timeout_generate = evaluate_on_private_tests('generated_tests', problem, solution, silent=True)

    logger.info(f"\ntest_failed_generate: {test_failed_generate}, test_failed_private: {test_failed_private}\n"
                f"test_timeout_generate: {test_timeout_generate}, test_timeout_private: {test_timeout_private}")

    return solution, test_results


if __name__ == "__main__":
        solve_and_test(dataset_name="deepmind/code_contests", split_name="valid",
                       #problem_name="1560_F1. Nearest Beautiful Number (easy version)",
                       problem_name="1548_D1. Gregor and the Odd Cows (Easy)",
                       evaluation_test_type="public_tests")

def evaluate_on_private_tests(evaluation_test_type, problem, solution, silent=True):
    # evaluate solution
    test_results = None
    if evaluation_test_type:
        test_results = eval_solution(evaluation_test_type=evaluation_test_type, example=problem, prediction=solution, silent=silent)

    test_passed = 0
    test_failed = 0
    test_timeout = 0

    if not test_results[1]:
        logger.info("No tests were run")
        return test_results, 0, 0

    for test in test_results[1].test_results:
        if test.program_status.name=='kTimeout':
            test_timeout += 1
        elif not test.passed:
            test_failed += 1
        else:
            test_passed += 1


    logger.info("=====================================")
    logger.info(f"test_passed: {test_passed}, test_failed: {test_failed}, test_timeout: {test_timeout}")
    logger.info("=====================================")

    return test_results, test_passed, test_failed, test_timeout

