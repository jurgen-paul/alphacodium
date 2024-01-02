import copy
import functools
import logging
import numpy as np
import yaml
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_choose_best_solution(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--choose best solution stage--")

            # get settings
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_choose_best_solution")

            # inference
            response_best_solution, _ = await send_inference(f)
            response_best_solution = response_best_solution.rstrip("` \n")
            response_best_solution_yaml = yaml.safe_load(response_best_solution)  # noqa

            # update best solution
            problem['s_best_solution'] = response_best_solution
            problem['s_other_solutions'] = []
            for solution in problem['s_possible_solutions']:
                if solution['name'] != response_best_solution_yaml['name']:
                    problem['s_other_solutions'].append(solution)

            return problem
        except Exception as e:
            logging.error(f"'run_choose_best_solution' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
