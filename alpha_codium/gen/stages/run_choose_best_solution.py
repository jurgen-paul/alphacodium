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
            use_recording =problem.get('use_recording', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')

            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_choose_best_solution")
            if use_recording:
                response_best_solution = np.load(recording_path + 'best_solution.npy', allow_pickle=True) \
                    .tolist()
                logger.info("Using recording")
                logger.debug(f"response_best_solution:\n{response_best_solution}")
            else:
                response_best_solution, _ = await send_inference(f)
                if do_recording:
                    np.save(recording_path + 'best_solution.npy', response_best_solution)
            response_best_solution = response_best_solution.rstrip("` \n")
            response_best_solution_yaml = yaml.safe_load(response_best_solution)  # noqa
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
