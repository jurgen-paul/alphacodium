import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_generate_possible_solutions(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--generate possible solutions stage--")

            # get settings
            problem['max_num_of_possible_solutions'] = get_settings().get('possible_solutions.max_num_of_possible_solutions')
            problem['use_test_explanations_possible_solutions'] = get_settings().get('possible_solutions.use_test_explanations')
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_generate_possible_solutions")

            # inference
            response_possible_solutions, _ = await send_inference(f)
            response_possible_solutions_yaml = yaml.safe_load(response_possible_solutions)
            problem['s_possible_solutions'] = response_possible_solutions_yaml['possible_solutions']
            problem['s_possible_solutions_str'] = response_possible_solutions.split('possible_solutions:')[1].strip()

            return problem
        except Exception as e:
            logging.error(f"'possible solutions' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
