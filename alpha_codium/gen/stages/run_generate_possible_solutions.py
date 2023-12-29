import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_generate_possible_solutions(self, problem):
    try:
        logger.info("--generate possible solutions stage--")
        use_recording = problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')

        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_generate_possible_solutions")
        if use_recording:
            response_possible_solutions = np.load(recording_path + 'possible_solutions.npy', allow_pickle=True) \
                .tolist()
            logger.info("Using recording")
            logger.debug(f"response_possible_solutions:\n{response_possible_solutions}")
        else:
            response_possible_solutions, _ = await retry_with_fallback_models(f)
            if do_recording:
                np.save(recording_path + 'possible_solutions.npy', response_possible_solutions)
        response_possible_solutions = response_possible_solutions.rstrip("` \n")
        response_possible_solutions_yaml = yaml.safe_load(response_possible_solutions)  # noqa

        problem['s_possible_solutions'] = response_possible_solutions_yaml['possible_solutions']
        problem['s_possible_solutions_str'] = response_possible_solutions.split('possible_solutions:')[1].strip()
        return problem
    except Exception as e:
        logging.error(f"Failed to generate possible solutions: {e}")
        raise e
