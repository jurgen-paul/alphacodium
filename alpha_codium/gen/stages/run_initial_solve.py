import functools
import logging
import numpy as np
import yaml
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_initial_solve(self, problem):
    try:
        logger.info("--solve stage--")
        use_recording =problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')

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
        problem['code_initial_solution'] = response_solve
        problem['code_recent_solution'] = response_solve
        problem['code_prev_solution'] = response_solve
        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
