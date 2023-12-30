import functools
import logging
import numpy as np
import yaml
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_initial_solve(self, problem, enable_record=True):
    counter_retry = 0
    while True:
        try:
            logger.info("--initial solve stage--")
            use_recording =problem.get('use_recording', False) and enable_record
            do_recording = problem.get('do_recording', False) and enable_record
            recording_path = problem.get('recording_path', '')

            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_solve")
            if use_recording:
                response_solve = np.load(recording_path + 'initial_solve.npy', allow_pickle=True).tolist()
                logger.info("Using recording")
                logger.debug(f"response_solve:\n{response_solve}")
            else:
                response_solve, _ = await send_inference(f)
                if do_recording:
                    np.save(recording_path + 'initial_solve.npy', response_solve)

            # clean up the response
            response_solve = response_solve.rstrip("` \n")
            if response_solve.startswith("```python"):
                response_solve = response_solve[10:]
            elif response_solve.startswith("python"):
                response_solve = response_solve[6:]

            # save the response
            problem['code_recent_solution'] = response_solve
            problem['code_prev_solution'] = response_solve
            return problem
        except Exception as e:
            logging.error(f"'initial solve' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
