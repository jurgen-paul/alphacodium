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

            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_solve")
            response_solve, _ = await send_inference(f)

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
