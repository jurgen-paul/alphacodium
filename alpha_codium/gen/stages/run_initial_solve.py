import ast
import functools
import logging
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger
from alpha_codium.settings.config_loader import get_settings

logger = get_logger(__name__)


async def run_initial_solve(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--initial solve stage--")

            f = functools.partial(self._run, problem=problem, prompt=choose_prompt())
            response_solve, _ = await send_inference(f)

            # clean up the response
            response_solve = response_solve.rstrip("` \n").strip()
            if response_solve.startswith("```python"):
                response_solve2 = response_solve[10:]
                try:
                    ast.parse(response_solve2)
                    response_solve = response_solve2
                except:
                    pass
            if response_solve.startswith("python"):
                response_solve2 = response_solve[6:]
                try:
                    ast.parse(response_solve2)
                    response_solve = response_solve2
                except:
                    pass
            if '```python' in response_solve: # somehwhere in the middle there is a code block
                logger.info("try taking only the code part from the response")
                response_solve2 = response_solve.split('```python')[1].strip()
                if '```' in response_solve2:
                    response_solve2 = response_solve2.split('```')[0].strip()
                    try:
                        ast.parse(response_solve2)
                        response_solve = response_solve2
                        logger.info(f"response_solve: {response_solve}")
                    except Exception as e:
                        logger.error(f"Error while parsing the response: {e}")
                        pass

            # save the response
            problem['code_recent_solution'] = response_solve
            problem['code_prev_solution'] = response_solve
            return problem
        except Exception as e:
            logging.error(f"'initial solve' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e

def choose_prompt():
    if get_settings().get("solve.use_direct_solutions", False):
        return "code_contests_prompts_solve_direct"
    else:
        return "code_contests_prompts_solve"