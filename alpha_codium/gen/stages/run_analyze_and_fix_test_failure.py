import ast
import difflib
import functools
import logging
import numpy as np
import yaml

from alpha_codium.code_contests.eval.code_test_runners import eval_solution
from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_analyze_and_fix_test_failure(self, problem, error_str):
    try:
        problem['error_str'] = error_str
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_analyze_and_fix")
        response_analyze_failure, _ = await retry_with_fallback_models(f)
        try:
            response_analyze_failure = response_analyze_failure.rstrip("'` \n") # remove trailing spaces and newlines from yaml response
            response_analyze_failure_yaml = yaml.safe_load(response_analyze_failure)
            problem['response_analyze_failure'] = response_analyze_failure
            code_recent_solution = response_analyze_failure_yaml['fixed_code'].rstrip("'` \n")
            if code_recent_solution .startswith("```python"):
                code_recent_solution= code_recent_solution[10:]
            elif code_recent_solution.startswith("python"):
                code_recent_solution = code_recent_solution[6:]
            try:
                ast.parse(code_recent_solution)
            except:
                code_recent_solution_fallback = '\n'.join(code_recent_solution.splitlines()[:-1]).rstrip("'` \n")
                try:
                    ast.parse(code_recent_solution_fallback)
                    code_recent_solution = code_recent_solution_fallback
                except:
                    logger.error(f"Invalid code:\n{code_recent_solution}")
                    return problem
            problem['code_recent_solution'] = code_recent_solution

            diff = difflib.unified_diff(problem['code_prev_solution'].splitlines(keepends=True),
                                        problem['code_recent_solution'].splitlines(keepends=True))
            patch = ''.join(diff)
            # problem['diff_patch'] = patch
            problem['specific_test_explanation'] = ''
            # problem['passed_tests_str'] = ''
            if get_settings().solve.reduce_verbose:
                logger.debug(f"diff:\n{patch}")
            else:
                logger.info(f"diff:\n{patch}")

        except yaml.YAMLError:
            logger.error(f"Failed to parse yaml:\n{response_analyze_failure}")
            # result = response_fixed_code
    except Exception as e:
        logging.error(f"Error: {e}")
        # exit(-1)

    return problem
