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


async def run_fix_code_from_tests_failure(self, problem,error_str, trace_str):
    try:
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_fix_solution")
        response_fixed_code, _ = await retry_with_fallback_models(f)
        try:
            response_fixed_code = response_fixed_code.rstrip("'` \n") # remove trailing spaces and newlines from yaml response
            if response_fixed_code.startswith("```python"):
                response_fixed_code = response_fixed_code[10:]

            problem['code_recent_solution'] = response_fixed_code
            diff = difflib.unified_diff(problem['code_prev_solution'].splitlines(keepends=True),
                                        response_fixed_code.splitlines(keepends=True))
            patch = ''.join(diff)
            problem['diff_patch'] = patch
            problem['diff_that_didnt_help'] = ''
            logger.info(f"diff:\n{patch}")

            # result = remove_if_main(result)
        except yaml.YAMLError:
            logger.error(f"Failed to parse yaml:\n{response_fixed_code}")
            # result = response_fixed_code
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)

    return problem
