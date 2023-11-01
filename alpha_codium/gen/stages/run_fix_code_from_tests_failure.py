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
        problem['error_str'] = error_str
        if error_str:
            logger.debug(f"error string:\n{error_str}")
        if get_settings().code_tester.use_trace:
            problem['trace_str'] = trace_str
        else:
            problem['trace_str'] = ''
        problem['possible_test_error'] = ''
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_fix_solution")
        response_fixed_code, _ = await retry_with_fallback_models(f)
        try:
            response_fixed_code_yaml = yaml.safe_load(response_fixed_code)
            recent_solution = response_fixed_code_yaml['new_solution_code']
            problem['code_recent_solution'] = recent_solution
            # result = remove_if_main(result)
        except yaml.YAMLError:
            logger.error(f"Failed to parse yaml:\n{response_fixed_code}")
            # result = response_fixed_code
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)

    return problem
