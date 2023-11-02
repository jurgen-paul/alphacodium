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


async def run_analyze_test_failure(self, problem,error_str, trace_str, counter):
    try:
        problem['error_str'] = error_str
        if get_settings().code_tester.use_trace and counter % 2 == 0:
            logger.info(f"Using trace_str")
            problem['trace_str'] = trace_str
        else:
            problem['trace_str'] = ''
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_analyze_trace")
        response_analyze_failure, _ = await retry_with_fallback_models(f)
        try:
            response_analyze_failure = response_analyze_failure.rstrip("'` \n") # remove trailing spaces and newlines from yaml response
            response_analyze_failure_yaml = yaml.safe_load(response_analyze_failure)
            problem['response_analyze_failure'] = response_analyze_failure
            problem['what_went_wrong'] = response_analyze_failure_yaml['what_went_wrong']
            problem['input_output_fixed_flow'] = response_analyze_failure_yaml['input_output_fixed_flow']
        except yaml.YAMLError:
            logger.error(f"Failed to parse yaml:\n{response_analyze_failure}")
            # result = response_fixed_code
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)

    return problem
