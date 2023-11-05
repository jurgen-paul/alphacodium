import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.gen.stages.run_analyze_tests_failure import run_analyze_test_failure
from alpha_codium.gen.stages.run_fix_code_from_tests_failure import run_fix_code_from_tests_failure
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_simple_test(self, problem):
    try:
        logger.info("--run simple test stage--")
        use_recording =problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')
        MAX_COUNTER = 2
        if use_recording:
            code_recent_solution = np.load(recording_path + 'problem_run_simple_test.npy', allow_pickle=True).tolist()
            problem['code_recent_solution'] = code_recent_solution
            logger.info("Using recording simple test")
            logger.debug(f"code_recent_solution:\n{code_recent_solution}")
        else:
            counter = 0
            last_code_solution = copy.deepcopy(problem['code_recent_solution'])
            test_input = [problem['problem_ai_simple_test']['input']]
            test_output = [problem['problem_ai_simple_test']['output']]

            # run the solution on the simple test
            problem, passed_simple_test, non_empty_output, error_str, trace_str, tests_timeout \
                = run_tests(self, problem, counter, test_input, test_output)

            while not passed_simple_test:
                counter += 1
                if counter > MAX_COUNTER:
                    logger.error(f"Failed to pass simple tests after {counter - 1} attempts. exiting")
                    break
                logger.error(f"Failed to pass simple tests. trying to fix code")
                problem['diff_that_didnt_help'] = ''
                problem = await run_analyze_test_failure(self, problem, error_str, trace_str, counter)

                problem = await run_fix_code_from_tests_failure(self, problem, error_str, trace_str)

                problem, passed_simple_test, non_empty_output, error_str, trace_str, tests_timeout \
                    = run_tests(self, problem, counter, test_input, test_output)

            if not passed_simple_test:
                problem['code_recent_solution'] = last_code_solution

            if do_recording:
                np.save(recording_path + 'problem_run_simple_test.npy', problem['code_recent_solution'])

        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
