import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.gen.stages.run_analyze_tests_failure import run_analyze_test_failure
from alpha_codium.gen.stages.run_fix_code_from_tests_failure import run_fix_code_from_tests_failure
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_evaluate_all_ai_tests(self, problem):
    try:
        logger.info("--iterate on all ai tests stage--")

        use_recording = problem.get('use_recording', False)
        if get_settings().solve.disable_recording_public_tests:
            use_recording = False
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')

        # evaluate ai tests
        ai_tests = problem['problem_ai_tests']
        for test in ai_tests:
            test_inputs = test['input']
            test_outputs = test['output']
            if not isinstance(test_inputs, list):
                test_inputs = [test_inputs]
                test_outputs = [test_outputs]
            counter = 0
            # run the solution on the tests
            problem, test_passed, non_empty_output, error_str, trace_str, tests_timeout \
                = run_tests(self, problem, counter, test_inputs, test_outputs)

            # we passed without changing the code. Add the test to the passed tests list
            if test_passed:
                if test_inputs not in problem['passed_tests']['inputs']:
                    logger.info(f"Passed ai tests without code fixing. adding to passed tests list")
                    problem['passed_tests']['inputs'] += test_inputs
                    problem['passed_tests']['outputs'] += test_outputs

            else:
                if get_settings().solve.disable_recording_public_tests:
                    logger.error(f"Failed to pass ai tests. moving on")
                    continue
                logger.error(f"Failed to pass ai tests. trying to fix code")
                last_code_solution = copy.deepcopy(problem['code_recent_solution'])

                # run 'analyze_test_failure' stage
                problem = await run_analyze_test_failure(self, problem, error_str, trace_str, counter)

                # run 'fix_code_from_tests_failure' stage
                problem = await run_fix_code_from_tests_failure(self, problem, error_str, trace_str)

                problem, test_passed, non_empty_output, error_str, trace_str, tests_timeout \
                    = run_tests(self, problem, counter, test_inputs, test_outputs)

                if not test_passed:
                    logger.error(f"Failed to pass ai tests after trying to fix code. reverting to last solution")
                    problem['code_recent_solution'] = last_code_solution
                else:
                    # running passed tests again to make sure we didn't break anything
                    if problem['passed_tests']['inputs']:
                        problem, all_passed_prev, non_empty_output, error_str, trace_str, tests_timeout \
                            = run_tests(self, problem, counter,
                                        problem['passed_tests']['inputs'],
                                        problem['passed_tests']['outputs'])
                        if not all_passed_prev:
                            logger.error(f"The fix broke prev passed tests. reverting to last solution")
                            problem['code_recent_solution'] = last_code_solution
                            continue

                    logger.info(f"Passed all ai tests after trying to fix code. using new solution")
                    if test_inputs not in problem['passed_tests']['inputs']:
                        problem['passed_tests']['inputs'] += test_inputs
                        problem['passed_tests']['outputs'] += test_outputs



        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
