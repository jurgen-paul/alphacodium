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


async def run_evaluate_public_tests(self, problem):
    try:
        logger.info("--iterate on public tests stage--")

        use_recording = problem.get('use_recording', False)
        if get_settings().solve.disable_recording_public_tests:
            use_recording = False
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')
        MAX_ALLOWED_COUNTER = get_settings().solve.get("max_counter_public_tests", 3)

        if use_recording:
            code_recent_solution = np.load(recording_path + 'problem_run_public_tests.npy', allow_pickle=True).tolist()
            problem['code_recent_solution'] = code_recent_solution
            logger.info("Using recording public tests")
            logger.debug(f"code_recent_solution:\n{code_recent_solution}")
        else:

            # evaluate public tests
            test_inputs_all = problem['public_tests']['input']
            test_outputs_all = problem['public_tests']['output']
            all_passed_public = True
            for test_inputs, test_outputs in zip(test_inputs_all, test_outputs_all):
                if not isinstance(test_inputs, list):
                    test_inputs = [test_inputs]
                    test_outputs = [test_outputs]
                counter = 0
                passed_specific_test = False
                last_error_str = None
                problem['diff_that_didnt_help'] = ''

                # loop to fix specific test
                last_code_solution = copy.deepcopy(problem['code_recent_solution'])
                while not passed_specific_test:

                    # run the solution on the tests
                    problem, passed_specific_test, non_empty_output, error_str, trace_str, tests_timeout \
                        = run_tests(self, problem, counter, test_inputs, test_outputs)

                    # analyze the tests results
                    counter += 1
                    logger.info(f"counter: {counter}")
                    if passed_specific_test:
                        logger.info(f"Passed public tests after {counter-1} attempts")
                        if test_inputs not in problem['passed_tests']['inputs']:
                            problem['passed_tests']['inputs'] += test_inputs
                            problem['passed_tests']['outputs'] += test_outputs
                        break
                    elif tests_timeout:
                        logger.error("timeout (no output). reverting to last solution")
                        problem['code_recent_solution'] = problem['code_last_solution']
                        continue
                    elif counter > MAX_ALLOWED_COUNTER:
                        logger.error(f"Failed to pass public tests after {MAX_ALLOWED_COUNTER} attempts")
                        break
                    elif not non_empty_output:
                        logging.info("Failed to pass public tests. actual_output is empty")
                        problem['recent_solution'] = problem['last_solution_code']
                        continue
                    else:
                        # tests run. save the last solution
                        problem['code_prev_solution'] = problem['code_recent_solution']

                    # if last fix didn't change anything, add the diff patch to prompt
                    if (not passed_specific_test) and (last_error_str == error_str):
                        logger.error("error string did not change.")
                        problem['diff_that_didnt_help'] = self.clip_string(problem['diff_patch'], get_settings().code_tester.get("max_trace_lines"))
                    else:
                        problem['diff_that_didnt_help'] = ''
                    last_error_str = error_str

                    # run 'analyze_test_failure' stage
                    problem = await run_analyze_test_failure(self, problem, error_str, trace_str, counter)

                    # run 'fix_code_from_tests_failure' stage
                    problem = await run_fix_code_from_tests_failure(self, problem, error_str, trace_str)

                    # evaluate previous tests that passed. if they fail, revert to last solution
                    if problem['passed_tests']['inputs']:
                        problem, passed_prev_test, non_empty_output, error_str, trace_str, tests_timeout \
                            = run_tests(self, problem, counter,
                                        problem['passed_tests']['inputs'],
                                        problem['passed_tests']['outputs'])
                        if not passed_prev_test:
                            logger.error(f"The fix broke prev passed tests. reverting to last solution")
                            problem['code_recent_solution'] = last_code_solution
                            continue

                if not passed_specific_test and get_settings().solve.revert_to_last_solution_on_failure:
                    logger.error('Public test - reverting to initial solution')
                    problem['code_recent_solution'] = last_code_solution
                all_passed_public = all_passed_public and passed_specific_test

            if all_passed_public:
                logger.info(f"==================")
                logger.info(f"Passed all public tests")
                logger.info(f"==================")
            else:
                logger.info(f"==================")
                logger.info(f"Failed to pass all public tests")
                logger.info(f"==================")
            if do_recording:
                np.save(recording_path + 'problem_run_public_tests.npy', problem['code_recent_solution'])

        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
