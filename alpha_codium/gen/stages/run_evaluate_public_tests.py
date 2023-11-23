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
    counter_retry = 0
    while True:
        try:
            logger.info("--iterate on public tests stage--")

            use_recording = problem.get('use_recording', False)
            if get_settings().public_tests.disable_recording_public_tests:
                use_recording = False
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')
            MAX_ALLOWED_FIXES_COUNTER = get_settings().public_tests.get("max_counter_public_tests", 3)

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
                max_allowed_calls = get_settings().get("public_tests.max_allowed_calls", 10)
                actual_number_of_calls = 0
                problem['present_short_description'] = False
                for test_inputs, test_outputs in zip(test_inputs_all, test_outputs_all):
                    if not isinstance(test_inputs, list):
                        test_inputs = [test_inputs]
                        test_outputs = [test_outputs]
                    counter = 0
                    passed_specific_test = False

                    # loop to fix specific test
                    last_code_solution = copy.deepcopy(problem['code_recent_solution'])
                    best_solution = copy.deepcopy(problem['code_recent_solution'])
                    best_d = float('inf')
                    while not passed_specific_test:
                        if counter > 2:
                            problem['present_short_description'] = True
                        else:
                            problem['present_short_description'] = False
                        # run the solution on the tests
                        problem, passed_specific_test, non_empty_output, error_str, trace_str, tests_timeout, d_tot \
                            = run_tests(self, problem, counter, test_inputs, test_outputs)

                        # save the best solution so far
                        if -1 < d_tot < best_d:
                            best_solution = copy.deepcopy(problem['code_recent_solution'])
                            best_d = d_tot

                        # cap the number of calls to the ai
                        if not passed_specific_test and actual_number_of_calls >= max_allowed_calls:
                            logger.error(f"Failed to pass public test. reached max number of calls")
                            break

                        # analyze the tests results
                        counter += 1
                        logger.info(f"counter: {counter}")
                        if passed_specific_test:
                            logger.info(f"Passed public tests after {counter-1} attempts")
                            if test_inputs not in problem['passed_tests']['inputs']:
                                problem['passed_tests']['inputs'] += test_inputs
                                problem['passed_tests']['outputs'] += test_outputs
                            break
                        # elif tests_timeout and 'code_last_solution' in problem:
                        #     logger.error("timeout (no output). reverting to last solution")
                        #     problem['code_recent_solution'] = problem['code_last_solution']
                        #     continue
                        elif counter > MAX_ALLOWED_FIXES_COUNTER:
                            logger.error(f"Failed to pass public tests after {MAX_ALLOWED_FIXES_COUNTER} attempts")
                            break
                        elif not non_empty_output:
                            logging.info("Failed to pass public tests. actual_output is empty")
                            problem['code_recent_solution'] = last_code_solution
                            continue
                        else:
                            # tests run. save the last solution
                            problem['code_prev_solution'] = problem['code_recent_solution']


                        # run 'analyze_test_failure' stage
                        problem = await run_analyze_test_failure(self, problem, error_str)

                        # run 'fix_code_from_tests_failure' stage
                        problem = await run_fix_code_from_tests_failure(self, problem, error_str)
                        actual_number_of_calls += 2

                        # evaluate previous tests that passed. if they fail, revert to last solution
                        if problem['passed_tests']['inputs']:
                            problem, passed_prev_test, non_empty_output, error_str, trace_str, tests_timeout, d_tot \
                                = run_tests(self, problem, counter,
                                            problem['passed_tests']['inputs'],
                                            problem['passed_tests']['outputs'])
                            if not passed_prev_test:
                                logger.error(f"The fix broke prev passed tests. reverting to last solution")
                                problem['code_recent_solution'] = last_code_solution
                                continue

                    if not passed_specific_test:
                        if problem['passed_tests']['inputs']:
                            logger.error(f"Public test - reverting to initial solution, where: '{problem['passed_tests']['inputs']}' passed")
                            problem['code_recent_solution'] = last_code_solution
                        else: # no solution passed so far.
                            logger.error(f'Public test -  Reverting to best solution so far, d_tot: {best_d}')
                            problem['code_recent_solution'] = best_solution
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
            logging.error(f"'public tests' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
