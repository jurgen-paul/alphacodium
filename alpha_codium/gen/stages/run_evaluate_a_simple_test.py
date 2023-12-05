import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.gen.stages.run_initial_solve import run_initial_solve
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_evaluate_a_simple_test(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--run simple test stage--")
            use_recording = problem.get('use_recording', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')
            MAX_COUNTER = 6
            if use_recording and False:
                code_recent_solution = np.load(recording_path + 'problem_run_simple_test.npy',
                                               allow_pickle=True).tolist()
                problem['code_recent_solution'] = code_recent_solution
                logger.info("Using recording simple test")
                logger.debug(f"code_recent_solution:\n{code_recent_solution}")
            else:
                counter = 0
                # test_input = [problem['problem_ai_simple_test']['input']]
                # test_output = [problem['problem_ai_simple_test']['output']]

                # first public test. maybe take this if there is more than one public test
                # test_input = [problem['public_tests']['input'][0]]
                # test_output = [problem['public_tests']['output'][0]]

                # All public tests
                test_input = problem['public_tests']['input']
                test_output = problem['public_tests']['output']

                best_solution = copy.deepcopy(problem['code_recent_solution'])
                best_d = float('inf')

                # run the solution on the simple test
                problem, passed_simple_test, non_empty_output, error_str, trace_str, tests_timeout, d_tot \
                    = run_tests(self, problem, counter, test_input, test_output)

                # set the distance to the correct solution
                if -1 < d_tot < best_d:
                    best_solution = copy.deepcopy(problem['code_recent_solution'])
                    best_d = d_tot

                while not passed_simple_test:
                    counter += 1
                    if counter > MAX_COUNTER:
                        logger.error(f"Failed to pass simple test after {counter - 1} attempts. exiting the stage")
                        break

                    s_best_solution_original = problem['s_best_solution']
                    problem['s_best_solution'] = problem['s_possible_solutions'][counter % len(problem['s_possible_solutions'])]
                    problem = await run_initial_solve(self, problem, enable_record=False)
                    problem['s_best_solution'] = s_best_solution_original


                    problem, passed_simple_test, non_empty_output, error_str, trace_str, tests_timeout, d_tot \
                        = run_tests(self, problem, counter, test_input, test_output)

                    if -1 < d_tot < best_d:
                        best_solution = copy.deepcopy(problem['code_recent_solution'])
                        best_d = d_tot

                if not passed_simple_test and best_d < float('inf'):
                    logger.error(f'Reverting to best solution so far, d_tot: {best_d}')
                    problem['code_recent_solution'] = best_solution

                if do_recording and False:
                    np.save(recording_path + 'problem_run_simple_test.npy', problem['code_recent_solution'])

            return problem
        except Exception as e:
            logging.error(f"'simple test stage' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
