import functools
import logging
import numpy as np
import yaml

from alpha_codium.code_contests.eval.code_test_runners import eval_solution
from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


def run_tests(self, problem, counter, test_inputs, test_outputs):
    try:
        # run the solution on the public tests
        logging.info(f"evaluating public tests. attempt {counter}")
        test_inputs, results = eval_solution(example=problem,
                                             prediction=problem['code_recent_solution'],
                                             test_inputs=test_inputs,
                                             test_outputs=test_outputs, )

        # analyze the tests results
        error_str = trace_str = ""
        all_passed = True
        non_empty_output = True
        tests_timeout = False
        if str(results.test_results[0].program_status) == 'ProgramStatus.kTimeout':
            tests_timeout = True
        elif str(results.test_results[0].program_status) == 'ProgramStatus.kFailed':
            logger.error("failed to run solution")
            error_str = results.test_results[0].sandbox_result
            trace_str = f"trace information:\n{self.render_trace(results.test_results[0].trace)}\n\n"
            all_passed = False
        else: # ProgramStatus.passed
            # initially assume all tests passed
            all_passed = True
            non_empty_output = True

            # build the error string
            error_str = ""
            trace_str = ""
            for i, t in enumerate(results.test_results):
                error_str += f"test input:\n{test_inputs[i]}\n" \
                             f"expected output:\n{t.expected_output}\n" \
                             f"code output:\n{t.actual_output}\n" \
                             f"====================\n====================\n"

                trace_str += f"trace:\n{self.render_trace(t.trace)}\n" \
                             f"====================\n====================\n"

                # if get_settings().code_tester.calc_trace:
                #     logger.debug(f"trace_str:\n{trace_str}")

                # is_all_passed_public = actual_output == expected_output
                all_passed = all_passed and t.passed
                non_empty_output = non_empty_output and t.actual_output
        return problem, all_passed, non_empty_output, error_str, trace_str, tests_timeout
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
