import abc
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.local_exec import (
    MultiTestResult,
    calculate_tests_pass_fail,
    execute_candidate_code,
)
from alpha_codium.config_loader import get_settings


class PythonTestsRunner(abc.ABC):
    test_program = "x=input()\nprint(x)"
    test_inputs = ["hello"]
    test_outputs = test_inputs

    @staticmethod
    def factory(runner_type, *args, **kwargs):
        if runner_type == 'local':
            return LocalPythonTestsRunner(*args, **kwargs)
        elif runner_type == 'code_contests':
            return CodeContestsOfficialPythonTestsRunner(*args, **kwargs)
        else:
            raise ValueError(f"Unknown method type: {runner_type}")

    @abstractmethod
    def test_interpreter(self):
        pass

    @abstractmethod
    def run_tests(self, test_id, candidate_id, candidate, test_inputs, tests_outputs):
        pass


    @abstractmethod
    def create_executor(self):
        pass

    def print_test_results(self, result: MultiTestResult, test_inputs: List[str] = None):
        if result.compilation_result:
            print(
                f"compilation results:{result.compilation_result.program_status if result.compilation_result else ''}")
            print(result.compilation_result.sandbox_result)
            print(result.compilation_result.stderr)

        for i, test_res in enumerate(result.test_results):
            print(f"input:\n{test_inputs[i]}")
            print(f"expected output:\n{test_res.expected_output}")
            print(f"actual output:\n{test_res.actual_output}")
            print(
                f"test-{i} :: status={test_res.program_status}, pased={test_res.passed}"
            )
            print(
                "====================================================================="
            )
            # print(test_res.stdout)
            # print(
            #     "====================================================================="
            # )


class LocalPythonTestsRunner(PythonTestsRunner):

    def __init__(self):
        super().__init__()
        self.sandbox = get_settings().code_tester.sandbox

    def test_interpreter(self):
        _, _, result = self.run_tests(0, "test interpreter", PythonTestsRunner.test_program,
                              PythonTestsRunner.test_inputs, PythonTestsRunner.test_outputs)
        super().print_test_results(result)

    def run_tests(self, test_id, candidate_id, candidate, test_inputs, tests_outputs):
        multi_result = execute_candidate_code(candidate=candidate, inputs=test_inputs,
                                              test_id=test_id, timeout=2, sandbox=self.sandbox)
        tests_results = calculate_tests_pass_fail(multi_result, expected_results=tests_outputs)
        return test_id, candidate_id, tests_results

    def create_executor(self):
        return ProcessPoolExecutor, {}#{'initializer': reliability_guard}


class CodeContestsOfficialPythonTestsRunner(PythonTestsRunner):
    def __init__(
            self,
            path_to_python_bin: str = "/usr/bin/python3.11",
            path_to_python_lib: List[str] = ["/usr/lib64", "/usr/lib64/python3.11"],  # noqa: B006
            num_threads: int = 4,  # noqa: B006
            stop_on_first_failure: bool = True
    ):

        try:
            from code_contests_tester import Py3TesterSandboxer, TestOptions  # noqa: F401
        except ImportError as e:
            raise ValueError("Error: cannot import the test sandbox on your environment") from e

        options = TestOptions()
        options.num_threads = num_threads
        options.stop_on_first_failure = stop_on_first_failure

        def compare_func(a, b):
            return a == b

        self.tester = Py3TesterSandboxer(path_to_python_bin, path_to_python_lib)
        self.options = options
        self.compare_func = compare_func
        self.test_interpreter()

    def test_interpreter(self):
        result = self.tester.test(
            PythonTestsRunner.test_program,
            PythonTestsRunner.test_inputs,
            self.options,
            PythonTestsRunner.test_outputs,
            self.compare_func,
        )
        super().print_test_results(result)

    def run_tests(self, test_id, candidate_id, candidate, test_inputs, tests_outputs):
        result = self.tester.test(
            candidate, test_inputs, self.options, tests_outputs, self.compare_func
        )
        return test_id, candidate_id, result

    def create_executor(self):
        return ThreadPoolExecutor, {}


def eval_solution(evaluation_test_type: str = "private_tests",
                  example: dict = {},  # the code contest problem
                  prediction: str = '',  # python code to be evaluated
                  test_inputs: Optional[List[str]] = None,
                  tests_outputs: Optional[List[str]] = None,
                  ):
    if not test_inputs or not tests_outputs:
        test_inputs = example.get(evaluation_test_type).get("input") if example.get(evaluation_test_type) else None
        test_outputs = example.get(evaluation_test_type).get("output") if example.get(evaluation_test_type) else None
    if test_inputs and test_outputs:
        test_runner = PythonTestsRunner.factory(get_settings().code_tester.tester_type)
        _, _, results = test_runner.run_tests(
            test_id=example["name"],
            candidate_id="id",
            candidate=prediction,
            test_inputs=test_inputs,
            tests_outputs=test_outputs
        )
        test_runner.print_test_results(results, test_inputs)
    else:
        print("example doesn't have inputs or outputs")


if __name__ == '__main__':
    cc = CodeContestDataProvider("assaf_test")
    problem = CodeContestDataProvider.find_problem(cc.dataset, problem_name= "1551_E. Fixed Points",
                                                   split_name= "valid", evaluation_test_type="private_tests",)
    sols = problem['solutions']
    #for solution in sols.get('solution'):
    solution = sols.get('solution')[1]
    eval_solution(evaluation_test_type="private_tests", example=problem, prediction=solution)

    for test_input, test_output in zip(problem['private_tests']['input'], problem['private_tests']['output'] ):
        print("===========")
        print(test_input)
        print("***")
        print(test_output)
        print("===========")
