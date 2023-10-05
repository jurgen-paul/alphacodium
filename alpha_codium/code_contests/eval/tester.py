from alpha_codium.code_contests.eval.code_contests_eval import TestsRunner


def test_solution(evaluation_test_type, example, prediction):
    test_inputs = example.get(evaluation_test_type).get("input") if example.get(evaluation_test_type) else None
    test_outputs = example.get(evaluation_test_type).get("output") if example.get(evaluation_test_type) else None
    if test_inputs and test_outputs:
        test_runner = TestsRunner(
            path_to_python_bin="/usr/bin/python3.11",
            path_to_python_lib=["/usr/lib64", "/usr/lib64/python3.11"],
            num_threads=4,
            stop_on_first_failure=True,
        )
        _, _, results = test_runner.run_test(
            test_id=example["name"],
            candidate_id="id",
            candidate=prediction,
            test_inputs=test_inputs,
            tests_outputs=test_outputs,
        )
        test_case_results = [test_result.passed for test_result in results.test_results]
        candidate_pass_fail = all(test_case_results)

        print(f"Competitor solution:\n {prediction}")
        print("=====================================")
        print(f"Final pass/fail: {candidate_pass_fail}")
        print(f"Individual test results: {test_case_results}")

    else:
        print(f"The problem didn't have tests of type {evaluation_test_type}")