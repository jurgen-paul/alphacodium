import os

from evaluate import load as load_metric

from alpha_codium.code_contests.data.provider import CodeContestDataProvider

try:
    from code_contests_tester import Py3TesterSandboxer  # noqa: F401
except ImportError:
    print("Error: cannot import the test sandbox on your environment")


def calculate_metrics(ds, k_values=[1, 10, 100]):  # noqa: B006

    metric_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "code_contests_eval.py"
    )
    metric = load_metric(metric_path, module_type="metric")
    pass_at_k, _ = metric.compute(
        predictions=ds["predictions"], references=ds["references"], k=k_values
    )

    print(pass_at_k)

def evaluate_code_contest_dataset(
        dataset_name,
        split_name = 'valid',
        k_values=[1,10, 10],  # noqa: B006
        evaluation_test_type='private_tests',
        path_to_solutions_column='solutions.solution',
        task_name_column='name',
        sample_rate=0.1,
    ):
    cc = CodeContestDataProvider(dataset_name)
    ds = cc.dataset[split_name]
    ds = cc.sample(ds, fraction=sample_rate)
    ds = CodeContestDataProvider.prepare_code_contest_split_for_eval(ds=ds,
                                                                     evaluation_test_type=evaluation_test_type,
                                                                     task_name_column=task_name_column,
                                                                     path_to_solutions_column=path_to_solutions_column)
    result = calculate_metrics(ds, k_values=k_values)
    print(result)


if __name__ == "__main__":
    evaluate_code_contest_dataset("assaf_test", evaluation_test_type="public_tests")
