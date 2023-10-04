import os
from evaluate import load as load_metric
import pyarrow as pa
from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from datasets import Dataset
import numpy as np


def evaluate_dataset(ds, k_values=[1, 10, 100], evaluation_test_type="public_tests", path_to_solutions_column="solutions.solution",
                     task_name_column="name", sample_rate=0.01 ):
    metric_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code_contests_eval.py')
    metric = load_metric(metric_path, module_type="metric")
    if sample_rate and sample_rate < 1:
        ds = CodeContestDataProvider.sample(ds, sample_rate)
    fl = ds.flatten()
    fl = fl.rename_column(f"{evaluation_test_type}.input", "tests_inputs")
    fl = fl.rename_column(f"{evaluation_test_type}.output", "tests_outputs")
    fl = fl.rename_column(path_to_solutions_column, "solution_candidates")
    fl = fl.select_columns([task_name_column, 'solution_candidates', 'tests_inputs', 'tests_outputs'])

    # Create new columns for the restructured data
    predictions_column = pa.array(
        [{'task_name': tn, 'solution_candidates': pl} for tn, pl in zip(fl['name'], fl['solution_candidates'])])
    references_column = pa.array([{'tests_inputs': r1, 'tests_outputs': r2} for r1, r2 in
                                  zip(fl['tests_inputs'], fl['tests_outputs'])])

    new_arrow_table = pa.Table.from_arrays([predictions_column, references_column], names=['predictions', 'references'])
    df = new_arrow_table.to_pandas()
    restructured_dataset = Dataset.from_pandas(df)

    pass_at_k, _ = metric.compute(predictions=restructured_dataset['predictions'],
                                  references=restructured_dataset['references'], k=k_values)

    print(pass_at_k)


if __name__ == '__main__':
    ds = CodeContestDataProvider("assaf_test").dataset['valid']
    evaluate_dataset(ds)
