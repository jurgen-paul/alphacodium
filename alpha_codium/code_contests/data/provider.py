import os.path
from typing import Iterable

import duckdb
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
import json
from datasets.features.features import Value, Sequence
import pyarrow as pa
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

problem_translations = ("source", "difficulty")

solution_translations = ("solutions", "incorrect_solutions")


class CodeContestDataProvider:

    def __init__(self, dataset_location="deepmind/code_contests", from_disk=False, connection=None):
        self.dataset_location = dataset_location
        self.from_disk = from_disk
        self.dataset_name = self.dataset_location.split(os.path.sep)[-1]
        self.dataset = self.get_dataset()
        self.connection = connection or duckdb.connect()
        self.connect()

    def get_dataset(self):
        if self.from_disk:
            f = load_from_disk
        else:
            f = load_dataset

        return f(self.dataset_location)

    def connect(self):
        if hasattr(self.dataset, 'keys'):
            for split in self.dataset.keys():
                split_ds = self.dataset[split]
                table = split_ds.data.table
                self.connection.register(f"{split_ds.info.dataset_name}_{split}", table)
        else:
            self.connection.register(f"{self.dataset.info.dataset_name}", self.dataset.data.table)

    def get_splits(self):
        return self.dataset.keys()

    def sample(self, split_name, fraction=0.1):
        table = self.dataset[split_name]
        sample_size = int(len(table) * fraction)
        indices = np.random.choice(len(table), sample_size, replace=False)
        sampled_table = table.select(indices)
        return sampled_table

    def query(self, query_string) -> pd.DataFrame:
        return self.connection.query(query_string).df()

    def translate_references(self, ds):
        for col in problem_translations:
            translated_source = ds.features[col].int2str(ds[col])
            ds = ds.remove_columns([col])
            ds = ds.add_column(col, translated_source)

        def translate_sequence_references(example, ds):
            for col in solution_translations:
                translator = ds.features[col].feature['language']
                arr = example[col]['language']
                translated_solution = [translator.int2str(item) for item in arr]
                example[col]['language'] = translated_solution

            return example

        new_features = ds.features.copy()
        for col in solution_translations:
            new_features[col] = Sequence(feature={
                'language': Value('string'),
                'solution': Value('string')
            })

        ds = ds.map(lambda example: translate_sequence_references(example, ds), features=new_features)
        return ds

    def filter_solution_by_languages(self, ds, languages: Iterable[str], keep=True):
        def filter_solutions_by_languages(example):
            for sol_col in solution_translations:
                langs = np.array(example[sol_col]['language'])
                sols = np.array(example[sol_col]['solution'])
                indices = np.isin(langs, languages)
                if not keep:
                    indices = ~indices
                filtered_data = {
                    'language': list(langs[indices]),
                    'solution': list(sols[indices]),
                }
                example[sol_col] = filtered_data
            return example

        ds = ds.map(filter_solutions_by_languages)
        return ds


def get_evaluation_candidates(dataset, test_type):
    records = []

    for entry in dataset:
        name = entry['name']
        solution_list = entry['solutions']['solution']
        test_input_list = entry[f'{test_type}_tests']['input']
        test_output_list = entry[f'{test_type}_tests']['output']

        record = {
            "name": name,
            "solutions": solution_list,
            "test_inputs": test_input_list,
            "test_outputs": test_output_list
        }

        records.append(record)
    return records

if __name__ == '__main__':
    """    #ds = CodeContestDataProvider("/Users/assaf/projects/codium/data", from_disk=True).get_dataset()
    cc = CodeContestDataProvider()
    result = cc.query("select count(*) from code_contests_valid")
    print(result)
    train_sample = cc.sample("train")
    translated = cc.translate_references(train_sample)
    df = translated.to_pandas()
    example = translated[0]
    # example['solutions'] = None
    # example['incorrect_solutions'] = None
    print(json.dumps(example['solutions']['language'], indent=4))
    ds = cc.filter_solution_by_languages(ds=translated, languages=['PYTHON3'])
    ds.save_to_disk("/Users/assaf/projects/codium/data")
    #print(json.dumps(ds[0]['solutions']['solution'][0], indent=4))"""

    """cc = CodeContestDataProvider()
    train_sample = cc.sample("train")
    translated = cc.translate_references(train_sample)
    ds = cc.filter_solution_by_languages(ds=translated, languages=['PYTHON3'])"""
    ds = CodeContestDataProvider("/home/ec2-user/data/sample", from_disk=True).get_dataset()
    records = get_evaluation_candidates(ds, "public")
    from evaluate import load as load_metric
    metric = load_metric('../evaluation/code_contests_eval.py', module_type="metric")
    fl = ds.flatten()
    fl = fl.rename_column("public_tests.input", "public_tests_inputs")
    fl = fl.rename_column("public_tests.output", "public_tests_outputs")
    fl = fl.rename_column("solutions.solution", "solution_candidates")
    fl = fl.select_columns(['name','solution_candidates','public_tests_inputs','public_tests_outputs'])

    arrow_table = ds.data

    # Create new columns for the restructured data
    predictions_column = pa.array(
        [{'task_name': tn, 'solution_candidates': pl} for tn, pl in zip(fl['name'], fl['solution_candidates'])])
    references_column = pa.array([{'tests_inputs': r1, 'tests_outputs': r2} for r1, r2 in
                                  zip(fl['public_tests_inputs'], fl['public_tests_outputs'])])

    # Combine the new columns with the Arrow table
    new_arrow_table = pa.Table.from_arrays([predictions_column, references_column], names=['predictions', 'references'])
    df = new_arrow_table.to_pandas().head(10)

    # Convert the Arrow table back to a HuggingFace dataset
    restructured_dataset = Dataset.from_pandas(df)

    #print(json.dumps(fl[0], indent=4))
    pass_at_k, _ = metric.compute(predictions=restructured_dataset['predictions'], references=restructured_dataset['references'])

    print(pass_at_k)


