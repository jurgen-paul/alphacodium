import os
import os.path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, DatasetDict
from datasets.features.features import Value, Sequence

from alpha_codium.config_loader import get_settings

problem_translations = ("source", "difficulty")

solution_translations = ("solutions", "incorrect_solutions")


class CodeContestDataProvider:
    hf_dataset_name = "deepmind/code_contests"

    def __init__(self, dataset_location=hf_dataset_name, connection=None):
        self.private_datasets_root = os.path.expanduser(get_settings().etl.private_dataset_cache_dir)
        self.dataset_location, self.dataset_name, self.load_from_disk = self.parse_location(dataset_location)
        self.dataset = self.load_dataset()
        self.connection = connection or duckdb.connect()
        self.connect()

    def parse_location(self, dataset_location):
        result_location = dataset_location
        dataset_name = dataset_location.split(os.path.sep)[-1]
        load_from_disk = dataset_location != CodeContestDataProvider.hf_dataset_name
        if load_from_disk:
            if not result_location.startswith(os.path.sep):
                result_location = os.path.join(self.private_datasets_root, result_location)
        return result_location, dataset_name, load_from_disk

    def load_dataset(self):
        if self.load_from_disk:
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

    @staticmethod
    def sample(ds, fraction=0.1):
        table = ds
        sample_size = int(len(table) * fraction)
        indices = np.random.choice(len(table), sample_size, replace=False)
        sampled_table = table.select(indices)
        return sampled_table

    def query(self, query_string) -> pd.DataFrame:
        return self.connection.query(query_string).df()

    def translate_references(self, ds):
        expand = False
        if not isinstance(ds, DatasetDict):
            to_translate = {'ds': ds}
            expand = True
        else:
            to_translate = ds
        for ds_name, ds_val in to_translate.items():
            for col in problem_translations:
                translated_col = ds_val.features[col].int2str(ds_val[col])
                ds_val = ds_val.remove_columns([col])
                ds_val = ds_val.add_column(col, translated_col)

            def translate_sequence_references(example, ds):
                for col in solution_translations:
                    translator = ds.features[col].feature['language']
                    arr = example[col]['language']
                    translated_solution = [translator.int2str(item) for item in arr]
                    example[col]['language'] = translated_solution

                return example

            new_features = ds_val.features.copy()
            for col in solution_translations:
                new_features[col] = Sequence(feature={
                    'language': Value('string'),
                    'solution': Value('string')
                })

            ds_val = ds_val.map(lambda example: translate_sequence_references(example, ds_val), features=new_features)
            to_translate[ds_name] = ds_val
        result = to_translate
        if expand:
            result = result[ds]
        return result

    def filter_solution_by_languages(self, ds, languages: Iterable[str], keep=True):
        languages_set = set(languages)

        def filter_solutions_by_languages(example):
            for sol_col in solution_translations:
                langs = example[sol_col]['language']
                sols = example[sol_col]['solution']

                filtered_languages = [l for idx, l in enumerate(langs) if (l in languages_set) == keep]
                filtered_solutions = [s for idx, s in enumerate(sols) if (langs[idx] in languages_set) == keep]

                example[sol_col] = {
                    'language': filtered_languages,
                    'solution': filtered_solutions
                }

            return example

        ds = ds.map(filter_solutions_by_languages)
        return ds
