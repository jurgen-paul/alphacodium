import os.path

import duckdb
import pandas as pd
from datasets import load_dataset
import numpy as np
import json


problem_translations = {
    "source": {0: "UNKNOWN_SOURCE",
                1: "CODECHEF",
                2: "CODEFORCES",
                3: "HACKEREARTH",
                4: "CODEJAM",
                5: "ATCODER",
                6: "AIZU"},

    "difficulty": {0: "UNKNOWN_DIFFICULTY",
                   1: "EASY",
                   2: "MEDIUM",
                   3: "HARD",
                   4: "HARDER",
                   5: "HARDEST",
                   6: "EXTERNAL",
                   **{i: chr(64 + i - 6) for i in range(7, 29)}
                   },

}
solution_translations = {
    "language": {0: "UNKNOWN_LANGUAGE",
                1: "PYTHON2",
                2: "CPP",
                3: "PYTHON3",
                4: "JAVA"},
}

class CodeContestDataProvider:

    def __init__(self, dataset_location="deepmind/code_contests", connection=None):
        self.dataset_location = dataset_location
        self.dataset_name = self.dataset_location.split(os.path.sep)[-1]
        self.dataset = self.get_dataset()
        self.connection = connection or duckdb.connect()
        self.connect()

    def get_dataset(self):
        return load_dataset(self.dataset_location)

    def connect(self):
        for split in self.dataset.keys():
            split_ds = self.dataset[split]
            table = split_ds.data.table
            self.connection.register(f"{split_ds.info.dataset_name}_{split}", table)

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

    def translate_columns(self):
        def transform_language(example):
            for column_name in problem_translations.keys():
                current_val = example[column_name]
                example[f"{column_name}_"] = problem_translations[column_name].get(current_val, 'unknown')

            for sol in ['solutions', 'incorrect_solutions']:
                solutions = example[sol]
                for column_name in solution_translations.keys():
                    current_val = solutions[column_name]
                    example[sol][f"{column_name}_"] = [solution_translations[column_name].get(v, 'unknown') for v in current_val]

            return example

        for split in ['valid']:
            self.dataset[split] = self.dataset[split].map(transform_language)


if __name__ == '__main__':
    cc = CodeContestDataProvider()
    result = cc.query("select count(*) from code_contests_valid")
    print(result)
    train_sample = cc.sample("train")
    print(train_sample.to_pandas())
    cc.translate_columns()
    sample_problem = cc.dataset['valid'][0]
    pretty = json.dumps(sample_problem, indent=4)
    print(pretty)

