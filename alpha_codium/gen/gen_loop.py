import asyncio
import os

import datasets
from datasets import Dataset, Features, Sequence, Value

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.pass_at_k_evaluator import calculate_metrics
from alpha_codium.config_loader import get_settings
from alpha_codium.gen.coding_competitor import CodeContestsCompetitor
from alpha_codium.log import get_logger

logger = get_logger(__name__)
features = Features(
    {"task_name": Value("string"), "solution_candidates": Sequence(Value("string")),
     "public_test_results": Sequence(Value("bool"))}
)


async def generate_candidate_solutions(ds):
    competitor = CodeContestsCompetitor()

    async def prediction_wrapper(example):
        result, passed_all_public = await competitor.run(example)
        return {"task_name": example.get("name"), "solution_candidates": [result],
                "public_test_results":[passed_all_public]}

    # Collect all the tasks
    tasks = [prediction_wrapper(example) for example in ds]
    results = await asyncio.gather(*tasks)
    ds = Dataset.from_list(results, features=features)
    return ds


def solve_and_evaluate_dataset(dataset_name, split_name='valid',  sample_rate=0.1, evaluation_test_type=None):
    logger.info('solve_and_evaluate_dataset')
    cc = CodeContestDataProvider(dataset_location=dataset_name)
    ds = cc.dataset[split_name]
    ds = cc.sample(ds, sample_rate)
    predictions = asyncio.run(generate_candidate_solutions(ds))
    evaluation_set = cc.prepare_for_evaluation(
        predictions=predictions, source_of_truth=ds, evaluation_test_type=evaluation_test_type
    )

    if evaluation_test_type:
        evaluation_results = calculate_metrics(evaluation_set)
    return predictions, evaluation_results


if __name__ == "__main__":
    base_path = os.path.expanduser(get_settings().etl.private_dataset_cache_dir)
    output_path = os.path.join(base_path, "generated_solutions_test")
    if not os.path.exists(output_path):
        cc = CodeContestDataProvider(dataset_location="deepmind/code_contests")
        ds = cc.dataset['valid']
        sub_ds = ds.filter(lambda example: example['name'] == "1548_D1. Gregor and the Odd Cows (Easy)")
        solutions = asyncio.run(generate_candidate_solutions(sub_ds))
        evaluation_set = cc.prepare_for_evaluation(
            predictions=solutions, source_of_truth=ds, evaluation_test_type="private_tests"
        )
        print(f"saving the output dataset to {output_path}")
        evaluation_set.save_to_disk(output_path)
    else:
        evaluation_set = datasets.load_from_disk(output_path)

    pass_at_k, inputs, evaluation_results = calculate_metrics(evaluation_set)
    print(pass_at_k)
    print(evaluation_results)

