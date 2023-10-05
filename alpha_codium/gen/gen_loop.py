import asyncio

from datasets import Dataset, Features, Sequence, Value

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.pass_at_k_evaluator import calculate_metrics
from alpha_codium.gen.coding_competitor import CodeContestsCompetitor

features = Features(
    {"task_name": Value("string"), "solution_candidates": Sequence(Value("string"))}
)


async def generate_candidate_solutions(ds):
    competitor = CodeContestsCompetitor()

    async def prediction_wrapper(example):
        result = await competitor.run(example)
        return {"task_name": example.get("name"), "solution_candidates": [result]}

    # Collect all the tasks
    tasks = [prediction_wrapper(example) for example in ds]
    results = await asyncio.gather(*tasks)
    ds = Dataset.from_list(results, features=features)
    return ds


def solve_and_evaluate_dataset(dataset_name, split_name='valid',  sample_rate=0.1, evaluation_test_type=None):

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
    solve_and_evaluate_dataset(dataset_name="assaf_test", sample=0.01, split_name='valid', evaluation_test_type='private_tests')
