import click as click

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.pass_at_k_evaluator import (
    evaluate_code_contest_dataset,
    evaluate_gen_dataset,
)


@click.group(
    name="eval",
    help="Commands for evaluating results",
)
def eval():
    pass


@eval.command("eval_cc_solutions")
@click.option(
    "--dataset_name",
    default=CodeContestDataProvider.hf_dataset_name,
    show_default=True,
    required=False,
    help="Name of the dataset to load. Default is to load from huggingface",
)
@click.option("--split_name",
              default="valid",
              show_default=True,
              required=False)
@click.option(
    "--k_values",
    default=[1, 10, 100],
    multiple=True,
    show_default=True,
    required=False
)
@click.option(
    "--evaluation_test_type",
    default="public_tests",
    show_default=True,
    required=False
)
@click.option(
    "--path_to_solutions_column",
    default="solutions.solution",
    show_default=True,
    required=False,
)
@click.option("--task_name_column", default="name", show_default=True, required=False)
@click.option("--sample_rate", default=0.01, show_default=True, required=False)
def eval_cc_solutions(
        dataset_name,
        split_name,
        k_values,
        evaluation_test_type,
        path_to_solutions_column,
        task_name_column,
        sample_rate,
):
    return evaluate_code_contest_dataset(
        dataset_name,
        split_name=split_name,
        k_values=k_values,
        evaluation_test_type=evaluation_test_type,
        path_to_solutions_column=path_to_solutions_column,
        task_name_column=task_name_column,
        sample_rate=sample_rate)


@eval.command("eval_gen_solutions")
@click.option(
    "--solution_dataset",
    show_default=True,
    required=True,
)
@click.option(
    "--ground_truth_dataset",
    show_default=True,
    required=True,
)

@click.option(
    "--ground_truth_split",
    show_default=True,
    required=False,
)

@click.option(
    "--evaluation_test_type",
    show_default=True,
    required=False,
)
@click.option(
    "--k_values",
    default=[1, 10, 100],
    multiple=True,
    show_default=True,
    required=False
)
def eval_gen_solutions(solution_dataset, ground_truth_dataset, ground_truth_split, evaluation_test_type, k_values):
    return evaluate_gen_dataset(evaluation_test_type, ground_truth_dataset,
                                ground_truth_split, k_values, solution_dataset)


if __name__ == '__main__':
    #evaluate_code_contest_dataset(dataset_name="deepmind/code_contests", sample_rate=0.01)
    evaluate_gen_dataset("private_tests","assaf_test", "valid", [1,10,100], "assat_test_solutions")