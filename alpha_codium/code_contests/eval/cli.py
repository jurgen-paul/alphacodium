
import click as click

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.pass_at_k_evaluator import evaluate_code_contest_dataset


@click.group(
    name="eval",
    help="Commands for evaluating results",
)
def eval():
    pass


@eval.command("pass_at_k")
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
def evaluate_dataset_command(
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
