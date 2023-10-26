import asyncio

import click as click

from alpha_codium.gen.coding_competitor import solve_and_test

from alpha_codium.gen.gen_loop import solve_and_evaluate_dataset
from alpha_codium.gen.generators import SimplePrompt
from alpha_codium.log import get_logger

logger = get_logger(__name__)

@click.group(
    name="gen",
    help="Commands for generating code predictions",
)
def gen():
    pass


@gen.command("ask")
@click.option(
    "--prompt",
    default="text",
    show_default=True,
    required=True,
)
def run_generation(prompt):
    return run_internal(prompt)


def run_internal(prompt):
    p = SimplePrompt()
    asyncio.run(p.run(prompt))


@gen.command("solve_problem")
@click.option(
    "--dataset_name",
    show_default=True,
    required=True,
)
@click.option(
    "--split_name",
    show_default=True,
    default='train',
    required=False,
)
@click.option(
    "--problem_name",
    default=None,
    show_default=True,
    required=False
)
@click.option(
    "--evaluation_test_type",
    show_default=True,
    type=click.Choice(['private_tests', 'public_tests', 'generated_tests'], case_sensitive=False),
    help="Type of the test",
    required=False,
)

def solve_problem(dataset_name, split_name, problem_name, evaluation_test_type):
    return solve_and_test(dataset_name=dataset_name, split_name=split_name,
                          problem_name=problem_name, evaluation_test_type=evaluation_test_type)


@gen.command("solve_and_evaluate_set")
@click.option(
    "--dataset_name",
    show_default=True,
    required=True,
)
@click.option(
    "--split_name",
    show_default=True,
    default='train',
    required=False,
)
@click.option(
    "--evaluation_test_type",
    show_default=True,
    type=click.Choice(['private_tests', 'public_tests', 'generated_tests'], case_sensitive=False),
    default='private_tests',
    required=False,
)
@click.option(
    "--sample_rate",
    default=0.1,
    show_default=True,
    required=False
)
def solve_and_evaluate_set(dataset_name, split_name, evaluation_test_type, sample_rate):
    predictions, eval_results = solve_and_evaluate_dataset(dataset_name=dataset_name, split_name=split_name,
                                                           sample_rate=sample_rate,
                                                           evaluation_test_type=evaluation_test_type)

    if predictions:
        logger.info(f"generated {len(predictions)} predictions")
    if eval_results:
        logger.info(eval_results)


if __name__ == "__main__":
    run_internal(prompt="what is the code_contests dataset?")
