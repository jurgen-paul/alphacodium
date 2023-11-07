import asyncio

import click as click

from alpha_codium.gen import gen_loop
from alpha_codium.gen.coding_competitor import solve_and_test
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
    "--problem_number",
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
def solve_problem(dataset_name, split_name, problem_name, evaluation_test_type, problem_number):
    return solve_and_test(dataset_name=dataset_name, split_name=split_name,
                          problem_name=problem_name, evaluation_test_type=evaluation_test_type,
                          problem_number=problem_number)


@gen.command("solve_set")
@click.option(
    "--input_dataset",
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
    "--sample_rate",
    default=0.1,
    show_default=True,
    required=False
)
@click.option(
    "--output_dataset_name",
    show_default=True,
    required=True,
)
def solve_set(input_dataset, split_name, sample_rate, output_dataset_name):
    return gen_loop.solve_set(input_dataset, split_name=split_name, sample_rate=sample_rate,
                              output_dataset_name=output_dataset_name)


if __name__ == "__main__":
    gen_loop.solve_set(dataset_name="assaf_test", split_name="valid",
                       sample_rate=0.1, output_dataset_name="assat_test_solutions")
