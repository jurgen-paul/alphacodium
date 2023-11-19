import click as click

from alpha_codium.code_contests.data.etl import etl
from alpha_codium.code_contests.data.provider import CodeContestDataProvider


@click.group(
    name="data",
    help="Commands for generating datasets",
)
def data():
    pass


@data.command("etl")
@click.option(
    "--source",
    default=CodeContestDataProvider.hf_dataset_name,
    show_default=True,
    required=False,
    help="Name of the dataset to load. Default is to load from huggingface",
)
@click.option(
    "--output_dataset_name",
    show_default=True,
    required=False,
    help="Name for the output dataset, which will be cached locally",
)
@click.option("--translate_references", default=True, show_default=True, required=False)
@click.option(
    "--filter_languages",
    default=["PYTHON3"],
    multiple=True,
    show_default=True,
    required=False,
)
@click.option("--disable_filter_languages", default=False, show_default=True, required=False)

@click.option("--train_sample", default=0.1, show_default=True, required=False)
def etl_command(
    source, output_dataset_name, train_sample, translate_references, filter_languages, disable_filter_languages,
):
    etl(
        source,
        output_dataset_name,
        train_sample,
        translate_references,
        filter_languages,
        disable_filter_languages,
    )
