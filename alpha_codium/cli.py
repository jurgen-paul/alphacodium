import click

from alpha_codium.code_contests.data import cli as data_cli
from alpha_codium.code_contests.eval import cli as eval_cli
from alpha_codium.gen import cli as gen_cli


@click.group()
def cli():
    pass


cli.add_command(gen_cli.gen)
cli.add_command(data_cli.data)
cli.add_command(eval_cli.eval)

if __name__ == "__main__":
    cli()
