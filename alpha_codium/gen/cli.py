import asyncio

import click as click

from alpha_codium.gen.generators import SimplePrompt


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

if __name__ == '__main__':
    run_internal(prompt="what is the code_contests dataset?")