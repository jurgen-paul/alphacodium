import asyncio

import click as click

from alpha_codium.gen.generators import SimplePrompt


@click.group(
    name="gen",
    help="Commands for generating code predictions",
)

def gen():
    pass


@gen.command("test")
@click.option(
    "--prompt",
    default="text",
    show_default=True,
    required=True,
)
def test_generation(prompt):
    p = SimplePrompt()
    asyncio.run(p.run(prompt))
