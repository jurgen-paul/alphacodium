# alphaCodium

Generating high quality code solutions to challenging coding problems from the code_contests dataset

## Data

The data in the project is based on the [code_contests](https://github.com/google-deepmind/code_contests) dataset.
Luckily, there is a huggingface dataset [huggingface](https://huggingface.co/datasets/deepmind/code_contests) which reflects the same data.

The `code_contests.data` package contains convenience tools for loading, filtering, transforming and saving the data in a standard hugginface `Dataset` API.

## Gen

Generating solutions is done using a lighweight framework (leveraging assets and design patterns from the `pr_agent` project).

The generation should result in a dataset of problem_id -> solution candidates, which will then be evaluated against test cases.

## Eval

Once we've generated solution candidates, we would like to evaluate them against test cases, and score the resuls.

The `eval` package exposes a `Metric` object that accepts code solutions as well as input-output pairs, and calculates the `pass@k` metric.

The evaluation itself requires running the code in an interpreter, in a sandboxed environment, and achieving high throughput for execution.

Deepmind provided in the `code_contests` repo, a C++ based framework for managing candidate execution against inputs and outputs.
This tool was wrapped in a [python package](https://pypi.org/project/code-contests-tester/0.1.3/) to make it more usable within a development environment.

## CLI

The project includes a cli tool for running common development tasks.
It's not clear yet if this API will be useful going forward, but it will likely help new team members onboard to the project.

```python -m alpha_codium.cli```

The tool has three sub-commands:

```
Options:
  --help  Show this message and exit.

Commands:
  data  Commands for generating datasets
  eval  Commands for evaluating results
  gen   Commands for generating code predictions

```

### data etl
Base command: `python -m alpha_codium.cli data etl --output_dataset_name 101_test`

This command downloads the hugging face dataset and applies common transformations (filtering only Python3 solutions, sampling data, and translating references).
It then stores the derived dataset called `assaf_test` (locally)

### eval
Example command: `python -m alpha_codium.cli eval pass_at_k --dataset_name 101_test --split_name train --evaluation_test_type private_tests`

This command takes a dataset called `101_test` (stored locally) and evaluates the `train` split of the dataset against all `private_tests` that accompany the problems.
The result is a `pass@k` metric


### gen

For now the cli is more or less useless, it sends a prompt (with retry etc.) and prints the results.

## installation

Due to the dependency on C++, there are a few quirks:

1. Support only Python 3.9
2. Run only on a linux box (docker currently not be an option due to sandbox security issues). **Tested on AWS Linux 2023 only**
3. You need to also have a Python3.11 installed (but **not** for your virtual environment) - this is the interpreter used to run your Python tests.
4. Make sure it's `bin` is in `/usr/bin/python3.11` and it's lib is in `/usr/lib64/` and `/usr/lib64/python3.11`

We will likely improve this sitaution later on and achieve more flexibility.
