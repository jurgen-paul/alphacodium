# alphaCodium

Generating high quality code solutions to challenging coding problems from the code_contests dataset

## Data

The data in the project is based on the [code_contests](https://github.com/google-deepmind/code_contests) dataset.
Luckily, there is a huggingface dataset [huggingface](https://huggingface.co/datasets/deepmind/code_contests) which reflects the same data.

The `code_contests.data` package contains convenience tools for loading, filtering, transforming and saving the data in a standard hugginface `Dataset` API.

## Gen

Generating solutions is done using a lighweight framework (leveraging assets and design patterns from the `pr_agent` project).

The generation should result in a dataset of problem_id -> solution candidates, which will then be evaluated against test cases.

The main prompts are in `code_contests_prompts_baseline.toml`

## Eval

Once we've generated solution candidates, we would like to evaluate them against test cases, and score the resuls.

The `eval` package exposes a `Metric` object that accepts code solutions as well as input-output pairs, and calculates the `pass@k` metric.

The evaluation itself requires running the code in an interpreter, in a sandboxed environment, and achieving high throughput for execution.

Deepmind provided in the `code_contests` repo, a C++ based framework for managing candidate execution against inputs and outputs.
This tool was wrapped in a [python package](https://pypi.org/project/code-contests-tester/0.1.3/) to make it more usable within a development environment.

## CLI

The project includes a cli tool for running common development tasks.
It's not clear yet if this API will be useful going forward, but it will likely help new team members onboard to the project.

```gencode ```

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
Base command: `gencode data etl --source deepmind/code_contests --output_dataset_name 101_test`

This command downloads the hugging face dataset and applies common transformations (filtering only Python3 solutions, sampling data, and translating references).
It then stores the derived dataset called `101_test` (locally)


### gen

This is the part that generates predictions.

`ask` - is just a simple prompt for testing

`solve_problem` - you can specify dataset name, split and even problem name. The code will generate a solution to the problem and run the tests against it

for example, try:
```gencode gen solve_problem --dataset_name deepmind/code_contests --split_name valid --problem_name "1560_F1. Nearest Beautiful Number (easy version)" --evaluation_test_type private_tests```


`solve_and_evaluate_set` - specify a dataset, split, sample etc. and the code will generate solutions for all the problems in the dataset, and calulate `pass@k`

### eval
Example command: `gencode eval pass_at_k --dataset_name 101_test --split_name train --evaluation_test_type private_tests`

This command takes a dataset name (can be a dataset stored locally) and evaluates the  one of it's split of the dataset against specified test types that accompany the problems.
The result is a `pass@k` metric


## installation

general - `pip install -e .`

development - `pip install -r requirements-dev.txt`


## code execution

There are two ways to execute code for evaluation - `local` and `code_contests`.
You can set it via the configuration.toml under `tester_type`

**These are mostly equivalent but will not give 1-1 results**

### `local`
Can run on any machine which has Python.
You can also control whether the runs are sandboxed (best effort) or not by setting `sandbox=true` in the configuration. 

### `code_contests`
This is a Python binding for code_contests` C++ library and sadbox for code execution.

It's execution results will be more reliable in comparison to the `local` option (some discrepancies already found between the two)

**Note**:

Due to the `code_contests` dependency on C++, there are a few quirks:

1. Support only Python 3.9
2. Run only on a linux box  (**Tested on AWS Linux 2023 only**)
3. You need to also have a Python3.11 installed (but **not** for your virtual environment) - this is the interpreter used to run your Python tests.
4. Make sure it's `bin` is in `/usr/bin/python3.11` and it's lib is in `/usr/lib64/` and `/usr/lib64/python3.11`

We will likely improve this sitaution later on and achieve more flexibility.
