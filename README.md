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
This is a Python binding for code_contests` C++ library and sandbox for code execution.

It's execution results will be more reliable in comparison to the `local` option (some discrepancies already found between the two)

**Note**:

Due to the `code_contests` dependency on C++, there are a few quirks:

1. Support only Python 3.9
2. Run only on a linux box  (**Tested on AWS Linux 2023 only**)
3. You need to also have a Python3.11 installed (but **not** for your virtual environment) - this is the interpreter used to run your Python tests.
4. Make sure it's `bin` is in `/usr/bin/python3.11` and it's lib is in `/usr/lib64/` and `/usr/lib64/python3.11`

We will likely improve this sitaution later on and achieve more flexibility.



#### Running on amazon Linux 23

1. Start an Amazon Linux 2023 machine (add your ssh keys, choose m5.xlarge or larger)
2. SSH into the machine
3. Run following commands (one by one)
```
   sudo yum install openssl-devel bzip2-devel libffi-devel
   sudo dnf install python3.11
   curl -O https://bootstrap.pypa.io/get-pip.py
   python3 get-pip.py --user
   git clone git@github.com:Codium-ai/alphaCodium.git
   git checkout assaf/cc_eval
   cd alphaCodium/
   python3 -m pip install virtualenv
   virtualenv venv
   source venv/bin//activate
   python --version
   pip install -e .
   huggingface-cli login
   # possibly change the paths in the configuration.toml to point to the right places if running on different type of machine
   gencode eval pass_at_k --dataset_name test_101 --split_name train --evaluation_test_type private_tests --sample_rate 0.01
```

#### Running in docker

**Note**: 

You still need to run the docker from a linux machine!

The reason is that Mac (especially with Arm chip) is unable to emulate the underlying OS accurately enought to support the Google sandbox.


* From the root of the code, run:

```
docker run --security-opt seccomp=unconfined --privileged --cap-add=SYS_ADMIN --platform linux/amd64 \
-v ${PWD}:/app -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
597530458568.dkr.ecr.us-east-1.amazonaws.com/alphacode/code_contests:latest \
/bin/bash

```

* Inside the docker, run:
```
cd /app

```

* Now you can run the tests inside the docker


## Generating solutions to a dataset

The process includes s

* Selecting the problems you want to let the competitor solve and storing them in a dataset

* Running a generation loop on the dataset (async) where the competitor will return 0-n solutions to each item

* Preparing the results for evaluation - transforming the solutions to a dataset where the schema is:

  * Predictions
    - task_name
    - candidates
  * References
    * Test inputs
    * Test outputs
  
After this dataset is ready, you can store it in disk using `.save_to_disk(path)`

This code can be found in `gen_loop.py`


```python
cc = CodeContestDataProvider(dataset_location="deepmind/code_contests")
ds = cc.dataset['valid']
sub_ds = ds.filter(lambda example: example['name'] == "1548_D1. Gregor and the Odd Cows (Easy)")
solutions = asyncio.run(generate_candidate_solutions(sub_ds))
evaluation_set = cc.prepare_for_evaluation(
    predictions=solutions, source_of_truth=ds, evaluation_test_type="private_tests"
)
print(f"saving the output dataset to {output_path}")
evaluation_set.save_to_disk(output_path)


```


## Evaluating solutions at scale

Given a dataset of solutions in the schema described above, evaluate it.

This is done using the `gen_loop.py` module, e.g.:

```pass_at_k, inputs, evaluation_results = calculate_metrics(evaluation_set)```

the result includes

1. `pass@k` for multiple provided k values, as well as evaluation results.

2. The tests per task

3. The run results per candidate  (multi test results)


