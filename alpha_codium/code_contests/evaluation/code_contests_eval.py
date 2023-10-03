# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The CodeContestsEval metric estimates the pass@k metric for code synthesis.
This is an evaluation harness for the code_contests problem solving dataset
described in the paper "Evaluating Large Language Models Trained on Code"
(https://arxiv.org/abs/2107.03374)."""

import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import datasets
import numpy as np

import evaluate
from code_contests_tester import ProgramStatus, Py3TesterSandboxer, TestOptions

_CITATION = """\
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan \
and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards \
and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray \
and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf \
and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray \
and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser \
and Mohammad Bavarian and Clemens Winter and Philippe Tillet \
and Felipe Petroski Such and Dave Cummings and Matthias Plappert \
and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss \
and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak \
and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain \
and William Saunders and Christopher Hesse and Andrew N. Carr \
and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa \
and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati \
and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei \
and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

_DESCRIPTION = """\
This metric implements the evaluation harness for the HumanEval problem solving dataset
described in the paper "Evaluating Large Language Models Trained on Code"
(https://arxiv.org/abs/2107.03374).
"""

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of candidates to evaluate. Each candidates should be a list
        of strings with several code candidates to solve the problem.
    references: a list with a test for each prediction. Each test should evaluate the
        correctness of a code candidate.
    k: number of code candidates to consider in the evaluation (Default: [1, 10, 100])
    num_workers: number of workers used to evaluate the canidate programs (Default: 4).
    timeout:
Returns:
    pass_at_k: dict with pass rates for each k
    results: dict with granular results of each unittest
Examples:
    >>> code_eval = evaluate.load("code_eval")
    >>> test_cases = ["assert add(2,3)==5"]
    >>> candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
    >>> pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1, 2])
    >>> print(pass_at_k)
    {'pass@1': 0.5, 'pass@2': 1.0}
"""

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""

import os

if os.environ.get("REMOTE"):
    base_path = '/home/ec2-user/.pyenv/versions/3.10.13'
else:
    base_path = '/Users/assaf/.pyenv/versions/3.10.11'
# os.path.join(base_path, "lib/python3.10")
print(f"using interpreter in {base_path}")

# tester = Py3TesterSandboxer("/home/ec2-user/.pyenv/versions/3.10.13/bin/python3.10",
#                             ["/home/ec2-user/.pyenv/versions/3.10.13/lib"])

tester = Py3TesterSandboxer("/usr/bin/python3.11",
                             ["/usr/lib/python3.11"])

options = TestOptions()
options.num_threads = 4
options.stop_on_first_failure = True


class TestsRunner:
    def __init__(self, tester: Py3TesterSandboxer, test_options: TestOptions, compare_func: Callable):
        self.tester = tester
        self.test_options = test_options
        self.compare_func = compare_func

    def run_test(self, test_id, candidate_id, candidate, test_inputs, options, tests_outputs, compare_func):
        result = tester.test(candidate, test_inputs, options, tests_outputs, compare_func)
        return test_id, candidate_id, result


def compare_func(a, b):
    return a == b


program = """
x = input()
print(x)

"""


def test_interpreter():
    result = tester.test(program, ["hello"], options, ["hello\n"], compare_func)
    print(f"compilation results:{result.compilation_result.program_status}")
    print(result.compilation_result.sandbox_result)
    print(result.compilation_result.stderr)

    for i, test_res in enumerate(result.test_results):
        print(f"test-{i} :: status={test_res.program_status}, pased={test_res.passed}")
        print("=====================================================================")
        print(test_res.stdout)
        print("=====================================================================")


test_interpreter()


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CodeContestsEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": {
                        "task_name": datasets.Value("string"),
                        "solution_candidates": datasets.Sequence(datasets.Value("string"))
                    },
                    "references": {
                        "tests_inputs": datasets.Sequence(datasets.Value("string")),
                        "tests_outputs": datasets.Sequence(datasets.Value("string"))
                    }
                }
            ),
            homepage="https://github.com/openai/human-eval",
            codebase_urls=["https://github.com/openai/human-eval"],
            reference_urls=["https://github.com/openai/human-eval"],
            license=_LICENSE,
        )

    def _compute(self, predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0):
        """Returns the scores"""

        if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
            raise ValueError(_WARNING)

        if os.name == "nt":
            raise NotImplementedError("This metric is currently not supported on Windows.")

        runner = TestsRunner(tester, options, compare_func)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            for prediction, reference in zip(predictions, references):
                task_name = prediction['task_name']
                candidates = prediction['solution_candidates']
                test_inputs = reference['tests_inputs']
                tests_outputs = reference['tests_outputs']
                print(f"submitting task {task_name} with {len(candidates)}")
                for candidate_id, candidate in enumerate(candidates):
                    print(f"\tsubmitting candidate {candidate_id}")
                    args = (task_name, candidate_id,candidate, test_inputs, options, tests_outputs, compare_func)
                    future = executor.submit(runner.run_test, *args)
                    futures.append(future)
                    completion_id[task_name] += 1
                    n_samples += 1

            for future in as_completed(futures):
                task_id, candidate_id, test_result = future.result()
                results[task_id].append((candidate_id, test_result))

        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [all(test_result.passed for test_result in r[1].test_results) for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

        return pass_at_k, results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
