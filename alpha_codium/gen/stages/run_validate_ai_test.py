import functools
import logging
import yaml
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_validate_ai_tests(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--validate ai tests stage--")
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_validate_ai_tests")
            response_problem_tests, _ = await retry_with_fallback_models(f)
            response_problem_tests = response_problem_tests.rstrip("` \n")
            if response_problem_tests.startswith("```yaml"):
                response_problem_tests = response_problem_tests[8:]
            problem['problem_ai_tests'] = yaml.safe_load(response_problem_tests)['tests']
            for p in problem['problem_ai_tests']:
                p['input'] = p['input'].replace('\\n', '\n')
                p['output'] = p['output'].replace('\\n', '\n')
            return problem
        except Exception as e:
            logging.error(f"'validate ai tests' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
