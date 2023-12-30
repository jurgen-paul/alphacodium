import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.gen.stages.run_validate_ai_test import run_validate_ai_tests
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_generate_ai_tests(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--generate ai tests stage--")
            use_recording = problem.get('use_recording', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')
            validate_ai_tests = get_settings().get('ai_tests.validate_ai_tests', False)

            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_generate_ai_tests")
            if use_recording:
                response_problem_tests = np.load(recording_path + 'problem_ai_tests.npy', allow_pickle=True).tolist()
                logger.info("Using recording")
                logger.debug(f"response_solve:\n{response_problem_tests}")
            else:
                response_problem_tests, _ = await send_inference(f)
                if do_recording:
                    np.save(recording_path + 'problem_ai_tests.npy', response_problem_tests)

            # clean up and parse the response
            response_problem_tests = response_problem_tests.rstrip("` \n")
            if response_problem_tests.startswith("```yaml"):
                response_problem_tests = response_problem_tests[8:]
            problem['problem_ai_tests'] = yaml.safe_load(response_problem_tests)['tests']
            problem['problem_ai_simple_test'] =  problem['problem_ai_tests'][0]

            if validate_ai_tests:
                problem = await run_validate_ai_tests(self, problem)

            # adding public tests to the beginning of the list
            for public_input, public_output in zip(problem['public_tests']['input'],
                                                   problem['public_tests']['output']):
                # to the beginning of the list
                problem['problem_ai_tests'].insert(0, {'input': public_input, 'output': public_output})
                # to the end of the list
                problem['problem_ai_tests'].append({'input': public_input, 'output': public_output})

            return problem
        except Exception as e:
            logging.error(f"'generate ai tests' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
