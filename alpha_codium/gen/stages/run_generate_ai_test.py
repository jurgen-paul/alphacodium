import functools
import logging
import numpy as np
import yaml
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_generate_ai_tests(self, problem):
    try:
        logger.info("--generate ai tests stage--")
        use_recording =problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')

        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_generate_ai_tests")
        if use_recording:
            response_problem_tests = np.load(recording_path + 'problem_tests.npy', allow_pickle=True).tolist()
            logger.info("Using recording")
            logger.debug(f"response_solve:\n{response_problem_tests}")
        else:
            response_problem_tests, _ = await retry_with_fallback_models(f)
            if do_recording:
                np.save(recording_path + 'problem_tests.npy', response_problem_tests)
        response_problem_tests = response_problem_tests.rstrip("` \n")
        if response_problem_tests.startswith("```yaml"):
            response_problem_tests = response_problem_tests[8:]
        problem['problem_ai_tests'] = yaml.safe_load(response_problem_tests)['tests']
        problem['problem_ai_simple_test'] =  problem['problem_ai_tests'][0]

        # adding public tests to the beginning of the list
        for public_input, public_output in zip(problem['public_tests']['input'],
                                               problem['public_tests']['output']):
            problem['problem_ai_tests'].insert(0, {'input': public_input, 'output': public_output})
        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
