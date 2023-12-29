import copy
import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.gen.utils import postprocess_response
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_fix_self_reflect(self, problem):
    try:
        logger.info("--fix reflection stage--")
        use_recording = problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_fix_reflection")
        if use_recording:
            response_fix_reflect = np.load(recording_path + 'fix_reflect.npy', allow_pickle=True).tolist()
            logger.info("Using recording")
            logger.debug(f"response_fix_reflect:\n{response_fix_reflect}")
        else:
            response_fix_reflect, _ = await retry_with_fallback_models(f)
            if do_recording:
                np.save(recording_path + 'fix_reflect.npy', response_fix_reflect)
        response_fix_reflect = response_fix_reflect.rstrip("` \n")
        try:
            response_fix_reflect_yaml = yaml.safe_load(response_fix_reflect)
        except yaml.YAMLError:
            response_fix_reflect = postprocess_response(response_fix_reflect)  # try to include only the yaml part
            response_fix_reflect_yaml = yaml.safe_load(response_fix_reflect)

        actual_number_of_tests = len(problem['public_tests']['input'])
        calculated_number_of_tests = len(response_fix_reflect_yaml['fixed_tests_explanations'])
        if actual_number_of_tests != calculated_number_of_tests:
            logger.error(f"Error: number of tests in fix self-reflection ({calculated_number_of_tests}) "
                         f"does not match the actual number of tests ({actual_number_of_tests})")
            exit(-1)

        problem['response_fix_reflect'] = response_fix_reflect
        problem['tests_explanations'] = response_fix_reflect_yaml['fixed_tests_explanations']
        problem['tests_explanations_str'] = response_fix_reflect.split('tests_explanations:')[1]

        # re-order the public tests from easiest to hardest
        problem['public_tests']['original'] = copy.deepcopy(problem['public_tests'])
        problem['public_tests']['input'] = [t['input'] for t in problem['tests_explanations']]
        problem['public_tests']['output'] = [t['output'] for t in problem['tests_explanations']]
        problem['public_tests']['explanation'] = [t['explanation'] for t in problem['tests_explanations']]

        return problem
    except Exception as e:
        logging.error(f"Failed 'run_fix_self_reflect', Error: {e}. Continuing to next stage.")
        return problem
