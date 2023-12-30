import functools
import logging
import numpy as np
import yaml

from alpha_codium.gen.stages.run_fix_self_reflect import run_fix_self_reflect
from alpha_codium.gen.utils import postprocess_response
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_self_reflect(self, problem, double_validation=True):
    counter_retry = 0
    while True:
        try:
            logger.info("--reflection stage--")
            validate_self_reflection = problem.get('self_reflection.validate_self_reflection', False)
            use_recording = problem.get('use_recording', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')

            actual_number_of_tests = len(problem['public_tests']['input'])
            problem['actual_number_of_tests'] = actual_number_of_tests
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompt_reflect")
            if use_recording:
                response_reflect = np.load(recording_path + 'reflect.npy', allow_pickle=True).tolist()
                logger.info("Using recording")
                logger.debug(f"response_reflect:\n{response_reflect}")
            else:
                response_reflect, _ = await retry_with_fallback_models(f)
                if do_recording:
                    np.save(recording_path + 'reflect.npy', response_reflect)
            response_reflect = response_reflect.rstrip("` \n")
            try:
                response_reflect_yaml = yaml.safe_load(response_reflect)
            except yaml.YAMLError:
                response_reflect = postprocess_response(response_reflect)  # try to include only the yaml part
                response_reflect_yaml = yaml.safe_load(response_reflect)

            actual_number_of_tests = len(problem['public_tests']['input'])
            calculated_number_of_tests = len(response_reflect_yaml['tests_explanations'])
            if actual_number_of_tests != calculated_number_of_tests:
                raise (f"Error: number of tests in self-reflection ({calculated_number_of_tests}) "
                             f"does not match the actual number of tests ({actual_number_of_tests})")
            problem['response_reflect'] = response_reflect
            try:
                problem['self_reflection'] = '- ' + '\n- '.join(response_reflect_yaml['self_reflection'])
            except:
                problem['self_reflection'] = response_reflect_yaml['self_reflection']
            problem['tests_explanations'] = response_reflect_yaml['tests_explanations']
            problem['tests_explanations_str'] = response_reflect.split('tests_explanations:')[1]

            # double validation self-reflection
            if validate_self_reflection:
                problem = await run_fix_self_reflect(self, problem)

            for s in problem['tests_explanations']:
                s['input'] = s['input'].replace('\\n', '\n')
                s['output'] = s['output'].replace('\\n', '\n')
                s['explanation'] = s['explanation'].replace('\\n', '\n')

            return problem
        except Exception as e:
            logging.error(f"'run_self_reflect' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
