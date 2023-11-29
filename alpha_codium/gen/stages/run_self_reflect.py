import functools
import logging
import numpy as np
import yaml

from alpha_codium.config_loader import get_settings
from alpha_codium.gen.utils import postprocess_response
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_self_reflect(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--reflection stage--")
            use_recording = problem.get('use_recording', False)
            do_recording = problem.get('do_recording', False)
            recording_path = problem.get('recording_path', '')

            if get_settings().self_reflect.get('randomize_best_solution', False):
                problem['min_num_of_solutions'] = 2
            else:
                problem['min_num_of_solutions'] = 3

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
            problem['response_reflect'] = response_reflect

            try:
                problem['self_description'] = '- ' + '\n- '.join(response_reflect_yaml['self_description'])
            except:
                problem['self_description'] = response_reflect_yaml['self_description']

            problem['s_possible_solutions'] = response_reflect_yaml['possible_solutions']
            problem['s_possible_solutions_str'] = response_reflect.split('possible_solutions:')[1]

            if get_settings().self_reflect.get('prefer_dynamic_programming', False):
                for s in problem['s_possible_solutions']:
                    if 'dynamic' in s['name'].lower() or 'dfs' in s['name'].lower() or 'bfs' in s['name'].lower():
                        logger.info(f"Enforcing dynamic programming: {s['name']}")
                        problem['s_possible_solutions'] = [s]
                        problem['s_possible_solutions_str'] = s
                        break
            if get_settings().self_reflect.get('randomize_best_solution', False):
                i = problem['iteration'] % len(problem['s_possible_solutions'])
                s = problem['s_possible_solutions'][i]
                logger.info(f"Enforcing randomize best solution: {s['name']}")
                problem['s_possible_solutions'] = [s]
                problem['s_possible_solutions_str'] = s
            return problem
        except Exception as e:
            logging.error(f"'run_self_reflect' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
