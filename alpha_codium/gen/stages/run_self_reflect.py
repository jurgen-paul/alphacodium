import functools
import logging
import numpy as np
import yaml
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_self_reflect(self, problem):
    try:
        logger.info("--reflection stage--")
        use_recording =problem.get('use_recording', False)
        do_recording = problem.get('do_recording', False)
        recording_path = problem.get('recording_path', '')
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
            response_reflect = self.postprocess_response(response_reflect)  # try to include only the yaml part
            response_reflect_yaml = yaml.safe_load(response_reflect)
        problem['response_reflect'] = response_reflect
        problem['self_description'] = response_reflect_yaml['self_description']
        problem['s_possible_solutions'] = response_reflect_yaml['possible_solutions']
        return problem
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
