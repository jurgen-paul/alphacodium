import os

from alpha_codium.config_loader import get_settings
from alpha_codium.log import get_logger

logger = get_logger(__name__)


def set_configurations(problem):
    # configurations
    problem = {k: problem.get(k) for k in ["name", "description", "public_tests"]}
    do_recording = get_settings().get("solve.do_recording", False)
    use_recording = get_settings().get("solve.use_recording", False)
    if use_recording or do_recording:
        # recording_path = f"./code_contests/{problem['name']}/{get_settings().config['model']}/"
        recording_path = f"/Users/talrid/Git/alphaCodium/alpha_codium/gen/code_contests/{problem['name']}/{get_settings().config['model']}/"
        logger.info(f"recording_path: {recording_path}\ndo_record: {do_recording}\nuse_record: {use_recording}")
        if do_recording:
            os.makedirs(recording_path, exist_ok=True)
        problem["recording_path"] = recording_path
    else:
        problem["recording_path"] = ''
    problem["do_recording"] = do_recording
    problem["use_recording"] = use_recording
    problem['number_of_ai_tests'] = get_settings().get("solve.number_of_ai_tests", 6)
    problem['diff_that_didnt_help'] = ''

    # initialize passed tests field
    problem['passed_tests'] = {}
    problem['passed_tests']['inputs'] = []
    problem['passed_tests']['outputs'] = []
    return problem