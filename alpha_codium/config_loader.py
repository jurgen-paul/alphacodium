import pathlib
from os import listdir
from os.path import abspath, dirname, join, isfile
import glob

from dynaconf import Dynaconf

PR_AGENT_TOML_KEY = "pr-agent"

current_dir = dirname(abspath(__file__))
setting_dir = join(current_dir, "settings")


toml_files = list(pathlib.Path(join(setting_dir)).glob('*.toml')) # includes hidden files
global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    settings_files=toml_files,
)

# current_dir = dirname(abspath(__file__))
# global_settings = Dynaconf(
#     envvar_prefix=False,
#     merge_enabled=True,
#     settings_files=[
#         join(current_dir, f)
#         for f in [
#             "settings/.secrets.toml",
#             "settings/configuration.toml",
#             "settings/code_contests_prompts_baseline.toml",
#             "settings/code_contests_prompts_reflect.toml",
#             "settings/code_contests_prompts_solve.toml",
#             "settings/code_contests_prompts_fix_solution.toml",
#             "settings/code_contests_prompts_choose_best_solution.toml",
#             "settings/code_contests_prompts_analyze_trace.toml",
#             "settings/code_contests_prompts_generate_ai_tests.toml",
#         ]
#     ],
# )


def get_settings():
    return global_settings
