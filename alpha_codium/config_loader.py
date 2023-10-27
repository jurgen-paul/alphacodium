from os.path import abspath, dirname, join

from dynaconf import Dynaconf

PR_AGENT_TOML_KEY = "pr-agent"

current_dir = dirname(abspath(__file__))
global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    settings_files=[
        join(current_dir, f)
        for f in [
            "settings/.secrets.toml",
            "settings/configuration.toml",
            "settings/code_contests_prompts_baseline.toml",
            "settings/code_contests_prompts_reflect.toml",
            "settings/code_contests_prompt_more_test_cases.toml",
            "settings/code_contests_prompts_solve.toml",
            "settings/code_contests_prompts_possible_solutions.toml",
            "settings/code_contests_prompts_fix_solution.toml",
            "settings/code_contests_prompts_choose_best_solution.toml",
        ]
    ],
)


def get_settings():
    return global_settings
