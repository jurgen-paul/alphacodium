import asyncio
import copy
import json
import os
import shutil
from collections import OrderedDict
import numpy as np

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.config_loader import get_settings
from alpha_codium.log import get_logger, setup_logger

logger = get_logger(__name__)




def solve_dataset(dataset_name='valid_and_test', split_name='valid'):
    split_name = 'valid'
    base_path = os.path.expanduser(get_settings().etl.private_dataset_cache_dir)
    dataset_name = 'valid_and_test_processed'
    output_path = os.path.join(base_path, dataset_name)
    data_provider = CodeContestDataProvider(dataset_location=output_path)
    ds = data_provider.dataset[split_name]

    solution_path_database = f'/Users/talrid/Git/alphaCodium/{split_name}_test_database.json'

    with open(solution_path_database, 'r') as f:
        database_solutions = json.load(f)
        database_solutions[split_name] = OrderedDict(
            sorted(database_solutions[split_name].items(), key=lambda x: int(x[0])))
    total_passed = 0
    total_failed = 0
    possible_multiple_solutions = 0
    for sol in database_solutions[split_name]:
        try:
            key_str = sol
            key_int = int(key_str)
            problem = ds[key_int]
            if problem.get('is_valid_problem', True) is False:
                logger.info(f"problem {key_int} is not valid")
                continue
            codium_solution = database_solutions[split_name][sol]
            passed_current = -1

            # scanning the iterations
            v_iter =[v for v in codium_solution['codium'].values() if 'solution' in v]
            if 'simulation' in codium_solution['codium']:
                v_iter_simulation = [v for v in codium_solution['codium']['simulation'].values() if 'solution' in v]
                v_iter += v_iter_simulation
            for v in v_iter:
                if not v:
                    continue
                test_failed_generate = v['test_failed_generate']
                test_failed_private = v['test_failed_private']
                test_passed_generate = v['test_passed_generate']
                test_passed_private = v['test_passed_private']
                if 'test_timeout_generate' in v:
                    test_timeout_generate = v['test_timeout_generate']
                    test_timeout_private = v['test_timeout_private']
                else:
                    test_timeout_generate = 0
                    test_timeout_private = 0
                if ((test_failed_generate + test_timeout_generate + test_failed_private + test_timeout_private) == 0 and
                        (test_passed_generate + test_passed_private) > 0):
                    print(f"problem {key_int} passed all tests")
                    passed_current=1
                    break
                else:
                    # if ds[key_int]['multiple_solutions']:
                    #     passed_current = -1
                    #     # break
                    # else:
                    #     passed_current = 0
                    passed_current = 0
            if passed_current == 1:
                total_passed += 1
            elif passed_current == 0:
                total_failed += 1
            elif passed_current == -1:
                possible_multiple_solutions+=1
        except Exception as e:
            print(f"Error: {e}")
            pass

    print(f"total_passed: {total_passed}, total_failed: {total_failed}, possible_multiple_solutions: {possible_multiple_solutions}")
    print(f"pass rate: {total_passed/(total_passed+total_failed)}")



if __name__ == "__main__":
    solve_dataset()
