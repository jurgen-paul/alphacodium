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

    # process base dataset
    output_dataset_name = 'valid_and_test_processed'
    base_path = os.path.expanduser(get_settings().etl.private_dataset_cache_dir)
    output_path = os.path.join(base_path, output_dataset_name)
    if True:
        data_provider = CodeContestDataProvider(dataset_location=dataset_name)
        for split_name in ['valid', 'test']:
            multiple_solutions_list =np.array([False] * len(data_provider.dataset[split_name]))
            ds = data_provider.dataset[split_name]
            for i, p in enumerate(ds):
                d_output = p['description'].split('Output\n')[1]
                if ('multiple solutions' in p['description'] or 'multiple possible solutions' in p['description']
                        or 'multiple possible solutions' in p['description'] or 'multiple' in d_output):
                    print(f"problem {i} has multiple solutions")
                    # print(f"=========\n{p['description']}\n=======\n\n")
                    multiple_solutions_list[i] = True
                else:
                    multiple_solutions_list[i] = False

                # sorting so that 'python' solutions will be first
                np_lang = np.array(p['solutions']['language'])
                inds_sorted = np.concatenate(
                    (np.argwhere(np_lang == 'PYTHON3'), np.argwhere(np_lang == 'CPP'), np.argwhere(np_lang == 'JAVA')))
                p['solutions']['solution'] = [p['solutions']['solution'][i[0]] for i in inds_sorted]
                p['solutions']['language'] = [p['solutions']['language'][i[0]] for i in inds_sorted]
            data_provider.dataset[split_name]=data_provider.dataset[split_name].add_column('multiple_solutions', multiple_solutions_list)
        data_provider.dataset.save_to_disk(output_path)

    split_name = 'valid'
    data_provider = CodeContestDataProvider(dataset_location=output_path)
    ds = data_provider.dataset[split_name]
    num_problems = len(ds)
    path_database= f'/Users/talrid/Git/alphaCodium/{split_name}_test_database.json'

    with open(path_database, 'r') as f:
        database_solutions = json.load(f)
        database_solutions[split_name] = OrderedDict(sorted(database_solutions[split_name] .items(), key=lambda x: int(x[0])))
    total_passed = 0
    total_failed = 0
    possible_multiple_solutions = 0
    for sol in database_solutions[split_name]:
        try:
            key_str = sol
            key_int = int(key_str)
            value = database_solutions[split_name][sol]
            passed_current= -1
            # scanning the iterations
            for v in value['codium'].values():
                if not v:
                    continue
                test_failed_generate = v['test_failed_generate']
                test_failed_private = v['test_failed_private']
                test_passed_generate = v['test_passed_generate']
                test_passed_private = v['test_passed_private']
                if (test_failed_generate + test_failed_private) == 0 and (test_passed_generate + test_passed_private) > 0:
                    print(f"problem {key_int} passed all tests")
                    passed_current=1
                    break
                else:
                    if ds[key_int]['multiple_solutions']:
                        passed_current=-1
                        # break
                    else:
                        passed_current = 0
            if passed_current==1:
                total_passed+=1
            elif passed_current==0:
                total_failed+=1
            elif passed_current==-1:
                possible_multiple_solutions+=1
        except Exception as e:
            print(f"Error: {e}")
            pass

    print(f"total_passed: {total_passed}, total_failed: {total_failed}, possible_multiple_solutions: {possible_multiple_solutions}")



if __name__ == "__main__":
    solve_dataset()
