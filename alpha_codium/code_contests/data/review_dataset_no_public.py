import asyncio
import copy
import json
import os
import shutil
from collections import OrderedDict
import numpy as np
from datasets import Dataset

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.config_loader import get_settings
from alpha_codium.log import get_logger, setup_logger
from alpha_codium.gen.utils import evaluate_solution_on_subset
from alpha_codium.code_contests.data.valid_prob3 import run_problem_3
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


        # problem 3 valid
        ind_problem_valid = 3
        dataset_valid_dict = data_provider.dataset['valid'].to_dict()
        if True:
            p_3 = data_provider.dataset['valid'][ind_problem_valid]
            p_3_tests = p_3['generated_tests']
            is_valid_test = [True] * len(p_3_tests['input'])
            count_false = 0
            count_correct = 0
            for i, input in enumerate(p_3_tests['input']):
                n, m, x = input.splitlines()[0].split()
                n = int(n)
                m = int(m)
                a = input.splitlines()[1].split()
                b = input.splitlines()[2].split()
                if (n != len(a) or m != len(b)):
                    count_false += 1
                    is_valid_test[i] = False
                else:
                    count_correct += 1
            print(f"count_false: {count_false}, count_correct: {count_correct}")
            dataset_valid_dict['generated_tests'][ind_problem_valid]['is_valid_test'] = is_valid_test
            data_provider.dataset['valid'] = Dataset.from_dict(dataset_valid_dict)

        data_provider.dataset.save_to_disk(output_path)

    split_name = 'valid'
    data_provider = CodeContestDataProvider(dataset_location=output_path)
    ds = data_provider.dataset[split_name]

    path_database_solutions= f'/Users/talrid/Git/alphaCodium/{split_name}_test_database.json'
    with open(path_database_solutions, 'r') as f:
        database_solutions = json.load(f)
        database_solutions[split_name] = OrderedDict(sorted(database_solutions[split_name] .items(), key=lambda x: int(x[0])))

    for sol in database_solutions[split_name]:
        try:
            key_str = sol
            key_int = int(key_str)
            value = database_solutions[split_name][sol]
            # if value['public_solution']:
            #     continue


            possible_bad_generated_tests = False
            passed_problem = False
            iter=-1
            min_errors = 200
            for i,v in enumerate(value['codium'].values()):
                if list(value['codium'].keys())[i]=='simulation':
                    continue
                if not v:
                    continue
                if 'test_passed_public' in v:
                    test_passed_public = v['test_passed_public']
                    test_failed_public = v['test_failed_public']
                else:
                    test_passed_public = 1
                    test_failed_public = 0
                if 'test_timeout_generate' in v:
                    test_timeout_generate = v['test_timeout_generate']
                    test_timeout_private = v['test_timeout_private']
                else:
                    test_timeout_generate = 0
                    test_timeout_private = 0
                test_failed_generate = v['test_failed_generate']
                test_failed_private = v['test_failed_private']
                test_passed_generate = v['test_passed_generate']
                test_passed_private = v['test_passed_private']

                if test_failed_public == 0 and test_passed_public > 0:
                    if test_timeout_generate == test_timeout_private == 0:
                        if test_failed_generate + test_failed_private == 0:
                            passed_problem = True
                            break

                if test_failed_public== 0 and test_failed_private == 0 and test_failed_generate > 0:
                    if min_errors > test_failed_generate:
                        possible_bad_generated_tests = True
                        min_errors = test_failed_generate
                        iter = i
                    # break
                # if 'test_timeout_generate' in v:
                #     test_timeout_generate = v['test_timeout_generate']
                #     test_timeout_private = v['test_timeout_private']
                # else:
                #     test_timeout_generate = 0
                #     test_timeout_private = 0
                #



            if possible_bad_generated_tests and not passed_problem:
                v = list(value['codium'].values())[iter]
                print(f"\nproblem {key_int} is suspicious")
                if 'test_passed_public' in v:
                    print(f"test_passed_public: {v['test_passed_public']}, test_failed_public: {v['test_failed_public']}")
                print(f"test_passed_generate: {v['test_passed_generate']}, test_failed_generate: {v['test_failed_generate']}")
                print(f"test_passed_private: {v['test_passed_private']}, test_failed_private: {v['test_failed_private']}")
                print(v['solution'])

                test_results, test_passed_private, test_failed_private, test_timeout_private \
                    = evaluate_solution_on_subset('generated_tests', ds[key_int], v['solution'], silent=False)
                exit(-1)
        except Exception as e:
            print(f"Error: {e}")
            pass

    # print(f"total_passed: {total_passed}, total_failed: {total_failed}, possible_multiple_solutions: {possible_multiple_solutions}")



if __name__ == "__main__":
    solve_dataset()
