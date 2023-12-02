import asyncio
import json
import os
import shutil
from collections import OrderedDict

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.config_loader import get_settings
from alpha_codium.gen.coding_competitor import CodeContestsCompetitor
from alpha_codium.gen.utils import evaluate_solution_on_subset
from alpha_codium.log import get_logger, setup_logger

logger = get_logger(__name__)




def solve_dataset(dataset_name='valid_and_test_processed', split_name='test'):

    # load dataset
    data_provider = CodeContestDataProvider(dataset_location=dataset_name)
    num_problems = len(data_provider.dataset[split_name])
    base_path = os.getcwd()
    path_database= f'{base_path}/{split_name}_test_database.json'
    path_database_backup= f'{base_path}/{split_name}_test_database_backup.json'
    log_path = f'{base_path}/example.log'
    # working_dir = f'{base_path}/alpha_codium/gen'
    get_settings().solve.reduce_verbose = True

    ## load database
    try:
        with open(path_database, 'r') as f:
            database = json.load(f)
            database[split_name] = OrderedDict(sorted(database[split_name] .items(), key=lambda x: int(x[0])))
    except:
        print(f"Failed to load database from {path_database}")
        database = {split_name: {}}

    # for d in database[split_name]:
    #     if 'codium' in database[split_name][d]:
    #         for it in database[split_name][d]['codium']:
    #             if 'iteration' in it:
    #                 database[split_name][d]['codium'][it] = None

    # iterate on problems
    for problem_number in range(0, num_problems):
        # skip if already ran
        prev = database[split_name].get(str(problem_number), {}).get('codium', {}).get('iteration_0', {})
        if not ((prev=={}) or (prev is None)):
            print(f"problem_number {problem_number} already ran")
            continue

        if data_provider.dataset[split_name][problem_number].get('is_valid_problem', True) is False:
            logger.info(f"problem {problem_number} is not valid")
            continue

        sim_results = database[split_name].get(str(problem_number), {}).get('codium', {}).get('simulation',{})
        already_solved = False
        try:
            for it_name in sim_results:
                it_vals=sim_results[it_name]
                if (it_vals['test_failed_private'] + it_vals['test_timeout_private'] + it_vals['test_failed_generate'] + it_vals['test_timeout_generate'] == 0) \
                    and (it_vals['test_passed_private'] + it_vals['test_passed_generate'] > 0):
                    logger.info(f"problem {problem_number} already solved in simulation")
                    already_solved = True
                    break
        except:
            pass
        if already_solved:
            continue

        shutil.rmtree(log_path, ignore_errors=True)
        os.chdir(base_path)
        setup_logger()
        logger.info(f"problem_number: {problem_number}")

        problem_name = data_provider.dataset[split_name][int(problem_number)]['name']
        logger.info(f"problem_name: {problem_name}")
        problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name)
        logger.info(f"problem['cf_tags']: {problem['cf_tags']}")
        # if not problem['private_tests']['input']:
        #     logger.info("No private tests for this problem")

        problem_database = {problem_number: {}}

        # evaluate prev public solutions first
        evaluate_prev_solutions = get_settings().get("dataset.evaluate_prev_solutions", False)
        if evaluate_prev_solutions:
            try:
                if not problem['solutions']['solution']:
                    logger.info("No public solutions for this problem")
                found_solution = False
                for index_published, sol_published in enumerate(problem['solutions']['solution']):
                    logger.info(f"evaluating public solution {index_published} on private tests...")
                    test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset(
                        'private_tests', problem,
                        sol_published, silent=True)
                    logger.info(f"evaluating public solution {index_published} on generated tests...")
                    test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset(
                        'generated_tests',
                        problem,
                        sol_published,
                        silent=True)
                    if (
                            test_failed_private == test_failed_generate == test_timeout_private == test_timeout_generate == 0) \
                            and test_passed_private + test_passed_generate > 0:
                        logger.info(f"sol_published index {index_published} passed all tests:\n{sol_published}")
                        found_solution = True
                        problem_database[problem_number]['public_solution'] = {}
                        problem_database[problem_number]['public_solution']['index'] = index_published
                        problem_database[problem_number]['public_solution']['solution'] = sol_published
                        problem_database[problem_number]['public_solution']['test_passed_private'] = test_passed_private
                        problem_database[problem_number]['public_solution']['test_failed_private'] = test_failed_private
                        problem_database[problem_number]['public_solution']['test_passed_generate'] = test_passed_generate
                        problem_database[problem_number]['public_solution']['test_failed_generate'] = test_failed_generate
                        break
                if not found_solution:
                    problem_database[problem_number]['public_solution'] = {}
                    logger.info(f"None of the public solutions passed all tests")
            except:
                pass


        # solve problem
        problem_database[problem_number]['codium'] = {}
        if 'simulation' in database[split_name].get(str(problem_number), {}).get('codium', {}):
            problem_database[problem_number]['codium']['simulation'] = database[split_name].get(str(problem_number), {}).get('codium', {})['simulation']

        solver = CodeContestsCompetitor()
        setting = get_settings()
        for iteration in range(setting.get("solve.max_iterations", 1)):
            it_str = f"iteration_{iteration}"

            # run policy
            setting.self_reflect.randomize_best_solution = False
            setting.self_reflect.prefer_dynamic_programming = False
            if iteration == 1:
                # setting.self_reflect.randomize_best_solution = True
                pass
            elif iteration == 2:
                setting.self_reflect.prefer_dynamic_programming = True
                setting.self_reflect.randomize_best_solution = True
            elif iteration == 3:
                setting.self_reflect.randomize_best_solution = True

            problem_database[problem_number]['codium'][it_str] = {}

            solution = solver.solve_problem(problem, iteration)
            if not solution:
                logger.info(f"codium failed to solve problem {problem_number} in iteration {iteration}")
                continue
            logger.info(f"evaluating solution on public tests...")
            test_results, test_passed_public, test_failed_public, test_timeout_public = evaluate_solution_on_subset('public_tests', problem, solution, silent=True)

            logger.info(f"evaluating solution on private tests...")
            test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset('private_tests', problem, solution, silent=True)

            logger.info(f"evaluating solution on generated tests...")
            test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset('generated_tests', problem, solution, silent=True)


            logger.info(f"\ntest_passed_public: {test_passed_public}, test_failed_public: {test_failed_public}, test_timeout_public: {test_timeout_public}\n"
                        f"test_passed_private: {test_passed_private}, test_failed_private: {test_failed_private}, test_timeout_private: {test_timeout_private}\n"
                        f"test_passed_generate: {test_passed_generate}, test_failed_generate: {test_failed_generate}, test_timeout_generate: {test_timeout_generate}\n")

            problem_database[problem_number]['codium'][it_str]['solution'] = solution
            problem_database[problem_number]['codium'][it_str]['test_passed_private'] = test_passed_private
            problem_database[problem_number]['codium'][it_str]['test_failed_private'] = test_failed_private
            problem_database[problem_number]['codium'][it_str]['test_timeout_private'] = test_timeout_private
            problem_database[problem_number]['codium'][it_str]['test_passed_generate'] = test_passed_generate
            problem_database[problem_number]['codium'][it_str]['test_failed_generate'] = test_failed_generate
            problem_database[problem_number]['codium'][it_str]['test_timeout_generate'] = test_timeout_generate
            problem_database[problem_number]['codium'][it_str]['test_passed_public'] = test_passed_public
            problem_database[problem_number]['codium'][it_str]['test_failed_public'] = test_failed_public
            problem_database[problem_number]['codium'][it_str]['test_timeout_public'] = test_timeout_public
            with open(log_path, 'r') as f:
                log = f.read()
                problem_database[problem_number]['codium'][it_str]['log'] = log
                os.makedirs(f'./{split_name}_logs/', exist_ok=True)
                shutil.copyfile(log_path, f'./{split_name}_logs/test_{problem_number}_{it_str}.log')

            if (test_failed_private == 0 and test_failed_generate == 0 and
                    test_timeout_private == 0 and test_timeout_generate == 0 and
                    (test_passed_private + test_passed_generate) > 0):
                logger.info(f"codium solved problem {problem_number} in iteration {iteration}")
                break
            else:
                logger.info(f"codium failed to solve problem {problem_number} in iteration {iteration}")
        database[split_name][problem_number] = problem_database[problem_number]
        with open(path_database, 'w') as f:
            json.dump(database, f)
        with open(path_database_backup, 'w') as f:
            json.dump(database, f)



if __name__ == "__main__":
    solve_dataset()
