# cd /mnt/talr/alpha_codium_remote
# python3 -m alpha_codium.solve_dataset \
# --database_solution_path="./o1-preview-baseline.json" \
# --split_name=valid \
# --dataset_name="/mnt/talr/alpha_codium_remote/alpha_codium/code_contests/data/valid_and_test_processed"
#
#
# --database_solution_path="./claude3_opus.json" \
import argparse
import asyncio

from alpha_codium.gen.dataset_solver import solve_dataset
from alpha_codium.log import get_logger, setup_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="valid_and_test_processed") # /mnt/talr/alpha_codium_remote/alpha_codium/code_contests/data/valid_and_test_processed
parser.add_argument("--split_name", type=str, default="valid")
parser.add_argument("--database_solution_path", type=str, default="")
async def main():
    args = parser.parse_args()
    setup_logger()

    # set default database_solution_path
    args.database_solution_path = args.database_solution_path
    if not args.database_solution_path:
        args.database_solution_path = f"./{args.dataset_name}_{args.split_name}_solution_database.json"
        logger.info(f"args.database_solution_path: {args.database_solution_path}")

    await solve_dataset(dataset_name=args.dataset_name,
                  split_name=args.split_name,
                  database_solution_path=args.database_solution_path)

if __name__ == "__main__":
    asyncio.run(main())
