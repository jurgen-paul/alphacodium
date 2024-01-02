import argparse

from alpha_codium.gen.dataset_solver import solve_dataset
from alpha_codium.log import get_logger, setup_logger
from alpha_codium.settings.config_loader import get_settings

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="valid_and_test_processed")
parser.add_argument("--split_name", type=str, default="valid")
if __name__ == "__main__":
    args = parser.parse_args()
    setup_logger()
    solve_dataset(dataset_name=args.dataset_name,
                  split_name=args.split_name)
