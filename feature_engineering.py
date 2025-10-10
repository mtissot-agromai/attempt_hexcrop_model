import pandas as pd
from itertools import combinations
import logging, os, sys, argparse
from utils import extract_all_features
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

error_file_handler = logging.FileHandler('errors.log')
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(stdout_handler)
logger.addHandler(error_file_handler)

def main(args):
    parser = argparse.ArgumentParser(description="Performing feature analysis for Hexcrop models.")

    # parser.add_argument("--nosplit", 
    #                     action='store_true',
    #                     dest='SPLIT_DATASET',
    #                     default=True,
    #                     help='Do not split the dataset into training and validation datasets')

    # parser.add_argument("--input", nargs=1, type=str, help='The path to input folder.')

    args = parser.parse_args()

    # ========== Parsing arguments ==========
    SPLIT_DATASET = args.SPLIT_DATASET

    OUTPUT_PATH='.'
    if args.output:
        OUTPUT_PATH=args.output[0]
    # ======================================


if __name__ == "__main__":
    main(sys.argv)