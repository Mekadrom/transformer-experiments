from utils import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', type=str, required=True)

args = parser.parse_args()

average_checkpoints(None, args.run_name)
