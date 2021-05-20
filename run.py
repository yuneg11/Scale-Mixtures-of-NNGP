import os
import sys
import argparse

import experiments

import warnings

warnings.filterwarnings("ignore")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'


def main(raw_args):
    parser = argparse.ArgumentParser(description="NNGP Experiments")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    experiments.add_subparser(subparsers, increase_depth=False)

    args = parser.parse_args(raw_args)
    args.func(**vars(args))


if __name__ == "__main__":
    main(sys.argv[1:])
