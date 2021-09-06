import os
import bdb
import argparse
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="NNGP Experiments")
parser.add_argument("-g", "--gpu", type=int)
parser.add_argument("-f", "--fraction", type=float)
args, main_args = parser.parse_known_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if args.fraction is not None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.fraction)


import experiments


def main(raw_args):
    parser = argparse.ArgumentParser(description="NNGP Experiments")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    experiments.add_subparser(subparsers, increase_depth=False)

    args = parser.parse_args(raw_args)

    try:
        args.func(**vars(args))
    except KeyboardInterrupt or bdb.BdbQuit:
        print("Stopped.")


if __name__ == "__main__":
    main(main_args)
