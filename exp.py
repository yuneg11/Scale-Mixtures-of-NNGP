import os
import argparse

import yaml
from pprint import pprint


def find_optimal_with_nll(results):
    min_valid_nll = float('inf')
    min_valid_result = None

    for result in results:
        valid_nll = float(float(result["valid-nll"]))
        if valid_nll < min_valid_nll:
            min_valid_nll = valid_nll
            min_valid_result = result

    return min_valid_result


def extract_args(result):
    args = "-nh " + str(result["num-hiddens"]) \
         + " -wv " + str(result["w-variance"]) \
         + " -bv " + str(result["b-variance"]) \
         + (" -a " + str(result["alpha"]) if "alpha" in result else "") \
         + (" -b " + str(result["beta"]) if "beta" in result else "") \
         + (" -bc " + str(result["burr_c"]) if "burr_c" in result else "") \
         + (" -bd " + str(result["burr_d"]) if "burr_d" in result else "") \
         + (" -lv " + str(result["last-layer-var"]) if "last-layer-var" in result else "") \
         + " -act " + str(result["activation"]) \
         + " -e " + str(result["eps-log-var"])
    return args


def find_optimal(args):
    with open(args.filepath, "r") as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    if args.type == "all":
        const, invgamma, burr12 = True, True, True
    elif args.type == "const":
        const, invgamma, burr12 = True, False, False
    elif args.type == "invgamma":
        const, invgamma, burr12 = False, True, False
    elif args.type == "burr12":
        const, invgamma, burr12 = False, False, True
    else:
        const, invgamma, burr12 = False, False, False
        print("No result type selected")
        exit(-1)

    const_results = []
    invgamma_results = []
    burr12_results = []

    for result in results:
        result_type = result["type"]

        if result_type == "const" and const:
            const_results.append(result)
        elif result_type == "invgamma" and invgamma:
            invgamma_results.append(result)
        elif result_type == "burr12" and burr12:
            burr12_results.append(result)

    if const:
        print(len(const_results))
        min_const_result = find_optimal_with_nll(const_results)
        pprint(min_const_result)

    if invgamma:
        print(len(invgamma_results))
        min_invgamma_result = find_optimal_with_nll(invgamma_results)
        pprint(min_invgamma_result)

    if burr12:
        print(len(burr12_results))
        min_burr12_result = find_optimal_with_nll(burr12_results)
        pprint(min_burr12_result)

    if args.extract_args:
        if const:
            print(extract_args(min_const_result))
        if invgamma:
            print(extract_args(min_invgamma_result))
        if burr12:
            print(extract_args(min_burr12_result))


def replicate_stat(args):
    for filename in sorted(os.listdir(args.root)):
        with open(os.path.join(args.root, filename), "r") as f:
            results = yaml.load(f, Loader=yaml.FullLoader)[:args.num_replicates]

        val_mean = sum([result["valid-nll"]
                        for result in results
                        if result["valid-nll"] != "nan"]) / len(results)
        val_std = (sum([(result["valid-nll"] - val_mean) ** 2
                        for result in results
                        if result["valid-nll"] != "nan"]) / len(results)) ** 0.5

        test_mean = sum([result["test-nll"]
                         for result in results
                         if result["test-nll"] != "nan"]) / len(results)
        test_std = (sum([(result["test-nll"] - test_mean) ** 2
                         for result in results
                         if result["test-nll"] != "nan"]) / len(results)) ** 0.5

        print(f"{filename.split('.')[0]}")
        print(f"Valid: {val_mean:.3f} ± {val_std:.3f}")
        print(f"Test:  {test_mean:.3f} ± {test_std:.3f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Tool")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    opt_parser = subparsers.add_parser("optimal", aliases=["opt"], help="find optimal")
    opt_parser.set_defaults(func=find_optimal)
    opt_parser.add_argument("filepath", type=str)
    opt_parser.add_argument("type", choices=["all", "const", "invgamma", "burr12"], default="all", nargs="?")
    opt_parser.add_argument("-e", "--extract-args", action="store_true")

    rep_parser = subparsers.add_parser("replicate", aliases=["rep"], help="replicate stats")
    rep_parser.set_defaults(func=replicate_stat)
    rep_parser.add_argument("root", type=str)
    rep_parser.add_argument("-n", "--num-replicates", type=int, default=10)

    args = parser.parse_args()
    args.func(args)
