import argparse
from pprint import pprint

import yaml


def find_optimal_with_nll(results):
    min_valid_nll = float('inf')
    min_valid_result = None

    for result in results:
        valid_nll = float(result["valid-nll"])
        if valid_nll < min_valid_nll:
            min_valid_nll = valid_nll
            min_valid_result = result

    return min_valid_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("type", choices=["all", "const", "invgamma", "burr12"], default="all", nargs="?")
    args = parser.parse_args()

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
