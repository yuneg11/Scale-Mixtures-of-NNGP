from . import (
    regression,
    classification,
    classification_reg,
    importance,
)


def add_subparser(subparsers, increase_depth=False):

    if increase_depth:
        parser = subparsers.add_parser("experiments", help="experiments")
        subparsers = parser.add_subparsers(metavar="experiments")

    regression.add_subparser(subparsers)
    classification.add_subparser(subparsers)
    classification_reg.add_subparser(subparsers)
    importance.add_subparser(subparsers)
