from . import (
    sample,
    regression,
    classification,
)


def add_subparser(subparsers):
    sample.add_subparser(subparsers)
    regression.add_subparser(subparsers)
    classification.add_subparser(subparsers)
