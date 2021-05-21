from . import (
    regression,
    classification,
    
    classification_base, 
)


def add_subparser(subparsers, increase_depth=False):

    if increase_depth:
        parser = subparsers.add_parser("experiments", help="experiments")
        subparsers = parser.add_subparsers(metavar="experiments")

    regression.add_subparser(subparsers)
    classification.add_subparser(subparsers)
    
    classification_base.add_subparser(subparsers)
