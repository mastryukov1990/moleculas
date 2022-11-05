import argparse
from typing import Callable

from lib.logger import Logger

logger = Logger(__name__)


def get_parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="rec-offline-ranker CLI",
        formatter_class=formatter,
    )
    return parser


def run_pars(pars_arguments: Callable[[argparse.ArgumentParser], None]):
    parser = get_parser()
    pars_arguments(parser)
    args = parser.parse_args()
    args.func(args)
