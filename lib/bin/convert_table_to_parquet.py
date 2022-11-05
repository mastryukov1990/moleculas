import pandas as pd
import argparse

from lib.common import run_pars
from lib.logger import Logger

logger = Logger(__name__)


def convert(table_path: str, parquet_path: str):
    pd.read_table(table_path).to_parquet(parquet_path)


def run_task(args):
    convert(args.table_path, args.parquet_path)


def pars_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--table-path", type=str)
    parser.add_argument("--parquet-path", type=str)
    parser.set_defaults(func=run_task)


if __name__ == "__main__":
    run_pars(pars_arguments)
