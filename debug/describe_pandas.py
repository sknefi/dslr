#!/usr/bin/env python3
"""
Debug helper used to compare and validate our describe output.
It uses pandas helpers that are forbidden in the subject, but it's only in
debugger not in the main program.
"""

import argparse
import sys

try:
    import pandas as pd
except ModuleNotFoundError:
    print(
        "describe_pandas: missing dependency 'pandas'. Run `make install` first, "
        "or use a Python interpreter that already has pandas installed.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def print_section(title):
    print()
    print(title)
    print("=" * len(title))


def main():
    parser = argparse.ArgumentParser(
        description="Display dataset information using pandas describe()."
    )
    parser.add_argument("dataset", help="Path to a CSV dataset")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataset)
    except OSError as error:
        print(f"describe_pandas: cannot read {args.dataset}: {error}", file=sys.stderr)
        return 1
    except pd.errors.ParserError as error:
        print(f"describe_pandas: cannot parse {args.dataset}: {error}", file=sys.stderr)
        return 1

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    print_section("Shape")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")

    print_section("Columns And Types")
    print(df.dtypes)

    print_section("Missing Values")
    print(df.isna().sum())

    print_section("Numeric Describe")
    print(df.describe())

    print_section("All Columns Describe")
    print(df.describe(include="all"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
