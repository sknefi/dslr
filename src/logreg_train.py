#!/usr/bin/env python3

import sys
from typing import List

from constants import HOUSE_COLUMN_NAME
from database import Database


def train(database: Database) -> None:
    houses: List[str] = sorted(set(database.column(HOUSE_COLUMN_NAME)))
    feature_names: List[str] = database.numeric_columns_except_index()

    print(f"Rows: {database.row_count()}")
    print(f"Columns: {database.column_count()}")
    print(f"Target column: {HOUSE_COLUMN_NAME}")
    print(f"Houses: {', '.join(houses)}")
    print(f"Numeric features: {len(feature_names)}")
    for feature_name in feature_names:
        print(f"- {feature_name}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"logreg_train: {error}", file=sys.stderr)
        return 1

    if not database.has_column(HOUSE_COLUMN_NAME):
        print(f"logreg_train: missing column: {HOUSE_COLUMN_NAME}", file=sys.stderr)
        return 1

    train(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
