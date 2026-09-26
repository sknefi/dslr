#!/usr/bin/env python3

import sys
from typing import List

from constants import HOUSE_COLUMN_NAME
from database import Database


def selected_feature_names(database: Database) -> List[str]:
    return database.numeric_columns_except_index()


def print_dataset_info(database: Database, feature_names: List[str]) -> None:
    houses: List[str] = sorted(set(database.column(HOUSE_COLUMN_NAME)))
    print(f"Rows: {database.row_count()}")
    print(f"Columns: {database.column_count()}")
    print(f"Target column: {HOUSE_COLUMN_NAME}")
    print(f"Houses: {', '.join(houses)}")
    print(f"Selected numeric features: {len(feature_names)}")
    for feature_name in feature_names:
        print(f"- {feature_name}")


def train(database: Database) -> None:
    feature_names: List[str] = selected_feature_names(database)
    print_dataset_info(database, feature_names)


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

    if len(selected_feature_names(database)) == 0:
        print("logreg_train: no numeric features found", file=sys.stderr)
        return 1

    train(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
