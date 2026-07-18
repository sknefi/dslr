#!/usr/bin/env python3

import json
import sys

from database import Database


def database_to_dict(database: Database) -> dict:
    return {
        "path": database.path,
        "row_count": database.row_count(),
        "column_count": database.column_count(),
        "headers": database.column_names(),
        "numeric_columns": database.numeric_columns(),
        "rows": database.rows,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"a.py: {error}", file=sys.stderr)
        return 1

    print(json.dumps(database_to_dict(database), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
