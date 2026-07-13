#!/usr/bin/env python3

import sys

from database import Database


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"describe: {error}", file=sys.stderr)
        return 1

    # print(database.headers)
    # print(database.rows[1])
    # print(database.numeric_column("Divination"))
    # print(database.numeric_columns())
    # # print(database.numeric_column("Best Hand"))
    # # print(Database._is_number(132))
    print(database.rows[0])


if __name__ == "__main__":
    main()
