#!/usr/bin/env python3

import sys
from typing import List

import matplotlib.pyplot as plt

from database import Database


X_FEATURE = "Muggle Studies"
Y_FEATURE = "Defense Against the Dark Arts"
FIGURE_WIDTH_PX = 1280
FIGURE_HEIGHT_PX = 720
FIGURE_DPI = 100
POINT_ALPHA = 0.55


def paired_values(database: Database, x_feature: str, y_feature: str) -> tuple[List[float], List[float]]:
    x_values: List[float] = []
    y_values: List[float] = []

    for row in database.rows:
        x_value = row[x_feature]
        y_value = row[y_feature]
        if x_value == "" or y_value == "":
            continue
        try:
            x_values.append(float(x_value))
            y_values.append(float(y_value))
        except ValueError:
            continue

    return x_values, y_values


def scatter_plot(database: Database) -> None:
    x_values, y_values = paired_values(database, X_FEATURE, Y_FEATURE)

    plt.figure(
        figsize=(FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI),
        dpi=FIGURE_DPI,
    )
    plt.scatter(x_values, y_values, alpha=POINT_ALPHA)
    plt.title(f"{X_FEATURE} vs {Y_FEATURE}")
    plt.xlabel(X_FEATURE)
    plt.ylabel(Y_FEATURE)
    plt.show()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"scatter_plot: {error}", file=sys.stderr)
        return 1

    if not database.has_column(X_FEATURE):
        print(f"scatter_plot: missing column: {X_FEATURE}", file=sys.stderr)
        return 1
    if not database.has_column(Y_FEATURE):
        print(f"scatter_plot: missing column: {Y_FEATURE}", file=sys.stderr)
        return 1

    scatter_plot(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
