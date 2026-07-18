#!/usr/bin/env python3

import sys
from typing import List

import matplotlib.pyplot as plt

from constants import FIGURE_DPI, FIGURE_HEIGHT_PX, FIGURE_WIDTH_PX, \
                      HOUSE_COLORS, HOUSE_COLUMN, \
                      SCATTER_ALPHA, SCATTER_X_FEATURE, SCATTER_Y_FEATURE
from database import Database


def paired_values(
    database: Database,
    x_feature: str,
    y_feature: str,
    house_name: str,
) -> dict[str, tuple[List[float], List[float]]]:
    """
    Return paired x/y feature values grouped by house.

    Example:
    {
        "Gryffindor": ([1.0, 2.0, ...], [6.9, 5.0, ...]),
        "Hufflepuff": ([6.1, 4.3, ...], [8.0, 6.4, ...]),
        "Ravenclaw":  ([3.2, 4.2, ...], [6.0, 7.8, ...]),
        "Slytherin":  ([4.0, 5.8, ...], [5.5, 3.0, ...]),
    }
    """
    paired_values_by_house = {}

    for row in database.rows:
        house = row[house_name]
        x_value = row[x_feature]
        y_value = row[y_feature]
        if house == "" or x_value == "" or y_value == "":
            continue
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except ValueError:
            continue

        if house not in paired_values_by_house:
            paired_values_by_house[house] = ([], [])
        paired_values_by_house[house][0].append(x_float)
        paired_values_by_house[house][1].append(y_float)

    return paired_values_by_house


def scatter_plot(database: Database) -> None:
    paired_values_by_house = paired_values(
        database,
        SCATTER_X_FEATURE,
        SCATTER_Y_FEATURE,
        HOUSE_COLUMN,
    )

    plt.figure(
        figsize=(FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI),
        dpi=FIGURE_DPI,
    )
    for house in sorted(paired_values_by_house):
        x_values, y_values = paired_values_by_house[house]
        plt.scatter(
            x_values,
            y_values,
            alpha=SCATTER_ALPHA,
            label=house,
            color=HOUSE_COLORS.get(house),
        )
    plt.title(f"{SCATTER_X_FEATURE} vs {SCATTER_Y_FEATURE}")
    plt.xlabel(SCATTER_X_FEATURE)
    plt.ylabel(SCATTER_Y_FEATURE)
    plt.legend()
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

    if not database.has_column(SCATTER_X_FEATURE):
        print(f"scatter_plot: missing column: {SCATTER_X_FEATURE}", file=sys.stderr)
        return 1
    if not database.has_column(SCATTER_Y_FEATURE):
        print(f"scatter_plot: missing column: {SCATTER_Y_FEATURE}", file=sys.stderr)
        return 1
    if not database.has_column(HOUSE_COLUMN):
        print(f"scatter_plot: missing column: {HOUSE_COLUMN}", file=sys.stderr)
        return 1

    scatter_plot(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
