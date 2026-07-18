#!/usr/bin/env python3

import sys
from typing import List

import matplotlib.pyplot as plt

from database import Database
from histogram import HOUSE_COLORS, HOUSE_COLUMN


X_FEATURE = "Muggle Studies"
Y_FEATURE = "Defense Against the Dark Arts"
FIGURE_WIDTH_PX = 1280
FIGURE_HEIGHT_PX = 720
FIGURE_DPI = 100
POINT_ALPHA = .42


def paired_values(
    database: Database,
    x_feature: str,
    y_feature: str,
    house_name: str,
) -> dict[str, tuple[List[float], List[float]]]:
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
    paired_values_by_house = paired_values(database, X_FEATURE, Y_FEATURE, HOUSE_COLUMN)

    plt.figure(
        figsize=(FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI),
        dpi=FIGURE_DPI,
    )
    for house in sorted(paired_values_by_house):
        x_values, y_values = paired_values_by_house[house]
        plt.scatter(
            x_values,
            y_values,
            alpha=POINT_ALPHA,
            label=house,
            color=HOUSE_COLORS.get(house),
        )
    plt.title(f"{X_FEATURE} vs {Y_FEATURE}")
    plt.xlabel(X_FEATURE)
    plt.ylabel(Y_FEATURE)
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

    if not database.has_column(X_FEATURE):
        print(f"scatter_plot: missing column: {X_FEATURE}", file=sys.stderr)
        return 1
    if not database.has_column(Y_FEATURE):
        print(f"scatter_plot: missing column: {Y_FEATURE}", file=sys.stderr)
        return 1
    if not database.has_column(HOUSE_COLUMN):
        print(f"scatter_plot: missing column: {HOUSE_COLUMN}", file=sys.stderr)
        return 1

    scatter_plot(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
