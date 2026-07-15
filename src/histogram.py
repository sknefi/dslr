#!/usr/bin/env python3

import sys
from typing import Dict, List

import matplotlib.pyplot as plt

from database import Database


DEFAULT_FEATURE = "Defense Against the Dark Arts"
HOUSE_COLUMN = "Hogwarts House"
FIGURE_WIDTH_PX = 1280
FIGURE_HEIGHT_PX = 720
FIGURE_DPI = 100
HOUSE_COLORS = {
    "Gryffindor": "red",
    "Hufflepuff": "gold",
    "Ravenclaw":  "royalblue",
    "Slytherin":  "green",
}


def group_feature_by_house(
    database: Database,
    house_name: str,
    feature_name: str,
) -> Dict[str, List[float]]:
    grouped_values: Dict[str, List[float]] = {}

    for row in database.rows:
        house = row[house_name]
        value = row[feature_name]
        if house == "" or value == "":
            continue
        try:
            numeric_value = float(value)
        except ValueError:
            continue

        if house not in grouped_values:
            grouped_values[house] = []
        grouped_values[house].append(numeric_value)

    # print(grouped_values)
    return grouped_values


def histogram(database: Database, feature_name: str) -> None:
    grouped_values = group_feature_by_house(database, HOUSE_COLUMN, feature_name)

    if len(grouped_values) == 0:
        print(f"histogram: no numeric values found for feature: {feature_name}", file=sys.stderr)
        return

    plt.figure(
        figsize=(FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI),
        dpi=FIGURE_DPI,
    )
    for house in sorted(grouped_values):
        plt.hist(
            grouped_values[house],
            bins=40,
            alpha=0.42,
            label=house,
            color=HOUSE_COLORS.get(house),
        )

    plt.title(f"{feature_name} score distribution by house")
    plt.xlabel(feature_name)
    plt.ylabel("Number of students")
    plt.legend()
    plt.show()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"histogram: {error}", file=sys.stderr)
        return 1

    if not database.has_column(HOUSE_COLUMN):
        print(f"histogram: missing column: {HOUSE_COLUMN}", file=sys.stderr)
        return 1
    if not database.has_column(DEFAULT_FEATURE):
        print(f"histogram: missing column: {DEFAULT_FEATURE}", file=sys.stderr)
        return 1

    histogram(database, DEFAULT_FEATURE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
