#!/usr/bin/env python3

import sys
from typing import Dict, List

import matplotlib.pyplot as plt

from constants import BIN_COUNT, \
                      FIGURE_DPI, FIGURE_HEIGHT_PX, FIGURE_WIDTH_PX, \
                      HISTOGRAM_ALPHA, \
                      HOUSE_COLORS, HOUSE_COLUMN_NAME, INDEX_COLUMN
from database import Database
from describe import max, mean, min, std
from utils import label_width


def group_feature_by_house(
    database: Database,
    house_name: str,
    feature_name: str,
) -> Dict[str, List[float]]:
    """
    Return one feature's numeric values grouped by house.

    Example:
    {
        "Gryffindor": [1.2, 2.4, ...],
        "Hufflepuff": [4.2, 6.9, ...],
        "Ravenclaw":  [9.1, 8.6, ...],
        "Slytherin":  [7.5, 7.4, ...],
    }
    """
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


def homogeneity_score(grouped_values: Dict[str, List[float]]) -> float:
    """
    Return a simple homogeneity score for one feature.

    The values are already split by house. A homogeneous feature should have
    similar means and similar standard deviations for all houses. The score is
    normalized by the full feature range so features with different scales can
    be compared. Lower score means more homogeneous.
    """
    all_values = []
    for values in grouped_values.values():
        all_values += values

    feature_range = max(all_values) - min(all_values)
    if feature_range == 0:
        return 0.0

    house_means = []
    house_stds = []
    for values in grouped_values.values():
        house_means.append(mean(values))
        house_stds.append(std(values))

    mean_spread = max(house_means) - min(house_means)
    std_spread = max(house_stds) - min(house_stds)
    return (mean_spread + std_spread) / feature_range


def most_homogeneous_feature(database: Database) -> str:
    best_feature = ""
    best_score = None
    numeric_features = []

    for feature_name in database.numeric_columns():
        if feature_name != INDEX_COLUMN:
            numeric_features.append(feature_name)
    feature_name_width = label_width(numeric_features)

    for feature_name in numeric_features:
        grouped_values = group_feature_by_house(database, HOUSE_COLUMN_NAME, feature_name)
        if len(grouped_values) == 0:
            continue
        score = homogeneity_score(grouped_values)
        print(f"{feature_name:<{feature_name_width}} {score:.6f}")
        if best_score is None or score < best_score:
            best_score = score
            best_feature = feature_name

    print("-" * (feature_name_width + 9))
    if best_score is not None:
        print(f"{best_feature:<{feature_name_width}} {best_score:.6f}")
    return best_feature


def histogram(database: Database, feature_name: str) -> None:
    grouped_values = group_feature_by_house(database, HOUSE_COLUMN_NAME, feature_name)

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
            bins=BIN_COUNT,
            alpha=HISTOGRAM_ALPHA,
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

    if not database.has_column(HOUSE_COLUMN_NAME):
        print(f"histogram: missing column: {HOUSE_COLUMN_NAME}", file=sys.stderr)
        return 1
    feature_name = most_homogeneous_feature(database)
    if feature_name == "":
        print("histogram: could not find a numeric feature", file=sys.stderr)
        return 1

    print(f"Most homogeneous feature: {feature_name}")
    histogram(database, feature_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
