#!/usr/bin/env python3

import sys
from typing import List

import matplotlib.pyplot as plt

from constants import FIGURE_DPI, FIGURE_HEIGHT_PX, FIGURE_WIDTH_PX,  \
                      SCATTER_ALPHA, \
                      HOUSE_COLORS, HOUSE_COLUMN_NAME, INDEX_COLUMN 
from database import Database
from describe import mean, std
from utils import label_width


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


def feature_pair_values(
    database: Database,
    x_feature: str,
    y_feature: str,
) -> tuple[List[float], List[float]]:
    x_values: List[float] = []
    y_values: List[float] = []

    for row in database.rows:
        x_value = row[x_feature]
        y_value = row[y_feature]
        if x_value == "" or y_value == "":
            continue
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except ValueError:
            continue

        x_values.append(x_float)
        y_values.append(y_float)

    return x_values, y_values


def covariance(x_values: List[float], y_values: List[float]) -> float:
    """
    Measure whether two features move together.

    Positive covariance means they tend to move in the same direction.
    Negative covariance means they tend to move in opposite directions.
    """
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    total = 0.0

    for i in range(len(x_values)):
        total += (x_values[i] - x_mean) * (y_values[i] - y_mean)
    return total / (len(x_values) - 1)


def correlation(x_values: List[float], y_values: List[float]) -> float:
    """
    Return normalized covariance between -1 and 1.

    Values close to 1 or -1 mean a strong linear relation.
    Values close to 0 mean little or no linear relation.
    """
    x_std = std(x_values)
    y_std = std(y_values)
    if x_std == 0 or y_std == 0:
        return 0.0
    return covariance(x_values, y_values) / (x_std * y_std)


def absolute(value: float) -> float:
    if value < 0:
        return -value
    return value


def most_similar_features(database: Database) -> tuple[str, str]:
    best_feature_a = ""
    best_feature_b = ""
    best_score = None
    numeric_features = []

    for feature_name in database.numeric_columns():
        if feature_name != INDEX_COLUMN:
            numeric_features.append(feature_name)
    feature_width = label_width(numeric_features)

    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            feature_a = numeric_features[i]
            feature_b = numeric_features[j]
            x_values, y_values = feature_pair_values(database, feature_a, feature_b)
            if len(x_values) < 2: # y_values is the same size as x_values
                continue
            corr = correlation(x_values, y_values)
            score = absolute(corr)
            print(f"{feature_a:<{feature_width}} {feature_b:<{feature_width}} {corr:.6f}")
            if best_score is None or score > best_score:
                best_score = score
                best_feature_a = feature_a
                best_feature_b = feature_b

    print("-" * (feature_width * 2 + 11))
    if best_score is not None:
        x_values, y_values = feature_pair_values(database, best_feature_a, best_feature_b)
        corr = correlation(x_values, y_values)
        print(f"{best_feature_a:<{feature_width}} {best_feature_b:<{feature_width}} {corr:.6f}")

    return best_feature_a, best_feature_b


def scatter_plot(database: Database, x_feature: str, y_feature: str) -> None:
    paired_values_by_house = paired_values(
        database,
        x_feature,
        y_feature,
        HOUSE_COLUMN_NAME,
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
    plt.title(f"{x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
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

    if not database.has_column(HOUSE_COLUMN_NAME):
        print(f"scatter_plot: missing column: {HOUSE_COLUMN_NAME}", file=sys.stderr)
        return 1

    x_feature, y_feature = most_similar_features(database)
    if x_feature == "" or y_feature == "":
        print("scatter_plot: could not find similar numeric features", file=sys.stderr)
        return 1

    print(f"Most similar features: {x_feature} / {y_feature}")
    scatter_plot(database, x_feature, y_feature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
