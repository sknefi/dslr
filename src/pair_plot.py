#!/usr/bin/env python3

import sys
from typing import List

import matplotlib.pyplot as plt

from constants import BIN_COUNT, \
                      FIGURE_DPI, FIGURE_HEIGHT_PX, FIGURE_WIDTH_PX, \
                      SCATTER_ALPHA, HISTOGRAM_ALPHA, \
                      HOUSE_COLORS, HOUSE_COLUMN, INDEX_COLUMN, PAIR_PLOT_FEATURE_COUNT, \
                      PAIR_PLOT_MARKER_SIZE

from database import Database
from describe import max, mean, min, std
from histogram import group_feature_by_house
from scatter_plot import paired_values
from utils import label_width


def feature_separation_score(database: Database, feature_name: str) -> float:
    """
    Return how useful a feature looks for separating houses.

    The score compares how far apart the house means are against the average
    spread inside each house. Higher score means better visual separation.
    """
    grouped_values = group_feature_by_house(database, HOUSE_COLUMN, feature_name)
    if len(grouped_values) == 0:
        return 0.0

    house_means = []
    house_stds = []
    for values in grouped_values.values():
        if len(values) < 2:
            continue
        house_means.append(mean(values))
        house_stds.append(std(values))

    if len(house_means) == 0:
        return 0.0

    average_std = mean(house_stds)
    if average_std == 0:
        return 0.0

    mean_spread = max(house_means) - min(house_means)
    return mean_spread / average_std


def selected_features(database: Database) -> List[str]:
    """
    Select the best features to display in the pair plot.

    Every numeric feature except Index gets a separation score. The list is
    sorted from highest score to lowest, then only the first
    PAIR_PLOT_FEATURE_COUNT features are returned.
    """
    feature_scores = []
    numeric_features = []

    for feature_name in database.numeric_columns():
        if feature_name != INDEX_COLUMN:
            numeric_features.append(feature_name)
    feature_width = label_width(numeric_features)

    for feature_name in numeric_features:
        score = feature_separation_score(database, feature_name)
        feature_scores.append({
            "name": feature_name,
            "score": score,
        })
        print(f"{feature_name:<{feature_width}} {score:.6f}")

    feature_scores.sort(key=lambda feature: feature["score"], reverse=True)
    selected = []
    for feature in feature_scores[:PAIR_PLOT_FEATURE_COUNT]:
        selected.append(feature["name"])

    print("-" * (feature_width + 9))
    for feature in feature_scores[:PAIR_PLOT_FEATURE_COUNT]:
        print(f"{feature['name']:<{feature_width}} {feature['score']:.6f}")
    return selected


def pair_plot(database: Database) -> None:
    features = selected_features(database)
    size = len(features)
    figure, axes = plt.subplots(
        size,
        size,
        figsize=(FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI),
        dpi=FIGURE_DPI,
    )

    for row_index, y_feature in enumerate(features):
        for column_index, x_feature in enumerate(features):
            axis = axes[row_index][column_index]

            if row_index == column_index: # histogram on main diagonal
                values_by_house = group_feature_by_house(database, HOUSE_COLUMN, x_feature)
                for house in sorted(values_by_house):
                    axis.hist(
                        values_by_house[house],
                        bins=BIN_COUNT,
                        alpha=HISTOGRAM_ALPHA,
                        color=HOUSE_COLORS.get(house),
                    )
            else: # scatter plot everywhere else except main diagonal
                values_by_house = paired_values(database, x_feature, y_feature, HOUSE_COLUMN)
                for house in sorted(values_by_house):
                    x_values, y_values = values_by_house[house]
                    axis.scatter(
                        x_values,
                        y_values,
                        alpha=SCATTER_ALPHA,
                        s=PAIR_PLOT_MARKER_SIZE,
                        color=HOUSE_COLORS.get(house),
                    )

            # Display x and y labels just once on left and bottom of grid
            if row_index == size - 1:
                axis.set_xlabel(x_feature)
            else:
                axis.set_xticklabels([])

            if column_index == 0:
                axis.set_ylabel(y_feature)
            else:
                axis.set_yticklabels([])

    handles = []
    labels = []
    for house in sorted(HOUSE_COLORS):
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=HOUSE_COLORS[house]))
        labels.append(house)
    figure.legend(handles, labels, loc="upper right")
    figure.suptitle("Pair plot by Hogwarts house")
    figure.tight_layout()
    plt.show()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"pair_plot: {error}", file=sys.stderr)
        return 1

    if not database.has_column(HOUSE_COLUMN):
        print(f"pair_plot: missing column: {HOUSE_COLUMN}", file=sys.stderr)
        return 1

    pair_plot(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
