#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt

from constants import BIN_COUNT, \
                      FIGURE_DPI, FIGURE_HEIGHT_PX, FIGURE_WIDTH_PX, \
                      SCATTER_ALPHA, HISTOGRAM_ALPHA, \
                      HOUSE_COLORS, HOUSE_COLUMN, PAIR_PLOT_FEATURES, \
                      PAIR_PLOT_MARKER_SIZE

from database import Database
from histogram import group_feature_by_house
from scatter_plot import paired_values


def pair_plot(database: Database) -> None:
    # features = database.numeric_columns() # too much info, unclear
    features = PAIR_PLOT_FEATURES
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
    for feature_name in PAIR_PLOT_FEATURES:
        if not database.has_column(feature_name):
            print(f"pair_plot: missing column: {feature_name}", file=sys.stderr)
            return 1

    pair_plot(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
