#!/usr/bin/env python3

import sys
from typing import Dict, List

from constants import HOUSE_COLUMN_NAME
from database import Database
from describe import mean, std
from utils import label_width


def selected_feature_names(database: Database) -> List[str]:
    return database.numeric_columns_except_index()


def feature_fill_values(database: Database, feature_names: List[str]) -> Dict[str, float]:
    """
    Return the average value for every selected feature.
    These values will be used when a student has an empty cell.

    Example:
    {
        "Astronomy": 42.797131,
        "Herbology": 1.921020,
    }
    """
    fill_values: Dict[str, float] = {}
    for feature_name in feature_names:
        values: List[float] = database.numeric_column(feature_name)
        fill_values[feature_name] = mean(values)
    return fill_values


def print_dataset_info(database: Database, feature_names: List[str]) -> None:
    houses: List[str] = sorted(set(database.column(HOUSE_COLUMN_NAME)))
    print(f"Rows: {database.row_count()}")
    print(f"Columns: {database.column_count()}")
    print(f"Target column: {HOUSE_COLUMN_NAME}")
    print(f"Houses: {', '.join(houses)}")
    print(f"Selected numeric features: {len(feature_names)}")


def print_feature_info(database: Database, feature_names: List[str], fill_values: Dict[str, float]) -> None:
    feature_width = label_width(feature_names)
    print(f"{'Feature':<{feature_width}} {'Missing':>8} {'Fill value':>14}")
    for feature_name in feature_names:
        missing_count = database.missing_count(feature_name)
        print(f"{feature_name:<{feature_width}} {missing_count:>8} {fill_values[feature_name]:>14.6f}")


def training_data(
    database: Database,
    feature_names: List[str],
    fill_values: Dict[str, float],
) -> tuple[List[List[float]], List[str]]:
    """
    Return clean training data.

    x_train contains one list of numeric feature values per student.
    Empty cells are replaced with the feature average from fill_values.

    y_train contains the correct house for each matching x_train row.

    Example:
    x_train = [
        [49634.57, 39.79, 1.14, ...],
        [104956.0, -489.55, 4.20, ...],
    ]
    y_train = [
        "Ravenclaw",
        "Slytherin"
    ]

    x_train[0] and y_train[0] describe the same student.
    """
    x_train: List[List[float]] = []
    y_train: List[str] = []

    for row in database.rows:
        house = row[HOUSE_COLUMN_NAME]

        if house == "":
            continue

        feature_values: List[float] = []
        for feature_name in feature_names:
            value = row[feature_name]
            if value == "":
                feature_values.append(fill_values[feature_name])
            else:
                feature_values.append(float(value))

        x_train.append(feature_values)
        y_train.append(house)

    return x_train, y_train


def feature_normalization_values(
    x_train: List[List[float]],
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Return mean and std for every feature column.

    Example:
    {
        "Astronomy": {
            "mean": 39.797131,
            "std": 520.298268,
        }
    }
    """
    normalization_values: Dict[str, Dict[str, float]] = {}

    for feature_index, feature_name in enumerate(feature_names):
        values: List[float] = []
        for row in x_train:
            values.append(row[feature_index])

        feature_std = std(values)
        if feature_std == 0:
            feature_std = 1.0

        normalization_values[feature_name] = {
            "mean": mean(values),
            "std": feature_std,
        }

    return normalization_values


def normalize_training_data(
    x_train: List[List[float]],
    feature_names: List[str],
    normalization_values: Dict[str, Dict[str, float]],
) -> List[List[float]]:
    """
    Return normalized x_train.

    Formula:
    normalized_value = (value - feature_mean) / feature_std
    """
    x_train_normalized: List[List[float]] = []

    for row in x_train:
        normalized_row: List[float] = []
        for feature_index, feature_name in enumerate(feature_names):
            value = row[feature_index]
            feature_mean = normalization_values[feature_name]["mean"]
            feature_std = normalization_values[feature_name]["std"]
            normalized_row.append((value - feature_mean) / feature_std)
        x_train_normalized.append(normalized_row)

    return x_train_normalized


def train(database: Database) -> None:
    feature_names: List[str] = selected_feature_names(database)
    fill_values: Dict[str, float] = feature_fill_values(database, feature_names)
    x_train, y_train = training_data(database, feature_names, fill_values)
    normalization_values = feature_normalization_values(x_train, feature_names)
    x_train_normalized = normalize_training_data(
        x_train,
        feature_names,
        normalization_values,
    )

    print_dataset_info(database, feature_names)
    print(f"Training rows: {len(x_train)}")
    print(f"Labels: {len(y_train)}")
    print(f"Features per row: {len(feature_names)}")
    print(f"Normalized rows: {len(x_train_normalized)}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"logreg_train: {error}", file=sys.stderr)
        return 1

    if not database.has_column(HOUSE_COLUMN_NAME):
        print(f"logreg_train: missing column: {HOUSE_COLUMN_NAME}", file=sys.stderr)
        return 1

    if len(selected_feature_names(database)) == 0:
        print("logreg_train: no numeric features found", file=sys.stderr)
        return 1

    train(database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
