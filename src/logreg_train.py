#!/usr/bin/env python3

import sys
from math import exp
from typing import Dict, List

from constants import HOUSE_COLUMN_NAME, LEARNING_RATE, TRAINING_EPOCHS, TRAINING_HOUSE
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
    Return a new list of student rows with normalized feature values.

    Input example:
    feature_names = ["Astronomy", "Herbology"]
    x_train = [
        [8.0, 80.0],    # first student
        [10.0, 100.0],  # second student
        [12.0, 120.0],  # third student
    ]
    normalization_values = {
        "Astronomy": {"mean": 10.0, "std": 2.0},
        "Herbology": {"mean": 100.0, "std": 20.0},
    }

    Output for those inputs:
    [
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]

    Each cell uses (value - feature mean) / feature std. For example,
    Astronomy 8.0 becomes (8.0 - 10.0) / 2.0 = -1.0.
    The row and feature order stay the same; x_train is not changed.
    Use these same normalization_values later for test students.
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


def binary_labels(y_train: List[str], house: str) -> List[int]:
    """Return 1 for students in house and 0 for students in other houses.

    Example: ["Gryffindor", "Ravenclaw"] -> [1, 0] for Gryffindor.
    """
    return [1 if label == house else 0 for label in y_train]


def sigmoid(value: float) -> float:
    """Turn a model score into a probability between 0 and 1."""
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    positive_exp = exp(value)
    return positive_exp / (1.0 + positive_exp)


def train_binary_model(
    x_train: List[List[float]],
    y_binary: List[int],
    learning_rate: float,
    epochs: int,
) -> tuple[List[float], float]:
    """Train one house-versus-rest model; return its weights and bias."""
    weights: List[float] = [0.0] * len(x_train[0])
    bias                 = 0.0
    student_count        = len(x_train)

    for _ in range(epochs):
        weight_gradients: List[float] = [0.0] * len(weights)
        bias_gradient = 0.0

        for features, target in zip(x_train, y_binary):
            score = bias
            for i, value in enumerate(features):
                score += weights[i] * value
            error = sigmoid(score) - target

            for i, value in enumerate(features):
                weight_gradients[i] += error * value
            bias_gradient += error

        for i in range(len(weights)):
            weights[i] -= learning_rate * weight_gradients[i] / student_count
        bias -= learning_rate * bias_gradient / student_count

    return weights, bias


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
    y_binary = binary_labels(y_train, TRAINING_HOUSE)
    weights, bias = train_binary_model(
        x_train_normalized,
        y_binary,
        LEARNING_RATE,
        TRAINING_EPOCHS,
    )

    print(f"Model: {TRAINING_HOUSE} versus other houses")
    print(f"Bias: {bias:.6f}")
    feature_width = label_width(feature_names)
    for feature_name, weight in zip(feature_names, weights):
        print(f"  {feature_name:<{feature_width}} {weight:>10.6f}")


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
