#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

from constants import HOUSE_COLUMN_NAME, HOUSES, INDEX_COLUMN, PREDICTIONS_FILE
from database import Database
from utils import sigmoid


def load_model(path: str) -> dict:
    """Read the trained house models from a JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load model {path}: {error}") from error

    if not isinstance(model, dict):
        raise ValueError("model must be a JSON object")
    return model


def validate_inputs(database: Database, model: dict) -> None:
    """Check that the dataset has the features required by the model."""
    feature_names = model.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("model has no feature names")
    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError("model contains invalid feature names")

    models = model.get("models")
    if not isinstance(models, dict) or set(models) != set(HOUSES):
        raise ValueError("model must contain all four houses")
    for house in HOUSES:
        house_model = models[house]
        if not isinstance(house_model, dict):
            raise ValueError(f"invalid model for {house}")
        weights = house_model.get("weights")
        bias = house_model.get("bias")
        if not isinstance(weights, dict) or set(weights) != set(feature_names):
            raise ValueError(f"invalid weights for {house}")
        if not isinstance(bias, (int, float)) or not all(
            isinstance(weight, (int, float)) for weight in weights.values()
        ):
            raise ValueError(f"invalid numeric values for {house}")

    fill_values = model.get("fill_values")
    normalization_values = model.get("normalization_values")
    if not isinstance(fill_values, dict) or not isinstance(normalization_values, dict):
        raise ValueError("model has no preprocessing values")
    for name in feature_names:
        stats = normalization_values.get(name)
        if name not in fill_values or not isinstance(stats, dict):
            raise ValueError(f"model has no preprocessing values for {name}")
        if "mean" not in stats or "std" not in stats or stats["std"] == 0:
            raise ValueError(f"model has invalid normalization values for {name}")

    for name in [INDEX_COLUMN] + feature_names:
        if not database.has_column(name):
            raise ValueError(f"missing column in dataset: {name}")


def prepare_prediction_data(
    database: Database,
    model: dict
) -> List[tuple[str, List[float]]]:
    """Return each student's original Index and normalized feature values.

    Example: [("0", [-0.48, 1.27, ...]), ("1", [0.15, -0.79, ...])].
    """
    students: List[tuple[str, List[float]]] = []
    for row in database.rows:
        features: List[float] = []
        for name in model["feature_names"]:
            raw_value = row[name]
            value = model["fill_values"][name] if raw_value == "" else float(raw_value)
            stats = model["normalization_values"][name]
            features.append((value - stats["mean"]) / stats["std"])
        students.append((row[INDEX_COLUMN], features))
    return students


def predict_house(features: List[float], model: dict) -> tuple[str, Dict[str, float]]:
    """Return the most likely house and each house model's probability."""
    probabilities: Dict[str, float] = {}
    best_house = ""
    best_probability = -1.0

    for house in HOUSES:
        house_model = model["models"][house]
        score = house_model["bias"]
        for name, value in zip(model["feature_names"], features):
            score += house_model["weights"][name] * value

        probability = sigmoid(score)
        probabilities[house] = probability
        if probability > best_probability:
            best_house = house
            best_probability = probability

    return best_house, probabilities


def predict_students(
    students: List[tuple[str, List[float]]],
    model: dict
) -> List[tuple[str, str, Dict[str, float]]]:
    """Predict a house for each student while keeping the original Index."""
    predictions: List[tuple[str, str, Dict[str, float]]] = []
    for index, features in students:
        house, probabilities = predict_house(features, model)
        predictions.append((index, house, probabilities))
    return predictions


def print_prediction_preview(
    predictions: List[tuple[str, str, Dict[str, float]]]
) -> None:
    """Print the prediction count and the first five results."""
    print(f"Predicted {len(predictions)} students")
    for index, house, probabilities in predictions[::]:
        scores = ", ".join(f"{name}: {value:.1%}" for name, value in probabilities.items())
        print(f"{index}: {house} ({scores})")


def save_predictions(
    predictions: List[tuple[str, str, Dict[str, float]]], path: str
) -> None:
    """Write each student's Index and predicted house to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([INDEX_COLUMN, HOUSE_COLUMN_NAME])
        for index, house, _ in predictions:
            writer.writerow([index, house])


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            f"Usage: {sys.argv[0]} dataset_test.csv model_batch.json [houses.csv]",
            file=sys.stderr,
        )
        return 1

    output_path = sys.argv[3] if len(sys.argv) == 4 else PREDICTIONS_FILE
    try:
        database = Database(sys.argv[1])
        model = load_model(sys.argv[2])
        validate_inputs(database, model)
        students = prepare_prediction_data(database, model)
        predictions = predict_students(students, model)
        save_predictions(predictions, output_path)
    except (ValueError, OSError) as error:
        print(f"logreg_predict: {error}", file=sys.stderr)
        return 1

    print_prediction_preview(predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
