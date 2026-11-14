#!/usr/bin/env python3

import json
import sys
from typing import List

from constants import HOUSES, INDEX_COLUMN
from database import Database


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


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} dataset_test.csv output/model.json", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
        model = load_model(sys.argv[2])
        validate_inputs(database, model)
        students = prepare_prediction_data(database, model)
    except ValueError as error:
        print(f"logreg_predict: {error}", file=sys.stderr)
        return 1

    print(f"Prepared {len(students)} students for prediction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
