from math import exp
from typing import List

from constants import COLUMN_PRINT_PADDING


def format_number(value: float) -> str:
    return f"{value:.6f}"


def label_width(labels: List[str]) -> int:
    width: int = 0
    for label in labels:
        if len(label) > width:
            width = len(label)
    return width + COLUMN_PRINT_PADDING


def feature_width(feature, stat_names: List[str]) -> int:
    width: int = len(feature["name"])
    for stat_name in stat_names:
        value_width = len(format_number(feature[stat_name]))
        if value_width > width:
            width = value_width
    return width + COLUMN_PRINT_PADDING


def sigmoid(value: float) -> float:
    """Turn a model score into a probability between 0 and 1."""
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    positive_exp = exp(value)
    return positive_exp / (1.0 + positive_exp)
