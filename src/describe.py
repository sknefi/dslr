#!/usr/bin/env python3

import sys
from typing import List, Dict
from math import sqrt
from math import floor

from database import Database

def count(numeric_values: List[float]) -> float:
    return float(len(numeric_values)) # empty fields are already parsed

def mean(numeric_values: List[float]) -> float:
    sum_val: float = 0
    for x in numeric_values:
        sum_val += x
    return float(sum_val / count(numeric_values))

def std(numeric_values: List[float]) -> float:
    mean_val: float = mean(numeric_values)
    sum_diff: float = 0
    for x in numeric_values:
        sum_diff += (x - mean_val) ** 2
    return sqrt(sum_diff / (count(numeric_values) - 1))

def min(numeric_values: List[float]) -> float:
    min_val: float = numeric_values[0]
    for x in numeric_values:
        if x < min_val:
            min_val = x
    return min_val

def percentile(sorted_numeric_values: List[float], percent: float) -> float:
    index: float = (count(sorted_numeric_values) - 1) * percent
    lower_index: int = floor(index)
    upper_index: int = lower_index + 1

    if upper_index >= count(sorted_numeric_values):
        return sorted_numeric_values[lower_index]

    part: float = index - lower_index
    lower_value: float = sorted_numeric_values[lower_index]
    upper_value: float = sorted_numeric_values[upper_index]
    return lower_value + (upper_value - lower_value) * part

def q1(sorted_numeric_values: List[float]) -> float:
    return percentile(sorted_numeric_values, 0.25)

def median(sorted_numeric_values: List[float]) -> float:
    return percentile(sorted_numeric_values, 0.50)

def q3(sorted_numeric_values: List[float]) -> float:
    return percentile(sorted_numeric_values, 0.75)

def max(numeric_values: List[float]) -> float:
    max_val: float = numeric_values[0]
    for x in numeric_values:
        if x > max_val:
            max_val = x
    return max_val


def describe(database: Database) -> None:
    numeric_column_names: List[str] = database.numeric_columns()
    features: Dict[List[float]] = dict()

        

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        return 1

    try:
        database = Database(sys.argv[1])
    except ValueError as error:
        print(f"describe: {error}", file=sys.stderr)
        return 1

    describe(database)


if __name__ == "__main__":
    main()
