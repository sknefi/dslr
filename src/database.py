#!/usr/bin/env python3

import csv
from os import read


class Database:
    def __init__(self, path):
        self.path = path
        self.headers = []
        self.rows = []
        self._load_csv()

    def row_count(self):
        return len(self.rows)

    def column_count(self):
        return len(self.headers)

    def column_names(self):
        return list(self.headers)

    def has_column(self, name):
        return name in self.headers

    def column(self, name):
        self._require_column(name)
        return [row[name] for row in self.rows]

    def numeric_column(self, name):
        self._require_column(name)
        values = []
        for row in self.rows:
            value = row[name]
            if value == "":
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        return values

    def numeric_columns(self):
        names = []
        for name in self.headers:
            values = self.column(name)
            non_empty_count = 0
            numeric_count = 0
            for value in values:
                if value == "":
                    continue
                non_empty_count += 1
                if self._is_number(value):
                    numeric_count += 1
            if non_empty_count > 0 and non_empty_count == numeric_count:
                names.append(name)
        return names

    def missing_count(self, name):
        self._require_column(name)
        count = 0
        for value in self.column(name):
            if value == "":
                count += 1
        return count

    def _load_csv(self):
        try:
            with open(self.path, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    raise ValueError("empty CSV file")
                self.headers = list(reader.fieldnames)
                self.rows = [self._normalize_row(row) for row in reader]
        except OSError as error:
            raise ValueError(f"cannot read {self.path}: {error}") from error
        except csv.Error as error:
            raise ValueError(f"cannot parse {self.path}: {error}") from error

    def _normalize_row(self, row):
        normalized = {}
        for header in self.headers:
            value = row.get(header, "")
            normalized[header] = "" if value is None else value.strip()
        return normalized

    def _require_column(self, name):
        if name not in self.headers:
            raise ValueError(f"unknown column: {name}")

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
