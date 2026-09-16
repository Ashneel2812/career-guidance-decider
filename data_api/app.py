"""HTTP API for serving the synthetic training dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "synthetic_data.csv"
TARGET_COLUMN = "mbti_type"


def _load_dataset() -> tuple[list[str], list[dict[str, Any]]]:
    """Load and validate the CSV on each request so updates are picked up."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    with DATA_FILE.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        if not columns or TARGET_COLUMN not in columns:
            raise ValueError(f"Dataset must include a '{TARGET_COLUMN}' column")

        feature_columns = [column for column in columns if column != TARGET_COLUMN]
        if not feature_columns:
            raise ValueError("Dataset must include at least one feature column")

        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                record = {
                    column: int(row[column]) for column in feature_columns
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric feature at CSV row {row_number}"
                ) from error

            record[TARGET_COLUMN] = row[TARGET_COLUMN]
            records.append(record)

    return feature_columns, records


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise ValueError("shuffle must be one of: true, false, 1, 0, yes, no")


def _parse_rate(value: str | None, name: str) -> float:
    if value is None:
        return 0.0
    rate = float(value)
    if not 0 <= rate <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return rate


def _add_data_quality_issues(
    records: list[dict[str, Any]],
    feature_columns: list[str],
    missing_rate: float,
    invalid_rate: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Inject nulls and non-numeric feature values for pipeline testing."""
    dirty_records = [record.copy() for record in records]
    for record in dirty_records:
        if rng.random() < missing_rate:
            column = rng.choice(feature_columns + [TARGET_COLUMN])
            record[column] = None
        if rng.random() < invalid_rate:
            column = rng.choice(feature_columns)
            record[column] = rng.choice(["unknown", "not-a-number", "?"])
    return dirty_records

def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/v1/training-data")
    def training_data() -> Any:
        try:
            feature_columns, records = _load_dataset()
            limit_value = request.args.get("limit")
            limit = len(records) if limit_value is None else int(limit_value)
            if limit < 1:
                raise ValueError("limit must be greater than 0")
            if limit > len(records):
                raise ValueError(f"limit cannot exceed {len(records)}")

            shuffle = _parse_bool(request.args.get("shuffle"))
            missing_rate = _parse_rate(request.args.get("missing_rate"), "missing_rate")
            invalid_rate = _parse_rate(request.args.get("invalid_rate"), "invalid_rate")
            seed_value = request.args.get("seed")
            seed = None if seed_value is None else int(seed_value)
            rng = random.Random(seed)
            if shuffle:
                selected_records = records.copy()
                rng.shuffle(selected_records)
                selected_records = selected_records[:limit]
            else:
                selected_records = records[:limit]
            selected_records = _add_data_quality_issues(
                selected_records,
                feature_columns,
                missing_rate,
                invalid_rate,
                rng,
            )
        except (FileNotFoundError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

        return jsonify(
            {
                "dataset": DATA_FILE.name,
                "feature_columns": feature_columns,
                "target_column": TARGET_COLUMN,
                "row_count": len(selected_records),
                "quality": {
                    "missing_rate": missing_rate,
                    "invalid_rate": invalid_rate,
                    "seed": seed,
                },
                "data": selected_records,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
