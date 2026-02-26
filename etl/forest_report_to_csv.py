#!/usr/bin/env python3
"""Convert a forest report stored as JSON into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


NESTED_KEYS = {"metadata", "best_params", "forest_statistics"}


def _collect_fieldnames(data: Iterable[Dict[str, Any]]) -> List[str]:
    """Inspect the report entries and determine the CSV header."""

    top_level_keys: set[str] = set()
    metadata_keys: set[str] = set()
    params_keys: set[str] = set()
    statistics_keys: set[str] = set()

    for entry in data:
        for key in entry:
            if key in NESTED_KEYS:
                continue
            top_level_keys.add(key)

        metadata = entry.get("metadata") or {}
        metadata_keys.update(metadata.keys())

        best_params = entry.get("best_params") or {}
        params_keys.update(best_params.keys())

        forest_statistics = entry.get("forest_statistics") or {}
        statistics_keys.update(forest_statistics.keys())

    # Ensure dataset and status are always first.
    ordered_top_level = ["dataset", "status"]
    for key in sorted(top_level_keys):
        if key not in {"dataset", "status"}:
            ordered_top_level.append(key)

    ordered_metadata = [f"metadata_{key}" for key in sorted(metadata_keys)]
    ordered_params = [f"best_params_{key}" for key in sorted(params_keys)]
    ordered_statistics = [f"forest_statistics_{key}" for key in sorted(statistics_keys)]

    return ordered_top_level + ordered_metadata + ordered_params + ordered_statistics


def _serialise_sequence(value: Sequence[Any]) -> str:
    """Convert a sequence into a human-readable string."""

    return "|".join(str(item) for item in value)


def _format_value(value: Any) -> Any:
    """Prepare values for CSV serialization."""

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _serialise_sequence(value)

    return json.dumps(value, ensure_ascii=False)


def _flatten_entry(entry: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    """Create a flat dictionary containing all the CSV columns."""

    row: Dict[str, Any] = {field: "" for field in fieldnames}

    for key, value in entry.items():
        if key not in NESTED_KEYS:
            row[key] = _format_value(value)

    metadata = entry.get("metadata") or {}
    for key, value in metadata.items():
        column = f"metadata_{key}"
        if column in row:
            row[column] = _format_value(value)

    best_params = entry.get("best_params") or {}
    for key, value in best_params.items():
        column = f"best_params_{key}"
        if column in row:
            row[column] = _format_value(value)

    forest_statistics = entry.get("forest_statistics") or {}
    for key, value in forest_statistics.items():
        column = f"forest_statistics_{key}"
        if column in row:
            row[column] = _format_value(value)

    return row


def convert_report(input_path: Path, output_path: Path) -> None:
    """Read the JSON report from *input_path* and write a CSV to *output_path*."""

    with input_path.open("r", encoding="utf-8") as handle:
        data: List[Dict[str, Any]] = json.load(handle)

    fieldnames = _collect_fieldnames(data)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(_flatten_entry(entry, fieldnames))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="forest_report.json",
        type=Path,
        help="Path to the JSON report (default: forest_report.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination CSV file (default: replace .json with .csv).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path: Path = args.input

    if args.output is not None:
        output_path = args.output
    else:
        output_path = input_path.with_suffix(".csv")

    convert_report(input_path, output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
