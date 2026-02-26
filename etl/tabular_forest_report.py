#!/usr/bin/env python3
"""Generate ``forest_report.json`` (and optionally ``forest_report.csv``) for
the baseline tabular datasets.

For every dataset found in ``baseline/Classifiers-100-converted/`` the script:

1. Loads the dataset CSV from ``baseline/resources/datasets/<name>/<name>.csv``.
2. Loads the first available pre-trained JSON classifier from
   ``baseline/Classifiers-100-converted/<name>/``.
3. Evaluates accuracy on a held-out 30 % test split.
4. Collects forest statistics (n_estimators, max_depth).
5. Writes the aggregated results to ``forest_report.json``.

The JSON schema is identical to the one produced by the old
``etl/dataset_forest_report.py`` (which targeted Aeon datasets — now removed),
so all downstream consumers (``etl/tables.py``,
``etl/sort_datasets_by_complexity.py``, ``etl/forest_report_to_csv.py``,
``models_analysis.ipynb``) continue to work without modification.

Example usage::

    # Run from the repository root
    python etl/tabular_forest_report.py

    # Save to a custom path
    python etl/tabular_forest_report.py --output results/my_forest_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so that sibling top-level modules can be imported when
# the script is executed from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from load_rf_from_json import load_rf_from_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFIERS_ROOT = _PROJECT_ROOT / "baseline" / "Classifiers-100-converted"
DATASETS_ROOT = _PROJECT_ROOT / "baseline" / "resources" / "datasets"
DEFAULT_OUTPUT = _PROJECT_ROOT / "forest_report.json"
TEST_SIZE = 0.30
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path):
    """Load a dataset CSV and return (X, y, feature_names).

    The last column is treated as the class label, mirroring the convention
    used in ``baseline.xrf.Dataset``.  Labels are encoded to integers via
    :class:`~sklearn.preprocessing.LabelEncoder` when they are non-numeric.
    """
    import csv

    rows: List[List[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")

    # Feature names = all columns except the last
    feature_names: List[str] = header[:-1] if header else [f"f{i}" for i in range(len(rows[0]) - 1)]

    X_raw = np.array([row[:-1] for row in rows], dtype=float)
    labels_raw = [row[-1].strip() for row in rows]

    # Encode labels
    all_numeric = all(lbl.lstrip("-").replace(".", "", 1).isnumeric() for lbl in labels_raw)
    if all_numeric:
        y = np.array([int(float(lbl)) for lbl in labels_raw])
    else:
        le = LabelEncoder()
        y = le.fit_transform(labels_raw)

    return X_raw, y, feature_names


def _find_json_classifier(dataset_dir: Path) -> Optional[Path]:
    """Return the path to the first JSON classifier file found in *dataset_dir*."""
    candidates = sorted(dataset_dir.glob("*.json"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Forest statistics helpers
# ---------------------------------------------------------------------------

def _forest_statistics(rf) -> Dict[str, Any]:
    """Extract basic statistics from a fitted scikit-learn RF."""
    n_estimators = len(rf.estimators_)
    depths = [t.tree_.max_depth for t in rf.estimators_]
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max(depths)) if depths else 0,
        "avg_depth": float(np.mean(depths)) if depths else 0.0,
        "total_nodes": int(sum(t.tree_.node_count for t in rf.estimators_)),
        "total_leaves": int(sum(t.tree_.n_leaves for t in rf.estimators_)),
    }


def _best_params_from_rf(rf) -> Dict[str, Any]:
    """Extract the constructor parameters of a fitted scikit-learn RF."""
    params = rf.get_params()
    return {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in params.items()
            if k in ("n_estimators", "max_depth", "min_samples_split",
                     "min_samples_leaf", "max_features", "criterion", "bootstrap")}


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str) -> Dict[str, Any]:
    """Run the evaluation pipeline for a single baseline dataset.

    Returns a report entry dict compatible with the ``forest_report.json``
    schema expected by the downstream ETL tools.
    """
    csv_path = DATASETS_ROOT / dataset_name / f"{dataset_name}.csv"
    classifier_dir = CLASSIFIERS_ROOT / dataset_name
    json_path = _find_json_classifier(classifier_dir)

    if not csv_path.exists():
        return {
            "dataset": dataset_name,
            "status": "error",
            "error_message": f"CSV not found: {csv_path}",
            "timestamp": datetime.now().isoformat(),
        }

    if json_path is None:
        return {
            "dataset": dataset_name,
            "status": "error",
            "error_message": f"No JSON classifier found in {classifier_dir}",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        X, y, feature_names = _load_csv(csv_path)
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "error",
            "error_message": f"Failed to load dataset: {exc}",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        rf = load_rf_from_json(str(json_path))
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "error",
            "error_message": f"Failed to load RF from {json_path}: {exc}",
            "timestamp": datetime.now().isoformat(),
        }

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        # Fallback when stratification fails (e.g. too few samples per class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    # Evaluate
    try:
        accuracy = float(rf.score(X_test, y_test))
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "error",
            "error_message": f"Failed to evaluate RF: {exc}",
            "timestamp": datetime.now().isoformat(),
        }

    n_classes = len(np.unique(y))
    stats = _forest_statistics(rf)
    best_params = _best_params_from_rf(rf)

    return {
        "dataset": dataset_name,
        "status": "success",
        "accuracy": round(accuracy, 6),
        "best_params": best_params,
        "metadata": {
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": n_classes,
            "feature_names": feature_names,
            "series_length": None,   # Tabular dataset; no series length
            "classifier_source": str(json_path.name),
        },
        "forest_statistics": stats,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _discover_datasets() -> List[str]:
    """Return the sorted list of dataset names found in CLASSIFIERS_ROOT."""
    if not CLASSIFIERS_ROOT.exists():
        raise SystemExit(f"Classifiers directory not found: {CLASSIFIERS_ROOT}")
    return sorted(
        name for name in os.listdir(CLASSIFIERS_ROOT)
        if (CLASSIFIERS_ROOT / name).is_dir()
    )


def build_report(datasets: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
    """Process all datasets and return the aggregated report list."""
    report: List[Dict[str, Any]] = []
    total = len(datasets)

    for idx, name in enumerate(datasets, start=1):
        if verbose:
            print(f"[{idx:>3}/{total}] Processing {name}...", end="  ", flush=True)
        entry = process_dataset(name)
        report.append(entry)
        if verbose:
            status = entry.get("status", "?")
            if status == "success":
                acc = entry.get("accuracy", 0)
                n_est = entry.get("forest_statistics", {}).get("n_estimators", "?")
                print(f"OK  accuracy={acc:.4f}  n_estimators={n_est}")
            else:
                print(f"ERROR  {entry.get('error_message', '')}")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate forest_report.json for baseline tabular datasets. "
            "Run from the repository root."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the output JSON report (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-dataset progress (default: True).",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Suppress per-dataset progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = _discover_datasets()
    print(f"Found {len(datasets)} baseline datasets in {CLASSIFIERS_ROOT}")

    report = build_report(datasets, verbose=args.verbose)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    success_count = sum(1 for e in report if e.get("status") == "success")
    error_count = len(report) - success_count
    print(f"\nReport written to {output_path}")
    print(f"  Success: {success_count} datasets")
    if error_count:
        print(f"  Errors:  {error_count} datasets")
        for entry in report:
            if entry.get("status") != "success":
                print(f"    - {entry['dataset']}: {entry.get('error_message', 'unknown error')}")


if __name__ == "__main__":
    main()
