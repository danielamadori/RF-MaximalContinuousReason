#!/usr/bin/env python3
"""
Run PyXAI Random Forest explanation timing experiments on baseline datasets.

The script loads the converted scikit-learn RandomForest JSON files already
stored under baseline/Classifiers-100-converted, imports each model into PyXAI,
computes one explanation per sample, and writes both per-sample timings and
dataset-level aggregates.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from helpers import convert_numpy_types, parse_sample_indices


CLASSIFIERS_ROOT = Path("baseline") / "Classifiers-100-converted"
DATASETS_ROOT = Path("baseline") / "resources" / "datasets"
DEFAULT_OUTPUT_DIR = Path("results") / "pyxai_rf"

SAMPLE_FIELDNAMES = [
    "run_id",
    "timestamp",
    "dataset",
    "sample_index",
    "reason_method",
    "status",
    "prediction",
    "reason_length",
    "set_instance_seconds",
    "explanation_seconds",
    "total_seconds",
    "classifier_path",
    "dataset_path",
    "samples_path",
    "error",
    "reason",
    "reason_features",
]

AGGREGATE_FIELDNAMES = [
    "dataset",
    "reason_method",
    "n_samples",
    "n_ok",
    "n_no_reason",
    "n_error",
    "total_seconds_sum",
    "total_seconds_mean",
    "total_seconds_median",
    "total_seconds_min",
    "total_seconds_max",
    "total_seconds_p95",
    "explanation_seconds_sum",
    "explanation_seconds_mean",
    "explanation_seconds_median",
    "explanation_seconds_min",
    "explanation_seconds_max",
    "explanation_seconds_p95",
]


class DatasetRecord(object):
    def __init__(self, name, classifier_path, dataset_path, samples_path):
        self.name = name
        self.classifier_path = Path(classifier_path)
        self.dataset_path = Path(dataset_path)
        self.samples_path = Path(samples_path)


def resolve_dataset_name(name, datasets_root):
    candidates = [name]
    if "_" in name:
        candidates.append(name.replace("_", "-"))
    if "-" in name:
        candidates.append(name.replace("-", "_"))

    for candidate in candidates:
        dataset_dir = datasets_root / candidate
        dataset_path = dataset_dir / ("%s.csv" % candidate)
        samples_path = dataset_dir / ("%s.samples" % candidate)
        if dataset_path.exists() and samples_path.exists():
            return candidate, dataset_path, samples_path
    return None, None, None


def first_json_file(classifier_dir):
    json_files = sorted(classifier_dir.glob("*.json"))
    if not json_files:
        return None
    return json_files[0]


def discover_datasets(classifiers_root=CLASSIFIERS_ROOT, datasets_root=DATASETS_ROOT):
    records = OrderedDict()
    if not classifiers_root.exists():
        return records

    for classifier_dir in sorted(classifiers_root.iterdir()):
        if not classifier_dir.is_dir():
            continue

        classifier_path = first_json_file(classifier_dir)
        if classifier_path is None:
            continue

        dataset_name, dataset_path, samples_path = resolve_dataset_name(
            classifier_dir.name, datasets_root
        )
        if dataset_name is None:
            continue

        existing = records.get(dataset_name)
        if existing is None:
            records[dataset_name] = DatasetRecord(
                dataset_name, classifier_path, dataset_path, samples_path
            )
            continue

        # Prefer the classifier directory whose name matches the dataset exactly.
        if classifier_dir.name == dataset_name and existing.classifier_path.parent.name != dataset_name:
            records[dataset_name] = DatasetRecord(
                dataset_name, classifier_path, dataset_path, samples_path
            )

    return records


def select_datasets(records, requested):
    if not requested or "all" in requested:
        return list(records.values())

    selected = []
    missing = []
    aliases = {}
    for name, record in records.items():
        aliases[name] = record
        aliases[name.replace("-", "_")] = record
        aliases[name.replace("_", "-")] = record
        aliases[record.classifier_path.parent.name] = record

    seen = set()
    for name in requested:
        record = aliases.get(name)
        if record is None:
            missing.append(name)
            continue
        if record.name not in seen:
            selected.append(record)
            seen.add(record.name)

    if missing:
        available = ", ".join(records.keys())
        raise ValueError(
            "Unknown or incomplete dataset(s): %s. Available: %s"
            % (", ".join(missing), available)
        )

    return selected


def load_samples(record, separator=","):
    with record.dataset_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=separator)
        header = next(reader)
    feature_names = header[:-1]

    samples = np.loadtxt(str(record.samples_path), delimiter=separator)
    samples = np.atleast_2d(samples)

    expected_features = len(feature_names)
    if samples.shape[1] == expected_features + 1:
        samples = samples[:, :-1]
    elif samples.shape[1] != expected_features:
        raise ValueError(
            "Sample feature count mismatch for %s: expected %d, found %d"
            % (record.name, expected_features, samples.shape[1])
        )

    return feature_names, np.asarray(samples, dtype=np.float32)


def load_pyxai_modules():
    try:
        from pyxai import Explainer, Learning
    except ImportError as exc:
        raise RuntimeError(
            "PyXAI is required. Install it with `pip install pyxai` "
            "or `pip install -r requirements.txt`."
        ) from exc
    return Learning, Explainer


def import_model_into_pyxai(sklearn_rf, feature_names):
    Learning, _ = load_pyxai_modules()
    try:
        learner, model = Learning.import_models(sklearn_rf, feature_names=feature_names)
    except TypeError:
        learner, model = Learning.import_models(sklearn_rf, feature_names)
    if isinstance(model, (list, tuple)):
        model = model[0]
    return learner, model


def initialize_explainer(pyxai_model, first_instance=None):
    _, Explainer = load_pyxai_modules()
    try:
        return Explainer.initialize(pyxai_model)
    except TypeError:
        if first_instance is None:
            raise
        return Explainer.initialize(pyxai_model, instance=first_instance)


def build_reason_kwargs(args):
    kwargs = {}
    if args.time_limit is not None:
        kwargs["time_limit"] = args.time_limit

    if args.reason_method == "majoritary_reason":
        kwargs["n_iterations"] = args.majoritary_iterations
        kwargs["seed"] = args.seed

    return kwargs


def compute_reason(explainer, reason_method, kwargs):
    method = getattr(explainer, reason_method, None)
    if method is None:
        raise AttributeError("PyXAI explainer has no method `%s`" % reason_method)
    return method(**kwargs)


def reason_length(reason):
    if reason is None:
        return ""
    try:
        return len(reason)
    except TypeError:
        return 1


def json_dumps_cell(value):
    if value in (None, ""):
        return ""
    return json.dumps(convert_numpy_types(value), ensure_ascii=True)


def safe_prediction(model, instance):
    pred = model.predict(np.asarray([instance]))
    if isinstance(pred, np.ndarray):
        pred = pred[0]
    return convert_numpy_types(pred)


def format_float(value):
    if value in (None, ""):
        return ""
    return "%.9f" % float(value)


def write_sample_row(sample_csv, row):
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = sample_csv.exists()
    with sample_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def load_completed_keys(sample_csv, reason_method, retry_errors=False):
    completed = set()
    if not sample_csv.exists():
        return completed

    with sample_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reason_method") != reason_method:
                continue
            status = row.get("status", "")
            if retry_errors and status == "error":
                continue
            completed.add((row.get("dataset"), int(row.get("sample_index"))))
    return completed


def as_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, percent):
    if not values:
        return ""
    return float(np.percentile(np.asarray(values, dtype=float), percent))


def metric_summary(values):
    if not values:
        return {
            "sum": "",
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "p95": "",
        }
    return {
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95),
    }


def write_aggregates(sample_csv, aggregate_csv, reason_method):
    latest_rows = OrderedDict()
    if not sample_csv.exists():
        return

    with sample_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reason_method") == reason_method:
                latest_rows[(row.get("dataset"), row.get("sample_index"))] = row

    rows_by_dataset = defaultdict(list)
    for row in latest_rows.values():
        rows_by_dataset[row.get("dataset")].append(row)

    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDNAMES)
        writer.writeheader()
        for dataset in sorted(rows_by_dataset):
            rows = rows_by_dataset[dataset]
            statuses = [row.get("status", "") for row in rows]
            total_values = [
                value
                for value in (as_float(row.get("total_seconds")) for row in rows)
                if value is not None
            ]
            explanation_values = [
                value
                for value in (as_float(row.get("explanation_seconds")) for row in rows)
                if value is not None
            ]
            total_summary = metric_summary(total_values)
            explanation_summary = metric_summary(explanation_values)

            writer.writerow({
                "dataset": dataset,
                "reason_method": reason_method,
                "n_samples": len(rows),
                "n_ok": statuses.count("ok"),
                "n_no_reason": statuses.count("no_reason"),
                "n_error": statuses.count("error"),
                "total_seconds_sum": format_float(total_summary["sum"]),
                "total_seconds_mean": format_float(total_summary["mean"]),
                "total_seconds_median": format_float(total_summary["median"]),
                "total_seconds_min": format_float(total_summary["min"]),
                "total_seconds_max": format_float(total_summary["max"]),
                "total_seconds_p95": format_float(total_summary["p95"]),
                "explanation_seconds_sum": format_float(explanation_summary["sum"]),
                "explanation_seconds_mean": format_float(explanation_summary["mean"]),
                "explanation_seconds_median": format_float(explanation_summary["median"]),
                "explanation_seconds_min": format_float(explanation_summary["min"]),
                "explanation_seconds_max": format_float(explanation_summary["max"]),
                "explanation_seconds_p95": format_float(explanation_summary["p95"]),
            })


def write_manifest(output_dir, run_id, args, selected_records):
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "reason_method": args.reason_method,
        "time_limit": args.time_limit,
        "majoritary_iterations": args.majoritary_iterations,
        "seed": args.seed,
        "datasets": [record.name for record in selected_records],
        "sample_csv": str(output_dir / "pyxai_rf_sample_times.csv"),
        "aggregate_csv": str(output_dir / "pyxai_rf_dataset_aggregates.csv"),
    }

    try:
        import importlib.metadata as importlib_metadata
        manifest["pyxai_version"] = importlib_metadata.version("pyxai")
    except Exception:
        manifest["pyxai_version"] = "unknown"

    manifest_path = output_dir / ("manifest_%s.json" % run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def sample_indices_for(samples, args):
    if args.sample_index:
        indices = parse_sample_indices(args.sample_index)
    else:
        indices = list(range(len(samples)))

    valid = []
    for index in indices:
        if index < 0 or index >= len(samples):
            raise ValueError(
                "Sample index %d outside valid range 0-%d"
                % (index, len(samples) - 1)
            )
        valid.append(index)

    if args.max_samples is not None:
        valid = valid[:args.max_samples]
    return valid


def run_dataset(record, args, output_dir, sample_csv, completed_keys, run_id):
    from load_rf_from_json import load_rf_from_json

    if not args.quiet:
        print("[DATASET] %s" % record.name)
        print("[INFO] Classifier: %s" % record.classifier_path)

    feature_names, samples = load_samples(record, separator=args.separator)
    selected_indices = sample_indices_for(samples, args)

    sklearn_rf = load_rf_from_json(record.classifier_path)
    import_start = time.perf_counter()
    _, pyxai_model = import_model_into_pyxai(sklearn_rf, feature_names)
    import_seconds = time.perf_counter() - import_start

    explainer = initialize_explainer(
        pyxai_model,
        first_instance=samples[selected_indices[0]] if selected_indices else None,
    )
    reason_kwargs = build_reason_kwargs(args)

    if not args.quiet:
        print("[INFO] Samples: %d, PyXAI import: %.3fs" % (len(selected_indices), import_seconds))

    for ordinal, sample_index in enumerate(selected_indices, start=1):
        key = (record.name, sample_index)
        if args.resume and key in completed_keys:
            if not args.quiet and (ordinal == 1 or ordinal % args.progress_every == 0):
                print("[SKIP] %s sample %d already recorded" % (record.name, sample_index))
            continue

        instance = samples[sample_index]
        timestamp = datetime.now().isoformat()
        set_instance_seconds = None
        explanation_seconds = None
        total_seconds = None
        reason = None
        features = None
        error = ""
        status = "ok"
        prediction = ""
        total_start = None
        explanation_start = None

        try:
            prediction = safe_prediction(sklearn_rf, instance)
            total_start = time.perf_counter()

            set_start = time.perf_counter()
            try:
                explainer.set_instance(instance)
            finally:
                set_instance_seconds = time.perf_counter() - set_start

            explanation_start = time.perf_counter()
            try:
                reason = compute_reason(explainer, args.reason_method, reason_kwargs)
            finally:
                explanation_seconds = time.perf_counter() - explanation_start
            total_seconds = time.perf_counter() - total_start

            if reason is None:
                status = "no_reason"
            elif args.store_features:
                try:
                    features = explainer.to_features(reason)
                except Exception as feature_exc:
                    features = ["to_features failed: %s: %s" % (
                        type(feature_exc).__name__,
                        feature_exc,
                    )]
        except Exception as exc:
            if total_start is not None:
                total_seconds = time.perf_counter() - total_start
            if explanation_start is not None and explanation_seconds is None:
                explanation_seconds = time.perf_counter() - explanation_start
            status = "error"
            error = "%s: %s" % (type(exc).__name__, exc)
            if args.fail_fast:
                raise

        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset": record.name,
            "sample_index": sample_index,
            "reason_method": args.reason_method,
            "status": status,
            "prediction": json_dumps_cell(prediction),
            "reason_length": reason_length(reason),
            "set_instance_seconds": format_float(set_instance_seconds),
            "explanation_seconds": format_float(explanation_seconds),
            "total_seconds": format_float(total_seconds),
            "classifier_path": str(record.classifier_path),
            "dataset_path": str(record.dataset_path),
            "samples_path": str(record.samples_path),
            "error": error,
            "reason": json_dumps_cell(reason),
            "reason_features": json_dumps_cell(features),
        }
        write_sample_row(sample_csv, row)
        completed_keys.add(key)

        if not args.quiet and (ordinal == 1 or ordinal % args.progress_every == 0):
            print(
                "[SAMPLE] %s %d/%d idx=%d status=%s total=%s"
                % (
                    record.name,
                    ordinal,
                    len(selected_indices),
                    sample_index,
                    status,
                    row["total_seconds"] or "n/a",
                )
            )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run PyXAI RF explanation timings on baseline datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to run. Use `all` or omit for every complete baseline dataset.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets with classifier, CSV, and samples, then exit.",
    )
    parser.add_argument(
        "--reason-method",
        default="majoritary_reason",
        choices=[
            "majoritary_reason",
            "minimal_majoritary_reason",
            "sufficient_reason",
            "minimal_sufficient_reason",
            "direct_reason",
        ],
        help="PyXAI explanation method to time.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Per-sample PyXAI time limit in seconds when supported by the method.",
    )
    parser.add_argument(
        "--majoritary-iterations",
        type=int,
        default=50,
        help="Iterations for majoritary_reason greedy search.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for PyXAI where supported.")
    parser.add_argument(
        "--sample-index",
        default=None,
        help="Comma-separated sample indices/ranges, e.g. 0,3-5.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of selected samples per dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV separator for datasets and samples.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip rows already present in the per-sample CSV.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With resume enabled, retry rows whose previous status was error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing PyXAI output CSV files before running.",
    )
    parser.add_argument(
        "--store-features",
        action="store_true",
        help="Also store explainer.to_features(reason). Not included in timing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first sample error.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    output_dir = Path(args.output_dir)
    sample_csv = output_dir / "pyxai_rf_sample_times.csv"
    aggregate_csv = output_dir / "pyxai_rf_dataset_aggregates.csv"

    records = discover_datasets()
    if args.list_datasets:
        for name in records:
            print(name)
        return 0

    selected_records = select_datasets(records, args.datasets)
    if not selected_records:
        print("[ERROR] No complete datasets found.", file=sys.stderr)
        return 1

    if args.overwrite:
        for path in [sample_csv, aggregate_csv]:
            if path.exists():
                path.unlink()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    completed_keys = load_completed_keys(
        sample_csv,
        args.reason_method,
        retry_errors=args.retry_errors,
    )
    manifest_path = write_manifest(output_dir, run_id, args, selected_records)

    if not args.quiet:
        print("[RUN] %s" % run_id)
        print("[INFO] Datasets: %s" % ", ".join(record.name for record in selected_records))
        print("[INFO] Output: %s" % output_dir)
        print("[INFO] Manifest: %s" % manifest_path)

    for record in selected_records:
        run_dataset(record, args, output_dir, sample_csv, completed_keys, run_id)

    write_aggregates(sample_csv, aggregate_csv, args.reason_method)
    if not args.quiet:
        print("[DONE] Per-sample timings: %s" % sample_csv)
        print("[DONE] Dataset aggregates: %s" % aggregate_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
