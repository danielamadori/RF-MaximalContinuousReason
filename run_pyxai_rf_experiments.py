#!/usr/bin/env python3
"""
Run PyXAI Random Forest explanation timing experiments.

By default the script loads the converted scikit-learn RandomForest JSON files
already stored under baseline/Classifiers-100-converted. It can also reuse the
dataset loading conventions from init_uci.py, init_openml.py, and init_pmlb.py,
train a Random Forest in memory, import it into PyXAI, compute one explanation
per sample, and write both per-sample timings and dataset-level aggregates.
"""

from __future__ import print_function

import argparse
import csv
import importlib
import json
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
INIT_TYPE_MODULES = OrderedDict([
    ("baseline", "init_baseline"),
    ("uci", "init_uci"),
    ("openml", "init_openml"),
    ("pmlb", "init_pmlb"),
])

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
    def __init__(
        self,
        name,
        classifier_path=None,
        dataset_path=None,
        samples_path=None,
        init_type="baseline",
        sklearn_rf=None,
        feature_names=None,
        samples=None,
        labels=None,
        sample_source_indices=None,
        metadata=None,
    ):
        self.name = name
        self.classifier_path = Path(classifier_path) if classifier_path else None
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.samples_path = Path(samples_path) if samples_path else None
        self.init_type = init_type
        self.sklearn_rf = sklearn_rf
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.samples = samples
        self.labels = labels
        self.sample_source_indices = sample_source_indices
        self.metadata = metadata or {}


def normalize_init_type(value):
    value = (value or "baseline").strip().lower()
    if value.endswith(".py"):
        value = value[:-3]
    if value.startswith("init_"):
        value = value[5:]
    if value not in INIT_TYPE_MODULES:
        choices = ", ".join(INIT_TYPE_MODULES.keys())
        raise ValueError("Unknown init type `%s`. Choose one of: %s" % (value, choices))
    return value


def load_init_module(init_type):
    module_name = INIT_TYPE_MODULES[init_type]
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "Could not import %s for --init-type %s: %s. Install its dependencies first."
            % (module_name, init_type, exc)
        ) from exc


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
    if record.feature_names is not None and record.samples is not None:
        return record.feature_names, np.asarray(record.samples, dtype=np.float32)

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


def labels_equal(left, right):
    if str(left) == str(right):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def parse_max_features(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("none", "null"):
        return None
    if text.lower() in ("sqrt", "log2"):
        return text.lower()
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("Invalid boolean value `%s`" % value)


def build_rf_params(args):
    params = {"random_state": args.random_state}

    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.criterion:
        params["criterion"] = args.criterion
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.min_samples_split is not None:
        params["min_samples_split"] = args.min_samples_split
    if args.min_samples_leaf is not None:
        params["min_samples_leaf"] = args.min_samples_leaf
    if args.max_leaf_nodes is not None:
        params["max_leaf_nodes"] = args.max_leaf_nodes
    if args.max_features is not None:
        params["max_features"] = parse_max_features(args.max_features)
    if args.min_impurity_decrease is not None:
        params["min_impurity_decrease"] = args.min_impurity_decrease
    if args.bootstrap is not None:
        params["bootstrap"] = parse_bool(args.bootstrap)
    if args.rf_max_samples is not None:
        params["max_samples"] = args.rf_max_samples
    if args.ccp_alpha is not None:
        params["ccp_alpha"] = args.ccp_alpha

    return params


def train_random_forest(X_train, y_train, X_test, y_test, feature_names, args):
    from sklearn.ensemble import RandomForestClassifier

    rf_params = build_rf_params(args)

    if args.optimize:
        from rf_utils import get_rf_search_space, optimize_rf_hyperparameters

        if not args.quiet:
            print("[INFO] Optimizing RF hyperparameters, n_calls=%d" % args.n_calls)
        best_params, _, _, _ = optimize_rf_hyperparameters(
            X_train,
            y_train,
            get_rf_search_space(),
            n_iter=args.n_calls,
            random_state=args.random_state,
            X_test=X_test,
            y_test=y_test,
            verbose=0 if args.quiet else 1,
        )
        rf_params.update(best_params)

    if not args.quiet:
        print("[INFO] Training RandomForestClassifier with params: %s" % rf_params)

    sklearn_rf = RandomForestClassifier(**rf_params)
    sklearn_rf.fit(X_train, y_train)

    if not args.quiet:
        train_score = sklearn_rf.score(X_train, y_train)
        test_score = sklearn_rf.score(X_test, y_test) if len(X_test) else float("nan")
        print("[INFO] RF accuracy train=%.3f test=%.3f" % (train_score, test_score))
        print("[INFO] Features: %d" % len(feature_names))

    return sklearn_rf


def split_loaded_arrays(X, y, args, source_name):
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(str)
    source_indices = np.arange(len(X))
    original_count = len(X)

    if args.sample_pct <= 0 or args.sample_pct > 100:
        raise ValueError("--sample-pct must be in the range (0, 100].")

    if args.sample_pct < 100.0 and args.test_sample_index is None:
        n_keep = max(1, int(len(X) * (args.sample_pct / 100.0)))
        rng = np.random.default_rng(args.random_state)
        kept = rng.choice(len(X), size=n_keep, replace=False)
        X = X[kept]
        y = y[kept]
        source_indices = source_indices[kept]
        if not args.quiet:
            print("[INFO] %s: sampled %d/%d rows" % (source_name, n_keep, original_count))
    elif args.sample_pct < 100.0:
        print(
            "[WARNING] --sample-pct ignored because --test-sample-index preserves source indices."
        )

    if args.test_sample_index is not None:
        indices = parse_sample_indices(args.test_sample_index)
        if any(index < 0 or index >= len(X) for index in indices):
            raise ValueError(
                "--test-sample-index outside valid range 0-%d for %s"
                % (len(X) - 1, source_name)
            )

        test_rows = np.asarray(indices, dtype=int)
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[test_rows] = False
        return (
            X[train_mask],
            y[train_mask],
            X[test_rows],
            y[test_rows],
            source_indices[test_rows],
        )

    try:
        X_train, X_test, y_train, y_test, _, test_source_indices = train_test_split(
            X,
            y,
            source_indices,
            test_size=args.test_split,
            random_state=args.random_state,
            stratify=y,
        )
    except ValueError as exc:
        if not args.quiet:
            print("[WARNING] Stratified split failed (%s); falling back to plain split." % exc)
        X_train, X_test, y_train, y_test, _, test_source_indices = train_test_split(
            X,
            y,
            source_indices,
            test_size=args.test_split,
            random_state=args.random_state,
            stratify=None,
        )

    return X_train, y_train, X_test, y_test, test_source_indices


def build_openml_record(dataset_name, args):
    module = load_init_module("openml")
    X_df, y, actual_name = module.load_and_prepare_dataset(dataset_name)
    if X_df is None or y is None:
        raise ValueError("OpenML dataset `%s` could not be loaded." % dataset_name)

    feature_names = [str(column) for column in X_df.columns]
    y_values = np.asarray(y) if not hasattr(y, "to_numpy") else y.to_numpy()
    X_train, y_train, X_test, y_test, source_indices = split_loaded_arrays(
        np.asarray(X_df, dtype=float),
        y_values,
        args,
        actual_name,
    )
    sklearn_rf = train_random_forest(X_train, y_train, X_test, y_test, feature_names, args)

    return DatasetRecord(
        "openml:%s" % actual_name,
        init_type="openml",
        sklearn_rf=sklearn_rf,
        feature_names=feature_names,
        samples=X_test,
        labels=y_test,
        sample_source_indices=source_indices,
        metadata={"actual_name": actual_name, "source": "openml"},
    )


def build_pmlb_record(dataset_name, args):
    module = load_init_module("pmlb")
    X_df, y, actual_name = module.load_and_prepare_dataset(
        dataset_name,
        shuffle=args.test_sample_index is None,
        random_state=args.random_state,
    )
    if X_df is None or y is None:
        raise ValueError("PMLB dataset `%s` could not be loaded." % dataset_name)

    feature_names = [str(column) for column in X_df.columns]
    y_values = np.asarray(y) if not hasattr(y, "to_numpy") else y.to_numpy()
    X_train, y_train, X_test, y_test, source_indices = split_loaded_arrays(
        np.asarray(X_df, dtype=float),
        y_values,
        args,
        actual_name,
    )
    sklearn_rf = train_random_forest(X_train, y_train, X_test, y_test, feature_names, args)

    return DatasetRecord(
        "pmlb:%s" % actual_name,
        init_type="pmlb",
        sklearn_rf=sklearn_rf,
        feature_names=feature_names,
        samples=X_test,
        labels=y_test,
        sample_source_indices=source_indices,
        metadata={"actual_name": actual_name, "source": "pmlb"},
    )


def build_uci_record(dataset_name, args):
    module = load_init_module("uci")
    X_train, y_train, X_test, y_test, feature_names, class_names = module.load_and_prepare_dataset(
        dataset_name=dataset_name,
        dataset_id=args.uci_id,
        feature_prefix=args.feature_prefix,
        test_split=args.test_split,
        random_state=args.random_state,
    )

    if args.test_sample_index is not None:
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        indices = parse_sample_indices(args.test_sample_index)
        if any(index < 0 or index >= len(X_all) for index in indices):
            raise ValueError(
                "--test-sample-index outside valid range 0-%d for UCI dataset"
                % (len(X_all) - 1)
            )
        test_rows = np.asarray(indices, dtype=int)
        train_mask = np.ones(len(X_all), dtype=bool)
        train_mask[test_rows] = False
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_rows]
        y_test = y_all[test_rows]
        source_indices = test_rows
    else:
        source_indices = np.arange(len(X_test))

    actual_name = dataset_name or "ID=%s" % args.uci_id
    feature_names = [str(name) for name in feature_names]
    sklearn_rf = train_random_forest(X_train, y_train, X_test, y_test, feature_names, args)

    return DatasetRecord(
        "uci:%s" % actual_name,
        init_type="uci",
        sklearn_rf=sklearn_rf,
        feature_names=feature_names,
        samples=X_test,
        labels=y_test,
        sample_source_indices=source_indices,
        metadata={
            "actual_name": actual_name,
            "source": "uci",
            "classes": list(convert_numpy_types(class_names)),
        },
    )


def build_init_records(args):
    if args.init_type == "baseline":
        records = discover_datasets(
            classifiers_root=Path(args.classifiers_root),
            datasets_root=Path(args.datasets_root),
        )
        return select_datasets(records, args.datasets)

    requested = list(args.datasets or [])
    if args.init_type == "uci" and args.uci_id is not None:
        if len(requested) > 1:
            raise ValueError("--uci-id can only be used with zero or one --datasets value.")
        requested = [requested[0] if requested else None]

    if not requested or "all" in requested:
        raise ValueError(
            "--datasets is required for --init-type %s; use --list-datasets to inspect options."
            % args.init_type
        )

    builders = {
        "openml": build_openml_record,
        "pmlb": build_pmlb_record,
        "uci": build_uci_record,
    }
    return [builders[args.init_type](dataset_name, args) for dataset_name in requested]


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
            completed.add((row.get("dataset"), str(row.get("sample_index"))))
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
        "init_type": args.init_type,
        "reason_method": args.reason_method,
        "time_limit": args.time_limit,
        "majoritary_iterations": args.majoritary_iterations,
        "seed": args.seed,
        "class_label": args.class_label,
        "class_filter": args.class_filter,
        "test_split": args.test_split,
        "test_sample_index": args.test_sample_index,
        "sample_pct": args.sample_pct,
        "random_state": args.random_state,
        "rf_params": build_rf_params(args),
        "datasets": [record.name for record in selected_records],
        "dataset_metadata": [
            {
                "name": record.name,
                "init_type": record.init_type,
                "classifier_path": str(record.classifier_path) if record.classifier_path else "",
                "dataset_path": str(record.dataset_path) if record.dataset_path else "",
                "samples_path": str(record.samples_path) if record.samples_path else "",
                "metadata": convert_numpy_types(record.metadata),
            }
            for record in selected_records
        ],
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


def sample_display_index(record, sample_row):
    if record.sample_source_indices is None:
        return int(sample_row)
    value = record.sample_source_indices[sample_row]
    return convert_numpy_types(value)


def selected_sample_rows(record, samples, sklearn_rf, args):
    if args.sample_index:
        rows = parse_sample_indices(args.sample_index)
    else:
        rows = list(range(len(samples)))

    valid = []
    for row in rows:
        if row < 0 or row >= len(samples):
            raise ValueError(
                "Sample index %d outside valid range 0-%d"
                % (row, len(samples) - 1)
            )
        valid.append(row)

    if args.class_label is not None:
        filtered = []
        if args.class_filter == "actual":
            if record.labels is None:
                raise ValueError(
                    "--class-filter actual requires labels, unavailable for %s" % record.name
                )
            values = record.labels
        else:
            values = sklearn_rf.predict(samples)

        for row in valid:
            if labels_equal(values[row], args.class_label):
                filtered.append(row)
        valid = filtered

    if args.max_samples is not None:
        valid = valid[:args.max_samples]
    return valid


def run_dataset(record, args, output_dir, sample_csv, completed_keys, run_id):
    if not args.quiet:
        print("[DATASET] %s" % record.name)
        if record.classifier_path is not None:
            print("[INFO] Classifier: %s" % record.classifier_path)
        else:
            print("[INFO] Classifier: trained in memory from %s" % record.init_type)

    feature_names, samples = load_samples(record, separator=args.separator)

    if record.sklearn_rf is not None:
        sklearn_rf = record.sklearn_rf
    else:
        from load_rf_from_json import load_rf_from_json
        sklearn_rf = load_rf_from_json(record.classifier_path)

    selected_rows = selected_sample_rows(record, samples, sklearn_rf, args)
    if not selected_rows:
        if not args.quiet:
            print("[WARNING] No samples selected for %s" % record.name)
        return

    import_start = time.perf_counter()
    _, pyxai_model = import_model_into_pyxai(sklearn_rf, feature_names)
    import_seconds = time.perf_counter() - import_start

    explainer = initialize_explainer(
        pyxai_model,
        first_instance=samples[selected_rows[0]],
    )
    reason_kwargs = build_reason_kwargs(args)

    if not args.quiet:
        print("[INFO] Samples: %d, PyXAI import: %.3fs" % (len(selected_rows), import_seconds))

    for ordinal, sample_row in enumerate(selected_rows, start=1):
        sample_index = sample_display_index(record, sample_row)
        key = (record.name, str(sample_index))
        if args.resume and key in completed_keys:
            if not args.quiet and (ordinal == 1 or ordinal % args.progress_every == 0):
                print("[SKIP] %s sample %s already recorded" % (record.name, sample_index))
            continue

        instance = samples[sample_row]
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
            "classifier_path": str(record.classifier_path) if record.classifier_path else "",
            "dataset_path": str(record.dataset_path) if record.dataset_path else "",
            "samples_path": str(record.samples_path) if record.samples_path else "",
            "error": error,
            "reason": json_dumps_cell(reason),
            "reason_features": json_dumps_cell(features),
        }
        write_sample_row(sample_csv, row)
        completed_keys.add(key)

        if not args.quiet and (ordinal == 1 or ordinal % args.progress_every == 0):
            print(
                "[SAMPLE] %s %d/%d idx=%s status=%s total=%s"
                % (
                    record.name,
                    ordinal,
                    len(selected_rows),
                    sample_index,
                    status,
                    row["total_seconds"] or "n/a",
                )
            )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run PyXAI RF explanation timings using baseline or init_* dataset loaders."
    )
    parser.add_argument(
        "--init-type",
        default="baseline",
        help="Dataset/model source: baseline, uci, openml, pmlb, or init_*.py name.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Datasets to run. For baseline, use `all` or omit for every complete "
            "baseline dataset. For uci/openml/pmlb, specify at least one dataset."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets for the selected --init-type, then exit.",
    )
    parser.add_argument(
        "--classifiers-root",
        default=str(CLASSIFIERS_ROOT),
        help="Baseline classifier root used by --init-type baseline.",
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DATASETS_ROOT),
        help="Baseline dataset root used by --init-type baseline.",
    )
    parser.add_argument(
        "--uci-id",
        type=int,
        default=None,
        help="UCI dataset ID for --init-type uci.",
    )
    parser.add_argument(
        "--feature-prefix",
        default="f",
        help="Feature prefix passed to init_uci.py.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.3,
        help="Train/test split for init_uci/init_openml/init_pmlb sources.",
    )
    parser.add_argument(
        "--test-sample-index",
        default=None,
        help="Rows to hold out as test samples for non-baseline init sources, e.g. 0,3-5.",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=100.0,
        help="Percentage of loaded rows to keep before splitting for openml/pmlb.",
    )
    parser.add_argument(
        "--class-label",
        default=None,
        help="Optionally restrict measured samples to this class label.",
    )
    parser.add_argument(
        "--class-filter",
        choices=["predicted", "actual"],
        default="predicted",
        help="When --class-label is set, filter by RF prediction or actual label.",
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
    parser.add_argument("--random-state", type=int, default=42, help="RF/split random seed.")
    parser.add_argument("--n-estimators", type=int, default=10, help="RF number of trees.")
    parser.add_argument(
        "--criterion",
        default="gini",
        choices=["gini", "entropy"],
        help="RF split criterion.",
    )
    parser.add_argument("--max-depth", type=int, default=None, help="RF max depth.")
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="RF min_samples_split.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="RF min_samples_leaf.",
    )
    parser.add_argument("--max-leaf-nodes", type=int, default=None, help="RF max_leaf_nodes.")
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help='RF max_features: sqrt, log2, None, int, or float. Default matches init_*.py.',
    )
    parser.add_argument(
        "--min-impurity-decrease",
        type=float,
        default=0.0,
        help="RF min_impurity_decrease.",
    )
    parser.add_argument(
        "--bootstrap",
        default="True",
        help="RF bootstrap flag, True or False.",
    )
    parser.add_argument(
        "--rf-max-samples",
        type=float,
        default=None,
        help="RF max_samples for bootstrap. Separate from PyXAI --max-samples.",
    )
    parser.add_argument("--ccp-alpha", type=float, default=0.0, help="RF ccp_alpha.")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Bayesian RF hyperparameter optimization before PyXAI import.",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=20,
        help="Optimization iterations for --optimize.",
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
    args = parser.parse_args(argv)

    try:
        args.init_type = normalize_init_type(args.init_type)
        parse_bool(args.bootstrap)
    except ValueError as exc:
        parser.error(str(exc))

    if args.test_split <= 0 or args.test_split >= 1:
        parser.error("--test-split must be in the range (0, 1).")
    if args.sample_pct <= 0 or args.sample_pct > 100:
        parser.error("--sample-pct must be in the range (0, 100].")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be >= 0.")
    if args.progress_every <= 0:
        parser.error("--progress-every must be > 0.")
    if args.uci_id is not None and args.init_type != "uci":
        parser.error("--uci-id is only valid with --init-type uci.")

    return args


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    output_dir = Path(args.output_dir)
    sample_csv = output_dir / "pyxai_rf_sample_times.csv"
    aggregate_csv = output_dir / "pyxai_rf_dataset_aggregates.csv"

    if args.list_datasets:
        if args.init_type == "baseline":
            records = discover_datasets(
                classifiers_root=Path(args.classifiers_root),
                datasets_root=Path(args.datasets_root),
            )
            for name in records:
                print(name)
        else:
            try:
                load_init_module(args.init_type).list_available_datasets()
            except Exception as exc:
                print("[ERROR] %s" % exc, file=sys.stderr)
                return 1
        return 0

    try:
        selected_records = build_init_records(args)
    except Exception as exc:
        print("[ERROR] %s" % exc, file=sys.stderr)
        return 1

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
