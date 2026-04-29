#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments_baseline_timeseries.py

Experiment runner for baseline Random Forest explanation analysis on
univariate time-series datasets loaded through aeon.

This script mirrors the structure of run_experiments_baseline.py but adapts
loading and preprocessing to aeon/UCR time-series datasets so they can be
analyzed with the same explanation backends used for tabular data.

Main design choice:
- load train/test splits from aeon
- flatten each univariate series into a tabular vector
- build a lightweight dataset wrapper exposing the subset of the tabular
  Dataset API required by the baseline experiment runner
- train RFSklearn/RFBreiman or load a converted JSON classifier when needed
- generate explanations on selected test samples
"""

from __future__ import print_function
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False

from load_rf_from_json import load_rf_from_json

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Install redis package to enable Redis support.")

from baseline.xrf import XRF
from baseline.xrf import RFBreiman, RFSklearn

SUPPORTED_EXPLAINERS = ('xrf', 'infxp')
INFXP_EXPLAINER_CLASS = None

DEFAULT_OUTPUT_DIR = 'baseline/resources/experiments_timeseries'
DEFAULT_FEATURE_PREFIX = 't'


AVAILABLE_DATASETS = [
    'Coffee', 'ECG200', 'GunPoint', 'ItalyPowerDemand', 'Lightning2', 'Lightning7',
    'MedicalImages', 'MoteStrain', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
    'Symbols', 'SyntheticControl', 'TwoLeadECG', 'Wafer', 'Wine', 'Yoga',
    'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    'ChlorineConcentration', 'CinCECGTorso', 'Computers', 'CricketX', 'CricketY', 'CricketZ',
    'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW', 'Earthquakes', 'ECG5000', 'ECGFiveDays', 'ElectricDevices',
    'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',
    'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound',
    'LargeKitchenAppliances', 'Mallat', 'Meat', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
    'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
    'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
    'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'StarlightCurves',
    'Strawberry', 'SwedishLeaf', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
    'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ', 'WordSynonyms', 'Worms', 'WormsTwoClass'
]


def _add_sys_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_infxp_explainer_class():
    global INFXP_EXPLAINER_CLASS
    if INFXP_EXPLAINER_CLASS is not None:
        return INFXP_EXPLAINER_CLASS

    root_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.join(root_dir, 'baseline')
    infxp_dir = os.path.join(baseline_dir, 'infxp')

    _add_sys_path(baseline_dir)
    _add_sys_path(infxp_dir)

    try:
        from baseline.infxp.Infxpl import INFXRF
    except ImportError as e:
        raise ImportError(
            "Could not import the infxp backend from baseline/infxp/Infxpl.py. "
            "Make sure the infxp dependencies are installed."
        ) from e

    INFXP_EXPLAINER_CLASS = INFXRF
    return INFXP_EXPLAINER_CLASS


class AeonTimeSeriesDataset(object):
    """
    Lightweight dataset wrapper exposing the API expected by the baseline
    experiment runner while sourcing data from aeon univariate datasets.
    """

    def __init__(self, dataset_name, feature_prefix='t', test_size=None,
                 random_state=42, sample_percentage=100.0, verbose=True):
        if not AEON_AVAILABLE:
            raise ImportError("aeon is not installed. Install it with: pip install aeon")

        self.dataset_name = dataset_name
        self.feature_prefix = feature_prefix
        self.random_state = random_state
        self.sample_percentage = sample_percentage
        self.verbose = verbose

        X_train_raw, y_train = load_classification(dataset_name, split='train')
        X_test_raw, y_test = load_classification(dataset_name, split='test')

        if X_train_raw.ndim != 3:
            raise ValueError(
                "Only 3D aeon classification arrays are supported: "
                "(n_samples, n_channels, n_timepoints)."
            )
        if X_train_raw.shape[1] != 1:
            raise ValueError(
                f"Dataset '{dataset_name}' is not univariate (n_channels={X_train_raw.shape[1]})."
            )

        self.n_channels = int(X_train_raw.shape[1])
        self.series_length = int(X_train_raw.shape[2])

        X_train = self._flatten_series(X_train_raw)
        X_test = self._flatten_series(X_test_raw)
        y_train = np.asarray(y_train).astype(str)
        y_test = np.asarray(y_test).astype(str)

        if sample_percentage is not None and sample_percentage < 100.0:
            X_train, y_train = self._subsample(X_train, y_train, sample_percentage, random_state)
            X_test, y_test = self._subsample(X_test, y_test, sample_percentage, random_state + 1)

        if test_size is not None:
            X_all = np.vstack([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=test_size,
                random_state=random_state,
                stratify=y_all if len(np.unique(y_all)) > 1 else None,
            )

        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.X_test = np.asarray(X_test, dtype=np.float32)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

        self.data = np.vstack([self.X_train, self.X_test])
        self.labels = np.concatenate([self.y_train, self.y_test])
        self.targets = sorted(np.unique(self.labels).tolist())

        width = len(str(self.series_length - 1))
        self.features = [f"{feature_prefix}_{i:0{width}d}" for i in range(self.series_length)]
        self.m_features_ = list(self.features)
        self.categorical_features = []
        self.categorical_names = {}
        self.class_names = list(self.targets)
        self.cat_data = False

        if self.verbose:
            print(f"Loaded aeon dataset: {dataset_name}")
            print(f"  Train shape: {self.X_train.shape}")
            print(f"  Test shape:  {self.X_test.shape}")
            print(f"  Series length: {self.series_length}")
            print(f"  Classes: {self.targets}")

    @staticmethod
    def _flatten_series(X):
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def _subsample(X, y, sample_percentage, random_state):
        if sample_percentage <= 0:
            raise ValueError("sample_percentage must be > 0")
        n_samples = len(X)
        n_keep = max(1, int(np.ceil(n_samples * sample_percentage / 100.0)))
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_samples, size=n_keep, replace=False)
        return X[indices], y[indices]

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)

    def transform_inverse(self, X):
        return np.asarray(X, dtype=np.float32)

    def readable_sample(self, x):
        return x

    def train_test_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


class TimeSeriesExperimentRunner:
    """Run baseline explanation experiments on aeon time-series datasets."""

    def __init__(self, dataset_name, output_dir=DEFAULT_OUTPUT_DIR,
                 feature_prefix=DEFAULT_FEATURE_PREFIX, algo='sklearn', verbose=True,
                 use_redis=False, redis_host='localhost', redis_port=6379, redis_db=0,
                 classifier_json_dirs=None, classifier_json_required=False,
                 explainer_backend='xrf', test_size=None, random_state=42,
                 sample_percentage=100.0):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.feature_prefix = feature_prefix
        self.algo = algo
        self.verbose = verbose
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.explainer_backend = self._normalize_explainer_backend(explainer_backend)
        self.test_size = test_size
        self.random_state = random_state
        self.sample_percentage = sample_percentage

        if classifier_json_dirs is None:
            classifier_json_dirs = [
                os.path.join('baseline', 'Classifiers-100-converted'),
                os.path.join('baseline', 'Classifiers-converted')
            ]
        self.classifier_json_dirs = self._normalize_classifier_dirs(classifier_json_dirs)
        self.classifier_json_required = classifier_json_required

        self.redis_conn = None
        if self.use_redis:
            try:
                self.redis_conn = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                self.redis_conn.ping()
                print(f"Connected to Redis at {redis_host}:{redis_port} (DB {redis_db})")
            except Exception as e:
                print(f"Warning: Failed to connect to Redis: {e}")
                self.use_redis = False
                self.redis_conn = None

        self.data = self._load_dataset()
        self.bench_name = self.dataset_name
        if self.explainer_backend == 'xrf':
            self.bench_dir = os.path.join(output_dir, self.bench_name)
        else:
            self.bench_dir = os.path.join(output_dir, self.explainer_backend, self.bench_name)
        os.makedirs(self.bench_dir, exist_ok=True)

        self.results = []

    @staticmethod
    def _normalize_explainer_backend(explainer_backend):
        explainer_backend = (explainer_backend or 'xrf').lower()
        if explainer_backend not in SUPPORTED_EXPLAINERS:
            choices = ', '.join(SUPPORTED_EXPLAINERS)
            raise ValueError(f"Unsupported explainer backend '{explainer_backend}'. Choose one of: {choices}")
        return explainer_backend

    @staticmethod
    def _normalize_classifier_dirs(classifier_json_dirs):
        if classifier_json_dirs is None:
            return []
        if isinstance(classifier_json_dirs, str):
            return [classifier_json_dirs]
        return list(classifier_json_dirs)

    def _load_dataset(self):
        return AeonTimeSeriesDataset(
            dataset_name=self.dataset_name,
            feature_prefix=self.feature_prefix,
            test_size=self.test_size,
            random_state=self.random_state,
            sample_percentage=self.sample_percentage,
            verbose=self.verbose
        )

    def _find_classifier_json_path(self, n_estimators, max_depth):
        if not self.classifier_json_dirs:
            return None

        filename = f"{self.bench_name}_nbestim_{n_estimators}_maxdepth_{max_depth}.mod.json"
        for base_dir in self.classifier_json_dirs:
            json_path = os.path.join(base_dir, self.bench_name, filename)
            if os.path.exists(json_path):
                return json_path
        return None

    def _create_explainer(self, model, verb):
        if self.explainer_backend == 'xrf':
            explainer_cls = XRF
        else:
            explainer_cls = _load_infxp_explainer_class()

        explainer = explainer_cls(model, self.data.m_features_, self.data.targets, verb=verb)
        if getattr(self.data, 'cat_data', False):
            explainer.ffnames = self.data.m_features_
            explainer.readable_data = lambda x: self.data.readable_sample(self.data.transform_inverse(x)[0])
        return explainer

    def _explain_sample(self, explainer, sample, xtype, etype, smallest, sample_index):
        return explainer.explain(
            sample,
            xtype=xtype,
            etype=etype,
            smallest=smallest,
            sample_index=sample_index
        )

    @staticmethod
    def _jsonable_feature_indices(expl):
        if expl is None:
            return []
        return [int(idx) for idx in expl]

    @staticmethod
    def _get_explanation_backend_state(explainer):
        return getattr(explainer, 'x', None)

    def _build_explanation_record(self, explainer, expl, sample_index):
        backend_state = self._get_explanation_backend_state(explainer)
        interval_rule = getattr(backend_state, 'interval_explanation', None)
        interval_terms = getattr(backend_state, 'interval_preamble', None)
        rule = getattr(backend_state, 'explanation_rule', None) or interval_rule
        infxp_coverage = getattr(backend_state, 'infxp_coverage', None)
        axp_domain_coverage = getattr(backend_state, 'axp_domain_coverage', None)

        record = {
            'sample_index': None if sample_index is None else int(sample_index),
            'feature_indices': self._jsonable_feature_indices(expl),
        }
        if rule is not None:
            record['explanation_rule'] = rule
        if interval_rule is not None:
            record['interval_explanation'] = interval_rule
        if interval_terms is not None:
            record['interval_terms'] = interval_terms
        if infxp_coverage is not None:
            record['infxp_coverage'] = float(infxp_coverage)
        if axp_domain_coverage is not None:
            record['axp_domain_coverage'] = float(axp_domain_coverage)

        return record

    def _evaluate_model(self, model):
        X_train, X_test, y_train, y_test = self.data.train_test_split()
        X_train = self.data.transform(X_train)
        X_test = self.data.transform(X_test)
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)
        return train_acc, test_acc

    @staticmethod
    def _get_forest_model(model):
        if hasattr(model, 'forest'):
            return model.forest
        return model

    @staticmethod
    def _parse_json_filename(json_path):
        name = os.path.basename(json_path)
        if '_nbestim_' not in name or '_maxdepth_' not in name:
            return None, None
        try:
            n_part = name.split('_nbestim_', 1)[1]
            n_estimators_str, depth_part = n_part.split('_maxdepth_', 1)
            max_depth_str = depth_part.split('.', 1)[0]
            return int(n_estimators_str), int(max_depth_str)
        except (ValueError, IndexError):
            return None, None

    def run_single_experiment(self, n_estimators, max_depth, test_index_list=None, classifier_json_path=None):
        print(f"\n{'=' * 60}")
        print(
            f"Running experiment: dataset={self.dataset_name}, n_estimators={n_estimators}, "
            f"max_depth={max_depth}, explainer={self.explainer_backend}"
        )
        print(f"{'=' * 60}")

        json_path = classifier_json_path
        model_source = 'trained'
        cls = None

        if json_path:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

        if json_path or self.classifier_json_dirs:
            if json_path is None:
                json_path = self._find_classifier_json_path(n_estimators, max_depth)
            if json_path:
                if self.verbose:
                    print(f"Loading classifier from JSON: {json_path}")
                cls = load_rf_from_json(json_path)
                train_accuracy, test_accuracy = self._evaluate_model(cls)
                training_time = 0.0
                model_source = 'json'
            elif self.classifier_json_required:
                raise FileNotFoundError(
                    f"No converted classifier found for {self.bench_name} "
                    f"(n_estimators={n_estimators}, max_depth={max_depth})"
                )
            elif self.verbose:
                print("No converted classifier found; training a new model.")

        if cls is None:
            start_time = time.time()
            params = {'n_trees': n_estimators, 'depth': max_depth}
            if self.algo == 'breiman':
                cls = RFBreiman(**params)
            else:
                cls = RFSklearn(**params)
            train_accuracy, test_accuracy = cls.train(self.data)
            training_time = time.time() - start_time

        verb = 0 if not self.verbose else 1
        explainer = self._create_explainer(cls, verb)

        rf_model = self._get_forest_model(cls)
        estimators = rf_model.estimators_
        total_nodes = int(sum(tree.tree_.node_count for tree in estimators))
        total_leaves = int(sum(tree.tree_.n_leaves for tree in estimators))
        avg_depth = float(np.mean([tree.tree_.max_depth for tree in estimators]))

        _, X_test, _, _ = self.data.train_test_split()
        X_test = self.data.transform(X_test)
        if test_index_list:
            num_samples = len(test_index_list)
        else:
            num_samples = len(X_test)
        explanation_results = self._generate_explanations(
            explainer,
            X_test,
            num_samples=num_samples,
            test_index_list=test_index_list
        )

        result = {
            'dataset_name': self.dataset_name,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_time': float(training_time),
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'avg_tree_depth': avg_depth,
            'series_length': int(self.data.series_length),
            'n_train_samples': int(len(self.data.X_train)),
            'n_test_samples': int(len(self.data.X_test)),
            'timestamp': datetime.now().isoformat(),
            'explanations': explanation_results,
            'model_source': model_source,
            'explainer': self.explainer_backend
        }
        if json_path:
            result['classifier_json_path'] = json_path

        if self.verbose:
            print("\nResults:")
            print(f"  Train Accuracy: {train_accuracy:.4f} ({100 * train_accuracy:.2f}%)")
            print(f"  Test Accuracy:  {test_accuracy:.4f} ({100 * test_accuracy:.2f}%)")
            print(f"  Training Time:  {training_time:.2f}s")
            print(f"  Total Nodes:    {total_nodes}")
            print(f"  Total Leaves:   {total_leaves}")
            print(f"  Avg Tree Depth: {avg_depth:.2f}")
            if explanation_results['avg_explanation_time']:
                print(f"  Avg Expl Time:  {explanation_results['avg_explanation_time']:.3f}s")
                print(f"  Avg Expl Len:   {explanation_results['avg_explanation_length']:.1f}")

        if model_source == 'json':
            result['model_path'] = json_path
        else:
            model_filename = os.path.join(
                self.bench_dir,
                f"{self.bench_name}_nest{n_estimators}_depth{max_depth}.mod.pkl"
            )
            cls.save_model(model_filename)
            result['model_path'] = model_filename

        return result

    def _generate_explanations(self, explainer, X_test, num_samples=5,
                               test_index_list=None, xtype='abd', etype='sat', smallest=False):
        num_samples = min(num_samples, len(X_test))

        explanations = []
        explanation_details = []
        interval_explanations = []
        infxp_coverages = []
        axp_domain_coverages = []
        explanation_times = []
        explanation_lengths = []

        if self.verbose:
            print(f"\n  Generating explanations for {num_samples} test samples...")

        indices = test_index_list if test_index_list is not None else list(range(num_samples))

        for i in indices:
            sample = X_test[i]
            try:
                expl_start = time.time()
                expl = self._explain_sample(
                    explainer,
                    sample,
                    xtype=xtype,
                    etype=etype,
                    smallest=smallest,
                    sample_index=i
                )
                expl_time = time.time() - expl_start

                explanations.append(expl)
                explanation_record = self._build_explanation_record(explainer, expl, i)
                explanation_details.append(explanation_record)
                interval_explanations.append(explanation_record.get('interval_explanation'))
                infxp_coverages.append(explanation_record.get('infxp_coverage'))
                axp_domain_coverages.append(explanation_record.get('axp_domain_coverage'))
                explanation_times.append(float(expl_time))
                explanation_lengths.append(len(expl) if expl else 0)

                if self.verbose:
                    print(f"    Sample {i}: explanation length={len(expl) if expl else 0}, time={expl_time:.3f}s")
            except Exception as e:
                if self.verbose:
                    print(f"    Sample {i}: Error generating explanation: {e}")
                infxp_coverages.append(None)
                axp_domain_coverages.append(None)
                explanation_times.append(None)
                explanation_lengths.append(None)

        valid_times = [t for t in explanation_times if t is not None]
        valid_lengths = [l for l in explanation_lengths if l is not None]
        valid_infxp_coverages = [c for c in infxp_coverages if c is not None]
        valid_axp_domain_coverages = [c for c in axp_domain_coverages if c is not None]

        result = {
            'num_samples_explained': len(indices),
            'list_explained_indices': indices,
            'num_successful': len(valid_times),
            'avg_explanation_time': float(np.mean(valid_times)) if valid_times else None,
            'min_explanation_time': float(np.min(valid_times)) if valid_times else None,
            'max_explanation_time': float(np.max(valid_times)) if valid_times else None,
            'avg_explanation_length': float(np.mean(valid_lengths)) if valid_lengths else None,
            'min_explanation_length': int(np.min(valid_lengths)) if valid_lengths else None,
            'max_explanation_length': int(np.max(valid_lengths)) if valid_lengths else None,
            'explanation_times': explanation_times,
            'explanation_lengths': explanation_lengths,
            'explanation_indices': explanations,
            'interval_explanations': interval_explanations,
            'infxp_coverages': infxp_coverages,
            'avg_infxp_coverage': float(np.mean(valid_infxp_coverages)) if valid_infxp_coverages else None,
            'min_infxp_coverage': float(np.min(valid_infxp_coverages)) if valid_infxp_coverages else None,
            'max_infxp_coverage': float(np.max(valid_infxp_coverages)) if valid_infxp_coverages else None,
            'axp_domain_coverages': axp_domain_coverages,
            'avg_axp_domain_coverage': float(np.mean(valid_axp_domain_coverages)) if valid_axp_domain_coverages else None,
            'min_axp_domain_coverage': float(np.min(valid_axp_domain_coverages)) if valid_axp_domain_coverages else None,
            'max_axp_domain_coverage': float(np.max(valid_axp_domain_coverages)) if valid_axp_domain_coverages else None,
            'explanation_details': explanation_details,
            'full_explanations': explanation_details
        }

        if self.verbose and valid_times:
            print("  Explanation statistics:")
            print(f"    Successful: {len(valid_times)}/{len(indices)}")
            print(f"    Avg time: {result['avg_explanation_time']:.3f}s")
            print(f"    Avg length: {result['avg_explanation_length']:.1f}")
            if result['avg_infxp_coverage'] is not None:
                print(f"    Avg INFXP coverage: {result['avg_infxp_coverage']:.2f}%")

        return result

    def run_grid_experiment(self, n_estimators_list, max_depth_list, test_index_list=None):
        print(f"\n{'#' * 60}")
        print("# Running Time-Series Grid Experiment")
        print(f"# Dataset: {self.bench_name}")
        print(f"# Explainer: {self.explainer_backend}")
        print(f"# n_estimators: {n_estimators_list}")
        print(f"# max_depth: {max_depth_list}")
        print(f"# Total experiments: {len(n_estimators_list) * len(max_depth_list)}")
        print(f"{'#' * 60}\n")

        for n_est in n_estimators_list:
            for depth in max_depth_list:
                try:
                    result = self.run_single_experiment(n_est, depth, test_index_list=test_index_list)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in experiment (n_est={n_est}, depth={depth}): {e}")
                    self.results.append({
                        'dataset_name': self.dataset_name,
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'explainer': self.explainer_backend,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        self.save_results()
        if self.use_redis:
            self.save_results_to_redis()
        self.print_summary()

    def save_results(self):
        results_file = os.path.join(self.bench_dir, f"{self.bench_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'bench_name': self.bench_name,
                'algorithm': self.algo,
                'explainer': self.explainer_backend,
                'feature_prefix': self.feature_prefix,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'sample_percentage': self.sample_percentage,
                'experiments': self.results
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    def save_results_to_redis(self):
        if not self.redis_conn:
            print("Warning: Redis connection not available. Skipping Redis save.")
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if self.explainer_backend == 'xrf':
                redis_key = f"timeseries_experiment:{self.bench_name}:{timestamp}"
                index_key = f"timeseries_experiment_index:{self.bench_name}"
            else:
                redis_key = f"timeseries_experiment:{self.explainer_backend}:{self.bench_name}:{timestamp}"
                index_key = f"timeseries_experiment_index:{self.explainer_backend}:{self.bench_name}"

            data = {
                'dataset_name': self.dataset_name,
                'bench_name': self.bench_name,
                'algorithm': self.algo,
                'explainer': self.explainer_backend,
                'feature_prefix': self.feature_prefix,
                'timestamp': timestamp,
                'experiments': self.results
            }
            self.redis_conn.set(redis_key, json.dumps(data))
            self.redis_conn.sadd(index_key, redis_key)
            print(f"Results saved to Redis: {redis_key}")
        except Exception as e:
            print(f"Error saving results to Redis: {e}")

    def print_summary(self):
        if not self.results:
            print("No results to summarize.")
            return

        print(f"\n{'=' * 70}")
        print("TIME-SERIES EXPERIMENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'n_est':<8} {'depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Time(s)':<10}")
        print(f"{'-' * 70}")

        for result in self.results:
            if 'error' in result:
                print(f"{result['n_estimators']:<8} {result['max_depth']:<8} ERROR: {result['error']}")
            else:
                print(
                    f"{result['n_estimators']:<8} {result['max_depth']:<8} "
                    f"{result['train_accuracy']:.4f}      {result['test_accuracy']:.4f}      "
                    f"{result.get('training_time', 0):.2f}"
                )

        valid_results = [r for r in self.results if 'error' not in r]
        if valid_results:
            best = max(valid_results, key=lambda x: x.get('test_accuracy', 0))
            print(f"\n{'=' * 70}")
            print("Best Configuration (by test accuracy):")
            print(f"  dataset: {self.dataset_name}")
            print(f"  n_estimators: {best['n_estimators']}")
            print(f"  max_depth: {best['max_depth']}")
            print(f"  Test Accuracy: {best['test_accuracy']:.4f} ({100 * best['test_accuracy']:.2f}%)")
            print(f"{'=' * 70}\n")


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_optional_int_list(value):
    if value is None or value.strip() == '':
        return None
    return parse_int_list(value)


def expand_explainers(explainer):
    if explainer == 'all':
        return list(SUPPORTED_EXPLAINERS)
    return [explainer]


def list_available_datasets():
    print("Available aeon univariate datasets:")
    for name in AVAILABLE_DATASETS:
        print(f"  - {name}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Run baseline RF explanation experiments on aeon univariate time-series datasets.'
    )
    parser.add_argument('dataset_name', nargs='?', help='Aeon dataset name, e.g. ECG200 or Coffee')
    parser.add_argument('--list-datasets', action='store_true', help='List supported aeon dataset names and exit')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--feature-prefix', default=DEFAULT_FEATURE_PREFIX, help='Feature prefix for flattened time points')
    parser.add_argument('--algo', choices=['sklearn', 'breiman'], default='sklearn', help='RF algorithm')
    parser.add_argument('--n-estimators', default='100', help='Comma-separated list of n_estimators values')
    parser.add_argument('--max-depth', default='6', help='Comma-separated list of max_depth values')
    parser.add_argument('--test-index-list', default=None, help='Comma-separated test indices to explain')
    parser.add_argument('--test-size', type=float, default=None, help='Optional custom test split after combining train+test')
    parser.add_argument('--sample-percentage', type=float, default=100.0, help='Percentage of train and test samples to retain')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--explainer', choices=SUPPORTED_EXPLAINERS + ('all',), default='xrf', help='Explanation backend')
    parser.add_argument('--classifier-json-dir', action='append', default=None,
                        help='Directory containing converted classifier JSON files; can be specified multiple times')
    parser.add_argument('--classifier-json-required', action='store_true',
                        help='Fail if converted classifier JSON is not found')
    parser.add_argument('--redis', action='store_true', help='Save results to Redis')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis DB')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        list_available_datasets()
        return 0

    if not args.dataset_name:
        parser.error('dataset_name is required unless --list-datasets is used')

    n_estimators_list = parse_int_list(args.n_estimators)
    max_depth_list = parse_int_list(args.max_depth)
    test_index_list = parse_optional_int_list(args.test_index_list)
    verbose = not args.quiet

    for explainer_backend in expand_explainers(args.explainer):
        runner = TimeSeriesExperimentRunner(
            dataset_name=args.dataset_name,
            output_dir=args.output,
            feature_prefix=args.feature_prefix,
            algo=args.algo,
            verbose=verbose,
            use_redis=args.redis,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            classifier_json_dirs=args.classifier_json_dir,
            classifier_json_required=args.classifier_json_required,
            explainer_backend=explainer_backend,
            test_size=args.test_size,
            random_state=args.random_state,
            sample_percentage=args.sample_percentage
        )
        runner.run_grid_experiment(
            n_estimators_list=n_estimators_list,
            max_depth_list=max_depth_list,
            test_index_list=test_index_list
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
