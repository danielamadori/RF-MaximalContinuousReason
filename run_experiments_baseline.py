#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## run_experiments.py
##
##  Experiment runner for Random Forest complexity analysis
##

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

from rf_utils import get_rf_search_space, optimize_rf_hyperparameters
from load_rf_from_json import load_rf_from_json

try:
    import redis
    from redis_helpers.connection import connect_redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Install redis package to enable Redis support.")

from baseline.xrf import XRF, Dataset
from baseline.xrf import RFBreiman, RFSklearn


class ExperimentRunner:
    """
    Run experiments with different RF complexities on datasets.
    """
    
    def __init__(self, dataset_path=None, output_dir='experiments', 
                 separator=',', use_categorical=False, algo='sklearn', verbose=True,
                 use_redis=False, redis_host='localhost', redis_port=6379, redis_db=0,
                 classifier_json_dirs=None, classifier_json_required=False):
        """
        Initialize experiment runner.
        
        Args:
            dataset_path: Path to dataset file
            output_dir: Directory to save results
            separator: CSV separator
            use_categorical: Whether to use categorical encoding
            algo: 'sklearn' or 'breiman'
            verbose: Print detailed output
            use_redis: Whether to save results to Redis
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            classifier_json_dirs: Directories to search for converted classifier JSON files
            classifier_json_required: Fail if a converted classifier is not found
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.separator = separator
        self.use_categorical = use_categorical
        self.algo = algo
        self.verbose = verbose
        self.use_redis = use_redis and REDIS_AVAILABLE
        if classifier_json_dirs is None:
            classifier_json_dirs = [
                os.path.join('baseline', 'Classifiers-100-converted'),
                os.path.join('baseline', 'Classifiers-converted')
            ]
        self.classifier_json_dirs = self._normalize_classifier_dirs(classifier_json_dirs)
        self.classifier_json_required = classifier_json_required
        
        # Initialize Redis connection if requested
        self.redis_conn = None
        if self.use_redis:
            try:
                self.redis_conn = redis.Redis(host=redis_host, port=redis_port, 
                                             db=redis_db, decode_responses=True)
                self.redis_conn.ping()
                print(f"Connected to Redis at {redis_host}:{redis_port} (DB {redis_db})")
            except Exception as e:
                print(f"Warning: Failed to connect to Redis: {e}")
                self.use_redis = False
                self.redis_conn = None
        
        # Load dataset
        self.data = self._load_dataset(dataset_path, separator, use_categorical)
        
        # Create output directory
        self.bench_name = os.path.basename(dataset_path)
        if self.bench_name.endswith('.csv'):
            self.bench_name = os.path.splitext(self.bench_name)[0]
        
        self.bench_dir = os.path.join(output_dir, self.bench_name)
        os.makedirs(self.bench_dir, exist_ok=True)

        self.sample_data = None
        self.sample_path = None
        if self.data is not None:
            self.sample_data, self.sample_path = self._load_samples_file(dataset_path, separator)
            if self.sample_data is not None:
                expected_features = len(self.data.features)
                if self.sample_data.shape[1] == expected_features + 1:
                    if self.verbose:
                        print("Sample file includes labels; dropping last column.")
                    self.sample_data = self.sample_data[:, :-1]
                elif self.sample_data.shape[1] != expected_features:
                    raise ValueError(
                        f"Sample file feature count mismatch: expected {expected_features}, "
                        f"found {self.sample_data.shape[1]} in {self.sample_path}"
                    )
                if self.verbose:
                    print(f"Loaded {len(self.sample_data)} samples from {self.sample_path}")
        
        # Results storage
        self.results = []

    @staticmethod
    def _normalize_classifier_dirs(classifier_json_dirs):
        if classifier_json_dirs is None:
            return []
        if isinstance(classifier_json_dirs, str):
            return [classifier_json_dirs]
        return list(classifier_json_dirs)

    def _find_classifier_json_path(self, n_estimators, max_depth):
        if not self.classifier_json_dirs:
            return None

        filename = f"{self.bench_name}_nbestim_{n_estimators}_maxdepth_{max_depth}.mod.json"
        for base_dir in self.classifier_json_dirs:
            json_path = os.path.join(base_dir, self.bench_name, filename)
            if os.path.exists(json_path):
                return json_path

        return None

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

    def _load_samples_file(self, dataset_path, separator):
        if not dataset_path:
            return None, None

        dataset_dir = os.path.dirname(dataset_path)
        stem = os.path.splitext(os.path.basename(dataset_path))[0]
        candidates = [
            os.path.join(dataset_dir, f"{stem}.samples"),
            os.path.join(dataset_dir, f"{stem}.sample")
        ]

        for path in candidates:
            if not os.path.exists(path):
                continue
            if os.path.getsize(path) == 0:
                if self.verbose:
                    print(f"Warning: sample file is empty: {path}")
                return None, path
            try:
                samples = np.loadtxt(path, delimiter=separator)
            except Exception as e:
                raise ValueError(f"Failed to load samples from {path}: {e}")
            samples = np.atleast_2d(samples)
            return samples, path

        return None, None

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

    def run_json_experiments(self, json_files, test_index_list=None):
        if not json_files:
            print("No JSON classifiers provided.")
            return

        print(f"\n{'#'*60}")
        print(f"# Running JSON Experiments")
        print(f"# Dataset: {self.bench_name}")
        print(f"# Total classifiers: {len(json_files)}")
        print(f"{'#'*60}\n")

        for json_path in json_files:
            json_path = str(json_path)
            n_estimators, max_depth = self._parse_json_filename(json_path)
            if n_estimators is None or max_depth is None:
                print(f"Warning: could not parse parameters from {json_path}")
                continue
            try:
                result = self.run_single_experiment(
                    n_estimators,
                    max_depth,
                    test_index_list=test_index_list,
                    classifier_json_path=json_path
                )
                self.results.append(result)
            except Exception as e:
                print(f"Error in experiment ({json_path}): {e}")
                self.results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'classifier_json_path': json_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        self.save_results()
        if self.use_redis:
            self.save_results_to_redis()
        self.print_summary()
    
    def _load_dataset(self, dataset_path, separator, use_categorical):
        """
        Load dataset from Redis cache or file with validation.
        
        Args:
            dataset_path: Path to dataset file
            separator: CSV separator
            use_categorical: Whether to use categorical encoding
            
        Returns:
            Dataset object
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset path is empty or invalid
        """
        # Validate dataset path
        if not dataset_path or not dataset_path.strip():
            raise ValueError("Dataset path cannot be empty")
        
        dataset_path = dataset_path.strip()
        
        # Try to load from Redis first if available
        if self.redis_conn:
            redis_data_key = f"dataset_data:{os.path.basename(dataset_path)}"
            try:
                cached_data = self.redis_conn.get(redis_data_key)
                if cached_data:
                    print(f"Loading complete dataset from Redis cache: {redis_data_key}")
                    # Deserialize the pickled Dataset object
                    data = pickle.loads(cached_data.encode('latin1'))
                    print(f"Successfully loaded dataset from Redis")
                    return data
            except Exception as e:
                print(f"Note: Could not load from Redis cache, will load from file: {e}")
        
        # Validate file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        if not os.path.isfile(dataset_path):
            raise ValueError(f"Dataset path is not a file: {dataset_path}")
        
        # Check if file is readable and not empty
        try:
            file_size = os.path.getsize(dataset_path)
            if file_size == 0:
                raise ValueError(f"Dataset file is empty: {dataset_path}")
        except OSError as e:
            raise ValueError(f"Cannot access dataset file: {dataset_path} - {e}")
        
        # Load dataset from file
        print(f"Loading dataset from file: {dataset_path}")
        try:
            data = Dataset(filename=dataset_path, separator=separator,
                         use_categorical=use_categorical)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
        
        # Cache complete dataset to Redis if available
        if self.redis_conn:
            redis_data_key = f"dataset_data:{os.path.basename(dataset_path)}"
            redis_meta_key = f"dataset_meta:{os.path.basename(dataset_path)}"
            try:
                # Pickle the entire Dataset object and save to Redis
                pickled_data = pickle.dumps(data)
                # Convert bytes to latin1 string for Redis text mode
                self.redis_conn.set(redis_data_key, pickled_data.decode('latin1'))
                
                # Also save metadata for quick reference
                dataset_metadata = {
                    'path': dataset_path,
                    'separator': separator,
                    'use_categorical': use_categorical,
                    'n_samples': len(data.data),
                    'n_features': len(data.features),
                    'cached_at': datetime.now().isoformat()
                }
                self.redis_conn.set(redis_meta_key, json.dumps(dataset_metadata))
                print(f"Cached complete dataset to Redis: {redis_data_key}")
            except Exception as e:
                print(f"Warning: Could not cache dataset to Redis: {e}")
        
        return data
        
    def run_single_experiment(self, n_estimators, max_depth, test_index_list=None, classifier_json_path=None):
        """
        Run a single experiment with given RF parameters.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            classifier_json_path: Optional path to a converted classifier JSON
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"{'='*60}")
        
        json_path = classifier_json_path
        model_source = 'trained'
        cls = None

        if json_path:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

        if json_path or self.classifier_json_dirs:
            if self.data is None:
                raise ValueError("Converted classifier loading requires a tabular dataset.")
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

            # Initialize classifier
            params = {'n_trees': n_estimators, 'depth': max_depth}

            if self.algo == 'breiman':
                cls = RFBreiman(**params)
            else:
                cls = RFSklearn(**params)

            # Train model
            train_accuracy, test_accuracy = cls.train(self.data)

            training_time = time.time() - start_time
        
        # Create XRF explainer
        verb = 0 if not self.verbose else 1
        xrf = XRF(cls, self.data.m_features_, self.data.targets, verb=verb)
        
        # Get model complexity metrics
        rf_model = self._get_forest_model(cls)
        estimators = rf_model.estimators_
        total_nodes = rf_model.tree_.node_count if hasattr(rf_model, 'tree_') else int(sum(tree.tree_.node_count for tree in estimators))
        total_leaves = int(sum(tree.tree_.n_leaves for tree in estimators))
        avg_depth = float(np.mean([tree.tree_.max_depth for tree in estimators]))
        
        # Generate explanations for test samples
        if self.sample_data is not None and len(self.sample_data) > 0:
            X_samples = self.data.transform(np.asarray(self.sample_data, dtype=np.float32))
            if test_index_list:
                num_samples = len(test_index_list)
            else:
                num_samples = len(X_samples)
            explanation_results = self._generate_explanations(
                xrf,
                X_samples,
                num_samples=num_samples,
                test_index_list=test_index_list
            )
        else:
            _, X_test, _, y_test = self.data.train_test_split()
            X_test = self.data.transform(X_test)
            explanation_results = self._generate_explanations(xrf, X_test, test_index_list=test_index_list)
        
        result = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_time': float(training_time),
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'avg_tree_depth': avg_depth,
            'timestamp': datetime.now().isoformat(),
            'explanations': explanation_results,
            'model_source': model_source
        }
        if json_path:
            result['classifier_json_path'] = json_path
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Train Accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")
            print(f"  Test Accuracy:  {test_accuracy:.4f} ({100*test_accuracy:.2f}%)")
            print(f"  Training Time:  {training_time:.2f}s")
            print(f"  Total Nodes:    {total_nodes}")
            print(f"  Total Leaves:   {total_leaves}")
            print(f"  Avg Tree Depth: {avg_depth:.2f}")
            if explanation_results['avg_explanation_time']:
                print(f"  Avg Expl Time:  {explanation_results['avg_explanation_time']:.3f}s")
                print(f"  Avg Expl Len:   {explanation_results['avg_explanation_length']:.1f}")
        
        # Save model
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
    
    def _generate_explanations(self, xrf, X_test, num_samples=5, test_index_list = None, xtype='abd', etype='sat', smallest=False):
        """
        Generate explanations for a subset of test samples.
        
        Args:
            xrf: XRF explainer instance
            num_samples: Number of test samples to explain
            xtype: Type of explanation ('abd' or 'con')
            etype: Encoding type ('sat' or 'maxsat')
            smallest: Whether to compute smallest explanations
            
        Returns:
            Dictionary with explanation statistics
        """
        
        # Limit number of samples to explain
        num_samples = min(num_samples, len(X_test))
        
        explanations = []
        explanation_times = []
        explanation_lengths = []
        
        if self.verbose:
            print(f"\n  Generating explanations for {num_samples} test samples...")
        if test_index_list:
            for i in test_index_list:
                sample = X_test[i]
                
                try:
                    expl_start = time.time()
                    expl = xrf.explain(sample, xtype=xtype, etype=etype, smallest=smallest, sample_index=i)
                    expl_time = time.time() - expl_start
                    
                    explanations.append(expl)
                    explanation_times.append(float(expl_time))
                    explanation_lengths.append(len(expl) if expl else 0)
                    
                    if self.verbose:
                        print(f"    Sample {i+1}: explanation length={len(expl) if expl else 0}, time={expl_time:.3f}s")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    Sample {i+1}: Error generating explanation: {e}")
                    explanation_times.append(None)
                    explanation_lengths.append(None)
        else:
            for i in range(num_samples):
                sample = X_test[i]
                
                try:
                    expl_start = time.time()
                    expl = xrf.explain(sample, xtype=xtype, etype=etype, smallest=smallest, sample_index=i)
                    expl_time = time.time() - expl_start
                    
                    explanations.append(expl)
                    explanation_times.append(float(expl_time))
                    explanation_lengths.append(len(expl) if expl else 0)
                    
                    if self.verbose:
                        print(f"    Sample {i+1}: explanation length={len(expl) if expl else 0}, time={expl_time:.3f}s")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    Sample {i+1}: Error generating explanation: {e}")
                    explanation_times.append(None)
                    explanation_lengths.append(None)
        
        # Calculate statistics
        valid_times = [t for t in explanation_times if t is not None]
        valid_lengths = [l for l in explanation_lengths if l is not None]
        
        result = {
            'num_samples_explained': num_samples,
            'list_explained_indices': test_index_list if test_index_list is not None else list(range(num_samples)),
            'num_successful': len(valid_times),
            'avg_explanation_time': float(np.mean(valid_times)) if valid_times else None,
            'min_explanation_time': float(np.min(valid_times)) if valid_times else None,
            'max_explanation_time': float(np.max(valid_times)) if valid_times else None,
            'avg_explanation_length': float(np.mean(valid_lengths)) if valid_lengths else None,
            'min_explanation_length': int(np.min(valid_lengths)) if valid_lengths else None,
            'max_explanation_length': int(np.max(valid_lengths)) if valid_lengths else None,
            'explanation_times': explanation_times,
            'explanation_lengths': explanation_lengths,
            'full_explanations': explanations
        }
        
        if self.verbose and valid_times:
            print(f"  Explanation statistics:")
            print(f"    Successful: {len(valid_times)}/{num_samples}")
            print(f"    Avg time: {result['avg_explanation_time']:.3f}s")
            print(f"    Avg length: {result['avg_explanation_length']:.1f}")
        
        return result
    
    def run_grid_experiment(self, n_estimators_list, max_depth_list, test_index_list=None):
        """
        Run experiments over a grid of hyperparameters.
        
        Args:
            n_estimators_list: List of n_estimators values to try
            max_depth_list: List of max_depth values to try
        """
        print(f"\n{'#'*60}")
        print(f"# Running Grid Experiment")
        print(f"# Dataset: {self.bench_name}")
        print(f"# n_estimators: {n_estimators_list}")
        print(f"# max_depth: {max_depth_list}")
        print(f"# Total experiments: {len(n_estimators_list) * len(max_depth_list)}")
        print(f"{'#'*60}\n")
        
        for n_est in n_estimators_list:
            for depth in max_depth_list:
                try:
                    result = self.run_single_experiment(n_est, depth, test_index_list=test_index_list)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in experiment (n_est={n_est}, depth={depth}): {e}")
                    self.results.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        self.save_results()
        if self.use_redis:
            self.save_results_to_redis()
        self.print_summary()
    
    def save_results(self):
        """Save experiment results to JSON file."""
        results_file = os.path.join(self.bench_dir, f"{self.bench_name}_results.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': self.dataset_path,
                'bench_name': self.bench_name,
                'algorithm': self.algo,
                'use_categorical': self.use_categorical,
                'experiments': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def save_results_to_redis(self):
        """Save experiment results to Redis."""
        if not self.redis_conn:
            print("Warning: Redis connection not available. Skipping Redis save.")
            return
        
        try:
            # Create a Redis key based on dataset name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            redis_key = f"experiment:{self.bench_name}:{timestamp}"
            
            # Store the full experiment data
            data = {
                'dataset': self.dataset_path,
                'bench_name': self.bench_name,
                'algorithm': self.algo,
                'use_categorical': self.use_categorical,
                'timestamp': timestamp,
                'experiments': self.results
            }
            
            self.redis_conn.set(redis_key, json.dumps(data))
            print(f"Results saved to Redis: {redis_key}")
            
            # Also add to an index of all experiments for this dataset
            index_key = f"experiment_index:{self.bench_name}"
            self.redis_conn.sadd(index_key, redis_key)
            
            # Store individual experiment results with separate keys for easy querying
            for idx, result in enumerate(self.results):
                if 'error' not in result:
                    result_key = f"experiment:{self.bench_name}:{timestamp}:result:{idx}"
                    self.redis_conn.hset(result_key, mapping=result)
                    
            print(f"Saved {len(self.results)} individual results to Redis")
            
        except Exception as e:
            print(f"Error saving results to Redis: {e}")
    
    def print_summary(self):
        """Print summary of all experiments."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        # Check if we have accuracy data
        has_accuracy = any('train_accuracy' in r for r in self.results if 'error' not in r)
        
        if has_accuracy:
            print(f"{'n_est':<8} {'depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Time(s)':<10}")
        else:
            print(f"{'n_est':<8} {'depth':<8} {'Nodes':<12} {'Leaves':<12} {'Avg Depth':<12}")
        print(f"{'-'*60}")
        
        for result in self.results:
            if 'error' in result:
                print(f"{result['n_estimators']:<8} {result['max_depth']:<8} ERROR: {result['error']}")
            elif has_accuracy:
                print(f"{result['n_estimators']:<8} {result['max_depth']:<8} "
                      f"{result['train_accuracy']:.4f}      {result['test_accuracy']:.4f}      "
                      f"{result.get('training_time', 0):.2f}")
            else:
                # Experiments without accuracy metrics
                print(f"{result['n_estimators']:<8} {result['max_depth']:<8} "
                      f"{result.get('total_nodes', 'N/A'):<12} {result.get('total_leaves', 'N/A'):<12} "
                      f"{result.get('avg_tree_depth', 0):.2f}")
        
        # Find best configuration
        valid_results = [r for r in self.results if 'error' not in r]
        if valid_results and has_accuracy:
            best = max(valid_results, key=lambda x: x.get('test_accuracy', 0))
            print(f"\n{'='*60}")
            print(f"Best Configuration (by test accuracy):")
            print(f"  n_estimators: {best['n_estimators']}")
            print(f"  max_depth: {best['max_depth']}")
            print(f"  Test Accuracy: {best['test_accuracy']:.4f} ({100*best['test_accuracy']:.2f}%)")
            print(f"{'='*60}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description='Run Random Forest experiments with different complexities'
#     )
    
#     # Dataset arguments
#     parser.add_argument('dataset', type=str, 
#                        help='Path to dataset CSV file')
#     parser.add_argument('-s', '--separator', type=str, default=',',
#                        help='CSV separator (default: comma)')
#     parser.add_argument('-c', '--categorical', action='store_true',
#                        help='Use categorical encoding')
    
#     # RF parameters
#     parser.add_argument('-n', '--n-estimators', type=str, default='10,50,100',
#                        help='Comma-separated list of n_estimators values (default: 10,50,100)')
#     parser.add_argument('-d', '--max-depth', type=str, default='3,5,10',
#                        help='Comma-separated list of max_depth values (default: 3,5,10)')
#     parser.add_argument('-a', '--algo', type=str, default='sklearn', choices=['sklearn', 'breiman'],
#                        help='RF algorithm (default: sklearn)')
    
#     # Output arguments
#     parser.add_argument('-o', '--output', type=str, default='experiments',
#                        help='Output directory (default: experiments)')
#     parser.add_argument('-v', '--verbose', action='store_true',
#                        help='Verbose output')
    
#     # Redis arguments
#     parser.add_argument('--redis', action='store_true',
#                        help='Save results to Redis')
#     parser.add_argument('--redis-host', type=str, default='localhost',
#                        help='Redis host (default: localhost)')
#     parser.add_argument('--redis-port', type=int, default=6379,
#                        help='Redis port (default: 6379)')
#     parser.add_argument('--redis-db', type=int, default=0,
#                        help='Redis database number (default: 0)')
    
#     args = parser.parse_args()
    
#     # Parse parameter lists
#     n_estimators_list = [int(x.strip()) for x in args.n_estimators.split(',')]
#     max_depth_list = [int(x.strip()) for x in args.max_depth.split(',')]
    
#     # Run experiments
#     runner = ExperimentRunner(
#         dataset_path=args.dataset,
#         output_dir=args.output,
#         separator=args.separator,
#         use_categorical=args.categorical,
#         algo=args.algo,
#         verbose=args.verbose,
#         use_redis=args.redis,
#         redis_host=args.redis_host,
#         redis_port=args.redis_port,
#         redis_db=args.redis_db
#     )
    
#     runner.run_grid_experiment(n_estimators_list, max_depth_list)

def main():
    # Run experiments on a tabular baseline dataset
    name_dataset = 'iris'
    dataset = f'baseline/resources/datasets/{name_dataset}/{name_dataset}.csv'
    output = 'baseline/resources/experiments'
    separator = ','
    categorical = False
    algo = 'sklearn'
    verbose = True
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0
    n_estimators_list = [100]
    max_depth_list = [6]
    test_index_list = [0, 1, 2, 3, 4]
    runner = ExperimentRunner(
        dataset_path=dataset,
        output_dir=output,
        separator=separator,
        use_categorical=categorical,
        algo=algo,
        verbose=verbose,
        use_redis=False,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
    )
    runner.run_grid_experiment(n_estimators_list, max_depth_list, test_index_list=test_index_list)

def main_from_converted():
    converted_dir = os.path.join('baseline', 'Classifiers-100-converted')
    datasets_root = os.path.join('baseline', 'resources', 'datasets')
    output = 'baseline/resources/experiments'
    separator = ','
    categorical = False
    algo = 'sklearn'
    verbose = True
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0

    if not os.path.isdir(converted_dir):
        print(f"Converted classifiers directory not found: {converted_dir}")
        return

    dataset_dirs = [
        name for name in os.listdir(converted_dir)
        if os.path.isdir(os.path.join(converted_dir, name))
    ]

    for dataset_name in sorted(dataset_dirs):
        dataset_path = os.path.join(datasets_root, dataset_name, f"{dataset_name}.csv")
        if not os.path.exists(dataset_path):
            print(f"Warning: dataset CSV not found for {dataset_name}: {dataset_path}")
            continue

        json_dir = os.path.join(converted_dir, dataset_name)
        json_files = [
            os.path.join(json_dir, fname)
            for fname in os.listdir(json_dir)
            if fname.endswith('.json')
        ]
        if not json_files:
            print(f"Warning: no JSON classifiers found in {json_dir}")
            continue

        runner = ExperimentRunner(
            dataset_path=dataset_path,
            output_dir=output,
            separator=separator,
            use_categorical=categorical,
            algo=algo,
            verbose=verbose,
            use_redis=False,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            classifier_json_dirs=[converted_dir],
            classifier_json_required=False
        )
        runner.run_json_experiments(sorted(json_files))
if __name__ == '__main__':
    if '--from-converted' in sys.argv:
        sys.argv.remove('--from-converted')
        main_from_converted()
    else:
        main()
