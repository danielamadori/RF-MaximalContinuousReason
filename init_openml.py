#!/usr/bin/env python3
"""
Initialize Redis with an OpenML Classification Dataset

This script:
1.  Connects to Redis (clearing DBs by default)
2.  Downloads a dataset from OpenML by NAME
3.  Preprocesses data (Imputation, Encoding)
4.  Trains a Random Forest (with Bayesian Optimization)
5.  Stores everything in Redis (Dataset, Forest, Endpoints, Initial Candidate)

Usage:
    python init_openml.py [DATASET_NAME] [OPTIONS]
    python init_openml.py --list-datasets
    python init_openml.py credit-g --class-label "good" --test-sample-index "0, 5-8, 20"
"""

import sys
import argparse
import pandas as pd
import numpy as np
import redis
import json
import datetime
import openml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Shared utilities
from helpers import convert_numpy_types, parse_sample_indices
from redis_helpers.connection import connect_redis
from rf_utils import (
    create_forest_params, 
    get_rf_search_space, 
    optimize_rf_hyperparameters, 
    train_and_convert_forest
)
from init_utils import (
    store_training_set,
    store_dataset_total_samples,
    store_forest_and_endpoints,
    process_all_classified_samples,
    initialize_seed_candidate,
)
from helpers import convert_numpy_types


def list_available_datasets():
    """List popular active classification datasets on OpenML"""
    print("\nFetching popular OpenML datasets (Active, Classification)...")
    try:
        # Get active classification datasets with >1000 runs to ensure quality
        # Simplify query to debug
        datalist = openml.datasets.list_datasets(
            output_format="dataframe",
            status="active"
        )
        
        if datalist.empty:
            print("No datasets found matching criteria.")
            return

        # Sort by number of runs/likes if possible, otherwise just by ID
        # Default API doesn't return 'runs' easily in DF, sorting by ID
        print(f"\nFound {len(datalist)} datasets. Showing top 20 by size:")
        
        view = datalist[['did', 'name', 'version', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']]
        view = view.sort_values(by='NumberOfInstances', ascending=False).head(20)
        
        print(view.to_string(index=False))
        print("\nUse the NAME to initialize, e.g.: python init_openml.py iris")
        
    except Exception as e:
        print(f"[ERROR] Failed to list datasets: {e}")


def get_dataset_by_name(name):
    """
    Find dataset ID by name, preferring the latest version.
    Returns (dataset_id, dataset_name) or None
    """
    try:
        # Search for exact name match
        datalist = openml.datasets.list_datasets(
            output_format="dataframe",
            status="active"
        )
        
        # Filter locally case-insensitive
        matches = datalist[datalist['name'].str.lower() == name.lower()]
        
        if matches.empty:
            print(f"[ERROR] No active dataset found with name '{name}'")
            sys.exit(1)
        
        # Sort by version descending to get latest
        latest = matches.sort_values('version', ascending=False).iloc[0]
        
        return int(latest['did']), latest['name']
        
    except Exception as e:
        print(f"[ERROR] Failed to find dataset: {e}")
        sys.exit(1)


def load_and_prepare_dataset(dataset_name):
    """
    Load dataset from OpenML and prepare X, y.
    Handles missing values and categorical features.
    """
    print(f"\n[START] Loading dataset '{dataset_name}' from OpenML...")
    
    # Resolve name to ID
    result = get_dataset_by_name(dataset_name)
    if not result:
        return None, None, None
    
    dataset_id, actual_name = result
    print(f"[INFO] Resolved '{dataset_name}' to ID {dataset_id} (Version: latest)")

    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        
        # Get data as dataframe
        # target_name might be None if dataset doesn't explicitly state it
        target = dataset.default_target_attribute
        if not target:
            print("[ERROR] Dataset has no default target attribute.")
            return None, None, None

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )
        
        print(f"[INFO] Loaded shape: {X.shape}")
        
        # Handle Missing Values
        if X.isnull().values.any():
            print("[INFO] Missing values detected. Imputing...")
            # Split numerical and categorical for imputation
            num_cols = X.select_dtypes(include=[np.number]).columns
            cat_cols = X.select_dtypes(exclude=[np.number]).columns
            
            if len(num_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                X[num_cols] = imputer_num.fit_transform(X[num_cols])
                
            if len(cat_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

        # Handle Categorical Features (Label Encoding)
        # Random Forest in sklearn needs numerical input
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            print(f"[INFO] Encoding {len(cat_cols)} categorical features...")
            le_dict = {}
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        return X, y, actual_name

    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Initialize Redis with OpenML Dataset')
    
    # Dataset selection
    parser.add_argument('dataset_name', nargs='?', help='Name of the OpenML dataset (e.g. iris)')
    parser.add_argument('--list-datasets', action='store_true', help='List available popular datasets')
    
    # Core arguments
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--class-label', type=str, default=None, help='Target class label (optional, interactive if missing)')
    parser.add_argument('--sample-pct', type=float, default=100.0, help='Percentage of samples to keep (0-100)')
    parser.add_argument('--test-split', type=float, default=0.3, help='Test set split ratio (default: 0.3)')
    parser.add_argument('--test-sample-index', type=str, default=None, help='Index (or comma-separated indices) of samples to use for testing (Leave-One-Out). All others used for training.')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--no-clean', action='store_true', help='Do not clean Redis databases before starting')

    # Forest arguments
    parser.add_argument('--n-estimators', type=int, default=10, help='Number of trees')
    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy'], help='Splitting criterion')
    parser.add_argument('--max-depth', type=int, default=None, help='Max depth of trees')
    parser.add_argument('--min-samples-split', type=int, default=2, help='Min samples split')
    parser.add_argument('--min-samples-leaf', type=int, default=1, help='Min samples leaf')
    parser.add_argument('--max-leaf-nodes', type=int, default=None, help='Max leaf nodes')
    parser.add_argument('--max-features', type=str, default='sqrt', help='Max features (int, float, "sqrt", "log2")')
    parser.add_argument('--min-impurity-decrease', type=float, default=0.0, help='Min impurity decrease')
    parser.add_argument('--bootstrap', type=str, default='True', help='Bootstrap samples (True/False)')
    parser.add_argument('--max-samples', type=float, default=None, help='Max samples for bootstrap')
    parser.add_argument('--ccp-alpha', type=float, default=0.0, help='Complexity parameter used for Minimal Cost-Complexity Pruning')
    
    # Optimization arguments
    parser.add_argument('--optimize', action='store_true', help='Run Bayesian Optimization for RF')
    parser.add_argument('--n-calls', type=int, default=20, help='Number of optimization steps')
    
    args = parser.parse_args()

    # List datasets mode
    if args.list_datasets:
        list_available_datasets()
        return

    if not args.dataset_name:
        print("[ERROR] Please specify a dataset name or use --list-datasets")
        return

    # 1. Connect to Redis
    print("\n[START] Initializing Random Path Worker System (OpenML)")
    connections, db_mapping = connect_redis(port=args.redis_port)
    if not connections:
        return

    if not args.no_clean:
        from redis_helpers.utils import clean_all_databases
        clean_all_databases(connections, db_mapping)
        print("[INFO] Redis databases cleaned")

    # 2. Load Dataset
    X_df, y, dataset_actual_name = load_and_prepare_dataset(args.dataset_name)
    if X_df is None:
        return

    feature_names = list(X_df.columns)
    
    # Convert to numpy for processing (explicit to avoid pyarrow-backed arrays in pandas 3+)
    X = np.asarray(X_df, dtype=float)
    y_raw = np.asarray(y) if not hasattr(y, 'to_numpy') else y.to_numpy()
    y = y_raw.astype(str)
    
    print(f"\n[INFO] Dataset: {dataset_actual_name}")
    print(f"[INFO] Features: {len(feature_names)}")
    print(f"[INFO] Samples: {len(X)}")

    # 3. Handle Target Label
    classes = np.unique(y)
    print(f"[INFO] Classes detected: {classes}")

    target_label = args.class_label
    if target_label is None:
        if len(classes) == 2:
            # Check if classes are numeric-like or string
            # For OpenML, often '1', '2' or 'class1', 'class2'
            # Default to the second one (usually positive) or first
            target_label = str(classes[0]) # Default first
            print(f"[INFO] No class label specified. Defaulting to first class: '{target_label}'")
        else:
            print(f"\nAvailable classes: {classes}")
            print("[ERROR] Please specify --class-label")
            return
    
    # Verify label exists
    if target_label not in classes:
        print(f"[ERROR] Class label '{target_label}' not found in dataset. Available: {classes}")
        return

    print(f"[INFO] Target Class Label: {target_label}")

    # 4. Sampling & Split
    # 4. Sampling & Split
    if args.sample_pct < 100.0 and args.test_sample_index is None:
        n_keep = int(len(X) * (args.sample_pct / 100.0))
        indices = np.random.choice(len(X), n_keep, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"[INFO] Sampled {n_keep} instances ({args.sample_pct}%)")
    elif args.sample_pct < 100.0 and args.test_sample_index is not None:
        print(f"[WARNING] --sample-pct ignored because --test-sample-index is set. Using full dataset to preserve indexing.")

    
    if args.test_sample_index is not None:
        try:
            indices = parse_sample_indices(args.test_sample_index)
        except ValueError as e:
            print(f"[ERROR] Invalid format for --test-sample-index: {e}")
            return
            
        if any(i < 0 or i >= len(X) for i in indices):
            print(f"[ERROR] One or more indices out of bounds (0-{len(X)-1})")
            return
            
        print(f"Applying strict Split: Test sample indices {indices}")
        
        # Select test samples
        X_test = X[indices]
        y_test = y[indices]
        
        # Use all others for training
        X_train = np.delete(X, indices, axis=0)
        y_train = np.delete(y, indices, axis=0)
        
        print(f"LOO Training set: {X_train.shape[0]} samples")
        print(f"LOO Test set: {X_test.shape[0]} samples (Indices: {indices})")
        
        # Verify class membership
        mismatches = [i for i, label in enumerate(y_test) if label != target_label]
        if mismatches:
            print(f"[WARNING] The following selected samples do NOT belong to target class '{target_label}':")
            for i in mismatches:
                print(f"  - Index {indices[i]}: Class '{y_test[i]}'")
            
            if len(mismatches) == len(indices):
                print(f"[ERROR] No selected samples belong to the target class! Aborting.")
                sys.exit(1)
        
        print(f"LOO Training set: {X_train.shape[0]} samples")
        print(f"LOO Test set: {X_test.shape[0]} samples (Indices: {indices})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split, random_state=args.random_state, stratify=y
        )
    
    store_dataset_total_samples(connections, len(X_test))

    # Store Training Set Metadata
    store_training_set(connections, X_train, y_train, feature_names, dataset_actual_name, dataset_type='openml')

    # 5. Train/Optimize Forest
    print("\n--- Training Random Forest ---")
    forest_params = create_forest_params(args)
    
    if args.optimize:
        print(f"Starting Bayesian Optimization (n_calls={args.n_calls})...")
        best_params, best_score, _, _ = optimize_rf_hyperparameters(
            X_train, y_train, get_rf_search_space(), 
            n_iter=args.n_calls, random_state=args.random_state,
            X_test=X_test, y_test=y_test # Optional validation
        )
        forest_params.update(best_params)
    
    # Train and Store
    sklearn_rf, our_forest, X_train, y_train = train_and_convert_forest(
        X_train, y_train, X_test, y_test, forest_params, feature_names, 
        random_state=args.random_state
    )
    
    # Store Forest and EU in Redis
    # store_forest_and_endpoints returns the endpoints universe (eu_data)
    eu_data = store_forest_and_endpoints(connections, our_forest)
    
    # 6. Process Samples
    print("\n--- Processing Samples ---")
    # Identify target samples in TEST set
    target_indices = np.where(y_test == target_label)[0]
    target_samples = X_test[target_indices]
    
    print(f"Found {len(target_samples)} samples of class '{target_label}' in test set")
    
    stored_samples, summary = process_all_classified_samples(connections, dataset_actual_name, target_label, our_forest, X_test, y_test, feature_names, eu_data, dataset_type='openml')
    
    # 7. Initialize Seed Candidates
    # Ideally pick a correctly predicted sample
    seed_sample = None
    for s in stored_samples:
        if s['prediction_correct']:
            # We need the full sample dict which is in the stored metadata or we can reconstruct/retrieve
            # process_all_classified_samples returns simplified list.
            # But wait, initialize_seed_candidate needs 'sample_dict', 'test_index', 'prediction_correct'.
            # 's' has 'sample_key', 'icf_bitmap', 'prediction_correct', 'test_index'.
            # It does NOT have 'sample_dict'.
            pass
    
    # Actually, process_all_classified_samples stores everything in Redis.
    # But initialize_seed_candidate takes a dict with 'sample_dict'.
    # Retrying logic: let's pick the first correct sample from the loop where we had sample_dict?
    # process_all_classified_samples does not return sample_dict.
    
    # Workaround: Fetch the sample data from Redis for the seed
    if stored_samples:
        for s in stored_samples:
            meta_json = connections['DATA'].get(f"{s['sample_key']}_meta")
            if meta_json:
                meta = json.loads(meta_json)
                initialize_seed_candidate(connections, meta, our_forest, eu_data)
    else:
        print("[WARNING] No samples processed, cannot initialize seed candidate.")

    # Store the target label for worker compatibility
    connections['DATA'].set('label', target_label)
    print(f"[INFO] Target label '{target_label}' set for worker processing")

    print(f"\n[SUCCESS] Successfully initialized {dataset_actual_name}")


if __name__ == "__main__":
    main()
