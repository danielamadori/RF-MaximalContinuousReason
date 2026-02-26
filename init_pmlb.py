#!/usr/bin/env python3
"""
Initialize Redis with a PMLB (Penn Machine Learning Benchmarks) Dataset

This script:
1.  Connects to Redis (clearing DBs by default)
2.  Downloads a dataset from PMLB by NAME
3.  Preprocesses data (Imputation, Encoding)
4.  Trains a Random Forest (with Bayesian Optimization)
5.  Stores everything in Redis (Dataset, Forest, Endpoints, Initial Candidate)

Usage:
    python init_pmlb.py [DATASET_NAME] [OPTIONS]
    python init_pmlb.py --list-datasets
    python init_pmlb.py sonar --class-label "1" --test-sample-index "0, 5-8, 20"
"""

import sys
import argparse
import pandas as pd
import numpy as np
import redis
import json
import datetime
import pmlb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Shared modules
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
from helpers import convert_numpy_types, parse_sample_indices


def list_available_datasets():
    """List available classification datasets on PMLB"""
    print("\nFetching PMLB datasets...")
    try:
        # pmlb.dataset_names is a list of all datasets
        all_datasets = pmlb.dataset_names
        classification_datasets = [d for d in pmlb.classification_dataset_names]
        
        print(f"\nFound {len(classification_datasets)} classification datasets.")
        print("Showing first 20:")
        for name in classification_datasets[:20]:
            print(f" - {name}")
            
        print("\nUse the NAME to initialize, e.g.: python init_pmlb.py glin_1")
        
    except Exception as e:
        print(f"[ERROR] Failed to list datasets: {e}")


def load_and_prepare_dataset(dataset_name, shuffle=True, random_state=42):
    """
    Load dataset from PMLB and prepare X, y.
    Handles missing values and categorical features.
    """
    print(f"\n[START] Loading dataset '{dataset_name}' from PMLB...")
    
    # Debug info
    if dataset_name not in pmlb.dataset_names:
        print(f"[WARNING] Dataset '{dataset_name}' not found in pmlb.dataset_names list.")
        # Try anyway as list might be outdated or different
    
    try:
        # Fetch data as dataframe to get feature names
        df = pmlb.fetch_data(dataset_name, return_X_y=False)
        
        # PMLB datasets assume the target is the last column or named 'target' class.
        # Actually pmlb.fetch_data processes it.
        # If return_X_y=False, it returns a dataframe. 
        # By convention in PMLB, the target is the 'target' column if formatted, 
        # but fetch_data might just return the raw data.
        # Let's use return_X_y=True to be safe for separation, but we need feature names.
        
        X, y = pmlb.fetch_data(dataset_name, return_X_y=True)
        
        # To get feature names, we fetch as DF again or inspect logic
        # fetch_data(local_cache_dir=..., return_X_y=False) -> DF
        df = pmlb.fetch_data(dataset_name, return_X_y=False)
        
        # PMLB documentation says: "The last column is the target column" usually.
        # But fetch_data(return_X_y=True) handles the separation.
        # So X is numpy array (or DF?), y is numpy array.
        # Let's check X type. pmlb returns numpy arrays by default for X, y.
        
        feature_names = [c for c in df.columns if c != 'target']
        # If 'target' column exists in DF, it's the target.
        # If not, PMLB logic for separation is applied. 
        
        # Let's trust X and y from return_X_y=True, but we need feature names.
        # Ideally X is a DataFrame if we don't convert it?
        # pmlb source: 
        # if return_X_y: return X, y
        # X is df.drop('target', axis=1).values
        
        if 'target' in df.columns:
            feature_names = [c for c in df.columns if c != 'target']
        else:
            # Fallback if target column name is different
            feature_names = list(df.columns[:-1]) 
        
        print(f"[INFO] Loaded shape: {X.shape}")
        
        # Convert to DataFrame for easier processing (handling NaNs etc consistently)
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Handle Missing Values
        if X_df.isnull().values.any():
            print("[INFO] Missing values detected. Imputing...")
            num_cols = X_df.select_dtypes(include=[np.number]).columns
            cat_cols = X_df.select_dtypes(exclude=[np.number]).columns
            
            if len(num_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                X_df[num_cols] = imputer_num.fit_transform(X_df[num_cols])
                
            if len(cat_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                X_df[cat_cols] = imputer_cat.fit_transform(X_df[cat_cols])

        # Handle Categorical Features (Label Encoding)
        cat_cols = X_df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            print(f"[INFO] Encoding {len(cat_cols)} categorical features...")
            for col in cat_cols:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
        
        return X_df, pd.Series(y), dataset_name

    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Initialize Redis with PMLB Dataset')
    
    # Dataset selection
    parser.add_argument('dataset_name', nargs='?', help='Name of the PMLB dataset (e.g. glin_1)')
    parser.add_argument('--list-datasets', action='store_true', help='List available classification datasets')
    
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
    print("\n[START] Initializing Random Path Worker System (PMLB)")
    connections, db_mapping = connect_redis(port=args.redis_port)
    if not connections:
        sys.exit(1)

    if not args.no_clean:
        from redis_helpers.utils import clean_all_databases
        clean_all_databases(connections, db_mapping)
        print("[INFO] Redis databases cleaned")

    # 2. Load Dataset
    # If using specific index, we want predictable order (no shuffle)
    use_shuffle = args.test_sample_index is None
    X_df, y, dataset_actual_name = load_and_prepare_dataset(args.dataset_name, shuffle=use_shuffle, random_state=args.random_state)
    if X_df is None:
        sys.exit(1)

    feature_names = list(X_df.columns)
    
    # Convert to numpy for processing
    X = X_df.values
    y = y.values
    
    # Ensure y is string for consistency with logic
    y = y.astype(str)
    
    print(f"\n[INFO] Dataset: {dataset_actual_name}")
    print(f"[INFO] Features: {len(feature_names)}")
    print(f"[INFO] Samples: {len(X)}")

    # 3. Handle Target Label
    classes = np.unique(y)
    print(f"[INFO] Classes detected: {classes}")

    target_label = args.class_label
    if target_label is None:
        if len(classes) == 2:
            target_label = str(classes[0]) # Default first
            print(f"[INFO] No class label specified. Defaulting to first class: '{target_label}'")
        else:
            print(f"\nAvailable classes: {classes}")
            print(f"\nAvailable classes: {classes}")
            print("[ERROR] Please specify --class-label")
            sys.exit(1)
    
    # Verify label exists
    if target_label not in classes:
        print(f"[ERROR] Class label '{target_label}' not found in dataset. Available: {classes}")
        sys.exit(1)

    print(f"[INFO] Target Class Label: {target_label}")

    # 4. Sampling & Split
    # 4. Sampling & Split
    # If test_sample_index is used, we MUST skip random sampling to preserve indices
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
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split, random_state=args.random_state, stratify=y
        )
    
    store_dataset_total_samples(connections, len(X_test))

    # Store Training Set Metadata
    store_training_set(connections, X_train, y_train, feature_names, dataset_actual_name, dataset_type='pmlb')

    # 5. Train/Optimize Forest
    print("\n--- Training Random Forest ---")
    forest_params = create_forest_params(args)
    
    if args.optimize:
        print(f"Starting Bayesian Optimization (n_calls={args.n_calls})...")
        best_params, best_score, _, _ = optimize_rf_hyperparameters(
            X_train, y_train, get_rf_search_space(), 
            n_iter=args.n_calls, random_state=args.random_state,
            X_test=X_test, y_test=y_test
        )
        forest_params.update(best_params)
    
    # Train and Store
    sklearn_rf, our_forest, X_train, y_train = train_and_convert_forest(
        X_train, y_train, X_test, y_test, forest_params, feature_names, 
        random_state=args.random_state
    )
    
    # Store Forest and EU in Redis
    eu_data = store_forest_and_endpoints(connections, our_forest)
    
    # 6. Process Samples
    print("\n--- Processing Samples ---")
    # Identify target samples in TEST set
    target_indices = np.where(y_test == target_label)[0]
    target_samples = X_test[target_indices]
    
    print(f"Found {len(target_samples)} samples of class '{target_label}' in test set")
    
    stored_samples, summary = process_all_classified_samples(connections, dataset_actual_name, target_label, our_forest, X_test, y_test, feature_names, eu_data, dataset_type='pmlb')
    
    # 7. Initialize Seed Candidates
    # 7. Initialize Seed Candidates
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
