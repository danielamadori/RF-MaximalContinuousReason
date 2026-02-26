#!/usr/bin/env python3
"""
Initialize Redis with Baseline Pre-trained Classifiers

This script:
1. Connects to Redis (clearing DBs by default)
2. Loads a pre-trained Random Forest from baseline/Classifiers-100-converted
3. Loads dataset from baseline/resources/datasets
4. Loads test samples from baseline/resources/datasets
5. Stores everything in Redis (Dataset, Forest, Endpoints, Initial Candidate)

Directory Structure:
- Classifiers: baseline/Classifiers-100-converted/<dataset_name>/*.json
- Datasets: baseline/resources/datasets/<dataset_name>/<dataset_name>.csv
- Samples: baseline/resources/datasets/<dataset_name>/<dataset_name>.samples

Usage:
    python init_baseline.py --list-datasets
    python init_baseline.py iris --class-label "0"
    python init_baseline.py sonar --class-label "1" --test-sample-index "0,5-8,20"
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import redis
import json
import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Shared modules
from redis_helpers.connection import connect_redis
from redis_helpers.utils import clean_all_databases
from init_utils import (
    store_forest_and_endpoints,
    initialize_seed_candidate
)
from helpers import convert_numpy_types, parse_sample_indices
from load_rf_from_json import load_rf_from_json
from baseline.xrf import Dataset
from rf_utils import sklearn_forest_to_forest
from rcheck_cache import rcheck_cache, saturate
from icf_eu_encoding import bitmap_mask_to_string, icf_to_bitmap_mask
from redis_helpers.samples import store_sample


# Constants
CLASSIFIERS_ROOT = os.path.join('baseline', 'Classifiers-100-converted')
DATASETS_ROOT = os.path.join('baseline', 'resources', 'datasets')
DATASETS = ['ann-thyroid', 'appendicitis', 'banknote', 'biodegradation', 'ecoli', 'glass2', 'heart-c', 'ionosphere', 'iris', 'karhunen', 'letter', 'magic', 'mofn-3-7-10', 'new-thyroid', 'pendigits', 'phoneme', 'ring', 'segmentation', 'shuttle', 'sonar', 'spambase', 'spectf', 'texture', 'threeOf9', 'twonorm', 'vowel', 'waveform-21', 'waveform-40', 'wdbc', 'wine-recog', 'wpbc', 'xd6']


def list_available_datasets():
    """List all datasets with pre-trained classifiers in baseline directory."""
    print("\nAvailable Baseline Datasets:")
    print("=" * 70)

    if not os.path.exists(CLASSIFIERS_ROOT):
        print(f"[ERROR] Classifiers directory not found: {CLASSIFIERS_ROOT}")
        return

    datasets = []
    for name in sorted(os.listdir(CLASSIFIERS_ROOT)):
        dataset_dir = os.path.join(CLASSIFIERS_ROOT, name)
        if not os.path.isdir(dataset_dir):
            continue

        # Check if dataset files exist
        csv_path = os.path.join(DATASETS_ROOT, name, f"{name}.csv")
        samples_path = os.path.join(DATASETS_ROOT, name, f"{name}.samples")

        # Count JSON classifiers
        json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

        status = "✓"
        notes = []
        if not os.path.exists(csv_path):
            status = "✗"
            notes.append("CSV missing")
        if not os.path.exists(samples_path):
            status = "✗"
            notes.append("samples missing")
        if len(json_files) == 0:
            status = "✗"
            notes.append("no classifiers")

        note_str = f" ({', '.join(notes)})" if notes else ""
        print(f"  {status} {name:<30} {len(json_files)} classifier(s){note_str}")

        if status == "✓":
            datasets.append(name)

    print(f"\nTotal: {len(datasets)} datasets ready to use")
    print("\nUsage: python init_baseline.py <dataset_name> --class-label <label>")


def find_classifier_json(dataset_name):
    """
    Find all classifier JSON files for a dataset.

    Returns:
        List of paths to JSON classifier files
    """
    classifier_dir = os.path.join(CLASSIFIERS_ROOT, dataset_name)

    # Try finding directory with hyphens if original not found
    if not os.path.exists(classifier_dir) and '_' in dataset_name:
        alt_name = dataset_name.replace('_', '-')
        alt_dir = os.path.join(CLASSIFIERS_ROOT, alt_name)
        if os.path.exists(alt_dir):
            classifier_dir = alt_dir
            # We don't change dataset_name here as it might be used for filtering inside


    if not os.path.exists(classifier_dir):
        return []

    json_files = [
        os.path.join(classifier_dir, fname)
        for fname in os.listdir(classifier_dir)
        if fname.endswith('.json')
    ]

    return sorted(json_files)


def load_dataset_from_baseline(dataset_name, separator=','):
    """
    Load dataset CSV and samples from baseline directory structure.

    Args:
        dataset_name: Name of the dataset
        separator: CSV separator (default: ',')

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, all_classes)
    """
    # Handle potential name mismatch (underscore vs hyphen)
    actual_name = dataset_name
    dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)

    if not os.path.exists(dataset_dir) and '_' in dataset_name:
        alt_name = dataset_name.replace('_', '-')
        if os.path.exists(os.path.join(DATASETS_ROOT, alt_name)):
            actual_name = alt_name

    dataset_path = os.path.join(DATASETS_ROOT, actual_name, f"{actual_name}.csv")
    samples_path = os.path.join(DATASETS_ROOT, actual_name, f"{actual_name}.samples")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    if os.path.getsize(samples_path) == 0:
        raise ValueError(f"Samples file is empty: {samples_path}")

    # Load dataset using baseline Dataset class
    print(f"[INFO] Loading dataset from: {dataset_path}")
    data = Dataset(filename=dataset_path, separator=separator, use_categorical=False)

    # Get train/test split from the Dataset
    X_train_raw, X_test_raw, y_train, y_test = data.train_test_split()

    # Transform data (apply any transformations the Dataset class has)
    X_train = data.transform(X_train_raw)
    X_test = data.transform(X_test_raw)

    # Load samples - these will be our actual test samples
    print(f"[INFO] Loading samples from: {samples_path}")
    samples = np.loadtxt(samples_path, delimiter=separator)
    samples = np.atleast_2d(samples)

    # Validate sample dimensions
    expected_features = len(data.features)
    if samples.shape[1] == expected_features + 1:
        print("[INFO] Sample file includes labels; dropping last column.")
        samples = samples[:, :-1]
    elif samples.shape[1] != expected_features:
        raise ValueError(
            f"Sample file feature count mismatch: expected {expected_features}, "
            f"found {samples.shape[1]} in {samples_path}"
        )

    print(f"[INFO] Loaded {len(samples)} test samples")
    print(f"[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Features: {data.features}")
    print(f"[INFO] Classes: {np.unique(y_train)}")

    # Convert class labels to strings for consistency (ensure int format '0' not '0.0')
    #y_train = y_train.astype(int).astype(str)
    #y_test = y_test.astype(int).astype(str)

    return X_train, samples, y_train, None, data.features, np.unique(y_train), data


def load_classifier_from_json(dataset_name):
    """
    Load a pre-trained classifier from JSON.

    Args:
        dataset_name: Name of the dataset
        classifier_index: Index of classifier to use if multiple exist (default: 0)

    Returns:
        (sklearn_rf, classifier_path) tuple
    """
    json_files = find_classifier_json(dataset_name)

    if not json_files:
        raise FileNotFoundError(
            f"No classifier JSON files found for dataset '{dataset_name}' "
            f"in {os.path.join(CLASSIFIERS_ROOT, dataset_name)}"
        )

    if len(json_files) > 1:
        raise ValueError(
            f"Classifier more than one RF found."
        )

    classifier_path = json_files[0]

    print(f"[INFO] Loading pre-trained classifier: {os.path.basename(classifier_path)}")

    # Parse parameters from filename
    filename = os.path.basename(classifier_path)
    try:
        parts = filename.split('_nbestim_')[1]
        n_estimators = int(parts.split('_maxdepth_')[0])
        max_depth = int(parts.split('_maxdepth_')[1].split('.')[0])
        print(f"[INFO] Classifier: {n_estimators} trees, max_depth={max_depth}")
    except (IndexError, ValueError):
        print(f"[WARNING] Could not parse classifier parameters from filename")

    # Load the classifier using load_rf_from_json
    sklearn_rf = load_rf_from_json(classifier_path)

    print(f"[INFO] Successfully loaded classifier with {sklearn_rf.n_estimators} trees")

    return sklearn_rf, classifier_path


def validation(connections, db_mapping, dataset_name):
    X_train, X_test_samples, y_train, _, feature_names, all_classes, data = load_dataset_from_baseline(dataset_name)
    sklearn_rf, classifier_path = load_classifier_from_json(dataset_name)

    our_forest = sklearn_forest_to_forest(sklearn_rf, feature_names)
    eu_data = store_forest_and_endpoints(connections, our_forest)

    X_test = [dict(zip(feature_names,  x_test) ) for x_test in X_test_samples]
    predictions = [our_forest.predict(x_test) for x_test in X_test ]
    validated= True
    not_validated = []
    validated_test = []
    for i in range(len(X_test)):
        nodes = []
        for tree in our_forest.trees:
            nodes.append(tree.root)
        caches = {
            'R': set(),
            'NR': set(),
            'GP': set(),
            'BP': set(),
            'AR': set(),
            'AP': set()
        }
        icf = our_forest.extract_icf(X_test[i])
        if not rcheck_cache(
                    connections=connections,
                    icf=icf,
                    label=predictions[i],
                    nodes=saturate(icf, nodes),
                    eu_data=eu_data,
                    forest=our_forest,
                    caches=caches,
                    info={}
                ):
            not_validated.append(icf)
            validated = False
        else:
            validated_test.append(X_test[i])

    print(40*"#")
    if not validated:
        print(f"{dataset_name} {len(not_validated)} over {len(X_test)} samples not validated")
    else:
        print(f"{dataset_name} FULLY VALIDATED")
    print(40*"#")


def process_all_classified_samples_baseline(
    connections,
    dataset_name,
    class_label,
    our_forest,
    X_test,
    eu_data
):
    """
    Process all test samples that are classified with the specified class label.
    Store samples in DATA and their ICF representations in R.

    Args:
        connections: Redis connections dict
        dataset_name: Name of the dataset
        class_label: Target class label to filter
        our_forest: Custom Forest object
        X_test: Test features array
        eu_data: Endpoints universe data
    Returns:
        tuple: (stored_samples list, summary dict)
    """
    print(f"\n=== Processing All Samples Classified as '{class_label}' ===")

    # Find all test samples that are classified as the target class
    target_samples_data = []
    current_time = datetime.datetime.now().isoformat()

    # Apply sample percentage filtering if specified


    for i, sample in enumerate(X_test):
        predicted_label = our_forest.predict(sample)

        # Store ALL samples classified with the target label (regardless of correctness)
        if predicted_label == class_label:
            target_samples_data.append({
                'test_index': i,
                'sample_dict': sample,
                'predicted_label': predicted_label,
            })

    print(f"Found {len(target_samples_data)} samples classified as '{class_label}'")

    if len(target_samples_data) == 0:
        print("[WARNING] No samples classified with the target label!")
        return [], {}

    # Store all samples and their ICF representations
    stored_samples = []
    correct_predictions = 0

    for idx, sample_data in enumerate(target_samples_data):
        sample_key = f"sample_{dataset_name}_{class_label}_{idx}"

        # Store sample in DATA with full metadata
        data_entry = {
            'sample_dict': sample_data['sample_dict'],
            'predicted_label': sample_data['predicted_label'],
            'test_index': sample_data['test_index'],
            'dataset_name': dataset_name,
            'timestamp': current_time,
        }

        # Store sample using our helper function
        if store_sample(connections['DATA'], sample_key, sample_data['sample_dict']):
            # Also store full metadata separately
            connections['DATA'].set(f"{sample_key}_meta", json.dumps(data_entry))

        # Calculate ICF and store in R
        try:
            sample_icf = our_forest.extract_icf(sample_data['sample_dict'])
            icf_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(sample_icf, eu_data))

            # Store ICF bitmap in R with metadata
            icf_metadata = {
                'sample_key': sample_key,
                'dataset_name': dataset_name,
                'class_label': class_label,
                'test_index': sample_data['test_index'],
                'timestamp': current_time
            }

            connections['R'].set(icf_bitmap, json.dumps(icf_metadata))

            stored_samples.append({
                'sample_key': sample_key,
                'icf_bitmap': icf_bitmap,
                'test_index': sample_data['test_index']
            })

        except Exception as e:
            print(f"[WARNING] Failed to process sample {idx}: {e}")
            continue

    # Store summary information
    summary = {
        'dataset_name': dataset_name,
        'target_class_label': class_label,
        'total_samples_processed': len(stored_samples),
        'total_test_samples': len(X_test),
        'samples_with_target_label': len(target_samples_data),
        'timestamp': current_time,
        'sample_keys': [s['sample_key'] for s in stored_samples]
    }

    connections['DATA'].set(f"summary_{dataset_name}_{class_label}", json.dumps(summary))

    print(f"[OK] Stored {len(stored_samples)} samples in DATA")
    print(f"[OK] Stored {len(stored_samples)} ICF representations in R")
    print(f"[OK] Summary stored in DATA['summary_{dataset_name}_{class_label}']")

    return stored_samples, summary


def main():
    parser = argparse.ArgumentParser(
        description='Initialize Redis with Baseline Pre-trained Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python init_baseline.py --list-datasets

  # Initialize with iris dataset
  python init_baseline.py iris --class-label "0"

  # Use specific test samples
  python init_baseline.py sonar --class-label "1" --test-sample-index "0,5,10"

  # Use sample ranges
  python init_baseline.py iris --class-label "0" --test-sample-index "0-10,20-30"

  # Select a different classifier if multiple exist
  python init_baseline.py iris --class-label "0" --classifier-index 1
        """
    )
    
    # Dataset selection
    parser.add_argument('dataset_name', nargs='?', 
                       help='Name of the baseline dataset (e.g., iris, sonar)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available baseline datasets')
    
    # Core arguments
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--class-label', type=str, required=False,
                       help='Target class label to process (required if dataset is provided)')
    parser.add_argument('--test-sample-index', type=str, default=None,
                       help='Index or indices of samples to use (e.g., "0,5,10" or "0-10")')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not clean Redis databases before initialization')
    #parser.add_argument('--preserve-ar', action='store_true',
    #                   help='Preserve AR database (DB5) when cleaning (implies --no-clean for DB5 only)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # List datasets mode
    if args.list_datasets:
        list_available_datasets()
        return 0
    
    # Validate arguments
    if not args.dataset_name:
        print("[ERROR] Please specify a dataset name or use --list-datasets")
        parser.print_help()
        return 1
    
    if not args.class_label:
        print("[ERROR] --class-label is required")
        return 1
    
    print(f"\n[START] Initializing Random Path Worker System (Baseline)")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] Target Class Label: {args.class_label}")
    
    try:
        # 1. Connect to Redis
        connections, db_mapping = connect_redis(port=args.redis_port)
        if not connections:
            return 1
        
        '''
        if not args.no_clean:
            if args.preserve_ar:
                # Clean all databases except DB5 (AR)
                print("[INFO] Cleaning Redis databases (preserving DB5/AR)...")
                for name, conn in connections.items():
                    if name not in ['AR']:  # Skip AR database
                        try:
                            conn.flushdb()
                        except Exception as e:
                            print(f"[WARNING] Could not clean {name}: {e}")
                print("[INFO] Redis databases cleaned (AR preserved)")
            else:
                # Clean all databases
                clean_all_databases(connections, db_mapping)
                print("[INFO] Redis databases cleaned")
        '''
        clean_all_databases(connections, db_mapping)
        #validation(connections, db_mapping, args.dataset_name)
        #clean_all_databases(connections, db_mapping)
        
        # 2. Load Dataset
        X_train, X_test_samples, y_train, _, feature_names, all_classes, data = load_dataset_from_baseline(
            args.dataset_name
        )
                
        print(f"[INFO] Dataset: {args.dataset_name}")
        print(f"[INFO] Features: {len(feature_names)}")
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Test samples: {len(X_test_samples)}")
        print(f"[INFO] Classes: {all_classes}")
        
        # 3. Load Pre-trained Classifier
        sklearn_rf, classifier_path = load_classifier_from_json(args.dataset_name)
             
        our_forest = sklearn_forest_to_forest(sklearn_rf, feature_names)
        
        # Normalize class_label to match forest prediction format (e.g. '1' → '1.0')
        forest_labels = [str(c) for c in sklearn_rf.classes_]
        if args.class_label not in forest_labels:
            for fl in forest_labels:
                try:
                    if float(fl) == float(args.class_label):
                        args.class_label = fl
                        print(f"[INFO] Normalizing class label to '{fl}'")
                        break
                except (ValueError, TypeError):
                    pass
        
        # 4. Store Training Set Metadata        
        print("\n[INFO] Storing forest and computing endpoints...")
        eu_data = store_forest_and_endpoints(connections, our_forest)

        # 5. Process Test Samples
        print("\n[INFO] Processing test samples...")

        X_test = [dict(zip(feature_names,  x_test) ) for x_test in X_test_samples]

        predictions = [our_forest.predict(x_test) for x_test in X_test ]
        


        label_samples = [sample for sample, pred in  zip(X_test, predictions) if pred == args.class_label  ]

        stored_samples, summary = process_all_classified_samples_baseline(
            connections,
            args.dataset_name,
            args.class_label,
            our_forest,
            label_samples,
            eu_data,
        )
        

        if not stored_samples:
            print("[WARNING] No samples processed, cannot initialize seed candidate.")
            return 1


        print("\n[INFO] Initializing seed candidates...")
        for s in stored_samples:
            meta_json = connections['DATA'].get(f"{s['sample_key']}_meta")
            if meta_json:
                meta = json.loads(meta_json)
                initialize_seed_candidate(connections, meta, our_forest, eu_data)
            else:
                print("[WARNING] No meta data found for a sample")
                
        # 9. Store target label for worker compatibility
        connections['DATA'].set('label', args.class_label)
        print(f"[INFO] Target label '{args.class_label}' set for worker processing")
        
        # 10. Store classifier metadata
        metadata = {
            'dataset': args.dataset_name,
            'classifier_path': classifier_path,
            'n_estimators': sklearn_rf.n_estimators,
            'max_depth': sklearn_rf.max_depth,
            'n_features': sklearn_rf.n_features_in_,
            'classes': list(sklearn_rf.classes_.astype(str)),
            'timestamp': datetime.datetime.now().isoformat()
        }
        connections['DATA'].set('classifier_metadata', json.dumps(metadata))
        
        print(f"\n[SUCCESS] Successfully initialized {args.dataset_name}")
        print(f"[SUCCESS] Pre-trained classifier loaded from: {os.path.basename(classifier_path)}")
        print(f"[SUCCESS] Ready for worker processing with {len(stored_samples)} samples")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[ABORT] Initialization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
