#!/usr/bin/env python3
"""
UCI Dataset Initializer Script

Initializes the random path worker system with any UCI ML Repository classification dataset.
Processes all test samples with a specified class label and stores their ICF representations.

Usage:
    python init_uci.py Iris --class-label "Iris-setosa" --n-estimators 50
    python init_uci.py --list-datasets
    python init_uci.py "Breast Cancer" --class-label "M" --optimize
    python init_uci.py Wine --class-label "1" --test-sample-index "0, 5-8, 20"
"""

import argparse
import json
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# UCI ML Repository imports
from ucimlrepo import fetch_ucirepo

# Shared utilities
from helpers import convert_numpy_types, parse_sample_indices
from redis_helpers.connection import connect_redis
from redis_helpers.utils import clean_all_databases
from rf_utils import (
    create_forest_params,
    get_rf_search_space,
    optimize_rf_hyperparameters,
    train_and_convert_forest,
)
from init_utils import (
    store_training_set,
    store_forest_and_endpoints,
    process_all_classified_samples,
    initialize_seed_candidate
)

# Curated list of popular UCI classification datasets
# Format: { 'name': (uci_id, 'category', 'description') }
AVAILABLE_DATASETS = {
    # Small datasets (good for testing, < 500 samples)
    'Iris': (53, 'small', '150 samples, 4 features, 3 classes - classic flower classification'),
    'Wine': (109, 'small', '178 samples, 13 features, 3 classes - wine cultivar recognition'),
    'Glass': (42, 'small', '214 samples, 9 features, 6 classes - glass type identification'),
    'Zoo': (111, 'small', '101 samples, 16 features, 7 classes - animal classification'),
    'Hayes-Roth': (44, 'small', '160 samples, 4 features, 3 classes - concept learning'),
    'Seeds': (236, 'small', '210 samples, 7 features, 3 classes - wheat seed varieties'),
    'Ecoli': (39, 'small', '336 samples, 7 features, 8 classes - protein localization'),
    'Liver Disorders': (60, 'small', '345 samples, 6 features, 2 classes - liver disease'),
    
    # Medium datasets (500-10K samples)
    'Breast Cancer Wisconsin': (17, 'medium', '569/699 samples, 30/9 features, 2 classes - tumor classification'),
    'Heart Disease': (45, 'medium', '303 samples, 13 features, 5 classes - cardiac diagnosis'),
    'Ionosphere': (52, 'medium', '351 samples, 34 features, 2 classes - radar signal classification'),
    'Diabetes': (34, 'medium', '768 samples, 8 features, 2 classes - Pima Indians diabetes'),
    'Car Evaluation': (19, 'medium', '1728 samples, 6 features, 4 classes - car acceptability'),
    'Mushroom': (73, 'medium', '8124 samples, 22 features, 2 classes - edible vs poisonous'),
    'Tic-Tac-Toe': (101, 'medium', '958 samples, 9 features, 2 classes - game outcome'),
    'Balance Scale': (12, 'medium', '625 samples, 4 features, 3 classes - balance tip direction'),
    'Banknote Authentication': (267, 'medium', '1372 samples, 4 features, 2 classes - genuine vs forged'),
    'Blood Transfusion': (176, 'medium', '748 samples, 4 features, 2 classes - donor prediction'),
    'Haberman Survival': (43, 'medium', '306 samples, 3 features, 2 classes - surgery survival'),
    'Statlog German Credit': (144, 'medium', '1000 samples, 20 features, 2 classes - credit risk'),
    'Statlog Australian Credit': (143, 'medium', '690 samples, 14 features, 2 classes - credit approval'),
    
    # Large datasets (> 10K samples)
    'Adult': (2, 'large', '48842 samples, 14 features, 2 classes - income prediction'),
    'Letter Recognition': (59, 'large', '20000 samples, 16 features, 26 classes - letter identification'),
    'Covertype': (31, 'large', '581012 samples, 54 features, 7 classes - forest cover type'),
    'Magic Gamma Telescope': (159, 'large', '19020 samples, 10 features, 2 classes - particle detection'),
    'Spambase': (94, 'large', '4601 samples, 57 features, 2 classes - email spam detection'),
    'Waveform': (108, 'large', '5000 samples, 21 features, 3 classes - waveform classification'),
}


def list_available_datasets():
    """Print available UCI classification datasets."""
    print("Available UCI ML Repository Classification Datasets:")
    print("=" * 70)
    
    # Group by category
    small = [(name, info) for name, info in AVAILABLE_DATASETS.items() if info[1] == 'small']
    medium = [(name, info) for name, info in AVAILABLE_DATASETS.items() if info[1] == 'medium']
    large = [(name, info) for name, info in AVAILABLE_DATASETS.items() if info[1] == 'large']
    
    print("\n[SMALL] Small Datasets (good for testing, < 500 samples):")
    for name, (uci_id, _, desc) in small:
        print(f"  {name:<25} (ID: {uci_id:>3}) - {desc}")
    
    print(f"\n[MEDIUM] Medium Datasets (500-10K samples):")
    for name, (uci_id, _, desc) in medium:
        print(f"  {name:<25} (ID: {uci_id:>3}) - {desc}")
    
    print(f"\n[LARGE] Large Datasets (> 10K samples, use with caution):")
    for name, (uci_id, _, desc) in large:
        print(f"  {name:<25} (ID: {uci_id:>3}) - {desc}")
    
    print(f"\nTotal: {len(AVAILABLE_DATASETS)} curated datasets available")
    print("\nYou can also use any UCI dataset by its ID: python init_uci.py --id 53")


def get_dataset_info(dataset_name=None, dataset_id=None):
    """Get basic info about a dataset without fully loading it."""
    try:
        # Fetch by name or ID
        if dataset_id is not None:
            dataset = fetch_ucirepo(id=dataset_id)
        elif dataset_name is not None:
            # Check if it's in our curated list
            if dataset_name in AVAILABLE_DATASETS:
                dataset_id = AVAILABLE_DATASETS[dataset_name][0]
                dataset = fetch_ucirepo(id=dataset_id)
            else:
                # Try fetching by name directly
                dataset = fetch_ucirepo(name=dataset_name)
        else:
            return {'error': 'Either dataset_name or dataset_id must be provided'}
        
        # Extract metadata
        X = dataset.data.features
        y = dataset.data.targets
        
        # Get unique classes from target
        if y is not None and len(y.columns) > 0:
            target_col = y.columns[0]
            classes = y[target_col].dropna().unique().tolist()
        else:
            classes = []
        
        # Count categorical vs numeric features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Count missing values
        missing_count = X.isnull().sum().sum() + (y.isnull().sum().sum() if y is not None else 0)
        
        return {
            'name': dataset.metadata.name if hasattr(dataset.metadata, 'name') else 'Unknown',
            'uci_id': dataset.metadata.uci_id if hasattr(dataset.metadata, 'uci_id') else None,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'categorical_features': categorical_cols,
            'numeric_features': numeric_cols,
            'n_categorical': len(categorical_cols),
            'n_numeric': len(numeric_cols),
            'classes': classes,
            'n_classes': len(classes),
            'missing_values': int(missing_count),
            'target_column': y.columns[0] if y is not None and len(y.columns) > 0 else None
        }
    except Exception as e:
        return {'error': str(e)}


def encode_categorical_features(X, feature_encoders=None):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        X: pandas DataFrame with features
        feature_encoders: Optional dict of pre-fitted encoders
        
    Returns:
        X_encoded: DataFrame with encoded features
        feature_encoders: dict mapping column names to fitted LabelEncoders
    """
    X_encoded = X.copy()
    
    if feature_encoders is None:
        feature_encoders = {}
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col not in feature_encoders:
            le = LabelEncoder()
            # Handle NaN by converting to string first
            X_encoded[col] = X_encoded[col].fillna('_MISSING_').astype(str)
            X_encoded[col] = le.fit_transform(X_encoded[col])
            feature_encoders[col] = le
        else:
            X_encoded[col] = X_encoded[col].fillna('_MISSING_').astype(str)
            X_encoded[col] = feature_encoders[col].transform(X_encoded[col])
    
    # Convert to numeric and handle remaining NaN
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
    
    return X_encoded, feature_encoders


def handle_missing_values(X, y=None):
    """
    Handle missing values with simple imputation.
    
    Args:
        X: pandas DataFrame or numpy array with features
        y: Optional target array
        
    Returns:
        X_imputed: numpy array with imputed values
        y_clean: target array with rows removed if y had NaN
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    # Impute numeric features with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_values)
    
    # Handle y if provided
    if y is not None:
        if isinstance(y, pd.DataFrame):
            y_values = y.values.ravel()
        elif isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
        
        # Remove rows where y is NaN
        valid_mask = ~pd.isna(y_values)
        X_imputed = X_imputed[valid_mask]
        y_clean = y_values[valid_mask]
        
        return X_imputed, y_clean
    
    return X_imputed, None


def load_and_prepare_dataset(dataset_name=None, dataset_id=None, feature_prefix="f", test_split=0.3, random_state=42):
    """
    Load and prepare UCI dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_id: UCI ID of the dataset
        feature_prefix: Prefix for feature names (default: "f")
        test_split: Fraction of data to use for testing (default: 0.3)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test, feature_names, class_names
    """
    print(f"Loading dataset: {dataset_name or f'ID={dataset_id}'}")
    
    try:
        # Fetch dataset
        if dataset_id is not None:
            dataset = fetch_ucirepo(id=dataset_id)
        elif dataset_name is not None:
            if dataset_name in AVAILABLE_DATASETS:
                dataset_id = AVAILABLE_DATASETS[dataset_name][0]
                dataset = fetch_ucirepo(id=dataset_id)
            else:
                dataset = fetch_ucirepo(name=dataset_name)
        else:
            raise ValueError("Either dataset_name or dataset_id must be provided")
        
        X = dataset.data.features
        y = dataset.data.targets
        
        if y is None or len(y.columns) == 0:
            raise ValueError("Dataset has no target variable")
        
        # Get target column
        target_col = y.columns[0]
        y_series = y[target_col]
        
        print(f"Dataset: {dataset.metadata.name if hasattr(dataset.metadata, 'name') else 'Unknown'}")
        print(f"Original shape: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Encode categorical features
        print("Encoding categorical features...")
        X_encoded, feature_encoders = encode_categorical_features(X)
        
        # Handle missing values
        print("Handling missing values...")
        X_clean, y_clean = handle_missing_values(X_encoded, y_series)
        
        # Keep original class labels as strings (don't encode to integers)
        y_clean_str = y_clean.astype(str)
        class_names = np.unique(y_clean_str)
        
        print(f"Clean shape: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        print(f"Classes: {class_names}")
        
        # Create feature names with prefix
        original_feature_names = X.columns.tolist()
        feature_names = [f"{feature_prefix}_{name}" for name in original_feature_names]
        
        # Split into train/test
        print(f"Splitting data with test_split={test_split}")
        # Convert to numpy to avoid pyarrow-backed array indexing issues (pandas 3+)
        if hasattr(X_clean, 'to_numpy'):
            X_clean_np = X_clean.to_numpy(dtype=float, na_value=np.nan)
        else:
            X_clean_np = np.asarray(X_clean, dtype=float)
        if hasattr(y_clean_str, 'to_numpy'):
            y_clean_np = y_clean_str.to_numpy()
        else:
            y_clean_np = np.asarray(y_clean_str)
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean_np, y_clean_np, test_size=test_split, random_state=random_state, stratify=y_clean_np
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_test, y_test, feature_names, class_names
        
    except Exception as e:
        raise Exception(f"Failed to load dataset: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize random path worker system with UCI ML Repository datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python init_uci.py --list-datasets

  # Show dataset info
  python init_uci.py Iris --info

  # Initialize Iris with default parameters, process samples labeled "Iris-setosa"
  python init_uci.py Iris --class-label "Iris-setosa"

  # Use dataset by UCI ID instead of name
  python init_uci.py --id 53 --class-label "Iris-setosa"

  # Custom forest parameters
  python init_uci.py Wine --class-label "1" --n-estimators 100 --max-depth 5

  # Use Bayesian optimization with cross-validation (recommended)
  python init_uci.py "Breast Cancer Wisconsin" --class-label "M" --optimize --opt-n-iter 30

  # Custom train/test split
  python init_uci.py Adult --class-label ">50K" --test-split 0.2

  # Process only 5% of samples
  python init_uci.py Covertype --class-label "1" --sample-percentage 5
        """
    )
    
    parser.add_argument('dataset_name', nargs='?', 
                       help='Name of the UCI dataset to load')
    
    parser.add_argument('--id', type=int, dest='dataset_id',
                       help='UCI dataset ID (alternative to dataset_name)')
    
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available UCI classification datasets')
    
    parser.add_argument('--class-label', type=str, required=False,
                       help='Target class label to process (required if dataset is provided)')
    
    parser.add_argument('--info', action='store_true',
                       help='Show information about the dataset without processing')
    
    # Forest parameters
    forest_group = parser.add_argument_group('Random Forest Parameters')
    forest_group.add_argument('--n-estimators', type=int, default=50,
                            help='Number of trees in the forest (default: 50)')
    forest_group.add_argument('--criterion', type=str, choices=['gini', 'entropy'],
                            help='Split quality criterion (default: gini)')
    forest_group.add_argument('--max-depth', type=int,
                            help='Maximum depth of trees (default: None)')
    forest_group.add_argument('--min-samples-split', type=int,
                            help='Minimum samples required to split (default: 2)')
    forest_group.add_argument('--min-samples-leaf', type=int,
                            help='Minimum samples required at leaf (default: 1)')
    forest_group.add_argument('--max-features', type=str,
                            help='Number of features for best split (default: "sqrt")')
    forest_group.add_argument('--max-leaf-nodes', type=int,
                            help='Maximum number of leaf nodes (default: None)')
    forest_group.add_argument('--min-impurity-decrease', type=float,
                            help='Minimum impurity decrease for split (default: 0.0)')
    forest_group.add_argument('--bootstrap', type=str, choices=['True', 'False'],
                            help='Whether to use bootstrap samples (default: True)')
    forest_group.add_argument('--max-samples', type=float,
                            help='Fraction of samples for each tree if bootstrap=True (default: None)')
    forest_group.add_argument('--ccp-alpha', type=float,
                            help='Complexity parameter for pruning (default: 0.0)')

    # Bayesian optimization parameters
    opt_group = parser.add_argument_group('Bayesian Optimization Parameters')
    opt_group.add_argument('--optimize', action='store_true',
                          help='Use Bayesian optimization to find best RF hyperparameters')
    opt_group.add_argument('--opt-n-iter', type=int, default=50,
                          help='Number of iterations for Bayesian optimization (default: 50)')
    opt_group.add_argument('--opt-cv', type=int, default=5,
                          help='Number of cross-validation folds for optimization (default: 5)')
    opt_group.add_argument('--opt-n-jobs', type=int, default=-1,
                          help='Number of parallel jobs for optimization, -1 for all cores (default: -1)')
    opt_group.add_argument('--opt-use-test', action='store_true',
                          help='Use test set for validation during optimization instead of CV. '
                               'WARNING: May lead to overfitting on test data!')
    
    # Data processing parameters
    parser.add_argument('--sample-percentage', type=float, default=100.0,
                       help='Process only this percentage of samples (0-100, default: 100)')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Fraction of data to use for testing (default: 0.3)')
    parser.add_argument('--test-sample-index', type=str, default=None,
                       help='Index (or comma-separated indices) of samples to use for testing (Leave-One-Out). All others used for training.')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--feature-prefix', type=str, default='f',
                       help='Prefix for feature names (default: "f")')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not clean Redis databases before initialization')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis/KeyDB server port (default: 6379)')
    
    args = parser.parse_args()
    
    # Handle list datasets
    if args.list_datasets:
        list_available_datasets()
        return 0
    
    # Validate arguments
    if not args.dataset_name and args.dataset_id is None:
        parser.error("dataset_name or --id is required (or use --list-datasets)")
    
    # Determine dataset identifier for display
    dataset_identifier = args.dataset_name or f"ID={args.dataset_id}"
    
    # Show dataset info if requested
    if args.info:
        print(f"Getting information for dataset: {dataset_identifier}")
        info = get_dataset_info(dataset_name=args.dataset_name, dataset_id=args.dataset_id)
        if 'error' in info:
            print(f"[ERROR] Error loading dataset: {info['error']}")
            return 1
        
        print(f"\n[INFO] Dataset Information: {info['name']}")
        print(f"  UCI ID: {info['uci_id']}")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']} ({info['n_numeric']} numeric, {info['n_categorical']} categorical)")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Class values: {info['classes']}")
        print(f"  Missing values: {info['missing_values']}")
        print(f"  Target column: {info['target_column']}")
        print(f"\n  Features: {', '.join(info['feature_names'][:10])}", end='')
        if len(info['feature_names']) > 10:
            print(f" ... and {len(info['feature_names']) - 10} more")
        else:
            print()
        
        if args.class_label:
            if args.class_label in [str(c) for c in info['classes']]:
                print(f"\n[OK] Target class label '{args.class_label}' is valid")
            else:
                print(f"\n[ERROR] Target class label '{args.class_label}' not found in dataset classes")
                print(f"   Available classes: {info['classes']}")
        
        return 0
    
    # If test_sample_index is set, we strictly process THAT sample. 
    # Disable random sampling.
    if args.test_sample_index is not None and args.sample_percentage < 100:
        print(f"[WARNING] Overriding --sample-percentage to 100% because --test-sample-index is set.")
        args.sample_percentage = 100.0
    
    if not args.class_label:
        parser.error("--class-label is required when processing a dataset")
    
    print(f"[START] Initializing Random Path Worker System")
    print(f"[INFO] Dataset: {dataset_identifier}")
    print(f"[INFO] Target Class Label: {args.class_label}")
    print(f"[INFO] Sample Percentage: {args.sample_percentage}%")
    print(f"[INFO] Test Split: {args.test_split}")

    try:
        # Connect to Redis
        connections, db_mapping = connect_redis(port=args.redis_port)
        
        # Clean databases if requested
        if not args.no_clean:
            print("Cleaning Redis databases...")
            clean_all_databases(connections, db_mapping)
        
        use_shuffle = args.test_sample_index is None
        
        # Load and prepare dataset
        X_train, y_train, X_test, y_test, feature_names, class_names = load_and_prepare_dataset(
            dataset_name=args.dataset_name,
            dataset_id=args.dataset_id,
            feature_prefix=args.feature_prefix,
            test_split=args.test_split,
            random_state=args.random_state
        )
        
        # Handle strict single/multi sample test (LOO) if specified
        if args.test_sample_index is not None:
             # Combine all data first
            X_all = np.vstack([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            
            
            try:
                indices = parse_sample_indices(args.test_sample_index)
            except ValueError as e:
                print(f"[ERROR] Invalid format for --test-sample-index: {e}")
                return 1

            # Validate indices
            if any(i < 0 or i >= len(X_all) for i in indices):
                print(f"[ERROR] One or more indices out of bounds (0-{len(X_all)-1})")
                return 1
            
            print(f"Applying strict Split: Test sample indices {indices}")
            
            # Select test samples
            X_test = X_all[indices]
            y_test = y_all[indices]
            
            # Use all others for training
            X_train = np.delete(X_all, indices, axis=0)
            y_train = np.delete(y_all, indices, axis=0)
            
            print(f"LOO Training set: {X_train.shape[0]} samples")
            print(f"LOO Test set: {X_test.shape[0]} samples (Indices: {indices})")
            
            # Debug info on classes
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"    Test Classes: {dict(zip(unique, counts))}")
            
            # Verify class membership
            mismatches = [i for i, label in enumerate(y_test) if label != args.class_label]
            if mismatches:
                print(f"[WARNING] The following selected samples do NOT belong to target class '{args.class_label}':")
                for i in mismatches:
                    print(f"  - Index {indices[i]}: Class '{y_test[i]}'")
                
                if len(mismatches) == len(indices):
                    print(f"[ERROR] No selected samples belong to the target class! Aborting.")
                    sys.exit(1)

        total_samples = len(X_test)
        if args.sample_percentage is not None and args.sample_percentage < 100.0:
            total_samples = max(1, int(total_samples * args.sample_percentage / 100.0))

        # Validate class label
        if args.class_label not in class_names:
            print(f"[ERROR] Class label '{args.class_label}' not found in dataset")
            print(f"   Available classes: {list(class_names)}")
            return 1

        # Optionally optimize RF hyperparameters with Bayesian optimization
        if args.optimize:
            print("\n" + "="*70)
            print("BAYESIAN OPTIMIZATION MODE")
            print("="*70)

            # Get search space
            search_space = get_rf_search_space()

            # Run Bayesian optimization
            best_params, best_score, test_score, optimizer = optimize_rf_hyperparameters(
                X_train, y_train,
                search_space=search_space,
                n_iter=args.opt_n_iter,
                cv=args.opt_cv,
                n_jobs=args.opt_n_jobs,
                random_state=args.random_state,
                verbose=1,
                X_test=X_test,
                y_test=y_test,
                use_test_for_validation=args.opt_use_test
            )

            # Store optimization results
            opt_results = {
                'best_params': best_params,
                'best_cv_score': best_score if not args.opt_use_test else None,
                'test_score': test_score,
                'used_test_for_validation': args.opt_use_test,
                'n_iter': args.opt_n_iter,
                'cv_folds': args.opt_cv if not args.opt_use_test else None,
                'dataset_name': dataset_identifier,
                'dataset_type': 'uci',
                'timestamp': datetime.datetime.now().isoformat()
            }
            opt_results_serializable = convert_numpy_types(opt_results)
            connections['DATA'].set('RF_OPTIMIZATION_RESULTS', json.dumps(opt_results_serializable))
            print(f"[OK] Optimization results saved to DATA['RF_OPTIMIZATION_RESULTS']")

            # Use optimized parameters
            rf_params = {**best_params, 'random_state': args.random_state}
            print(f"\n[OPTIMIZE] Using optimized parameters for final model")

        else:
            # Use manually specified parameters
            rf_params = create_forest_params(args)

        print(f"[INFO] Forest Parameters: {rf_params}")

        # Train and convert forest
        sklearn_rf, our_forest, X_train_used, y_train_used = train_and_convert_forest(
            X_train, y_train, X_test, y_test, rf_params, feature_names,
            random_state=args.random_state,
            sample_percentage=args.sample_percentage
        )

        # Store training set in DATA database
        store_training_set(connections, X_train_used, y_train_used, feature_names, dataset_identifier, dataset_type='uci')

        # Store forest and endpoints
        eu_data = store_forest_and_endpoints(connections, our_forest)
        
        # Process all test samples classified with target label
        stored_samples, summary = process_all_classified_samples(
            connections, dataset_identifier, args.class_label,
            our_forest, X_test, y_test, feature_names, eu_data, 
            args.sample_percentage, dataset_type='uci'
        )
        
        # Initialize seed candidates 
        for sample in stored_samples:
            sample_key = sample['sample_key']
            sample_dict = json.loads(connections['DATA'].get(sample_key + "_meta"))
            initialize_seed_candidate(connections, sample_dict, our_forest, eu_data)
        
        # Store the target label for worker compatibility
        connections['DATA'].set('label', args.class_label)
        print(f"[INFO] Target label '{args.class_label}' set for worker processing")

        print(f"\n[SUCCESS] Successfully initialized {dataset_identifier}")
        print(f"[INFO] Dataset type: UCI")
        print(f"[INFO] Forest: {len(our_forest)} trees")
        print(f"[INFO] Features: {len(feature_names)}")
        if summary:
            print(f"[INFO] Processed: {summary['total_samples_processed']} samples with label '{args.class_label}'")
            print(f"[OK] Correct: {summary['correct_predictions']}")
            print(f"[INFO] Incorrect: {summary['incorrect_predictions']}")
            print(f"[INFO] Accuracy: {summary['accuracy']:.3f}")
        else:
            print(f"[WARNING] No samples were classified with label '{args.class_label}'")

        print(f"[INFO] Data stored in Redis databases")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
