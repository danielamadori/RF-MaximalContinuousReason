#!/usr/bin/env python3
"""
Self-Contained Aeon Univariate Dataset Initializer Script

Initializes the random path worker system with any aeon univariate time series dataset.
Processes all test samples with a specified class label and stores their ICF representations.

Usage:
    python init_aeon_univariate.py ECG200 --class-label "1" --n-estimators 50
    python init_aeon_univariate.py --list-datasets
    python init_aeon_univariate.py Coffee --class-label "0" --max-depth 4
"""

import argparse
import redis
import json
import time
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from aeon.datasets import load_classification

# Bayesian optimization imports (optional - will check availability)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Import our custom modules (from init_enhanced)
from skforest_to_forest import sklearn_forest_to_forest
from redis_helpers.forest import store_forest
from redis_helpers.endpoints import store_monotonic_dict
from redis_helpers.samples import store_sample
from redis_helpers.preferred import insert_to_pr
from sample_converter import sklearn_sample_to_dict
from redis_helpers.utils import clean_all_databases
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string

# List of popular aeon univariate datasets
AVAILABLE_DATASETS = [
    # Small datasets (good for testing)
    'Coffee', 'ECG200', 'GunPoint', 'ItalyPowerDemand', 'Lightning2', 'Lightning7',
    'MedicalImages', 'MoteStrain', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
    'Symbols', 'SyntheticControl', 'TwoLeadECG', 'Wafer', 'Wine', 'Yoga',
    
    # Medium datasets
    'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    'ChlorineConcentration', 'CinCECGTorso', 'Computers', 'CricketX', 'CricketY', 'CricketZ',
    'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW', 'Earthquakes', 'ECG5000', 'ECGFiveDays', 'ElectricDevices',
    'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',
    
    # Large datasets (use with caution)
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

if hasattr(np, "bool8"):
    _NUMPY_BOOL_TYPES = (np.bool_, np.bool8, bool)
else:
    _NUMPY_BOOL_TYPES = (np.bool_, bool)


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization

    Args:
        obj: Object to convert (can be dict, list, or scalar)

    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, _NUMPY_BOOL_TYPES):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def connect_redis(port=6379):
    """Establish Redis connections to all databases"""
    connections = {}
    db_mapping = {
        'DATA': 0,
        'CAN': 1,      # Candidate reasons (for positive samples)
        'R': 2,        # Reasons (confirmed positive ICFs)
        'NR': 3,       # Non-reasons (confirmed negative ICFs)
        'CAR': 4,      # Candidate Anti-Reasons (starts empty) - replaces DD
        'AR': 5,       # Anti-Reasons (confirmed ARs) - replaces DDS
        'GP': 6,       # Good Profiles (forest profiles of reasons)
        'BP': 7,       # Bad Profiles (forest profiles of non-reasons)
        'PR': 8,       # Preferred Reasons
        'AP': 9,       # Anti-Reason Profiles (forest profiles of anti-reasons)
        'LOGS': 10     # NEW: Worker iteration logs

    }

    for name, db_id in db_mapping.items():
        try:
            conn = redis.Redis(host='localhost', port=port, db=db_id, decode_responses=True)
            conn.ping()
            connections[name] = conn
            print(f"Connected to Redis DB {db_id} ({name}) on port {port}")
        except redis.ConnectionError:
            raise Exception(f"Failed to connect to Redis DB {db_id} ({name}) on port {port}")

    print(f"Established {len(connections)} Redis connections")
    return connections, db_mapping


def list_available_datasets():
    """Print available aeon univariate datasets"""
    print("Available Aeon Univariate Time Series Datasets:")
    print("=" * 50)
    
    print("\n[SMALL] Small Datasets (good for testing):")
    small_datasets = AVAILABLE_DATASETS[:16]
    for i, dataset in enumerate(small_datasets):
        if i % 4 == 0:
            print()
        print(f"  {dataset:<20}", end="")
    
    print(f"\n\n[MEDIUM] Medium Datasets:")
    medium_datasets = AVAILABLE_DATASETS[16:50]
    for i, dataset in enumerate(medium_datasets):
        if i % 3 == 0:
            print()
        print(f"  {dataset:<25}", end="")
    
    print(f"\n\n[LARGE] Large Datasets (use with caution):")
    large_datasets = AVAILABLE_DATASETS[50:]
    for i, dataset in enumerate(large_datasets):
        if i % 3 == 0:
            print()
        print(f"  {dataset:<25}", end="")
    
    print(f"\n\nTotal: {len(AVAILABLE_DATASETS)} datasets available")
    print("\nNote: Not all datasets may be available in your aeon installation.")


def get_dataset_info(dataset_name):
    """Try to get basic info about a dataset without fully loading it"""
    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
        
        classes = np.unique(np.concatenate([y_train, y_test]))
        
        return {
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'series_length': X_train.shape[2],
            'n_channels': X_train.shape[1],
            'classes': classes.tolist(),
            'n_classes': len(classes)
        }
    except Exception as e:
        return {'error': str(e)}


def load_and_prepare_dataset(dataset_name, feature_prefix="t"):
    """Load and prepare aeon dataset"""
    print(f"Loading dataset: {dataset_name}")

    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
    except Exception as e:
        raise Exception(f"Failed to load dataset {dataset_name}: {e}")

    print(f"Dataset: {dataset_name}")
    print(f"Training set: {X_train.shape} samples")
    print(f"Test set: {X_test.shape} samples")
    print(f"Series length: {X_train.shape[2]} time points")
    print(f"Classes: {np.unique(np.concatenate([y_train, y_test]))}")

    # Reshape data: (n_samples, n_channels, n_timepoints) -> (n_samples, n_timepoints)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # Calculate padding width for feature names
    series_length = X_train_2d.shape[1]
    padding_width = len(str(series_length - 1))
    feature_names = [f"{feature_prefix}_{i:0{padding_width}d}" for i in range(series_length)]

    # Get all unique classes
    all_classes = np.unique(np.concatenate([y_train, y_test])).astype(str)

    return X_train_2d, y_train, X_test_2d, y_test, feature_names, all_classes


def create_forest_params(args):
    """Create RandomForest parameters from command line arguments"""
    rf_params = {'random_state': args.random_state}

    # Basic parameters
    if args.n_estimators:
        rf_params['n_estimators'] = args.n_estimators
    if args.criterion:
        rf_params['criterion'] = args.criterion

    # Tree structure parameters
    if args.max_depth:
        rf_params['max_depth'] = args.max_depth
    if args.min_samples_split:
        rf_params['min_samples_split'] = args.min_samples_split
    if args.min_samples_leaf:
        rf_params['min_samples_leaf'] = args.min_samples_leaf
    if args.max_leaf_nodes:
        rf_params['max_leaf_nodes'] = args.max_leaf_nodes

    # Feature selection
    if args.max_features:
        rf_params['max_features'] = args.max_features

    # Split quality
    if args.min_impurity_decrease:
        rf_params['min_impurity_decrease'] = args.min_impurity_decrease

    # Sampling parameters
    if args.bootstrap:
        rf_params['bootstrap'] = (args.bootstrap == 'True')
    if args.max_samples:
        rf_params['max_samples'] = args.max_samples

    # Pruning
    if args.ccp_alpha:
        rf_params['ccp_alpha'] = args.ccp_alpha

    return rf_params


def get_rf_search_space(include_bootstrap=True):
    """
    Define the hyperparameter search space for Bayesian optimization

    Args:
        include_bootstrap: If True, includes bootstrap and max_samples in search space.
                          If False, excludes them to avoid constraint violations (default: True)
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize is not installed. Install with: pip install scikit-optimize")

    search_space = {
        # Number of trees
        'n_estimators': Integer(10, 300, name='n_estimators'),

        # Tree structure
        'max_depth': Integer(2, 50, name='max_depth'),
        'min_samples_split': Integer(2, 20, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 10, name='min_samples_leaf'),
        'max_leaf_nodes': Categorical([None, 10, 20, 30, 50, 100], name='max_leaf_nodes'),

        # Feature selection
        'max_features': Categorical(['sqrt', 'log2', None], name='max_features'),

        # Split quality
        'criterion': Categorical(['gini', 'entropy'], name='criterion'),
        'min_impurity_decrease': Real(0.0, 0.1, prior='uniform', name='min_impurity_decrease'),

        # Pruning
        'ccp_alpha': Real(0.0, 0.05, prior='uniform', name='ccp_alpha'),
    }

    # Only include bootstrap-related parameters if requested
    # Note: max_samples only works when bootstrap=True, so we keep bootstrap=True only
    if include_bootstrap:
        search_space['bootstrap'] = Categorical([True], name='bootstrap')  # Only True to avoid constraint
        search_space['max_samples'] = Categorical([None, 0.5, 0.7, 0.9], name='max_samples')

    return search_space


def optimize_rf_hyperparameters(X_train, y_train, search_space, n_iter=50, cv=5,
                                 n_jobs=-1, random_state=42, verbose=1,
                                 X_test=None, y_test=None, use_test_for_validation=False):
    """
    Perform Bayesian optimization to find best Random Forest hyperparameters

    Args:
        X_train: Training features
        y_train: Training labels
        search_space: Dictionary defining the search space for hyperparameters
        n_iter: Number of iterations for optimization (default: 50)
        cv: Number of cross-validation folds (default: 5)
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        X_test: Test features (optional, for validation)
        y_test: Test labels (optional, for validation)
        use_test_for_validation: If True, uses test set for validation instead of CV (default: False)

    Returns:
        best_params: Dictionary of best hyperparameters found
        best_score: Best cross-validation/test score achieved
        test_score: Test set score (if test data provided)
        optimizer: The fitted BayesSearchCV object (if CV used) or best RF model
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize is not installed. Install with: pip install scikit-optimize")

    print(f"[OPTIMIZE] Starting Bayesian Optimization for Random Forest hyperparameters")
    print(f"   Search space: {len(search_space)} hyperparameters")
    print(f"   Iterations: {n_iter}")

    if use_test_for_validation:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided when use_test_for_validation=True")
        print(f"   Validation: Test set ({X_test.shape[0]} samples)")
        print(f"   WARNING: Using test set for validation may lead to overfitting on test data!")
    else:
        print(f"   Validation: {cv}-fold cross-validation")

    print(f"   This may take a while...")

    if use_test_for_validation:
        # Manual optimization using test set for validation
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        # Convert search space to list format for gp_minimize
        dimensions = list(search_space.values())

        best_score = -np.inf
        best_params = None
        best_model = None

        @use_named_args(dimensions)
        def objective(**params):
            nonlocal best_score, best_params, best_model

            # Handle constraint: max_samples can only be used with bootstrap=True
            if 'bootstrap' in params and 'max_samples' in params:
                if not params['bootstrap'] and params['max_samples'] is not None:
                    params['max_samples'] = None

            # Train model with current parameters
            rf = RandomForestClassifier(**params, random_state=random_state)
            rf.fit(X_train, y_train)

            # Evaluate on test set
            score = rf.score(X_test, y_test)

            # Track best model
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = rf

            # Return negative score (gp_minimize minimizes)
            return -score

        print("\nRunning Bayesian optimization with test set validation...")
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iter,
            random_state=random_state,
            verbose=verbose > 0
        )

        print(f"\n[OK] Optimization complete!")
        print(f"   Best test score: {best_score:.4f}")
        print(f"   Best parameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

        return best_params, best_score, best_score, best_model

    else:
        # Standard cross-validation approach
        # Create base estimator
        rf = RandomForestClassifier(random_state=random_state)

        # BayesSearchCV doesn't automatically handle the bootstrap/max_samples constraint
        # We need to use a custom scorer or handle it differently
        # For now, we'll rely on sklearn's internal validation during fit

        # Create Bayesian optimizer
        optimizer = BayesSearchCV(
            estimator=rf,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            scoring='accuracy',
            return_train_score=True,
            error_score='raise'  # Raise errors instead of ignoring them
        )

        # Fit the optimizer
        print("\n[RUNNING] Running Bayesian optimization...")
        optimizer.fit(X_train, y_train)

        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Optionally evaluate on test set
        test_score = None
        if X_test is not None and y_test is not None:
            test_score = optimizer.best_estimator_.score(X_test, y_test)
            print(f"\n[OK] Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Test set score: {test_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")
        else:
            print(f"\n[OK] Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")

        return best_params, best_score, test_score, optimizer


def train_and_convert_forest(X_train, y_train, X_test, y_test, rf_params, feature_names,
                               test_split=None, random_state=42, sample_percentage=None):
    """
    Train Random Forest and convert to our format

    Args:
        X_train: Training features (from aeon)
        y_train: Training labels (from aeon)
        X_test: Test features (from aeon)
        y_test: Test labels (from aeon)
        rf_params: Random Forest parameters
        feature_names: List of feature names
        test_split: If provided (e.g., 0.3), combines train+test and does custom split.
                   If None, uses the original aeon train/test split (default behavior)
        random_state: Random seed for reproducibility
        sample_percentage: If provided (e.g., 0.05), uses only a percentage of the combined data

    Returns:
        sklearn_rf: Trained sklearn RandomForestClassifier
        our_forest: Converted Forest object
        final_X_train: The actual training data used (for saving to DB)
        final_y_train: The actual training labels used (for saving to DB)
    """
    print("Training Random Forest...")
    print(f"RF parameters: {rf_params}")

    if test_split is not None:
        # Custom split mode: combine train+test and re-split
        print(f"Custom split mode: combining datasets and splitting with test_size={test_split}")
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        # Apply sample percentage filtering if specified
        if sample_percentage is not None and sample_percentage < 100.0:
            print(f"Applying sample percentage filter: {sample_percentage}%")
            # Calculate how many samples to keep
            n_samples = len(X_combined)
            n_keep = int(n_samples * sample_percentage / 100.0)
            
            # Randomly select indices
            indices = np.random.choice(n_samples, size=n_keep, replace=False)
            
            # Filter the data
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            
            print(f"Reduced from {n_samples} to {len(X_combined)} samples ({sample_percentage}%)")

        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_combined, y_combined, test_size=test_split, random_state=random_state
        )
    else:
        # Default mode: use original aeon train/test split
        print("Using original aeon train/test split")
        X_train_final = X_train
        y_train_final = y_train
        X_test_final = X_test
        y_test_final = y_test

    print(f"Training set: {X_train_final.shape[0]} samples")
    print(f"Test set: {X_test_final.shape[0]} samples")

    # Train Random Forest
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_final, y_train_final)

    # Evaluate
    train_score = rf.score(X_train_final, y_train_final)
    test_score = rf.score(X_test_final, y_test_final)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

    # Convert to our Forest format
    print("Converting to custom Forest format...")
    our_forest = sklearn_forest_to_forest(rf, feature_names)
    print(f"Converted to Forest with {len(our_forest)} trees")

    return rf, our_forest, X_train_final, y_train_final


def store_training_set(connections, X_train, y_train, feature_names, dataset_name):
    """Store training set in DATA database under key TRAINING_SET"""
    print("Storing training set in DATA['TRAINING_SET']...")

    training_data = {
        'X_train': X_train.tolist(),  # Convert numpy array to list for JSON serialization
        'y_train': y_train.tolist(),
        'feature_names': feature_names,
        'dataset_name': dataset_name,
        'n_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'timestamp': datetime.datetime.now().isoformat()
    }

    try:
        connections['DATA'].set('TRAINING_SET', json.dumps(training_data))
        print(f"[OK] Training set saved successfully ({X_train.shape[0]} samples, {X_train.shape[1]} features)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save training set: {e}")
        return False


def store_forest_and_endpoints(connections, our_forest):
    """Store forest and extract/store feature thresholds"""
    # Store forest in Redis
    print("Storing Random Forest in DATA['RF']...")
    if store_forest(connections['DATA'], 'RF', our_forest):
        print("[OK] Forest saved successfully")
    else:
        raise Exception("Failed to save forest to Redis")

    # Extract and store feature thresholds (endpoints universe)
    print("Extracting feature thresholds...")
    feature_thresholds = our_forest.extract_feature_thresholds()
    print(f"Extracted thresholds for {len(feature_thresholds)} features")

    print("Storing endpoints universe in DATA['EU']...")
    if store_monotonic_dict(connections['DATA'], 'EU', feature_thresholds):
        print("Endpoints universe saved successfully")
    else:
        raise Exception("Failed to save endpoints universe to Redis")

    return feature_thresholds


def process_all_classified_samples(connections, dataset_name, class_label, our_forest, 
                                 X_test, y_test, feature_names, eu_data,  sample_percentage=None):
    """
    Process all test samples that are classified with the specified class label
    Store samples in DATA and their ICF representations in R
    
    If sample_percentage is provided, only process that percentage of samples
    """
    print(f"\n=== Processing All Samples Classified as '{class_label}' ===")
    
    # Find all test samples that are classified as the target class
    target_samples_data = []
    current_time = datetime.datetime.now().isoformat()
    
    # Apply sample percentage filtering if specified (preserve original indices)
    total_test_samples = len(X_test)
    if sample_percentage is not None and sample_percentage < 100.0:
        print(f"Applying sample percentage filter: {sample_percentage}% of {total_test_samples} test samples")
        n_keep = int(total_test_samples * sample_percentage / 100.0)
        indices = np.random.choice(total_test_samples, size=n_keep, replace=False)
        print(f"Reduced test samples from {total_test_samples} to {len(indices)} ({sample_percentage}%)")
    else:
        indices = np.arange(total_test_samples)
    
    for test_index in indices:
        sample = X_test[test_index]
        actual_label = y_test[test_index]
        sample_dict = sklearn_sample_to_dict(sample, feature_names)
        predicted_label = our_forest.predict(sample_dict)
        
        # Store ALL samples classified with the target label (regardless of correctness)
        if predicted_label == class_label:
            target_samples_data.append({
                'test_index': int(test_index),
                'sample_dict': sample_dict,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'prediction_correct': (predicted_label == actual_label)
            })
    
    print(f"Found {len(target_samples_data)} samples classified as '{class_label}'")
    
    if len(target_samples_data) == 0:
        print("[WARNING] No samples classified with the target label!")
        return [], {}
    
    # Store all samples and their ICF representations
    stored_samples = []
    correct_predictions = 0
    
    for idx, sample_data in enumerate(target_samples_data):
        sample_key = f"sample_{dataset_name}_{class_label}_{sample_data['test_index']}"
        
        # Store sample in DATA with full metadata
        data_entry = {
            'sample_dict': sample_data['sample_dict'],
            'predicted_label': sample_data['predicted_label'],
            'actual_label': sample_data['actual_label'],
            'test_index': sample_data['test_index'],
            'dataset_name': dataset_name,
            'timestamp': current_time,
            'prediction_correct': sample_data['prediction_correct'],
            
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
                'prediction_correct': sample_data['prediction_correct'],
                'timestamp': current_time,
            }
            
            connections['R'].set(icf_bitmap, json.dumps(icf_metadata))
            
            stored_samples.append({
                'sample_key': sample_key,
                'icf_bitmap': icf_bitmap,
                'prediction_correct': sample_data['prediction_correct'],
                'test_index': sample_data['test_index']#,
            })
            
            if sample_data['prediction_correct']:
                correct_predictions += 1
                
        except Exception as e:
            print(f"  Failed to process sample {idx}: {e}")
            continue
    
    # Store summary information
    summary = {
        'dataset_name': dataset_name,
        'target_class_label': class_label,
        'total_samples_processed': len(stored_samples),
        'total_test_samples': len(indices),
        'samples_with_target_label': len(target_samples_data),
        'correct_predictions': correct_predictions,
        'incorrect_predictions': len(stored_samples) - correct_predictions,
        'accuracy': correct_predictions / len(stored_samples) if len(stored_samples) > 0 else 0.0,
        'timestamp': current_time,
        'sample_keys': [s['sample_key'] for s in stored_samples]
    }
    
    connections['DATA'].set(f"summary_{dataset_name}_{class_label}", json.dumps(summary))
    
    print(f"[OK] Stored {len(stored_samples)} samples in DATA")
    print(f"[OK] Stored {len(stored_samples)} ICF representations in R")
    print(f"[OK] Correct predictions: {summary['correct_predictions']}")
    print(f"[OK] Incorrect predictions: {summary['incorrect_predictions']}")
    print(f"[OK] Accuracy: {summary['accuracy']:.3f}")
    print(f"[OK] Summary stored in DATA['summary_{dataset_name}_{class_label}']")

    return stored_samples, summary


def initialize_seed_candidate(connections, sample_dict, our_forest, eu_data, dataset_name, class_label):
    """Generate initial ICF bitmap and store in CAN and PR"""
    # print("Generating initial ICF and storing in CAN and PR...")

    # Extract ICF for the sample
    forest_icf = our_forest.extract_icf(sample_dict['sample_dict'])
    # print(f"ICF calculated for {len(forest_icf)} features")

    # Generate bitmap
    bitmap_mask = icf_to_bitmap_mask(forest_icf, eu_data)
    bitmap_string = bitmap_mask_to_string(bitmap_mask)

    # print(f"Generated bitmap with {len(bitmap_mask)} bits")

    # Store in CAN with timestamp
    current_timestamp = time.time()
    # Store ICF bitmap in R with metadata
    sample_key = f'sample_{dataset_name}_{class_label}_{sample_dict["test_index"]}_meta'
    icf_metadata = {
        'sample_key': sample_key,
        'dataset_name': dataset_name,
        'class_label': class_label,
        'test_index': sample_dict['test_index'],
        'prediction_correct': sample_dict['prediction_correct'],        
        'timestamp': current_timestamp,
    }
    connections['CAN'].set(bitmap_string, json.dumps(icf_metadata))
    # print(f"Stored initial candidate in CAN")

    # Also store in PR (Preferred Reasons) database - timestamp auto-generated
    if insert_to_pr(connections['PR'], bitmap_string, current_timestamp, icf_metadata):
        print(f"Stored initial candidate in PR")
    else:
        print(f"Warning: Failed to store candidate in PR")

    return bitmap_string, forest_icf


def main():
    parser = argparse.ArgumentParser(
        description="Initialize random path worker system with aeon univariate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python init_aeon_univariate.py --list-datasets

  # Initialize ECG200 with default parameters, process samples labeled "1"
  python init_aeon_univariate.py ECG200 --class-label "1" --redis-port 6380
  keydb-server --port 6380
  redis-commander --redis-port 6380 --port 8081
  # Custom forest parameters
  python init_aeon_univariate.py Coffee --class-label "0" --n-estimators 100 --max-depth 5

  # Use Bayesian optimization with cross-validation (recommended)
  python init_aeon_univariate.py ECG200 --class-label "1" --optimize-rf --opt-n-iter 30

  # Use Bayesian optimization with test set validation (use with caution!)
  python init_aeon_univariate.py ECG200 --class-label "1" --optimize-rf --opt-use-test

  # Custom train/test split with optimization
  python init_aeon_univariate.py Coffee --class-label "0" --test-split 0.3 --optimize-rf

  # Process only 5% of samples with sample percentage
  python init_aeon_univariate.py ECG200 --class-label "1" --sample-percentage 5

  # Process only 0.05% of samples (as in your example)
  python init_aeon_univariate.py ECG200 --class-label "1" --sample-percentage 0.05
        """
    )
    
    parser.add_argument('dataset_name', nargs='?', 
                       help='Name of the aeon univariate dataset to load')
    
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available aeon univariate datasets')
    
    parser.add_argument('--class-label', type=str, required=False,
                       help='Target class label to process (required if dataset_name is provided)')
    
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
    opt_group.add_argument('--optimize-rf', action='store_true',
                          help='Use Bayesian optimization to find best RF hyperparameters (requires scikit-optimize)')
    opt_group.add_argument('--opt-n-iter', type=int, default=50,
                          help='Number of iterations for Bayesian optimization (default: 50)')
    opt_group.add_argument('--opt-cv', type=int, default=5,
                          help='Number of cross-validation folds for optimization (default: 5)')
    opt_group.add_argument('--opt-n-jobs', type=int, default=-1,
                          help='Number of parallel jobs for optimization, -1 for all cores (default: -1)')
    opt_group.add_argument('--opt-use-test', action='store_true',
                          help='Use test set for validation during optimization instead of CV. '
                               'WARNING: May lead to overfitting on test data! Use with caution.')
    
    # New parameters for sample percentage filtering
    parser.add_argument('--sample-percentage', type=float, default=100.0,
                       help='Process only this percentage of samples (0-100, default: 100)')
    parser.add_argument('--test-split', type=float, default=None,
                       help='If specified, combines train+test and does custom split with this test fraction (e.g., 0.3). '
                            'If not specified, uses original aeon train/test split (default: None)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--feature-prefix', type=str, default='t',
                       help='Prefix for feature names (default: "t")')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not clean Redis databases before initialization')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis/KeyDB server port (default: 6379)')
    
    args = parser.parse_args()
    
    # Handle list datasets
    if args.list_datasets:
        list_available_datasets()
        return
    
    # Validate arguments
    if not args.dataset_name:
        parser.error("dataset_name is required (or use --list-datasets)")
    
    if not args.info and not args.class_label:
        parser.error("--class-label is required when processing a dataset")
    
    # Show dataset info if requested
    if args.info:
        print(f"Getting information for dataset: {args.dataset_name}")
        info = get_dataset_info(args.dataset_name)
        if 'error' in info:
            print(f"[ERROR] Error loading dataset: {info['error']}")
            print("Make sure the dataset name is correct and aeon is installed.")
            return
        
        print(f"\n[INFO] Dataset Information: {args.dataset_name}")
        print(f"  Training samples: {info['train_size']}")
        print(f"  Test samples: {info['test_size']}")
        print(f"  Series length: {info['series_length']}")
        print(f"  Number of channels: {info['n_channels']}")
        print(f"  Number of classes: {info['n_classes']}")
        print(f"  Classes: {info['classes']}")
        
        if args.class_label:
            if args.class_label in [str(c) for c in info['classes']]:
                print(f"[OK] Target class label '{args.class_label}' is valid")
            else:
                print(f"[ERROR] Target class label '{args.class_label}' not found in dataset classes")
                print(f"   Available classes: {info['classes']}")
        
        return
    
    print(f"[START] Initializing Random Path Worker System")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] Target Class Label: {args.class_label}")
    print(f"[INFO] Sample Percentage: {args.sample_percentage}%")

    try:
        # Connect to Redis
        connections, db_mapping = connect_redis(port=args.redis_port)
        
        # Clean databases if requested
        if not args.no_clean:
            print("Cleaning Redis databases...")
            clean_all_databases(connections, db_mapping)
        
        # Load and prepare dataset
        X_train, y_train, X_test, y_test, feature_names, class_names = load_and_prepare_dataset(
            args.dataset_name, args.feature_prefix
        )


        # Determine which training data to use for optimization
        # If test_split is specified, we'll combine and split later in train_and_convert_forest
        # For optimization, we use the original training set
        X_train_for_opt = X_train
        y_train_for_opt = y_train

        # Optionally optimize RF hyperparameters with Bayesian optimization
        if args.optimize_rf:
            if not SKOPT_AVAILABLE:
                print("[ERROR] Error: scikit-optimize is not installed.")
                print("   Install with: pip install scikit-optimize")
                return 1

            print("\n" + "="*70)
            print("BAYESIAN OPTIMIZATION MODE")
            print("="*70)

            # Get search space
            search_space = get_rf_search_space()

            # Run Bayesian optimization
            best_params, best_score, test_score, optimizer = optimize_rf_hyperparameters(
                X_train_for_opt, y_train_for_opt,
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

            # Store optimization results in DATA for future reference
            opt_results = {
                'best_params': best_params,
                'best_cv_score': best_score if not args.opt_use_test else None,
                'test_score': test_score,
                'used_test_for_validation': args.opt_use_test,
                'n_iter': args.opt_n_iter,
                'cv_folds': args.opt_cv if not args.opt_use_test else None,
                'dataset_name': args.dataset_name,
                'timestamp': datetime.datetime.now().isoformat()
            }
            # Convert numpy types to native Python types for JSON serialization
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
            test_split=args.test_split, random_state=args.random_state,
            sample_percentage=args.sample_percentage
        )

        # Store training set in DATA database
        store_training_set(connections, X_train_used, y_train_used, feature_names, args.dataset_name)

        # Store forest and endpoints
        eu_data = store_forest_and_endpoints(connections, our_forest)
        
        # Process all test samples classified with target label
        stored_samples, summary = process_all_classified_samples(
            connections, args.dataset_name, args.class_label,
            our_forest, X_test, y_test, feature_names, eu_data,  args.sample_percentage
        )
        
        # Initialize seed candidates 
        for sample in stored_samples:
            sample_key = sample['sample_key']
            sample_dict = json.loads(connections['DATA'].get(sample_key + "_meta"))
            seed_bitmap, seed_icf = initialize_seed_candidate(
                connections, sample_dict, our_forest, eu_data, args.dataset_name, args.class_label
            )
        
        # Store the target label for worker compatibility
        connections['DATA'].set('label', args.class_label)
        print(f"  Target label '{args.class_label}' set for worker processing")
        
        print(f"\n Successfully initialized {args.dataset_name}")
        print(f" Forest: {len(our_forest)} trees")
        print(f" Features: {len(feature_names)}")
        print(f" Processed: {summary['total_samples_processed']} samples with label '{args.class_label}'")
        print(f" Correct: {summary['correct_predictions']}")
        print(f" Incorrect: {summary['incorrect_predictions']}")
        print(f" Accuracy: {summary['accuracy']:.3f}")
        print(f" Data stored in Redis databases")
        
        print(f"\n System Ready for Worker Processing!")
        print(f"  Next step: run python worker_rcheck.py")
        
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
