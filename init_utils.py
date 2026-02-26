"""
Initialization utilities shared between init scripts.

Functions for storing training data, processing samples, and initializing
candidate reasons in Redis.
"""
import json
import time
import datetime
import numpy as np

from redis_helpers.forest import store_forest
from redis_helpers.endpoints import store_monotonic_dict
from redis_helpers.samples import store_sample
from redis_helpers.preferred import insert_to_pr
from sample_converter import sklearn_sample_to_dict
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string


def store_training_set(connections, X_train, y_train, feature_names, dataset_name, dataset_type='generic'):
    """
    Store training set in DATA database under key TRAINING_SET.
    
    Args:
        connections: Redis connections dict
        X_train: Training features array
        y_train: Training labels array
        feature_names: List of feature names
        dataset_name: Name of the dataset
        dataset_type: Type of dataset ('uci', 'pmlb', 'openml', 'baseline', etc.)
        
    Returns:
        bool: True if successful
    """
    print("Storing training set in DATA['TRAINING_SET']...")

    training_data = {
        'X_train': X_train.tolist(),
        'y_train': y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
        'feature_names': feature_names,
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
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


def store_dataset_total_samples(connections, total_samples: int):
    """
    Store the total number of test samples in Redis DATA.

    Args:
        connections: Redis connections dict
        total_samples: Number of test samples
    """
    try:
        connections['DATA'].set('dataset_total_samples', int(total_samples))
        print(f"[OK] Total test samples stored: {total_samples}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store total samples: {e}")
    return False


def store_forest_and_endpoints(connections, our_forest):
    """
    Store forest and extract/store feature thresholds (endpoints universe).
    
    Args:
        connections: Redis connections dict
        our_forest: Custom Forest object
        
    Returns:
        dict: Feature thresholds (endpoints universe)
    """
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
        print("[OK] Endpoints universe saved successfully")
    else:
        raise Exception("Failed to save endpoints universe to Redis")

    return feature_thresholds


def process_all_classified_samples(
    connections,
    dataset_name,
    class_label,
    our_forest,
    X_test,
    y_test,
    feature_names,
    eu_data,
    sample_percentage=None,
    dataset_type='generic',
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
        y_test: Test labels array
        feature_names: List of feature names
        eu_data: Endpoints universe data
        sample_percentage: Optional percentage of samples to process
        dataset_type: Type of dataset ('uci', 'pmlb', 'openml', 'baseline', etc.)
        
    Returns:
        tuple: (stored_samples list, summary dict)
    """
    print(f"\n=== Processing All Samples Classified as '{class_label}' ===")
    
    # Find all test samples that are classified as the target class
    target_samples_data = []
    current_time = datetime.datetime.now().isoformat()
    
    # Apply sample percentage filtering if specified
    total_test_samples = len(X_test)
    if sample_percentage is not None and sample_percentage < 100.0:
        print(f"Applying sample percentage filter: {sample_percentage}% of {total_test_samples} test samples")
        n_keep = max(1, int(total_test_samples * sample_percentage / 100.0))
        indices = np.random.choice(total_test_samples, size=n_keep, replace=False)
        X_test_filtered = X_test[indices]
        y_test_filtered = y_test[indices] if hasattr(y_test, '__getitem__') else np.array(y_test)[indices]
        print(f"Reduced test samples from {total_test_samples} to {len(X_test_filtered)} ({sample_percentage}%)")
    else:
        X_test_filtered = X_test
        y_test_filtered = y_test
    
    for i, (sample, actual_label) in enumerate(zip(X_test_filtered, y_test_filtered)):
        sample_dict = sklearn_sample_to_dict(sample, feature_names)
        predicted_label = our_forest.predict(sample_dict)
        
        # Store ALL samples classified with the target label (regardless of correctness)
        if predicted_label == class_label:
            target_samples_data.append({
                'test_index': i,
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
        sample_key = f"sample_{dataset_name}_{class_label}_{idx}"
        
        # Store sample in DATA with full metadata
        data_entry = {
            'sample_dict': sample_data['sample_dict'],
            'predicted_label': sample_data['predicted_label'],
            'actual_label': sample_data['actual_label'],
            'test_index': sample_data['test_index'],
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'timestamp': current_time,
            'prediction_correct': sample_data['prediction_correct']
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
                'dataset_type': dataset_type,
                'class_label': class_label,
                'test_index': sample_data['test_index'],
                'prediction_correct': sample_data['prediction_correct'],
                'timestamp': current_time
            }
            
            connections['R'].set(icf_bitmap, json.dumps(icf_metadata))
            
            stored_samples.append({
                'sample_key': sample_key,
                'icf_bitmap': icf_bitmap,
                'prediction_correct': sample_data['prediction_correct'],
                'test_index': sample_data['test_index']
            })
            
            if sample_data['prediction_correct']:
                correct_predictions += 1
                
        except Exception as e:
            print(f"[WARNING] Failed to process sample {idx}: {e}")
            continue
    
    # Store summary information
    summary = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'target_class_label': class_label,
        'total_samples_processed': len(stored_samples),
        'total_test_samples': len(X_test_filtered),
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


def initialize_seed_candidate(connections, sample_dict, our_forest, eu_data):
    """
    Generate initial ICF bitmap and store in CAN and PR.
    
    Args:
        connections: Redis connections dict
        sample_dict: Sample data dictionary with 'sample_dict', 'test_index', 'prediction_correct'
        our_forest: Custom Forest object
        eu_data: Endpoints universe data
        
    Returns:
        tuple: (bitmap_string, forest_icf)
    """
    print("Generating initial ICF and storing in CAN and PR...")

    # Extract ICF for the sample
    forest_icf = our_forest.extract_icf(sample_dict['sample_dict'])
    print(f"ICF calculated for {len(forest_icf)} features")

    # Generate bitmap
    bitmap_mask = icf_to_bitmap_mask(forest_icf, eu_data)
    bitmap_string = bitmap_mask_to_string(bitmap_mask)

    print(f"Generated bitmap with {len(bitmap_mask)} bits")

    # Store in CAN with timestamp
    current_timestamp = time.time()
    icf_metadata = {
        'test_index': sample_dict['test_index'],
        'timestamp': current_timestamp
    }
    connections['CAN'].set(bitmap_string, json.dumps(icf_metadata))
    print(f"[OK] Stored initial candidate in CAN")

    # Also store in PR (Preferred Reasons) database
    if insert_to_pr(connections['PR'], bitmap_string, current_timestamp, icf_metadata):
        print(f"[OK] Stored initial candidate in PR")
    else:
        print(f"[WARNING] Failed to store candidate in PR")

    return bitmap_string, forest_icf
