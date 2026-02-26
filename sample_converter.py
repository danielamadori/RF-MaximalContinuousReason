"""
Sample converter for sklearn data to dictionary format.
Handles conversion between sklearn array format and our dictionary representation.
"""
import numpy as np
from typing import List, Dict, Union, Optional


def sklearn_sample_to_dict(sample: Union[np.ndarray, List],
                          feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Convert sklearn sample (numpy array or list) to dictionary format.

    Args:
        sample: Input sample as numpy array or list of feature values
        feature_names: List of feature names. If None, uses "feature_0", "feature_1", etc.

    Returns:
        Dictionary with feature_name: float_value pairs

    Raises:
        ValueError: If sample is empty or feature_names length doesn't match sample length
    """
    # Convert to numpy array if it's a list
    if isinstance(sample, list):
        sample = np.array(sample)
    elif not isinstance(sample, np.ndarray):
        raise ValueError(f"sample must be numpy array or list, got {type(sample)}")

    # Handle 1D and 2D arrays (flatten if needed)
    if sample.ndim == 0:
        raise ValueError("sample cannot be a scalar")
    elif sample.ndim > 2:
        raise ValueError(f"sample must be 1D or 2D array, got {sample.ndim}D")
    elif sample.ndim == 2:
        if sample.shape[0] != 1:
            raise ValueError(f"2D sample must have shape (1, n_features), got {sample.shape}")
        sample = sample.flatten()

    n_features = len(sample)
    if n_features == 0:
        raise ValueError("sample cannot be empty")

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match "
                        f"sample length ({n_features})")

    # Convert to dictionary with float values
    result = {}
    for i, (name, value) in enumerate(zip(feature_names, sample)):
        try:
            result[name] = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert feature '{name}' at index {i} to float: {e}")

    return result


def dict_to_sklearn_sample(sample_dict: Dict[str, float],
                          feature_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert dictionary format back to sklearn sample (numpy array).

    Args:
        sample_dict: Dictionary with feature_name: float_value pairs
        feature_names: Ordered list of feature names. If None, uses sorted keys from dict

    Returns:
        Numpy array with feature values in the specified order

    Raises:
        ValueError: If feature_names contains keys not in sample_dict
    """
    if not isinstance(sample_dict, dict):
        raise ValueError(f"sample_dict must be a dictionary, got {type(sample_dict)}")

    if len(sample_dict) == 0:
        raise ValueError("sample_dict cannot be empty")

    # Use sorted keys if no feature names provided
    if feature_names is None:
        feature_names = sorted(sample_dict.keys())

    # Check that all feature names exist in the dictionary
    missing_features = set(feature_names) - set(sample_dict.keys())
    if missing_features:
        raise ValueError(f"Features not found in sample_dict: {sorted(missing_features)}")

    # Extract values in the specified order
    values = []
    for name in feature_names:
        try:
            value = float(sample_dict[name])
            values.append(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert feature '{name}' to float: {e}")

    return np.array(values)


def batch_sklearn_samples_to_dict(samples: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """
    Convert batch of sklearn samples to list of dictionaries.

    Args:
        samples: 2D numpy array of shape (n_samples, n_features)
        feature_names: List of feature names. If None, uses "feature_0", "feature_1", etc.

    Returns:
        List of dictionaries, one for each sample

    Raises:
        ValueError: If samples is not 2D or feature_names length doesn't match n_features
    """
    if not isinstance(samples, np.ndarray):
        raise ValueError(f"samples must be numpy array, got {type(samples)}")

    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D array, got {samples.ndim}D")

    n_samples, n_features = samples.shape

    if n_features == 0:
        raise ValueError("samples cannot have 0 features")

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match "
                        f"n_features ({n_features})")

    # Convert each sample
    result = []
    for i in range(n_samples):
        sample_dict = sklearn_sample_to_dict(samples[i], feature_names)
        result.append(sample_dict)

    return result


def batch_dict_to_sklearn_samples(sample_dicts: List[Dict[str, float]],
                                 feature_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert list of dictionary samples back to sklearn batch format.

    Args:
        sample_dicts: List of dictionaries with feature_name: float_value pairs
        feature_names: Ordered list of feature names. If None, uses sorted keys from first dict

    Returns:
        2D numpy array of shape (n_samples, n_features)

    Raises:
        ValueError: If sample_dicts is empty or dictionaries have inconsistent keys
    """
    if not isinstance(sample_dicts, list):
        raise ValueError(f"sample_dicts must be a list, got {type(sample_dicts)}")

    if len(sample_dicts) == 0:
        raise ValueError("sample_dicts cannot be empty")

    # Use sorted keys from first dictionary if no feature names provided
    if feature_names is None:
        feature_names = sorted(sample_dicts[0].keys())

    # Validate that all dictionaries have the same keys
    expected_keys = set(feature_names)
    for i, sample_dict in enumerate(sample_dicts):
        if set(sample_dict.keys()) != expected_keys:
            raise ValueError(f"Sample {i} has inconsistent keys. Expected: {sorted(expected_keys)}, "
                           f"got: {sorted(sample_dict.keys())}")

    # Convert each dictionary to array
    result = []
    for sample_dict in sample_dicts:
        sample_array = dict_to_sklearn_sample(sample_dict, feature_names)
        result.append(sample_array)

    return np.array(result)