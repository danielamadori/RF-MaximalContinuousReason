"""
General helper utilities used across the project.
"""
from typing import Dict, Any, Optional, List
import numpy as np

# Handle numpy bool type compatibility
if hasattr(np, "bool8"):
    _NUMPY_BOOL_TYPES = (np.bool_, np.bool8, bool)
else:
    _NUMPY_BOOL_TYPES = (np.bool_, bool)


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

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


def increment_info_counter(info: Optional[Dict[str, Any]], key: str, amount: int = 1, prefix: str = ""):
    """
    Helper to increment info counters with optional prefix.
    
    Args:
        info: Optional dictionary to store statistics (None is safe, does nothing)
        key: Counter key
        amount: Amount to increment (default: 1)
        prefix: Optional prefix for the key (default: "")
        
    Example:
        >>> info = {}
        >>> increment_info_counter(info, 'count', 1)
        >>> increment_info_counter(info, 'hits', 1, prefix='cache_')
        >>> info
        {'count': 1, 'cache_hits': 1}
    """
    if info is not None:
        full_key = f"{prefix}{key}" if prefix else key
        info[full_key] = info.get(full_key, 0) + amount


def parse_sample_indices(index_str: str) -> List[int]:
    """
    Parse a string of indices and ranges into a list of integers.
    Supports comma-separated values and ranges (e.g., '1,3-5,8').
    
    Args:
        index_str: String containing indices or ranges
        
    Returns:
        List of unique sorted integers
        
    Raises:
        ValueError: If the string format is invalid
    """
    if not index_str:
        return []
        
    indices = set()
    parts = str(index_str).split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            # Handle range (e.g. "3-5")
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                # Inclusive range
                indices.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: '{part}'. Expected 'start-end' with integers.")
        else:
            # Handle single integer
            try:
                indices.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid index format: '{part}'. Expected integer.")
                
    return sorted(list(indices))