import json
import math
from typing import Dict, List, Union

def validate_monotonic_array(arr: List[float], key_name: str) -> bool:
    """Validate that array starts with -inf, ends with +inf, and is monotonically increasing"""
    if not isinstance(arr, list):
        raise TypeError(f"Array for key '{key_name}' must be a list")

    if len(arr) < 2:
        raise ValueError(f"Array for key '{key_name}' must have at least 2 elements")

    if not math.isinf(arr[0]) or arr[0] >= 0:
        raise ValueError(f"Array for key '{key_name}' must start with -infinity")

    if not math.isinf(arr[-1]) or arr[-1] <= 0:
        raise ValueError(f"Array for key '{key_name}' must end with +infinity")

    # Check monotonically increasing
    for i in range(1, len(arr)):
        if arr[i] <= arr[i-1]:
            raise ValueError(f"Array for key '{key_name}' must be monotonically increasing at index {i}")

    return True

def store_monotonic_dict(redis_conn, key: str, data: Dict[str, List[float]]) -> bool:
    """Store a dictionary of key:array pairs where arrays are monotonically increasing from -inf to +inf"""
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")

    # Validate all arrays
    for dict_key, array in data.items():
        if not isinstance(dict_key, str):
            raise TypeError(f"Dictionary key '{dict_key}' must be a string")
        validate_monotonic_array(array, dict_key)

    try:
        # Convert to JSON string and store
        json_data = json.dumps(data)
        redis_conn.set(key, json_data)
        print(f"Successfully stored dictionary with {len(data)} keys in Redis key '{key}'")
        return True
    except Exception as e:
        print(f"Error storing data: {e}")
        return False

def retrieve_monotonic_dict(redis_conn, key: str) -> Union[Dict[str, List[float]], None]:
    """Retrieve a dictionary of monotonic arrays from Redis"""
    try:
        json_data = redis_conn.get(key)
        if json_data is None:
            print(f"No data found for key '{key}'")
            return None

        data = json.loads(json_data)

        # Validate retrieved data
        if not isinstance(data, dict):
            raise ValueError("Retrieved data is not a dictionary")

        for dict_key, array in data.items():
            validate_monotonic_array(array, dict_key)

        print(f"Successfully retrieved dictionary with {len(data)} keys from Redis key '{key}'")
        return data

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None