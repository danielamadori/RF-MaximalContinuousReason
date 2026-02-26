"""
Redis helper functions for storing and retrieving sample representations.
"""
import json
import redis
from typing import Dict, List, Optional, Any


def store_sample(redis_conn: redis.Redis, key: str, sample_dict: Dict[str, float]) -> bool:
    """
    Store a single sample dictionary in Redis.

    Args:
        redis_conn: Redis connection object
        key: Redis key to store the sample under
        sample_dict: Dictionary with feature_name: float_value pairs

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input
        if not isinstance(sample_dict, dict):
            raise ValueError(f"sample_dict must be a dictionary, got {type(sample_dict)}")

        if len(sample_dict) == 0:
            raise ValueError("sample_dict cannot be empty")

        # Validate that all values can be converted to float
        validated_dict = {}
        for feature, value in sample_dict.items():
            try:
                validated_dict[feature] = float(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert feature '{feature}' to float: {e}")

        # Store in Redis as JSON
        redis_conn.set(key, json.dumps(validated_dict))

        print(f"Successfully stored sample with {len(validated_dict)} features in Redis key '{key}'")
        return True

    except Exception as e:
        print(f"Error storing sample in Redis key '{key}': {e}")
        return False


def retrieve_sample(redis_conn: redis.Redis, key: str) -> Optional[Dict[str, float]]:
    """
    Retrieve a single sample dictionary from Redis.

    Args:
        redis_conn: Redis connection object
        key: Redis key to retrieve the sample from

    Returns:
        Dict with feature_name: float_value pairs if successful, None otherwise
    """
    try:
        # Get data from Redis
        data_str = redis_conn.get(key)

        if data_str is None:
            print(f"No sample found for key '{key}'")
            return None

        # Parse JSON
        sample_dict = json.loads(data_str)

        # Validate structure
        if not isinstance(sample_dict, dict):
            raise ValueError("Invalid sample data structure")

        print(f"Successfully retrieved sample with {len(sample_dict)} features from Redis key '{key}'")
        return sample_dict

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Redis key '{key}': {e}")
        return None
    except Exception as e:
        print(f"Error retrieving sample from Redis key '{key}': {e}")
        return None


def store_sample_batch(redis_conn: redis.Redis, base_key: str,
                      sample_dicts: List[Dict[str, float]]) -> bool:
    """
    Store a batch of sample dictionaries in Redis using indexed keys.

    Args:
        redis_conn: Redis connection object
        base_key: Base key for storing samples (will be suffixed with index)
        sample_dicts: List of dictionaries with feature_name: float_value pairs

    Returns:
        bool: True if all samples stored successfully, False otherwise
    """
    try:
        if not isinstance(sample_dicts, list):
            raise ValueError(f"sample_dicts must be a list, got {type(sample_dicts)}")

        if len(sample_dicts) == 0:
            raise ValueError("sample_dicts cannot be empty")

        # Store each sample
        success_count = 0
        for i, sample_dict in enumerate(sample_dicts):
            sample_key = f"{base_key}_{i}"
            if store_sample(redis_conn, sample_key, sample_dict):
                success_count += 1

        # Store batch metadata
        batch_metadata = {
            'num_samples': len(sample_dicts),
            'base_key': base_key,
            'success_count': success_count
        }

        metadata_key = f"{base_key}_batch_info"
        redis_conn.set(metadata_key, json.dumps(batch_metadata))

        if success_count == len(sample_dicts):
            print(f"Successfully stored batch of {success_count} samples with base key '{base_key}'")
            return True
        else:
            print(f"Partially stored batch: {success_count}/{len(sample_dicts)} samples with base key '{base_key}'")
            return False

    except Exception as e:
        print(f"Error storing sample batch with base key '{base_key}': {e}")
        return False


def retrieve_sample_batch(redis_conn: redis.Redis, base_key: str) -> Optional[List[Dict[str, float]]]:
    """
    Retrieve a batch of sample dictionaries from Redis.

    Args:
        redis_conn: Redis connection object
        base_key: Base key used when storing the batch

    Returns:
        List of sample dictionaries if successful, None otherwise
    """
    try:
        # Get batch metadata first
        metadata_key = f"{base_key}_batch_info"
        metadata_str = redis_conn.get(metadata_key)

        if metadata_str is None:
            print(f"No batch metadata found for base key '{base_key}'")
            return None

        batch_metadata = json.loads(metadata_str)
        num_samples = batch_metadata['num_samples']

        # Retrieve individual samples
        samples = []
        for i in range(num_samples):
            sample_key = f"{base_key}_{i}"
            sample_dict = retrieve_sample(redis_conn, sample_key)
            if sample_dict is not None:
                samples.append(sample_dict)

        print(f"Successfully retrieved {len(samples)}/{num_samples} samples with base key '{base_key}'")
        return samples

    except json.JSONDecodeError as e:
        print(f"Error parsing batch metadata JSON from Redis key '{metadata_key}': {e}")
        return None
    except Exception as e:
        print(f"Error retrieving sample batch with base key '{base_key}': {e}")
        return None