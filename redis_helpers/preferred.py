"""
Redis helper functions for Preferred Reasons (PR) database.
"""
import redis
import json
import datetime
from typing import Optional, Set, List, Tuple


def get_pr_candidate(pr_connection: redis.Redis, scan_count: int = 100) -> Optional[Tuple[str, Set[str]]]:
    """
    Get the best candidate from PR database based on selection criteria:
    1. Minimum number of timestamps in set
    2. If tie, minimum maximum timestamp (earliest last access)
    
    Uses SCAN to iterate through keys without blocking.
    
    Args:
        pr_connection: Redis connection to PR database
        scan_count: Number of keys to fetch per SCAN iteration
        
    Returns:
        Tuple of (key, set of timestamps) or None if PR is empty
    """
    try:
        # Build list of (key, timestamps_set, max_timestamp) using SCAN
        candidates = []
        cursor = 0
        
        while True:
            cursor, keys = pr_connection.scan(cursor, match='*', count=scan_count)
            
            for key in keys:
                value = pr_connection.get(key)
                if value:
                    data = json.loads(value)
                    timestamps = None
                    
                    if isinstance(data, dict):
                        timestamps = data.get('timestamp')
                    else:
                        timestamps = data
                        
                    if isinstance(timestamps, list):
                        timestamps_set = set(timestamps)
                    elif isinstance(timestamps, str):
                        timestamps_set = set([timestamps])
                    else:
                        timestamps_set = set()
                    
                    if len(timestamps_set) > 0:
                        max_timestamp = max(timestamps_set)
                        candidates.append((key, timestamps_set, len(timestamps_set), max_timestamp))
            
            if cursor == 0:
                # Completed full scan
                break
        
        if not candidates:
            return None
        
        # Sort by: 1) number of timestamps (ascending), 2) max timestamp (ascending)
        candidates.sort(key=lambda x: (x[2], x[3]))
        
        # Return the best candidate
        best_key, best_timestamps, _, _ = candidates[0]
        return (best_key, best_timestamps)
        
    except Exception as e:
        print(f"Error getting PR candidate: {e}")
        return None


def get_pr_candidate_exclusive(
    pr_connection: redis.Redis,
    data_connection: redis.Redis,
    scan_count: int = 100,
    worker_name: Optional[str] = None,
) -> Optional[str]:
    """
    Get a PR candidate exclusively by acquiring a lock in DATA using SET NX.

    Args:
        pr_connection: Redis connection to PR database
        data_connection: Redis connection to DATA database (lock storage)
        scan_count: Number of keys to fetch per SCAN iteration
        worker_name: Optional worker name to store in hash on successful lock

    Returns:
        Key if lock acquired, otherwise None
    """
    try:
        # Get synchronized timestamp from Redis
        redis_time = data_connection.time()
        vts = f"{redis_time[0]}.{redis_time[1]}"

        cursor = 0
        while True:
            cursor, keys = pr_connection.scan(cursor, match='*', count=scan_count)

            for key in keys:
                if data_connection.hsetnx(key, "vts", vts):
                    if worker_name is not None:
                        data_connection.hset(key, "worker", worker_name)
                    return key

            if cursor == 0:
                break

        return None

    except Exception as e:
        print(f"Error getting exclusive PR candidate: {e}")
        return None


def close_pr_candidate(
    data_connection: redis.Redis,
    key: str,
    coverage_value: float,
    globally_dominated: bool = False,
    timeout_occurred: bool = False
) -> bool:
    """
    Close a PR candidate by setting the vte (end timestamp) in DATA.

    Args:
        data_connection: Redis connection to DATA database
        key: The candidate key to close
        coverage_value: The coverage value to set

    Returns:
        True if successful, False otherwise
    """
    try:
        redis_time = data_connection.time()
        vte = f"{redis_time[0]}.{redis_time[1]}"
        data_connection.hset(key, "vte", vte)
        data_connection.hset(key, "coverage", f"{coverage_value}")
        data_connection.hset(key, "globally_dominated", f"{globally_dominated}")
        data_connection.hset(key, "timeout_occurred", f"{timeout_occurred}")
        return True
    except Exception as e:
        print(f"Error closing PR candidate: {e}")
        return False


def add_timestamp_to_pr(pr_connection: redis.Redis, key: str, timestamp: str, icf_metadata: dict = {}) -> bool:
    """
    Add a timestamp to a PR key's set of timestamps.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key
        timestamp: ISO format timestamp string
        
    Returns:
        bool: True if successful
    """
    try:
        value = pr_connection.get(key)
        data = {}
        timestamps_set = set()
        
        if value:
            loaded_data = json.loads(value)
            if isinstance(loaded_data, dict):
                data = loaded_data
                timestamps = data.get('timestamp')
            else:
                timestamps = loaded_data
                
            if isinstance(timestamps, list):
                timestamps_set = set(timestamps)
            elif isinstance(timestamps, str):
                timestamps_set = set([timestamps])
                
        timestamps_set.add(timestamp)
        
        # Update data with new timestamps list
        data['timestamp'] = list(timestamps_set)
        
        # Merge with provided metadata if any
        if icf_metadata:
            data.update(icf_metadata)
            
        pr_connection.set(key, json.dumps(data))
        return True
        
    except Exception as e:
        print(f"Error adding timestamp to PR: {e}")
        return False


def remove_from_pr(pr_connection: redis.Redis, key: str) -> bool:
    """
    Remove a key from PR database.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key to remove
        
    Returns:
        bool: True if successful
    """
    try:
        result = pr_connection.delete(key)
        return result > 0
    except Exception as e:
        print(f"Error removing from PR: {e}")
        return False


def insert_to_pr(pr_connection: redis.Redis, key: str, timestamp: Optional[str] = None, icf_metadata: dict = None) -> bool:
    """
    Insert a new key into PR database with initial timestamp.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key
        timestamp: ISO format timestamp (default: current time)
        
    Returns:
        bool: True if successful
    """
    try:
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            if icf_metadata is None:
                icf_metadata = {}
            icf_metadata['timestamp'] = timestamp 
        
        if icf_metadata is None:
            icf_metadata = {}
            
        pr_connection.set(key, json.dumps(icf_metadata))
        return True
        
    except Exception as e:
        print(f"Error inserting to PR: {e}")
        return False


def count_pr_keys(pr_connection: redis.Redis, scan_count: int = 100) -> int:
    """
    Count the number of keys in PR database using SCAN.
    
    Args:
        pr_connection: Redis connection to PR database
        scan_count: Number of keys to fetch per SCAN iteration
        
    Returns:
        int: Number of keys in PR database
    """
    try:
        count = 0
        cursor = 0
        
        while True:
            cursor, keys = pr_connection.scan(cursor, match='*', count=scan_count)
            count += len(keys)
            
            if cursor == 0:
                break
        
        return count
        
    except Exception as e:
        print(f"Error counting PR keys: {e}")
        return 0


def add_candidate_metadata(
    data_connection: redis.Redis,
    key: str,
    field: str,
    value: str
) -> bool:
    """
    Add metadata field to a candidate's hash in DATA database.
    
    Args:
        data_connection: Redis connection to DATA database
        key: The candidate key (bitmap string)
        field: The metadata field name
        value: The metadata value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data_connection.hset(key, field, value)
        return True
    except Exception as e:
        print(f"Error adding candidate metadata: {e}")
        return False
