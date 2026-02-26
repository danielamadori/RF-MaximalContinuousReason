"""
ICF (Interval Condition Features) Redis helper functions.
"""
import redis
from typing import Dict, List, Tuple, Optional, Set, Any
from icf_eu_encoding import bitmap_mask_to_string, icf_to_bitmap_mask
from helpers import increment_info_counter


def random_key_from_can(can_connection: redis.Redis) -> Optional[str]:
    """
    Get a random key from the CAN database using Redis RANDOMKEY command.

    Args:
        can_connection: Redis connection to CAN database

    Returns:
        str: Random key from CAN, or None if CAN is empty
    """
    try:
        # Use Redis RANDOMKEY command to get a random key directly
        random_key = can_connection.randomkey()
        return random_key

    except Exception as e:
        print(f"Error getting random key from CAN: {e}")
        return None


def bitmap_to_icf(bitmap_string: str, eu: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    """
    Convert bitmap string representation back to ICF intervals.

    Constraints:
    - Each feature must have at least two 1s in its bitmap stride (otherwise error)
    - Each feature's endpoints must start with -inf and end with +inf (otherwise error)

    Args:
        bitmap_string: String of 0s and 1s representing the bitmap
        eu: Endpoints universe dictionary

    Returns:
        Dict mapping feature names to (inf, sup) intervals

    Raises:
        ValueError: If constraints are violated
    """
    if not isinstance(bitmap_string, str):
        raise TypeError("bitmap_string must be a string")

    if not isinstance(eu, dict):
        raise TypeError("eu must be a dictionary")

    # Validate EU constraints
    for feature, endpoints in eu.items():
        if len(endpoints) < 2:
            raise ValueError(f"Feature '{feature}' must have at least 2 endpoints")
        if endpoints[0] != float('-inf'):
            raise ValueError(f"Feature '{feature}' endpoints must start with -inf, got {endpoints[0]}")
        if endpoints[-1] != float('inf'):
            raise ValueError(f"Feature '{feature}' endpoints must end with +inf, got {endpoints[-1]}")

    # Convert string to list of integers
    bitmap_mask = [int(bit) for bit in bitmap_string]

    # Process features in lexicographic order
    sorted_features = sorted(eu.keys())
    icf = {}

    start_idx = 0
    for feature in sorted_features:
        endpoints = eu[feature]
        feature_length = len(endpoints)
        end_idx = start_idx + feature_length

        if end_idx > len(bitmap_mask):
            raise ValueError(f"Bitmap too short for feature '{feature}'")

        # Get the bitmap portion for this feature
        feature_bitmap = bitmap_mask[start_idx:end_idx]

        # Count 1s in this feature's bitmap stride
        ones_count = sum(feature_bitmap)
        if ones_count < 2:
            raise ValueError(f"Feature '{feature}' must have at least 2 ones in bitmap stride, got {ones_count}")

        # Find first and last 1 in the bitmap
        first_one = None
        last_one = None

        for i, bit in enumerate(feature_bitmap):
            if bit == 1:
                if first_one is None:
                    first_one = i
                last_one = i

        # Set bounds based on endpoints (we know first_one and last_one exist due to constraint check)
        inf_bound = endpoints[first_one]
        sup_bound = endpoints[last_one]

        icf[feature] = (inf_bound, sup_bound)
        start_idx = end_idx

    return icf


def icfs_share_sample(icf1: Dict[str, Tuple[float, float]], 
                      icf2: Dict[str, Tuple[float, float]]) -> bool:
    """
    Check if two ICFs share at least one sample.
    
    Two ICFs share a sample if for ALL features, their intervals overlap.
    Intervals (b, e) and (b', e') overlap if NOT ((e <= b') OR (e' <= b)),
    which is equivalent to (b < e') AND (b' < e).
    
    Args:
        icf1: First ICF - dictionary mapping feature_name to (inf, sup) interval
        icf2: Second ICF - dictionary mapping feature_name to (inf, sup) interval
        
    Returns:
        bool: True if ICFs share at least one sample, False otherwise
        
    Example:
        >>> icf1 = {'f1': (0.0, 5.0), 'f2': (1.0, 3.0)}
        >>> icf2 = {'f1': (3.0, 7.0), 'f2': (2.0, 4.0)}
        >>> icfs_share_sample(icf1, icf2)
        True  # They overlap in both features
        
        >>> icf3 = {'f1': (0.0, 2.0), 'f2': (1.0, 3.0)}
        >>> icfs_share_sample(icf1, icf3)
        False  # f1 intervals are disjoint: (0,2) and (3,7) don't overlap
    """
    if not isinstance(icf1, dict):
        raise TypeError("icf1 must be a dictionary")
    if not isinstance(icf2, dict):
        raise TypeError("icf2 must be a dictionary")
    
    # Get all features from both ICFs
    all_features = set(icf1.keys()) | set(icf2.keys())
    
    for feature in all_features:
        # Get intervals, defaulting to (-inf, inf) if feature not in ICF
        b1, e1 = icf1.get(feature, (float('-inf'), float('inf')))
        b2, e2 = icf2.get(feature, (float('-inf'), float('inf')))
        
        # Validate intervals
        if b1 >= e1 or b2 >= e2:
            raise ValueError(f"Invalid interval for feature '{feature}': intervals must have inf < sup")
        
        # Check if intervals overlap
        # Intervals are disjoint if: (e1 <= b2) OR (e2 <= b1)
        # They overlap if NOT disjoint: (b1 < e2) AND (b2 < e1)
        if not (b1 < e2 and b2 < e1):
            # At least one feature has disjoint intervals
            # Therefore ICFs do not share any sample
            return False
    
    # All features have overlapping intervals
    # Therefore ICFs share at least one sample
    return True


def cache_shares_sample_with_r(ext_icf: Dict[str, Tuple[float, float]],
                               eu_data: Dict[str, List[float]],
                               r_connection: redis.Redis,
                               r_cache: Set[str],
                               scan: int = 10,
                               use_db: bool = True,
                               full: bool = True,
                               info: Optional[Dict[str, Any]] = None,
                               info_prefix: str = "") -> bool:
    """
    Check if an ICF shares a sample with any reason in R database, with caching.
    
    Populates the cache with R entries while checking, similar to cache_dominated_icf.
    
    Args:
        ext_icf: The ICF to check for sample sharing
        eu_data: Endpoints universe - dictionary mapping feature_name to monotonic arrays
        r_connection: Redis connection to R database
        r_cache: Set cache for R bitmaps (will be populated)
        scan: Scan count parameter - if 0 use KEYS, if >0 use SCAN with count
        use_db: Whether to check database if not found in cache
        full: If True and using SCAN, continue until end; if False, stop at first match
        info: Optional dict to populate with execution statistics (accumulated, not reset)
        info_prefix: Prefix to add to all info keys (e.g., "ar_ext_R_")
        
    Returns:
        bool: True if ext_icf shares a sample with any reason in R, False otherwise
    """
    if not isinstance(ext_icf, dict):
        raise TypeError("ext_icf must be a dictionary")
    if not isinstance(eu_data, dict):
        raise TypeError("eu_data must be a dictionary")
    if not isinstance(r_cache, set):
        raise TypeError("r_cache must be a set")
    
    try:
        # Check cache first
        for r_bitmap in r_cache:
            increment_info_counter(info, 'cache_checks', prefix=info_prefix)
            
            try:
                r_icf = bitmap_to_icf(r_bitmap, eu_data)
                if icfs_share_sample(ext_icf, r_icf):
                    increment_info_counter(info, 'cache_hits', prefix=info_prefix)
                    increment_info_counter(info, 'shares_sample', prefix=info_prefix)
                    return True
            except Exception:
                # Skip invalid bitmaps in cache
                continue
        
        # Not found in cache, check database if requested
        if not use_db:
            return False
        
        increment_info_counter(info, 'db_checks', prefix=info_prefix)
        
        # Scan database and populate cache
        if scan == 0:
            # Use KEYS method - get all at once
            keys = r_connection.keys('*')
            increment_info_counter(info, 'db_keys_found', len(keys), prefix=info_prefix)
            
            found_sharing = False
            
            for r_bitmap in keys:
                # Add to cache if not already present
                if r_bitmap not in r_cache:
                    r_cache.add(r_bitmap)
                    increment_info_counter(info, 'cache_keys_added', prefix=info_prefix)
                
                # Check for sample sharing
                try:
                    r_icf = bitmap_to_icf(r_bitmap, eu_data)
                    if icfs_share_sample(ext_icf, r_icf):
                        found_sharing = True
                        # Continue to populate cache completely
                except Exception:
                    continue
            
            if found_sharing:
                increment_info_counter(info, 'shares_sample', prefix=info_prefix)
            
            return found_sharing
        
        else:
            # Use SCAN method
            cursor = 0
            found_sharing = False
            
            while True:
                cursor, keys = r_connection.scan(cursor, match='*', count=scan)
                increment_info_counter(info, 'db_scans', prefix=info_prefix)
                
                if len(keys) > 0:
                    increment_info_counter(info, 'db_keys_found', len(keys), prefix=info_prefix)
                    
                    for r_bitmap in keys:
                        # Add to cache if not already present
                        if r_bitmap not in r_cache:
                            r_cache.add(r_bitmap)
                            increment_info_counter(info, 'cache_keys_added', prefix=info_prefix)
                        
                        # Check for sample sharing
                        try:
                            r_icf = bitmap_to_icf(r_bitmap, eu_data)
                            if icfs_share_sample(ext_icf, r_icf):
                                found_sharing = True
                                # If not full mode, return immediately
                                if not full:
                                    increment_info_counter(info, 'shares_sample', prefix=info_prefix)
                                    return True
                        except Exception:
                            continue
                
                if cursor == 0:
                    # Completed scan
                    break
            
            if found_sharing:
                increment_info_counter(info, 'shares_sample', prefix=info_prefix)
            
            return found_sharing
    
    except Exception as e:
        print(f"Error checking sample sharing: {e}")
        if info is not None:
            error_key = f"{info_prefix}error"
            if error_key not in info:
                info[error_key] = []
            info[error_key].append(str(e))
        return False


def key_dominates(k1: str, k2: str, strictly: bool = False, reverse: bool = False) -> bool:
    if not reverse:    
        if strictly: 
            strictly = False
            for i in range(len(k1)):
                if k1[i] == '0' and k2[i] == '1':
                    return False
                if k1[i] == '1' and k2[i] == '0':
                    strictly = True
            return strictly
        else:
            for i in range(len(k1)):
                if k1[i] == '0' and k2[i] == '1':
                    return False
            return True
    else:
        if strictly: 
            strictly = False
            for i in range(len(k2)):
                if k2[i] == '0' and k1[i] == '1':
                    return False
                if k2[i] == '1' and k1[i] == '0':
                    strictly = True
            return strictly
        else:
            for i in range(len(k2)):
                if k2[i] == '0' and k1[i] == '1':
                    return False
            return True


def cache_dominated_bitmap(bitmap_string: str, 
                        r_connection: redis.Redis, 
                        cache: Set[str], 
                        reverse: bool = False, 
                        scan: int = 10, 
                        use_db: bool = True,
                        full: bool = False,
                        info: Optional[Dict[str, Any]] = None,
                        info_prefix: str = "") -> bool:
    """
    Check if a bitmap is dominated by any bitmap in the database.
    
    Args:
        bitmap_string: The bitmap string to check
        r_connection: Redis connection
        cache: Set cache for storing bitmap strings
        reverse: Whether to reverse domination logic
        scan: Scan count parameter - if 0 use KEYS, if >0 use SCAN with count
        use_db: Whether to check the database (or just cache)
        full: If True and using SCAN, continue until end; if False, stop at first match
        info: Optional dict to populate with execution statistics (accumulated, not reset)
        info_prefix: Prefix to add to all info keys (e.g., "R_" or "GP_")
        
    Returns:
        bool: True if dominated, False otherwise
    """
    try:
        # Check cache first
        for key in cache:
            increment_info_counter(info, 'cache_checks', prefix=info_prefix)
            
            if key_dominates(key, bitmap_string, reverse=reverse):
                increment_info_counter(info, 'cache_hits', prefix=info_prefix)
                increment_info_counter(info, 'dominated', prefix=info_prefix)
                return True

        if not use_db:
            return False
        
        increment_info_counter(info, 'db_checks', prefix=info_prefix)
            
        # Replace 0s with ? for pattern matching
        pattern = bitmap_string.replace('0' if not reverse else '1', '?')
        
        if scan == 0:
            # Use KEYS method (get all matching keys at once)
            keys = r_connection.keys(pattern)
            
            increment_info_counter(info, 'db_keys_found', len(keys), prefix=info_prefix)
            
            if len(keys) == 0:
                return False
            
            for key_ext in keys:
                remove = []
                for key in cache:
                    if key_dominates(key_ext, key, reverse=reverse):
                        remove.append(key)
                
                for key in remove:
                    cache.remove(key)
                    increment_info_counter(info, 'cache_keys_removed', prefix=info_prefix)
                
                cache.add(key_ext)
                increment_info_counter(info, 'cache_keys_added', prefix=info_prefix)
            
            increment_info_counter(info, 'dominated', prefix=info_prefix)
            return True
        
        else:
            # Use SCAN method with given count parameter
            cursor = 0
            found_any = False
            
            while True:
                cursor, keys = r_connection.scan(cursor, match=pattern, count=scan)
                
                increment_info_counter(info, 'db_scans', prefix=info_prefix)
                
                if len(keys) > 0:
                    found_any = True
                    
                    increment_info_counter(info, 'db_keys_found', len(keys), prefix=info_prefix)
                    
                    # Add keys to cache
                    for key_ext in keys:
                        remove = []
                        for key in cache:
                            if key_dominates(key_ext, key, reverse=reverse):
                                remove.append(key)
                        
                        for key in remove:
                            cache.remove(key)
                            increment_info_counter(info, 'cache_keys_removed', prefix=info_prefix)
                        
                        cache.add(key_ext)
                        increment_info_counter(info, 'cache_keys_added', prefix=info_prefix)
                    
                    # If not full mode, return immediately on first match (original behavior)
                    if not full:
                        increment_info_counter(info, 'dominated', prefix=info_prefix)
                        return True
                
                if cursor == 0:
                    # Completed scan
                    break
            
            if found_any:
                increment_info_counter(info, 'dominated', prefix=info_prefix)
            
            return found_any
            
    except Exception as e:
        print(f"Error checking domination: {e}")
        if info is not None:
            error_key = f"{info_prefix}error"
            # For errors, we might want to store the message rather than count
            if error_key not in info:
                info[error_key] = []
            info[error_key].append(str(e))
        return False


def cache_dominated_icf(icf: Dict[str, Tuple[float, float]], 
                       eu: Dict[str, List[float]], 
                       r_connection: redis.Redis, 
                       cache: Set[str], 
                       reverse: bool = False, 
                       scan: int = 10, 
                       use_db: bool = True,
                       full: bool = False,
                       info: Optional[Dict[str, Any]] = None,
                       info_prefix: str = "") -> bool:
    """
    Check if ICF is dominated by any ICF in R database.
    Domination means that for any feature, the interval in R contains the corresponding interval in ICF.
    
    Args:
        icf: ICF to check for domination
        eu: Endpoints universe
        r_connection: Redis connection to R database
        cache: Set cache for storing bitmap strings
        reverse: Whether to reverse domination logic
        scan: Scan count parameter - if 0 use KEYS, if >0 use SCAN with count
        use_db: Whether to check the database (or just cache)
        full: If True and using SCAN, continue until end; if False, stop at first match
        info: Optional dict to populate with execution statistics (accumulated, not reset)
        info_prefix: Prefix to add to all info keys (e.g., "R_" or "GP_")
        
    Returns:
        bool: True if dominated, False otherwise
    """
    try:
        # Convert ICF to bitmap
        bitmap_mask = icf_to_bitmap_mask(icf, eu)
        bitmap_string = bitmap_mask_to_string(bitmap_mask)
        return cache_dominated_bitmap(
            bitmap_string=bitmap_string,
            r_connection=r_connection,
            cache=cache,
            reverse=reverse,
            scan=scan,
            use_db=use_db,
            full=full,
            info=info,
            info_prefix=info_prefix
        )
    except Exception as e:
        print(f"Error checking domination in R: {e}")
        if info is not None:
            error_key = f"{info_prefix}error"
            if error_key not in info:
                info[error_key] = []
            info[error_key].append(str(e))
        return False
    

def delete_from_can(can_connection: redis.Redis, key: str) -> bool:
    """
    Delete a key from the CAN database.

    Args:
        can_connection: Redis connection to CAN database
        key: Key to delete

    Returns:
        bool: True if successfully deleted, False otherwise
    """
    try:
        result = can_connection.delete(key)
        return result > 0

    except Exception as e:
        print(f"Warning: Could not delete key from CAN (possibly already deleted by another worker): {e}")
        return True  # Return True since this is expected behavior in multi-worker environment


def smart_delete_dominated(new_key: str, r_connection: redis.Redis, exclude_new_key: bool = True, dominates: bool = True, verbose: int = 0, scan: int = 100) -> int:
    """
    Delete keys in R database based on domination relationship with the new key.

    Args:
        new_key: The reference key (bitmap string)
        r_connection: Redis connection to R database
        exclude_new_key: Whether to exclude the new key itself from deletion (default True)
        dominates: If True, delete keys dominated by new_key (replace '1' with '?')
                  If False, delete keys that dominate new_key (replace '0' with '?')
        verbose: Verbosity level (0=no print, 1=print count only, 2=print details)
        scan: Scan count parameter - if 0 use KEYS (complete), if >0 use SCAN with count (batch processing)

    Returns:
        int: Number of keys deleted
    """
    try:
        deleted_count = 0

        # Create pattern based on domination direction
        if dominates:
            # Find keys dominated BY new_key: replace '1' with '?'
            pattern = new_key.replace('1', '?')
            direction_desc = "dominated by"
        else:
            # Find keys that DOMINATE new_key: replace '0' with '?'
            pattern = new_key.replace('0', '?')
            direction_desc = "that dominate"

        if verbose >= 2:
            print(f"Smart delete: Looking for keys {direction_desc} new key")
            print(f"Pattern: {pattern}")
            print(f"New key: {new_key}")
            print(f"Exclude new key: {exclude_new_key}")
            print(f"Scan mode: {'KEYS (complete)' if scan == 0 else f'SCAN (count={scan})'}")

        if scan == 0:
            # Use KEYS method - get all matching keys at once (complete deletion)
            keys = r_connection.keys(pattern)
            
            if len(keys) == 0:
                if verbose >= 1:
                    print(f"Smart delete: No keys found {direction_desc} reference key")
                return 0
            
            # Process all found keys
            keys_to_delete = []
            for key in keys:
                # Check if we should exclude the new key itself from deletion
                if exclude_new_key and key == new_key:
                    continue
                keys_to_delete.append(key)
            
            # Delete all collected keys
            for key_to_delete in keys_to_delete:
                result = r_connection.delete(key_to_delete)
                if result > 0:
                    deleted_count += 1
                    if verbose >= 2:
                        print(f"  Deleted key: {key_to_delete}")
        
        else:
            # Use SCAN method - process keys in batches
            cursor = 0
            while True:
                cursor, keys = r_connection.scan(cursor, match=pattern, count=scan)
                
                if len(keys) > 0:
                    # Process this batch of keys
                    batch_deleted = 0
                    for key in keys:
                        # Check if we should exclude the new key itself from deletion
                        if exclude_new_key and key == new_key:
                            continue
                        
                        # Delete the key immediately
                        result = r_connection.delete(key)
                        if result > 0:
                            deleted_count += 1
                            batch_deleted += 1
                            if verbose >= 2:
                                print(f"  Deleted key: {key}")
                    
                    if verbose >= 2 and batch_deleted > 0:
                        print(f"  Batch deleted: {batch_deleted} keys")
                
                if cursor == 0:
                    # Completed full scan
                    break

        if verbose >= 1 and deleted_count > 0:
            print(f"Smart delete: Deleted {deleted_count} keys {direction_desc} reference key")

        return deleted_count
    
    except Exception as e:
        print(f"Error in smart_delete_dominated: {e}")
        return 0