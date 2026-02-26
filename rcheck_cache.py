"""
Refactored rcheck_cache.py - Simplified propagation logic with AR sample-sharing filter

Key changes:
- Removed propagate_on_true and propagate_on_false parameters
- FALSE results: propagate immediately when determined (early stop bad, tie case)
- TRUE results: propagate only at top level after full verification
- Added AR sample-sharing check: ICFs sharing samples with ARs are rejected
- Cleaner, more maintainable code
"""

from typing import Dict, List, Any, Set, Tuple, Optional
import redis
import random
import datetime
from redis_helpers.icf import (
    cache_dominated_icf, 
    cache_dominated_bitmap,
    smart_delete_dominated,
    key_dominates
)
from icf_eu_encoding import bitmap_mask_to_string, icf_to_bitmap_mask
from forest import Forest
from helpers import increment_info_counter


def pick_random_node_index(unknown, nodes):
    """Pick a random index of an unknown node from the full nodes list."""
    if not unknown:
        return None
    node = random.choice(unknown)
    return nodes.index(node)


def saturate(icf, nodes):
    """
    Saturate nodes list by following tree paths as far as possible given ICF constraints.
    Replaces each non-leaf node with its child when the ICF determines which branch to take.
    """
    r = [node for node in nodes]
    for i in range(len(nodes)):
        while 'leaf_id' not in r[i]:
            feature, threshold = r[i]['feature'], r[i]['value']
            low, high = icf[feature]
            if high <= threshold:
                r[i] = r[i]['low_child']
            elif low >= threshold:
                r[i] = r[i]['high_child']
            else:
                break
    return r


def insert_result_bitmap(db_connection, bitmap_string, cache, info, db_name, reverse=False, delete_scans: int = 100):
    """
    Insert bitmap in database, removing dominated entries based on reverse flag.
    
    Args:
        db_connection: Redis connection to target database
        bitmap_string: Bitmap string to insert
        cache: In-memory cache (Set[str]) to update
        info: Info dictionary for statistics
        db_name: Database name for logging
        reverse: If False, remove entries dominated by new key (for R/GP)
                If True, remove entries that dominate new key (for NR/BP)
        delete_scans: Scan count for smart deletion
    """
    current_time = datetime.datetime.now().isoformat()
    
    # Store bitmap in database
    db_connection.set(bitmap_string, current_time)
    
    # Remove dominated entries from database
    deleted_count = smart_delete_dominated(bitmap_string, db_connection, 
                                          exclude_new_key=True, 
                                          dominates=not reverse, 
                                          scan=delete_scans)
    increment_info_counter(info, f'deleted_from_{db_name}', deleted_count)
    
    # Clean cache if provided
    if cache is not None:
        remove_from_cache = []
        for cached_key in cache:
            if key_dominates(bitmap_string, cached_key, reverse=reverse):
                remove_from_cache.append(cached_key)
        
        for key in remove_from_cache:
            cache.remove(key)
        
        # Add new key to cache
        cache.add(bitmap_string)


def insert_result_icf(db_connection, icf, eu_data, cache, info, db_name, reverse=False, delete_scans: int = 100):
    """Insert ICF in database by converting to bitmap and calling insert_result_bitmap."""
    icf_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(icf, eu_data))
    insert_result_bitmap(db_connection, icf_bitmap, cache, info, db_name, reverse, delete_scans)


def handle_result_and_cleanup(connections: Dict[str, redis.Redis], 
                             icf: Dict[str, Tuple[float, float]],
                             forest_icf_bitmap: str,
                             eu_data: Dict[str, List[float]],
                             caches: Dict[str, Set[str]], 
                             info: Dict[str, Any],
                             is_good: bool,
                             delete_scans: int = 100,
                             label: str = None,
                             nodes: List[Dict[str, Any]] = None,
                             forest: 'Forest' = None,
                             check_scans: int = 10,
                             check_use_db: bool = False,
                             check_ar: bool = True):
    """
    Handle result insertion and CAN cleanup for good/bad results.
    
    Always propagates results to databases (no conditional propagation flags).
    When is_good=False, optionally checks if the ICF is an anti-reason.
    
    Args:
        connections: Dictionary of Redis connections
        icf: The ICF being processed
        forest_icf_bitmap: Forest profile bitmap for this ICF
        eu_data: Endpoints universe
        caches: Dictionary of in-memory caches
        info: Info dictionary for statistics
        is_good: True if this is a good result (reason), False if bad (non-reason)
        delete_scans: Scan count for smart deletion
        label: Target label (needed for AR check)
        nodes: List of nodes (needed for AR check)
        forest: Forest object (needed for AR check)
        check_scans: Scan count for cache checking (for AR check)
        check_use_db: Whether to check database (for AR check)
        check_ar: Whether to check if non-reasons are anti-reasons
    """
    
    if is_good:
        # Insert good results to R and GP
        insert_result_icf(connections['R'], icf, eu_data, caches.get('R'), info, 'R', 
                         reverse=False, delete_scans=delete_scans)
        insert_result_bitmap(connections['GP'], forest_icf_bitmap, caches.get('GP'), info, 'GP', 
                           reverse=False, delete_scans=delete_scans)
    else:
        # Insert bad results to NR and BP
        insert_result_icf(connections['NR'], icf, eu_data, caches.get('NR'), info, 'NR', 
                         reverse=True, delete_scans=delete_scans)
        insert_result_bitmap(connections['BP'], forest_icf_bitmap, caches.get('BP'), info, 'BP', 
                           reverse=True, delete_scans=delete_scans)
        
        # Check if this non-reason is actually an anti-reason
        if check_ar and label is not None and nodes is not None and forest is not None:
            # Import here to avoid circular dependency
            from ar_check_cache import ar_check_cache
            
            increment_info_counter(info, 'ar_checks_from_rcheck')
            ar_result = ar_check_cache(connections, icf, label, nodes, eu_data, forest, caches, info,
                                      delete_scans=delete_scans, check_scans=check_scans, 
                                      check_use_db=check_use_db)
            if ar_result:
                increment_info_counter(info, 'ar_found_from_rcheck')

    # Remove dominated candidates from CAN
    icf_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(icf, eu_data))
    deleted_can = smart_delete_dominated(icf_bitmap, connections['CAN'], 
                                        exclude_new_key=False, 
                                        dominates=is_good, 
                                        scan=delete_scans)
    increment_info_counter(info, 'deleted_from_CAN', deleted_can)    


def check_domination_cache(icf: Dict[str, Tuple[float, float]], 
                          forest_icf_bitmap: str,
                          connections: Dict[str, redis.Redis],
                          eu_data: Dict[str, List[float]], 
                          caches: Dict[str, Set[str]],
                          info: Dict[str, Any],
                          check_scans: int = 10,
                          check_use_db: bool = False,
                          use_R: bool = True,
                          use_GP: bool = True,
                          use_NR: bool = True,
                          use_BP: bool = True) -> Optional[bool]:
    """
    Check all domination caches and return result if found, None if no domination.

    NEW: Also checks if ICF shares a sample with any AR (anti-reason).
    If it shares a sample with an AR, it cannot be a pure reason → return False.
    
    Returns: 
        - True if good (dominated by R or GP)
        - False if bad (dominated by NR or BP, OR shares sample with AR)
        - None if no domination found
    """
    #todo dani
    
    if use_R:
        # Check if ICF is dominated by something in R (good samples)
        if cache_dominated_icf(icf, eu_data, connections['R'], caches['R'], 
                            reverse=False, scan=check_scans, use_db=check_use_db):
            increment_info_counter(info, 'dominated_by_R')
            return True
    if use_GP:
        # Check if forest profile is dominated in GP 
        if cache_dominated_bitmap(forest_icf_bitmap, connections['GP'], caches['GP'], 
                                reverse=False, scan=check_scans, use_db=check_use_db):
            increment_info_counter(info, 'forest_dominated_in_GP')
            return True
    if use_NR:
        # Check if ICF is dominated by something in NR (bad samples)
        if cache_dominated_icf(icf, eu_data, connections['NR'], caches['NR'], 
                            reverse=True, scan=check_scans, use_db=check_use_db):
            increment_info_counter(info, 'dominated_by_NR')
            return False
    if use_BP:
        # Check if forest profile is dominated in BP
        if cache_dominated_bitmap(forest_icf_bitmap, connections['BP'], caches['BP'], 
                                reverse=True, scan=check_scans, use_db=check_use_db):
            increment_info_counter(info, 'forest_dominated_in_BP')
            return False
    
    # NEW: Check if ICF shares a sample with any anti-reason in AR
    # If it shares a sample with an AR, it cannot be a pure reason
    from redis_helpers.icf import cache_shares_sample_with_r
    if cache_shares_sample_with_r(icf, eu_data, connections['AR'], caches.get('AR', set()), 
                                  scan=check_scans, use_db=check_use_db):
        increment_info_counter(info, 'shares_sample_with_AR')
        return False

    return None  # No domination found


def rcheck_cache(connections: Dict[str, redis.Redis], 
                icf: Dict[str, Tuple[float, float]], 
                label: str, 
                nodes: List[Dict[str, Any]],
                eu_data: Dict[str, List[float]],
                forest: Forest,  
                caches: Dict[str, Set[str]],
                info: Dict[str, Any] = {},
                delete_scans: int = 100,
                check_scans: int = 10,
                check_use_db: bool = False,
                use_R: bool = True,
                use_GP: bool = True,
                use_NR: bool = True,
                use_BP: bool = True) -> bool:
    """
    Entry point for checking ICF with cache support.
    
    Propagation strategy:
    - FALSE results are propagated immediately during recursion (as soon as determined)
    - TRUE results are propagated only here at the top level (after full verification)
    
    Args:
        connections: Dictionary of Redis connections (DATA, CAN, R, NR, GP, BP, AR)
        icf: Dictionary mapping feature names to (inf, sup) intervals
        label: Target label to check against
        nodes: List of tree node dictionaries from the forest
        eu_data: Endpoints universe - dictionary mapping feature_name to monotonic arrays
        forest: Forest object for calculating ICF profile bitmaps
        caches: Dictionary of in-memory caches for each database
                Expected keys: 'R', 'NR', 'GP', 'BP', 'AR' etc.
        info: Dictionary for collecting debugging/logging information
        delete_scans: Number of scans for smart deletion
        check_scans: Number of scans for cache checking
        check_use_db: Whether to check database in addition to cache
        use_R: Whether to use R cache/database
        use_GP: Whether to use GP cache/database
        use_NR: Whether to use NR cache/database
        use_BP: Whether to use BP cache/database
        
    Returns:
        bool: True if ICF satisfies the label constraint, False otherwise
    """

    # Call recursive checker (FALSE propagates immediately inside, TRUE does not)
    r = rcheck_cache_recursive(connections, icf, label, nodes, eu_data, forest, caches, info,
                               delete_scans, check_scans, check_use_db, use_R, use_GP, use_NR, use_BP)

    # If TRUE, propagate at top level after full recursion completes
    if r:
        forest_icf_bitmap = bitmap_mask_to_string(forest.icf_profile_to_bitmap(icf))
        handle_result_and_cleanup(connections, icf, forest_icf_bitmap, eu_data, caches, info, 
                                is_good=True, delete_scans=delete_scans)

    return r


def rcheck_cache_recursive(connections: Dict[str, redis.Redis], 
                icf: Dict[str, Tuple[float, float]], 
                label: str, 
                nodes: List[Dict[str, Any]],
                eu_data: Dict[str, List[float]],
                forest: Forest,  
                caches: Dict[str, Set[str]],
                info: Dict[str, Any] = {},
                delete_scans: int = 100,
                check_scans: int = 10,
                check_use_db: bool = False,
                use_R: bool = True,
                use_GP: bool = True,
                use_NR: bool = True,
                use_BP: bool = True) -> bool:
    """
    Recursive checking with cache support.
    
    Propagation strategy:
    - FALSE results: propagate IMMEDIATELY (they determine final result right away)
    - TRUE results: do NOT propagate (only propagate at top level after full tree check)
    
    Args:
        connections: Dictionary of Redis connections
        icf: Dictionary mapping feature names to (inf, sup) intervals
        label: Target label to check against
        nodes: List of tree node dictionaries from the forest
        eu_data: Endpoints universe
        forest: Forest object for calculating ICF profile bitmaps
        caches: Dictionary of in-memory caches
        info: Dictionary for collecting debugging/logging information
        delete_scans: Number of scans for smart deletion
        check_scans: Number of scans for cache checking
        check_use_db: Whether to check database in addition to cache
        use_R: Whether to use R cache/database
        use_GP: Whether to use GP cache/database
        use_NR: Whether to use NR cache/database
        use_BP: Whether to use BP cache/database
        
    Returns:
        bool: True if ICF satisfies the label constraint, False otherwise
    """

    increment_info_counter(info, 'iterations')

    # Create forest ICF profile bitmap  
    forest_icf_bitmap = bitmap_mask_to_string(forest.icf_profile_to_bitmap(icf))
    
    # Check all domination caches first
    cache_result = check_domination_cache(icf, forest_icf_bitmap, connections, eu_data, caches, info,
                                         check_scans=check_scans, check_use_db=check_use_db, use_R=use_R,
                                         use_GP=use_GP, use_NR=use_NR, use_BP=use_BP)
    if cache_result is not None:
        # Found in cache - already in database, no need to propagate again
        return cache_result

    # All cache checks failed, do traditional voting logic
    known = []
    unknown = []

    # Separate leaf and non-leaf nodes
    for node in nodes:
        if 'leaf_id' in node:
            known.append(node)
        else:
            unknown.append(node)

    # Count votes for each label
    votes = {label: 0}
    for node in known:
        votes[node['label']] = votes.get(node['label'], 0) + 1

    if len(votes.keys()) > 1:
        filtered_values = [v for k, v in votes.items() if k != label]
        max_adversarial = max(filtered_values) 
    else:
        max_adversarial = 0

    # Early stopping - TRUE: don't propagate, FALSE: propagate immediately
    if votes[label] > max_adversarial + len(unknown):
        # Early stop GOOD - don't propagate (will propagate at top level)
        increment_info_counter(info, 'early_stop_good')
        return True

    if max_adversarial > votes[label] + len(unknown):
        # Early stop BAD - propagate immediately and check if AR
        increment_info_counter(info, 'early_stop_bad')
        handle_result_and_cleanup(connections, icf, forest_icf_bitmap, eu_data, caches, info, 
                                is_good=False, delete_scans=delete_scans,
                                label=label, nodes=nodes, forest=forest,
                                check_scans=check_scans, check_use_db=check_use_db)
        return False

    if len(unknown) == 0 and max_adversarial == votes[label]:
        # Tie case - propagate immediately as FALSE and check if AR
        increment_info_counter(info, 'tie_case')
        handle_result_and_cleanup(connections, icf, forest_icf_bitmap, eu_data, caches, info, 
                                is_good=False, delete_scans=delete_scans,
                                label=label, nodes=nodes, forest=forest,
                                check_scans=check_scans, check_use_db=check_use_db)
        return False

    # No early stopping possible, need to split
    j = pick_random_node_index(unknown, nodes)

    if j is None:
        raise ValueError("No unknown nodes left but no decision reached")

    feature, threshold = nodes[j]['feature'], nodes[j]['value']
    b, e = icf[feature]

    # Split ICF into low and high branches
    icf_low = {k: v for k, v in icf.items()}
    icf_low[feature] = (b, threshold)

    icf_high = {k: v for k, v in icf.items()}
    icf_high[feature] = (threshold, e)

    choices = [(icf_low, "L", 'l'), (icf_high, "H", 'h')]
    random.shuffle(choices)

    icf_choice = choices[0][0]
    icf_path = choices[0][1]
    info['path'] = info.get('path', "") + icf_path

    # Recursively check first choice
    if rcheck_cache_recursive(connections, icf_choice, label, saturate(icf_choice, nodes), 
                             eu_data, forest, caches, info,
                             delete_scans=delete_scans, check_scans=check_scans, 
                             check_use_db=check_use_db, use_R=use_R, use_GP=use_GP, use_NR=use_NR, use_BP=use_BP):
        # First branch TRUE, check second branch
        icf_choice = choices[1][0]
        icf_path = choices[1][2]
        info['path'] = info['path'][:-1] + icf_path
        
        if rcheck_cache_recursive(connections, icf_choice, label, saturate(icf_choice, nodes), 
                                 eu_data, forest, caches, info,
                                 delete_scans=delete_scans, check_scans=check_scans, 
                                 check_use_db=check_use_db, use_R=use_R, use_GP=use_GP, use_NR=use_NR, use_BP=use_BP):
            # Both branches TRUE - don't propagate (will propagate at top level)
            info['path'] = info['path'][:-1]
            return True
        else: 
            # First TRUE, second FALSE - second already propagated, just return FALSE
            info['path'] = info['path'][:-1]
            return False
    else:
        # First branch FALSE - already propagated, just return FALSE
        info['path'] = info['path'][:-1]
        return False