"""
Refactored ar_check_cache.py - Simplified propagation logic with random successor extraction

Key changes:
- Removed propagate_on_true and propagate_on_false parameters
- TRUE results (anti-reasons): propagate ONLY at top level after full verification
- FALSE results (not anti-reasons): never propagate
- Random extraction of successors like rcheck_cache
- Salvages ARs discovered in branches even when parent is not an AR
"""

from typing import Dict, List, Any, Set, Tuple, Optional
import redis
import random
import datetime
from redis_helpers.icf import (
    cache_dominated_icf, 
    cache_dominated_bitmap,
    icfs_share_sample,
    cache_shares_sample_with_r,
    bitmap_to_icf,
    smart_delete_dominated
)
from icf_eu_encoding import bitmap_mask_to_string, icf_to_bitmap_mask
from forest import Forest
from helpers import increment_info_counter

# Import shared functions from rcheck_cache
from rcheck_cache import (
    pick_random_node_index,
    saturate,
    insert_result_bitmap,
    insert_result_icf
)


def generate_and_filter_ar_extensions(ar_icf: Dict[str, Tuple[float, float]],
                                      connections: Dict[str, redis.Redis],
                                      eu_data: Dict[str, List[float]],
                                      forest: Forest,
                                      caches: Dict[str, Set[str]],
                                      info: Dict[str, Any],
                                      check_r_sharing: bool = True,
                                      check_scans: int = 10,
                                      check_use_db: bool = True) -> List[str]:
    """
    Generate all extensions of an anti-reason and filter them.
    
    An extension is filtered (rejected) if:
    1. There exists an AR' in AR that dominates the extension
    2. (Optional) The extension shares a sample with any reason in R
    
    Args:
        ar_icf: The anti-reason ICF to extend
        connections: Dictionary of Redis connections
        eu_data: Endpoints universe
        forest: Forest object
        caches: Dictionary of caches
        info: Info dictionary for statistics
        check_r_sharing: Whether to filter extensions that share samples with R
        check_scans: Scan count for cache checks
        check_use_db: Whether to check database
        
    Returns:
        List of bitmap strings for extensions that pass all filters
    """
    # First, generate all extensions
    extension_icfs = []
    for feature in eu_data.keys():
        for direction in ["low", "high"]:
            ext_icf = forest.inflate_interval(ar_icf, eu_data, feature, direction)
            if ext_icf is not None:
                extension_icfs.append(ext_icf)
    
    increment_info_counter(info, 'ar_extensions_total', len(extension_icfs))
    
    # Now filter them
    filtered_extensions = []
    
    for ext_icf in extension_icfs:
        # Filter 1: Check if dominated by existing AR
        if cache_dominated_icf(ext_icf, eu_data, connections['AR'], 
                              caches.get('AR', set()),
                              scan=check_scans, use_db=check_use_db,
                              full=True,
                              info=info, info_prefix='ar_ext_AR_'):
            increment_info_counter(info, 'ar_extensions_filtered_by_AR')
            continue
        
        # Filter 2: Check if shares sample with reasons in R (optional)
        if check_r_sharing:
            if cache_shares_sample_with_r(ext_icf, eu_data, connections['R'],
                                         caches.get('R', set()),
                                         scan=check_scans, use_db=check_use_db,
                                         full=True,
                                         info=info, info_prefix='ar_ext_R_'):
                increment_info_counter(info, 'ar_extensions_filtered_by_R')
                continue
        
        # Extension passed all filters
        ext_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(ext_icf, eu_data))
        filtered_extensions.append(ext_bitmap)
    
    increment_info_counter(info, 'ar_extensions_added', len(filtered_extensions))
    
    return filtered_extensions


def handle_result_and_cleanup(connections: Dict[str, redis.Redis], 
                             icf: Dict[str, Tuple[float, float]],
                             forest_icf_bitmap: str,
                             eu_data: Dict[str, List[float]],
                             caches: Dict[str, Set[str]], 
                             info: Dict[str, Any],
                             is_ar: bool,
                             delete_scans: int = 100):
    """
    Handle result insertion and cleanup for AR/not-AR results.
    
    Always propagates when called (no conditional flags).
    Only called for AR results (is_ar=True).
    
    Args:
        connections: Dictionary of Redis connections
        icf: The ICF being processed
        forest_icf_bitmap: Forest profile bitmap for this ICF
        eu_data: Endpoints universe
        caches: Dictionary of in-memory caches
        info: Info dictionary for statistics
        is_ar: True if this is an anti-reason (should always be True when called)
        delete_scans: Scan count for smart deletion
    """
    
    if is_ar:
        # Insert anti-reasons to AR and AP (use reverse=False for AR, as larger ICF is better AR)
        insert_result_icf(connections['AR'], icf, eu_data, caches.get('AR'), info, 'AR', 
                         reverse=False, delete_scans=delete_scans)
        insert_result_bitmap(connections['AP'], forest_icf_bitmap, caches.get('AP'), info, 'AP', 
                           reverse=False, delete_scans=delete_scans)
        
        # Remove dominated candidates from CAR
        icf_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(icf, eu_data))
        deleted_car = smart_delete_dominated(icf_bitmap, connections['CAR'], 
                                            exclude_new_key=False, 
                                            dominates=True, 
                                            scan=delete_scans)
        increment_info_counter(info, 'deleted_from_CAR', deleted_car)


def check_domination_cache(icf: Dict[str, Tuple[float, float]], 
                          forest_icf_bitmap: str,
                          connections: Dict[str, redis.Redis],
                          eu_data: Dict[str, List[float]], 
                          caches: Dict[str, Set[str]],
                          info: Dict[str, Any],
                          check_scans: int = 10,
                          check_use_db: bool = False) -> Optional[bool]:
    """
    Check all domination caches and return result if found, None if no domination.
    
    Returns: 
        - True if this is an anti-reason (dominated by AR/AP)
        - False if this is not an anti-reason (dominated by reason in R/GP)
        - None if no domination found
    """
    
    # Check if we are dominated by a known anti-reason in AR
    # If AR dominates us → we're a subset of a known AR → we're also an AR
    if cache_dominated_icf(icf, eu_data, connections['AR'], caches['AR'], 
                          reverse=False, scan=check_scans, use_db=check_use_db):
        increment_info_counter(info, 'ar_dominated_by_AR')
        return True
    
    # Check if our forest profile is dominated by a known anti-reason profile in AP
    # If AP dominates our profile → our profile is subset of known AR profile → we're an AR
    if cache_dominated_bitmap(forest_icf_bitmap, connections['AP'], caches['AP'], 
                             reverse=False, scan=check_scans, use_db=check_use_db):
        increment_info_counter(info, 'ar_profile_dominated_by_AP')
        return True

    # Check if we are dominated by (contained in) a known reason in R
    # If a reason in R dominates us → we're a subset of a reason → not an AR
    if cache_shares_sample_with_r(icf, eu_data, connections['R'], caches['R'], 
                          scan=check_scans, use_db=check_use_db):
        increment_info_counter(info, 'ar_share_by_reason_in_R')
        return False

    # Check if our forest profile is dominated by (contained in) a known good profile in GP
    # If a good profile in GP dominates us → we're a subset of good samples → not an AR
    if cache_dominated_bitmap(forest_icf_bitmap, connections['GP'], caches['GP'], 
                             reverse=False, scan=check_scans, use_db=check_use_db):
        increment_info_counter(info, 'ar_profile_dominated_by_GP')
        return False

    return None  # No domination found


def ar_check_cache(connections: Dict[str, redis.Redis], 
                  icf: Dict[str, Tuple[float, float]], 
                  label: str, 
                  nodes: List[Dict[str, Any]],
                  eu_data: Dict[str, List[float]],
                  forest: Forest,  
                  caches: Dict[str, Set[str]],
                  info: Dict[str, Any] = {},
                  delete_scans: int = 100,
                  check_scans: int = 10,
                  check_use_db: bool = False) -> bool:
    """
    Entry point for checking if ICF is an anti-reason.
    
    Propagation strategy:
    - TRUE results (anti-reasons): propagate ONLY at top level after full verification
    - FALSE results (not anti-reasons): never propagate
    
    Args:
        connections: Dictionary of Redis connections (DATA, CAR, R, NR, AR, AP, GP, BP)
        icf: Dictionary mapping feature names to (inf, sup) intervals
        label: Target label to check against
        nodes: List of tree node dictionaries from the forest
        eu_data: Endpoints universe - dictionary mapping feature_name to monotonic arrays
        forest: Forest object for calculating ICF profile bitmaps
        caches: Dictionary of in-memory caches for each database
                Expected keys: 'R', 'NR', 'AR', 'AP', 'GP', 'BP' etc.
        info: Dictionary for collecting debugging/logging information
        delete_scans: Number of scans for smart deletion
        check_scans: Number of scans for cache checking
        check_use_db: Whether to check database in addition to cache
        
    Returns:
        bool: True if ICF is an anti-reason (never induces label), False otherwise
    """

    # Call recursive checker (no propagation during recursion)
    r = ar_check_cache_recursive(connections, icf, label, nodes, eu_data, forest, caches, info,
                                 delete_scans, check_scans, check_use_db)

    # If TRUE (is anti-reason), propagate at top level and generate extensions
    if r:
        forest_icf_bitmap = bitmap_mask_to_string(forest.icf_profile_to_bitmap(icf))
        handle_result_and_cleanup(connections, icf, forest_icf_bitmap, eu_data, caches, info, 
                                is_ar=True, delete_scans=delete_scans)
        
        # Generate and filter extensions for this anti-reason
        ar_extensions = generate_and_filter_ar_extensions(
            icf, connections, eu_data, forest, caches, info, 
            check_r_sharing=True,
            check_scans=check_scans,
            check_use_db=check_use_db
        )
        
        # Add extensions to CAR database (candidate anti-reasons)
        current_time = datetime.datetime.now().isoformat()
        for ext_bitmap in ar_extensions:
            connections['CAR'].set(ext_bitmap, current_time)
        
        if ar_extensions:
            increment_info_counter(info, 'ar_extensions_added_to_CAR', len(ar_extensions))

    return r


def ar_check_cache_recursive(connections: Dict[str, redis.Redis], 
                            icf: Dict[str, Tuple[float, float]], 
                            label: str, 
                            nodes: List[Dict[str, Any]],
                            eu_data: Dict[str, List[float]],
                            forest: Forest,  
                            caches: Dict[str, Set[str]],
                            info: Dict[str, Any] = {},
                            delete_scans: int = 100,
                            check_scans: int = 10,
                            check_use_db: bool = False) -> bool:
    """
    Recursive AR checking with cache support.
    
    Propagation strategy:
    - TRUE results (anti-reasons): do NOT propagate (only propagate at top level)
    - FALSE results (not anti-reasons): do NOT propagate (never propagate)
    - Salvages ARs found in branches even when parent is not an AR
    
    Returns True if this ICF is an anti-reason (label NEVER wins majority).
    """

    increment_info_counter(info, 'ar_iterations')

    # Create forest ICF profile bitmap  
    forest_icf_bitmap = bitmap_mask_to_string(forest.icf_profile_to_bitmap(icf))
    
    # Check all domination caches first
    cache_result = check_domination_cache(icf, forest_icf_bitmap, connections, eu_data, caches, info,
                                         check_scans=check_scans, check_use_db=check_use_db)
    if cache_result is not None:
        # Found in cache - already in database, no need to propagate again
        return cache_result

    # All cache checks failed, do traditional voting logic with REVERSED conditions for AR
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

    # Early stopping conditions for anti-reason checking (REVERSED from rcheck):
    # An ICF is an anti-reason if the target label NEVER wins majority
    
    # Case 1: Adversarial votes are too strong - label can never catch up (IS anti-reason)
    if max_adversarial > votes[label] + len(unknown):
        increment_info_counter(info, 'ar_early_stop_true')
        return True

    # Case 2: Label votes are too strong - label will definitely win (NOT anti-reason)
    if votes[label] > max_adversarial + len(unknown):
        increment_info_counter(info, 'ar_early_stop_false')
        return False

    # Case 3: Tie or label loses with no unknowns remaining (IS anti-reason)
    if len(unknown) == 0 and max_adversarial >= votes[label]:
        increment_info_counter(info, 'ar_tie_case')
        return True

    # No early stopping possible - must split and recurse
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

    # Random extraction of successors (like rcheck_cache)
    choices = [(icf_low, "L", 'l'), (icf_high, "H", 'h')]
    random.shuffle(choices)

    icf_choice = choices[0][0]
    icf_path = choices[0][1]
    info['ar_path'] = info.get('ar_path', "") + icf_path

    # Recursively check first choice (random)
    first_icf = choices[0][0]
    if ar_check_cache_recursive(connections, icf_choice, label, saturate(icf_choice, nodes), 
                                eu_data, forest, caches, info,
                                delete_scans=delete_scans, check_scans=check_scans, 
                                check_use_db=check_use_db):
        # First branch is AR, check second branch
        icf_choice = choices[1][0]
        icf_path = choices[1][2]
        info['ar_path'] = info['ar_path'][:-1] + icf_path
        
        second_icf = choices[1][0]
        if ar_check_cache_recursive(connections, icf_choice, label, saturate(icf_choice, nodes), 
                                    eu_data, forest, caches, info,
                                    delete_scans=delete_scans, check_scans=check_scans, 
                                    check_use_db=check_use_db):
            # Both branches are AR - parent is AR, propagate at top level
            info['ar_path'] = info['ar_path'][:-1]
            return True
        else: 
            # First AR, second not AR - parent is not AR, but salvage the first AR!
            # first_forest_bitmap = bitmap_mask_to_string(forest.icf_profile_to_bitmap(first_icf))
            # handle_result_and_cleanup(connections, first_icf, first_forest_bitmap, eu_data, caches, info, 
            #                        is_ar=True, delete_scans=delete_scans)
            
            # Generate and filter extensions of the AR child
            #ar_extensions = generate_and_filter_ar_extensions(
            #    first_icf, connections, eu_data, forest, caches, info, 
            #    check_r_sharing=True,
            #    check_scans=check_scans,
            #    check_use_db=check_use_db
            #)
            
            # Add extensions to CAR database
            #current_time = datetime.datetime.now().isoformat()
            #for ext_bitmap in ar_extensions:
            #    connections['CAR'].set(ext_bitmap, current_time)
            
            #if ar_extensions:
            #    increment_info_counter(info, 'ar_extensions_added_to_CAR', len(ar_extensions))
            
            info['ar_path'] = info['ar_path'][:-1]
            return False
    else:
        # First branch not AR - parent cannot be AR (early stop)
        # Note: Second branch might be AR, but we don't check it (random path exploration)
        info['ar_path'] = info['ar_path'][:-1]
        return False