#!/usr/bin/env python3
"""
Random Path Worker with Raw Info Logging to Redis

This worker logs raw iteration data to Redis LOGS database as JSON:
- Raw info dicts from rcheck_cache and ar_check_cache calls (no aggregation)
- Outcomes observed in the worker
- Cache sizes at each iteration
- Timestamps
- Full bitmaps without truncation for complete analysis

Log key format: WORKERIP:WORKERNAME:ITERATION
Log value: JSON string (parse with json.loads())

Example retrieval:
    import redis, json
    logs_db = redis.Redis(host='localhost', port=6379, db=10, decode_responses=True)
    log_json = logs_db.get("hostname:worker_12345:150")
    log_data = json.loads(log_json)
    print(log_data['car_processing']['candidate_bitmap'])  # Full bitmap string
"""
import redis
import time
import datetime
import json
import socket
import os
import signal
from pathlib import Path
from redis_helpers.forest import retrieve_forest
from redis_helpers.endpoints import retrieve_monotonic_dict
from redis_helpers.icf import random_key_from_can, bitmap_to_icf, delete_from_can, cache_dominated_icf, cache_dominated_bitmap
from redis_helpers.preferred import (
    get_pr_candidate_exclusive,
    add_timestamp_to_pr,
    remove_from_pr,
    count_pr_keys,
    close_pr_candidate,
    add_candidate_metadata
)
from rcheck_cache import rcheck_cache, saturate, check_domination_cache
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string
import numpy as np
import random
import argparse


def get_worker_id():
    """Get unique worker identifier based on hostname and PID"""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}:worker_{pid}"


def get_cache_sizes(caches):
    """Get current cache sizes"""
    return {name: len(cache_set) for name, cache_set in caches.items()}


def get_db_sizes(connections):
    """Get current database sizes"""
    return {
        'PR': count_pr_keys(connections['PR']),
        'CAN': connections['CAN'].dbsize(),
        'R': connections['R'].dbsize(),
        'NR': connections['NR'].dbsize(),
        'CAR': connections['CAR'].dbsize(),
        'AR': connections['AR'].dbsize(),
        'GP': connections['GP'].dbsize(),
        'BP': connections['BP'].dbsize(),
        'AP': connections['AP'].dbsize()
    }


def log_iteration_to_redis(logs_db, worker_id, iteration, log_data):
    """
    Log iteration data to Redis LOGS database as JSON
    
    Stores the log entry as a JSON string that can be parsed back to a dict.
    Full bitmaps are stored without truncation for complete analysis.
    
    Args:
        logs_db: Redis connection to LOGS database
        worker_id: Unique worker identifier
        iteration: Iteration number
        log_data: Dictionary containing all log information
    """
    log_key = f"{worker_id}:{iteration}"
    try:
        # Store as JSON string - can be parsed back with json.loads()
        logs_db.set(log_key, json.dumps(log_data))
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration} to Redis: {e}")


def append_coverage_t(data_connection, candidate, coverage_value, timestamp):
    existing = data_connection.hget(candidate, "coverage_t")
    coverage_map = {}
    if existing:
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    try:
                        key_num = float(key)
                    except (TypeError, ValueError):
                        key_num = None
                    try:
                        value_num = float(value)
                    except (TypeError, ValueError):
                        value_num = None
                    # Heuristic: timestamps are large, coverage is 0-1.
                    if key_num is not None and key_num > 1_000_000:
                        coverage_map[key] = value
                    elif value_num is not None and value_num > 1_000_000:
                        coverage_map[str(value)] = key
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, list) and len(item) == 2:
                        coverage_map[str(item[1])] = item[0]
        except json.JSONDecodeError:
            coverage_map = {}
    coverage_map[timestamp] = f"{coverage_value}"
    data_connection.hset(candidate, "coverage_t", json.dumps(coverage_map))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Random Path Worker with Raw Info Redis Logging')
    parser.add_argument('--redis-host', default='localhost', 
                       help='Redis server host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port (default: 6379)')
    parser.add_argument('--worker-name', default=None,
                       help='Worker name (default: auto-generated from hostname:pid)')
    parser.add_argument('--verbose-stdout', action='store_true',
                       help='Also print summary to stdout')
    parser.add_argument('--log-cache-sizes', action='store_true',
                       help='Log cache sizes at each iteration (may be expensive)')
    parser.add_argument('--log-db-sizes', action='store_true',
                       help='Log database sizes at each iteration (expensive)')
    parser.add_argument('--one-solution', action='store_true', help='Stop worker when find a maximal reason')
    parser.add_argument('--sample-timeout', type=float, default=None,
                              help='Maximum time in seconds to process a single sample (default: None - no timeout)')
    parser.add_argument('--save-iteration-cov', type=int, default=1000,
                       help='Save coverage to DATA every N iterations (default: 1000)')
    parser.add_argument('--use-R-cache', action=argparse.BooleanOptionalAction, help='Use R cache/database', default=True)
    parser.add_argument('--use-GP-cache', action=argparse.BooleanOptionalAction, help='Use GP cache/database', default=True)
    parser.add_argument('--use-NR-cache', action=argparse.BooleanOptionalAction, help='Use NR cache/database', default=True)
    parser.add_argument('--use-BP-cache', action=argparse.BooleanOptionalAction, help='Use BP cache/database', default=True)
    args = parser.parse_args()
    
    # Get worker identifier
    if args.worker_name:
        worker_id = args.worker_name
    else:
        worker_id = get_worker_id()
    
    print(f"Worker ID: {worker_id}")

    print(f"Sample timeout: {args.sample_timeout}")

    # Establish Redis connections
    connections = {}
    db_mapping = {
        'DATA': 0,
        'CAN': 1,
        'R': 2,
        'NR': 3,
        'CAR': 4,
        'AR': 5,
        'GP': 6,
        'BP': 7,
        'PR': 8,
        'AP': 9,
        'LOGS': 10  # NEW: Logs database
    }

    print(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    print(f"For ablation R {args.use_R_cache}, GP {args.use_GP_cache}, NR {args.use_NR_cache}, BP {args.use_BP_cache}")
    for name, db_id in db_mapping.items():
        try:
            conn = redis.Redis(host=args.redis_host, port=args.redis_port, db=db_id, decode_responses=True)
            conn.ping()
            connections[name] = conn
            print(f"Connected to Redis DB {db_id} ({name})")
        except redis.ConnectionError:
            print(f"Failed to connect to Redis DB {db_id} ({name})")
            return False

    print(f"Established {len(connections)} Redis connections")

    # Download RF, EU and Label from DATA
    print("\n=== Worker Initialization ===")

    print("Loading Random Forest from DATA['RF']...")
    rf_data = retrieve_forest(connections['DATA'], 'RF')
    if rf_data is None:
        print("Failed to load Random Forest")
        return False
    print(f"Loaded Random Forest with {len(rf_data.trees)} trees")

    print("Loading Endpoints Universe from DATA['EU']...")
    eu_data = retrieve_monotonic_dict(connections['DATA'], 'EU')
    if eu_data is None:
        print("Failed to load Endpoints Universe")
        return False
    print(f"Loaded EU with {len(eu_data)} features")

    print("Loading target label from DATA['label']...")
    label = connections['DATA'].get('label')
    if label is None:
        print("Failed to load target label")
        return False
    print(f"Target label: {label}")

    # Get the nodes from the forest
    nodes = []
    for tree in rf_data.trees:
        nodes.append(tree.root)

    # Initialize caches
    caches = {
        'R': set(),
        'NR': set(),
        'GP': set(),
        'BP': set(),
        'AR': set(),
        'AP': set()
    }
    print(f"Initialized in-memory caches for 6 databases")

    # Main Worker Loop
    print("\n=== Starting Main Worker Loop ===")
    print(f"Logging to Redis LOGS database with key prefix: {worker_id} ")
    print(f"with args {args}")
    iteration = 0
    good = 0
    bad = 0

    start_time = time.time()

    def check_and_select_pr():
        """Check PR and select a candidate if available"""
        return get_pr_candidate_exclusive(connections['PR'], connections['DATA'], worker_name=worker_id)

    def close_pr(candidate, coverage_value, globally_dominated=False, timeout_occurred=False):
        """Check PR and select a candidate if available"""
        return close_pr_candidate(connections['DATA'], candidate, coverage_value, globally_dominated=globally_dominated, timeout_occurred=timeout_occurred)

    def check_and_select_car():
        """Check CAR and select a candidate AR if available"""
        car_key = random_key_from_can(connections['CAR'])
        return car_key
    info = {}
    globally_dominated_count = 0
    try:
        # Initialize

        candidate = check_and_select_pr()
        if candidate is not None:
            candidate_icf = bitmap_to_icf(candidate, eu_data)
        
        # Track sample processing time
        sample_start_time = time.time() if candidate is not None else None

        while True:

            print_count = 0

            if candidate is None:
                break

            # Check if we've exceeded the sample timeout
            if sample_start_time is not None and args.sample_timeout is not None:
                elapsed_sample_time = time.time() - sample_start_time
                if elapsed_sample_time > args.sample_timeout:
                    print(f"Sample timeout reached ({elapsed_sample_time:.2f}s > {args.sample_timeout}s), moving to next sample")
                    one_count = icf_to_bitmap_mask(candidate_icf,eu_data)
                    coverage_value = one_count.count(1)/len(one_count)                    
                    close_pr(candidate, coverage_value, timeout_occurred=True)
                    candidate = check_and_select_pr()
                    if candidate is not None:
                        candidate_icf = bitmap_to_icf(candidate, eu_data)
                        sample_start_time = time.time()  # Reset timer for new sample
                    else:
                        sample_start_time = None
                    continue

            feature_directions = [(feature, direction) for feature in eu_data.keys() for direction in ["low", "high"]]
            #random.shuffle(feature_directions)
            extension_icfs = [extension for extension in [rf_data.inflate_interval(candidate_icf, eu_data, feature, direction) for feature, direction in feature_directions] if extension is not None]
            
            global_dominated =  False
            if args.use_R_cache:
                for ext_icf in extension_icfs:
                    if cache_dominated_icf(ext_icf, eu_data, connections['R'], caches['R'], reverse=False, scan=10, use_db=True):
                        global_dominated = True
                        break

            if global_dominated:
                globally_dominated_count += 1
                one_count = icf_to_bitmap_mask(candidate_icf,eu_data)
                coverage_value = one_count.count(1)/len(one_count)
                close_pr(candidate, coverage_value, globally_dominated=True)
                candidate = check_and_select_pr()
                if candidate is not None:
                    candidate_icf = bitmap_to_icf(candidate, eu_data)
                    sample_start_time = time.time()  # Reset timer for new sample
                else:
                    sample_start_time = None
            else:
                if print_count % args.save_iteration_cov == 0:
                    one_count = icf_to_bitmap_mask(candidate_icf,eu_data)
                    coverage_value = one_count.count(1)/len(one_count)
                    redis_time = connections['DATA'].time()
                    t = f"{redis_time[0]}.{redis_time[1]}"
                    append_coverage_t(connections['DATA'], candidate, coverage_value, t)

                    print(f"coverage {coverage_value}")
                    log_data = {
                        'iteration': iteration,
                        'coverage': coverage_value,
                        'candidate': candidate,
                        'timestamp': t,
                        **{k: v for k, v in info.items()},
                    }
                    log_iteration_to_redis(connections['LOGS'], worker_id, iteration, log_data)
                print_count += 1
                iteration += 1
                extension_is_reason = False
                for ext_icf in extension_icfs:
                    result = rcheck_cache(
                        connections=connections,
                        icf=ext_icf,
                        label=label,
                        nodes=saturate(ext_icf, nodes),
                        eu_data=eu_data,
                        forest=rf_data,
                        caches=caches,
                        info=info,
                        use_R=args.use_R_cache,
                        use_GP=args.use_GP_cache,
                        use_NR=args.use_NR_cache,
                        use_BP=args.use_BP_cache
                    )
                    if result:
                        candidate_icf = ext_icf
                        extension_is_reason = True
                        break
                if not extension_is_reason:
                    one_count = icf_to_bitmap_mask(candidate_icf,eu_data)
                    coverage_value = one_count.count(1)/len(one_count)
                    close_pr(candidate, coverage_value)
                    candidate = check_and_select_pr()
                    if candidate is not None:
                        candidate_icf = bitmap_to_icf(candidate, eu_data)
                        sample_start_time = time.time()  # Reset timer for new sample
                    else:
                        sample_start_time = None


    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Worker Loop Completed ===")
    
    # Final summary
    final_time = time.time()
    total_elapsed = final_time - start_time

    print(f"\n{'='*80}")
    print(f"=== Final Statistics ===")
    print(f"Worker ID: {worker_id}")
    print(f"Total runtime: {total_elapsed:.2f}s")
    print(f"\nAll iteration logs saved to Redis LOGS database")
    if iteration > 0:
        print(f"Log keys: {worker_id}:0 through {worker_id}:{iteration - 1} ({iteration} log entries)")
    else:
        print(f"No iterations logged (worker exited without processing candidates)")
    print(f'Ablation settings used - R: {args.use_R_cache}, GP: {args.use_GP_cache}, NR: {args.use_NR_cache}, BP: {args.use_BP_cache}')
    print(f'info dict: {info}')
    print(f'Globally dominated candidates skipped: {globally_dominated_count}')
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
