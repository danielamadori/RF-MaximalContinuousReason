import os
import time
import json
import subprocess
from pathlib import Path
import csv
import sys
import base64
from init_baseline import load_classifier_from_json, load_dataset_from_baseline
from init_utils import store_forest_and_endpoints
import redis
import shutil
import datetime
from etl.tables import _decode_dump_string
from skforest_to_forest import sklearn_forest_to_forest
CAN_DECODE_DUMP = True

try:
    import redis_backup
except ImportError:
    print("[ERROR] Could not import redis_backup.py. Ensure it is in the same directory.")
    sys.exit(1)
def is_pid_running(pid):
    """Check if process with PID exists."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False

    if os.name == 'nt':
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid_int}", "/NH"],
                capture_output=True,
                text=True,
                check=False
            )
            output = (result.stdout or "").strip()
            if not output or output.lower().startswith("info:"):
                return False
            return str(pid_int) in output
        except Exception:
            return False

    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True



def wait_for_workers():
    """
    Monitor worker_pids.json and wait for all workers to exit.
    Returns True if all workers finished.
    """
    pids_file = Path('workers/worker_pids.json')
    # Wait for pids file to appear (give launch_workers a moment)
    time.sleep(1) 
    
    if not pids_file.exists():
        print("[EXPERIMENT] Warning: workers/worker_pids.json not found after launch.")
        return True # Assume finished or failed to start
        
    print("[EXPERIMENT] Waiting for workers to finish...")
    
    while True:
        if not pids_file.exists():
            print("[EXPERIMENT] Workers file gone. Finished.")
            return True
            
        try:
            with open(pids_file, 'r') as f:
                try:
                    pids_data = json.load(f)
                except json.JSONDecodeError:
                    # File might be empty or partial write                    
                    continue
            
            running_count = 0
            for worker_id, info in pids_data.items():
                pid = info.get('pid')
                if pid and is_pid_running(pid):
                    running_count += 1
            
            if running_count == 0:
                print(f"[EXPERIMENT] All workers exited.")
                return True
                
        except Exception as e:
            print(f"[EXPERIMENT] Error monitoring workers: {e}")
def backup_state(config):
    """Backup all generic databases (0-10)."""
    print("[EXPERIMENT] Creating Redis Backup...")
    return redis_backup.create_multi_database_backup(config)

def append_to_csv_log(filepath, row_dict, fieldnames=None):
    """
    Append a row to the global CSV log.
    row_dict keys must match header.
    """
    file_exists = Path(filepath).exists()
    if fieldnames is None:
        fieldnames = ['Timestamp', 'Dataset', 'Sample_Index', 'Workers', 'Duration', 'Reasons_Count', 'Reasons']
    
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
        print(f"[EXPERIMENT] Logged result to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to write to CSV log: {e}")


def is_binary_key(value):
    return isinstance(value, str) and value and set(value) <= {'0', '1'}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_coverage_values_from_json(value):
    if value is None:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []

    coverage_values = []
    if isinstance(parsed, dict):
        for key, val in parsed.items():
            val_num = _safe_float(val)
            if val_num is None:
                val_num = _safe_float(key)
            if val_num is not None:
                coverage_values.append(val_num)
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                val_num = _safe_float(item[1])
                if val_num is None:
                    val_num = _safe_float(item[0])
                if val_num is not None:
                    coverage_values.append(val_num)
    return coverage_values


def extract_coverage_values(metadata):
    """
    Extract coverage values from a candidate metadata dict.
    Handles 'coverage', 'coverage_t' (JSON map), and 'coverage_*' fields.
    """
    if not isinstance(metadata, dict):
        return []

    coverage_values = []
    for key, value in metadata.items():
        if key == "coverage":
            val_num = _safe_float(value)
            if val_num is not None:
                coverage_values.append(val_num)
        elif key == "coverage_t":
            coverage_values.extend(_extract_coverage_values_from_json(value))
        elif key.startswith("coverage_"):
            if key == "coverage_t":
                continue
            val_num = _safe_float(value)
            if val_num is not None:
                coverage_values.append(val_num)
    return coverage_values


def collect_binary_keys_data(redis_config, db_index=0, scan_count=1000):
    """
    Collect all binary keys from a Redis DB and return a dict of key -> value.
    """
    redis_params = {
        'host': redis_config.get('host', 'localhost'),
        'port': redis_config.get('port', 6379),
        'password': redis_config.get('password', None),
        'db': db_index
    }
    if redis_config.get('username'):
        redis_params['username'] = redis_config['username']
    if redis_config.get('password'):
        redis_params['password'] = redis_config['password']

    r = redis.Redis(**redis_params, decode_responses=True)
    binary_keys_data = {}

    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, count=scan_count)
        for key in keys:
            if not is_binary_key(key):
                continue
            key_type = r.type(key)
            if key_type == "hash":
                value = r.hgetall(key)
            elif key_type == "string":
                value = r.get(key)
            elif key_type == "set":
                value = sorted(list(r.smembers(key)))
            elif key_type == "list":
                value = r.lrange(key, 0, -1)
            elif key_type == "zset":
                value = r.zrange(key, 0, -1, withscores=True)
            else:
                value = None
            binary_keys_data[key] = value
        if cursor == 0:
            break

    return binary_keys_data


def max_coverage_from_binary_keys(binary_keys_data):
    """
    Return the maximum coverage value found in binary key metadata.
    """
    max_coverage = None
    min_vts = None
    max_vte = None
    for metadata in binary_keys_data.values():
        if isinstance(metadata, dict):
            vts_val = _safe_float(metadata.get("vts"))
            vte_val = _safe_float(metadata.get("vte"))
            if vts_val is not None:
                if min_vts is None or vts_val < min_vts:
                    min_vts = vts_val
            if vte_val is not None:
                if max_vte is None or vte_val > max_vte:
                    max_vte = vte_val
            for val in extract_coverage_values(metadata):
                val_num = _safe_float(val)
                if val_num is not None and (max_coverage is None or val_num > max_coverage):
                    max_coverage = val_num
        elif isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                vts_val = _safe_float(parsed.get("vts"))
                vte_val = _safe_float(parsed.get("vte"))
                if vts_val is not None:
                    if min_vts is None or vts_val < min_vts:
                        min_vts = vts_val
                if vte_val is not None:
                    if max_vte is None or vte_val > max_vte:
                        max_vte = vte_val
                for val in extract_coverage_values(parsed):
                    val_num = _safe_float(val)
                    if val_num is not None and (max_coverage is None or val_num > max_coverage):
                        max_coverage = val_num
    duration = None
    if min_vts is not None and max_vte is not None:
        duration = max_vte - min_vts
    return max_coverage, duration


def save_result_to_disk(backup_data, filepath):
    """Save backup to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Using the directory structure save provided by redis_backup might be complex 
    # for a single file per experiment. Let's use json dump directly?
    # redis_backup has save_multi_database_backup_to_directory but that makes multiple files.
    # User said "salvo su file il contenuto". 
    # Let's save as one big JSON for simplicity or use the directory structure.
    # Directory structure is robust. Let's use save_multi_database_backup_to_directory.
    # Actually, saving one JSON per checkpoint is easier to manage than folder per checkpoint?
    # redis_backup.save_backup_to_file is for single DB.
    # Let's just create a folder "checkpoints/sample_X_workers_Y/" and save there.
    redis_backup.save_multi_database_backup_to_directory(backup_data, path)
    print(f"[EXPERIMENT] Saved result to {filepath}")



def save_readable_dump(backup_data, result_dir, redis_config=None):
    """
    Save a human-readable JSON version of the Redis dump.
    Uses etl.tables to decode RDB DUMP payloads.
    """
    if not CAN_DECODE_DUMP:
        return

    redis_clients = {}
    if redis_config:
        for db_id in backup_data.keys():
            db_config = dict(redis_config)
            db_config["db"] = db_id
            try:
                redis_clients[db_id] = redis.Redis(
                    host=db_config.get("host"),
                    port=db_config.get("port", 6379),
                    password=db_config.get("password"),
                    db=db_id,
                    decode_responses=True
                )
            except Exception:
                redis_clients[db_id] = None

    readable_data = {}
    
    for db_id, payload in backup_data.items():
        redis_client = redis_clients.get(db_id)
        db_content = {}
        entries = payload.get('entries', [])
        for entry in entries:
            # Decode Key
            try:
                key_b64 = entry.get('key')
                key = base64.b64decode(key_b64).decode('utf-8', errors='replace')
            except Exception:
                key = f"RAW:{entry.get('key')}"
            
            # Decode Value
            val_readable = "<binary>"
            key_type = entry.get("type")

            # Prefer live Redis decoding when available (handles hashes/sets/etc.)
            if redis_client is not None and key_type:
                try:
                    if key_type == "string":
                        val_readable = redis_client.get(key)
                        if isinstance(val_readable, str):
                            try:
                                val_readable = json.loads(val_readable)
                            except json.JSONDecodeError:
                                pass
                    elif key_type == "hash":
                        val_readable = redis_client.hgetall(key)
                    elif key_type == "set":
                        val_readable = sorted(list(redis_client.smembers(key)))
                    elif key_type == "list":
                        val_readable = redis_client.lrange(key, 0, -1)
                    elif key_type == "zset":
                        val_readable = redis_client.zrange(key, 0, -1, withscores=True)
                    elif key_type == "stream":
                        val_readable = redis_client.xrange(key)
                    else:
                        val_readable = f"<unsupported type: {key_type}>"
                except Exception:
                    val_readable = "<binary>"
            else:
                try:
                    decoded_bytes = _decode_dump_string(entry)
                    if decoded_bytes:
                        # Try UTF-8 string
                        try:
                            val_str = decoded_bytes.decode('utf-8')
                            # Try JSON
                            try:
                                val_readable = json.loads(val_str)
                            except json.JSONDecodeError:
                                val_readable = val_str
                        except UnicodeDecodeError:
                            val_readable = f"<binary: {len(decoded_bytes)} bytes>"
                except Exception as e:
                    val_readable = f"<error decoding: {str(e)}>"
            
            db_content[key] = val_readable
            
        readable_data[str(db_id)] = db_content

    try:
        with open(result_dir / "redis_dump_readable.json", "w", encoding='utf-8') as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        print(f"[EXPERIMENT] Saved readable dump to {result_dir / 'redis_dump_readable.json'}")
    except Exception as e:
        print(f"[ERROR] Failed to save readable dump: {e}")


def save_log_csv(result_dir, dataset_name, num_workers, class_label, redis_config):
    # 7. Extract Results (Binary Keys / Coverage)
    binary_keys_data = {}
    coverage_max = None
    duration = None
    try:
        binary_keys_data = collect_binary_keys_data(redis_config, db_index=0)
        coverage_max, duration = max_coverage_from_binary_keys(binary_keys_data)
        print(f"[EXPERIMENT] Found {len(binary_keys_data)} binary keys in DB0 (DATA).")
    except Exception as e:
        print(f"[WARNING] Failed to extract binary keys/coverage from Redis: {e}")

    result_dir.mkdir(parents=True, exist_ok=True)
    binary_keys_path = result_dir / "binary_keys_data.json"
    try:
        with open(binary_keys_path, "w", encoding="utf-8") as f:
            json.dump(binary_keys_data, f, indent=2, ensure_ascii=True)
        print(f"[EXPERIMENT] Saved binary keys data to {binary_keys_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save binary keys data: {e}")

    coverage_str = f"{coverage_max:.6f}" if coverage_max is not None else ""
    duration_str = f"{duration:.6f}" if isinstance(duration, (int, float)) else ""
    log_entry = {
        'Timestamp': datetime.datetime.now().isoformat(),
        'Dataset': dataset_name,
        'Workers': num_workers,
        'Class_Label': class_label,
        'Duration': duration_str,
        'Coverage': coverage_str,
        'Binary_Keys_Count': len(binary_keys_data),
        'Binary_Keys_File': str(binary_keys_path)

    }
    log_fieldnames = [
        'Timestamp', 'Dataset', 'Workers', 'Class_Label', 'Duration',
        'Coverage', 'Binary_Keys_Count', 'Binary_Keys_File'
    ]
    log_path = Path("results/experiments_log.csv")
    if log_path.exists():
        try:
            with open(log_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader)
            if existing_header != log_fieldnames:
                log_path = Path("results/experiments_log_coverage.csv")
        except Exception:
            log_path = Path("results/experiments_log_coverage.csv")
    append_to_csv_log(str(log_path), log_entry, fieldnames=log_fieldnames)



def run_experiment_loop(dataset_name, class_label, num_workers, redis_config,
                        one_solution=True, sample_timeout=None,                          
                        use_R: bool = True,
                        use_GP: bool = True,
                        use_NR: bool = True,
                        use_BP: bool = True):
    """
    Main loop: Init -> Snapshot -> [Restore -> Run(N) -> Save] * N_CPU

    Args:
        num_workers: If specified, run with this fixed worker count only (new hierarchy)
        preserve_ar: If True, preserve AR database (DB5) when restoring state
        dataset_timeout: Max time in seconds per dataset; converted to per-sample timeout and passed to workers
        sample_timeout: Max time in seconds per sample; passed to workers

    """
    if sample_timeout is not None:
        print(f"[EXPERIMENT] Sample timeout set to {sample_timeout} seconds (worker-level).")
    init_script = 'init_baseline.py'
    print(f"[EXPERIMENT] Target Experiment: {init_script} ")
  
    # 2. Run Init baseline
    print(f"[EXPERIMENT] Step 1: Initializing Environment ({init_script})...")
    cmd_init = [sys.executable, init_script, dataset_name, '--class-label',class_label]

    try:
        subprocess.check_call(cmd_init)
    except subprocess.CalledProcessError as e:
        print(f"[EXPERIMENT] Init script failed with code {e.returncode}. Aborting.")
        sys.exit(e.returncode)
        

    # Check if this worker iteration already exists (resume capability)
    # Directory structure: dataset → workers → class
    result_dir = Path(
        f"results/checkpoints/{dataset_name}/workers_{num_workers}/"
        f"class_{class_label}"
    )
    if use_BP and use_GP and use_NR and use_R:
        ablation_tag = "all_features"
    else:
        ablation_tag = f"ablation_R{int(use_R)}_NR{int(use_NR)}_GP{int(use_GP)}_BP{int(use_BP)}"
    result_dir = result_dir / ablation_tag
    metadata_file = result_dir / "experiment_metadata.json"
    
    if result_dir.exists() and metadata_file.exists():
        print(f"[SKIP] Workers {num_workers} already completed. Skipping to next iteration.")
        return
    
       
    # Capture logs before launch
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir()
    logs_before = set(log_dir.glob("*.log"))

    # 5. Launch Workers
    print(f"[EXPERIMENT] Launching {num_workers} Worker(s)...")
    cmd_launch = [
        sys.executable, 'launch_workers.py', 'start-legacy', 
        str(num_workers), '--sample-timeout', str(sample_timeout)
    ]
    
    if one_solution:
        cmd_launch.append('--one-solution')
    if not use_R:
        cmd_launch.append('--no-use-R-cache')
    if not use_GP:
        cmd_launch.append('--no-use-GP-cache')
    if not use_NR:
        cmd_launch.append('--no-use-NR-cache')
    if not use_BP:
        cmd_launch.append('--no-use-BP-cache')
    worker_args = []

    try:
        subprocess.check_call(cmd_launch)
    except subprocess.CalledProcessError as e:
        print(f"[EXPERIMENT] Launch failed with code {e.returncode}. Aborting.")
        # If launch fails, we probably should abort
        sys.exit(e.returncode)
        
    # 6. Wait for Completion
    print(f"[EXPERIMENT] Waiting for solution...")
    start_wait = time.time()
    wait_for_workers()
    duration = time.time() - start_wait

    print(f"[EXPERIMENT] Iteration {num_workers} completed in {duration:.2f} seconds.")

    
    # 8. Log to CSV
    # save_log_csv(result_dir, dataset_name, num_workers, class_label, redis_config)
    # 9. Save Results Snapshot
    print(f"[EXPERIMENT] Saving results to {result_dir}...")
    
    # We assume workers wrote to Redis, so we dump Redis.
    final_state = backup_state(redis_config)
    save_result_to_disk(final_state, result_dir)
    save_readable_dump(final_state, result_dir, redis_config=redis_config)


    print("[EXPERIMENT] cleaning DB (implicit in next restore)")
    
    # 11. Archive Logs
    logs_after = set(log_dir.glob("*.log"))
    new_logs = logs_after - logs_before
    
    if new_logs:
        dest_log_dir = result_dir / "logs"
        dest_log_dir.mkdir(exist_ok=True)
        for log_file in new_logs:
            try:
                shutil.copy2(log_file, dest_log_dir)
                print(f"[EXPERIMENT] Archived log {log_file.name}")
            except Exception as e:
                print(f"[WARNING] Failed to archive log {log_file}: {e}")


def get_classes(dataset_name, connections):
    """
    Get the list of class labels for a given dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        list: List of class labels
    """
    X_train, X_test_samples, y_train, _, feature_names, all_classes, data = load_dataset_from_baseline(dataset_name)
    sklearn_rf, classifier_path = load_classifier_from_json(dataset_name)

    our_forest = sklearn_forest_to_forest(sklearn_rf, feature_names)
    X_test = [dict(zip(feature_names,  x_test) ) for x_test in X_test_samples]
    predictions = [our_forest.predict(x_test) for x_test in X_test ]
    return list(set(predictions))