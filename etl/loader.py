import json
import math
import os
import re
import statistics
import zipfile
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt, dates as mdates

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

from etl.zip_inspector import scan_and_load, parse_zip_metadata, collect_archive_data, decode_key, try_decode_value, \
    summarise_entry_generic, DB_LABELS, _fetch_backup_metadata, summarise_entry, CANDIDATE_ANTI_REASONS_NAME, \
    CANDIDATE_REASONS_NAME, DB_DISPLAY_NAMES, BASE_COLUMNS, SCATTER_PLOTS, ADDITIONAL_SCATTER_PREFIX_PAIRS, BAR_PLOTS, \
    STACKED_BAR_CONFIG, HISTOGRAMS


def etl(zip_paths, results_dir):
    selected_zip_name = scan_and_load(zip_paths, results_dir)
    archives_metadata = [parse_zip_metadata(path) for path in zip_paths]
    archives_data = [collect_archive_data(path) for path in zip_paths]
    manifests_by_archive = {item['zip_name']: item['manifest'] for item in archives_data}
    manifest_prefix_by_archive = {item['zip_name']: item.get('manifest_prefix', '') for item in archives_data}
    backups_by_archive = {item['zip_name']: item['backups'] for item in archives_data}
    selected_archive_data = next((item for item in archives_data if item['zip_name'] == selected_zip_name), None)
    selected_manifest = manifests_by_archive.get(selected_zip_name)
    selected_manifest_prefix = manifest_prefix_by_archive.get(selected_zip_name, '')
    selected_backups = backups_by_archive.get(selected_zip_name)

    summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix)

    selected_db0_file_name = None
    if selected_manifest:
        files_map = selected_manifest.get('files', {}) or {}
        selected_db0_file_name = files_map.get('0')

    selected_db0_backup = None
    if selected_backups and selected_db0_file_name:
        selected_db0_backup = selected_backups.get(selected_db0_file_name)

    selected_db0_entries = []
    selected_db0_values = []
    if selected_db0_backup:
        selected_db0_entries = selected_db0_backup.get('entries') or []
        for entry in selected_db0_entries:
            try:
                key_bytes = decode_key(entry)
                key_text = key_bytes.decode('utf-8', errors='replace')
            except Exception as exc:
                key_text = f'<unable to decode key: {exc}>'
            preview, details = try_decode_value(entry)
            value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
            if isinstance(value_bytes, (bytes, bytearray)):
                value_text = value_bytes.decode('utf-8', errors='replace')
            else:
                value_text = str(preview)
            value_json = None
            if isinstance(value_text, str):
                try:
                    value_json = json.loads(value_text)
                except Exception:
                    value_json = None
            selected_db0_values.append({
                'key': key_text,
                'type': entry.get('type'),
                'ttl_ms': entry.get('pttl'),
                'value_text': value_text,
                'value_bytes': value_bytes,
                'value_json': value_json,
                'details': details,
            })

    selected_db0_values_by_key = {item['key']: item for item in selected_db0_values}

    db1_entries = []
    if selected_manifest and selected_backups is not None:
        files_map = selected_manifest.get('files', {}) or {}
        db1_file = files_map.get('1')
        if db1_file:
            data = selected_backups.get(db1_file) if isinstance(selected_backups, dict) else None
            if isinstance(data, dict):
                db1_entries = data.get('entries') or []

    db1_entries_summary = []
    for entry in db1_entries:
        try:
            key_bytes = decode_key(entry)
            key_text = key_bytes.decode('utf-8', errors='replace')
        except Exception as exc:
            key_text = f'<unable to decode key: {exc}>'
        preview, details = try_decode_value(entry)
        value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
        if isinstance(value_bytes, (bytes, bytearray)):
            value_text = value_bytes.decode('utf-8', errors='replace')
        else:
            value_text = str(preview)
        value_json = None
        if isinstance(value_text, str):
            try:
                value_json = json.loads(value_text)
            except Exception:
                value_json = None
        headline, extra = summarise_entry_generic(key_text, value_json, value_text)
        db1_entries_summary.append({
            'key': key_text,
            'type': entry.get('type'),
            'ttl_ms': entry.get('pttl'),
            'headline': headline,
            'details': extra,
        })

    other_db_summaries = []
    if selected_manifest and selected_backups is not None:
        files_map = selected_manifest.get('files', {}) or {}
        selected_zip_path = Path(selected_archive_data['zip_path']) if selected_archive_data else None
        for db_index in range(1, 10):
            file_name = files_map.get(str(db_index))
            if not file_name or selected_zip_path is None:
                continue
            label = DB_LABELS.get(db_index, 'Unknown')
            data = selected_backups.get(file_name) if isinstance(selected_backups, dict) else None
            metadata = {}
            if isinstance(data, dict):
                metadata = data.get('metadata') or {}
            if not metadata:
                member_name = f"{selected_manifest_prefix}{file_name}"
                metadata = _fetch_backup_metadata(selected_zip_path, member_name)
            key_count = metadata.get('key_count') if isinstance(metadata, dict) else None
            type_summary = metadata.get('type_summary') if isinstance(metadata, dict) else None
            other_db_summaries.append({
                'db_index': db_index,
                'label': label,
                'file_name': file_name,
                'key_count': key_count,
                'type_summary': type_summary if isinstance(type_summary, dict) else None,
            })

    selected_db0_summary = []
    for item in selected_db0_values:
        headline, extra = summarise_entry(item['key'], item['value_json'], item['value_text'], skip_sample_keys=True)
        if headline is None and extra is None:
            continue
        selected_db0_summary.append({
            'key': item['key'],
            'type': item['type'],
            'ttl_ms': item['ttl_ms'],
            'headline': headline,
            'details': extra,
        })

    if selected_db0_summary:
        print(f"DB 0 entries for {selected_zip_name}:")
        for entry in selected_db0_summary:
            ttl = entry['ttl_ms'] if isinstance(entry['ttl_ms'], int) else 'persistent'
            print(f"{entry['key']} (type={entry['type']}, ttl={ttl})")
            print(f"{entry['headline']}")
            for detail in entry['details']:
                print(f"    {detail}")
    else:
        print('No DB 0 data available for the current selection.')



    return selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix


def summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, *, show_summary=True):
        global DB10_WORKER_CACHE
        DB10_WORKER_CACHE = {}
        if not selected_zip_name:
            print("No archive selected.")
            return
        files_map = (selected_manifest or {}).get('files', {}) or {}
        db10_file = files_map.get('10')
        if not db10_file:
            print("No DB 10 available for the current selection.")
            return
        backups = selected_backups if isinstance(selected_backups, dict) else {}
        db10_data = backups.get(db10_file)
        if not isinstance(db10_data, dict):
            zip_path_str = selected_archive_data.get('zip_path') if selected_archive_data else None
            if zip_path_str:
                zip_path = Path(zip_path_str)
                if zip_path.exists():
                    try:
                        with zipfile.ZipFile(zip_path) as archive:
                            payload = archive.read(f"{selected_manifest_prefix}{db10_file}")
                        db10_data = json.loads(payload.decode('utf-8', errors='replace'))
                    except Exception:
                        db10_data = None
        if not isinstance(db10_data, dict):
            print("Unable to load DB 10 data.")
            return
        entries = db10_data.get('entries') or []
        if not entries:
            print("No entries available in DB 10.")
            DB10_WORKER_CACHE = {
                'zip_name': selected_zip_name,
                'db10_file': db10_file,
                'entries': entries,
                'db10_data': db10_data,
                'worker_stats': {},
                'worker_summaries': [],
            }
            return
        worker_stats = {}
        for entry in entries:
            preview, details = try_decode_value(entry)
            raw_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
            if isinstance(raw_bytes, (bytes, bytearray)):
                text = raw_bytes.decode('utf-8', errors='replace')
            elif isinstance(preview, (bytes, bytearray)):
                text = preview.decode('utf-8', errors='replace')
            else:
                text = str(preview)
            try:
                payload = json.loads(text)
            except Exception:
                continue
            worker_id = payload.get('worker_id')
            if not worker_id:
                try:
                    key_text = decode_key(entry).decode('utf-8', errors='replace')
                except Exception:
                    key_text = entry.get('key')
                if isinstance(key_text, str) and ':' in key_text:
                    worker_id = key_text.rsplit(':', 1)[0]
                else:
                    worker_id = str(key_text)
            stats = worker_stats.setdefault(worker_id, {
                'records': 0,
                'iterations': [],
                'queue_sizes': [],
                'car_queue_sizes': [],
                'total_seconds': [],
                'car_seconds': [],
                'can_seconds': [],
                'car_results': defaultdict(int),
                'can_results': defaultdict(int),
                'outcomes': defaultdict(int),
                'events': [],
                'raw_info_totals': defaultdict(float),
                'extensions_totals': defaultdict(float),
            })
            stats['records'] += 1
            iteration = payload.get('iteration')
            if isinstance(iteration, (int, float)):
                stats['iterations'].append(int(iteration))
            queue_size = payload.get('queue_size')
            if isinstance(queue_size, (int, float)):
                stats['queue_sizes'].append(float(queue_size))
            car_queue_size = payload.get('car_queue_size')
            if isinstance(car_queue_size, (int, float)):
                stats['car_queue_sizes'].append(float(car_queue_size))
            timings = payload.get('timings') or {}
            total_seconds = timings.get('total_seconds')
            if isinstance(total_seconds, (int, float)):
                stats['total_seconds'].append(float(total_seconds))
            car_seconds = timings.get('car_seconds')
            if isinstance(car_seconds, (int, float)):
                stats['car_seconds'].append(float(car_seconds))
            can_seconds = timings.get('can_seconds')
            if isinstance(can_seconds, (int, float)):
                stats['can_seconds'].append(float(can_seconds))
            car_result = (payload.get('car_processing') or {}).get('result')
            if car_result:
                stats['car_results'][car_result] += 1
            can_result = (payload.get('can_processing') or {}).get('result')
            if can_result:
                stats['can_results'][can_result] += 1
            for outcome_key, outcome_value in (payload.get('outcomes') or {}).items():
                if isinstance(outcome_value, bool):
                    label = f"{outcome_key}={'T' if outcome_value else 'F'}"
                    stats['outcomes'][label] += 1
            for prefix, processing in (( 'car', payload.get('car_processing') or {}), ('can', payload.get('can_processing') or {})):
                raw_info = processing.get('raw_info')
                if isinstance(raw_info, dict):
                    for key, value in raw_info.items():
                        if isinstance(value, (int, float)):
                            stats['raw_info_totals'][f'{prefix}_{key}'] += float(value)
                extensions = processing.get('extensions')
                if isinstance(extensions, dict):
                    for key, value in extensions.items():
                        if isinstance(value, (int, float)):
                            stats['extensions_totals'][f'{prefix}_{key}'] += float(value)
            stats['events'].append(payload)
        def summarize(values):
            if not values:
                return (None, None, None)
            return (min(values), statistics.mean(values), max(values))
        def format_number(value, digits=1):
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else '-'
        def format_hours(values):
            total = sum(values)
            return total / 3600 if total else 0.0
        def format_counter(mapping):
            if not mapping:
                return '-'
            items = sorted(dict(mapping).items())
            return ', '.join(f"{key}:{value}" for key, value in items)
        workers = []
        for worker_id, stats in worker_stats.items():
            queue_min, queue_avg, queue_max = summarize(stats['queue_sizes'])
            car_min, car_avg, car_max = summarize(stats['car_queue_sizes'])
            workers.append({
                'worker_id': worker_id,
                'records': stats['records'],
                'iter_min': min(stats['iterations']) if stats['iterations'] else None,
                'iter_max': max(stats['iterations']) if stats['iterations'] else None,
                'queue_avg': queue_avg,
                'car_queue_avg': car_avg,
                'queue_range': (queue_min, queue_max),
                'car_queue_range': (car_min, car_max),
                'total_hours': format_hours(stats['total_seconds']),
                'car_hours': format_hours(stats['car_seconds']),
                'can_hours': format_hours(stats['can_seconds']),
                'car_results': dict(stats['car_results']),
                'can_results': dict(stats['can_results']),
                'outcomes': dict(stats['outcomes']),
            })
        workers.sort(key=lambda item: item['worker_id'])
        DB10_WORKER_CACHE = {
            'zip_name': selected_zip_name,
            'db10_file': db10_file,
            'entries': entries,
            'db10_data': db10_data,
            'worker_stats': worker_stats,
            'worker_summaries': workers,
        }
        if show_summary:
            print(f"DB 10: {len(entries)} events, {len(workers)} workers for {selected_zip_name}")
        if not workers:
            return
        if not show_summary:
            return
        for index, worker in enumerate(workers, start=1):
            label = f"W{index:02d}"
            iter_range = f"{worker['iter_min']}–{worker['iter_max']}" if worker['iter_min'] is not None else '-'
            queue_range = worker['queue_range']
            car_queue_range = worker['car_queue_range']
            queue_range_text = f"{format_number(queue_range[0])}-{format_number(queue_range[1])}" if queue_range[0] is not None else '-'
            car_range_text = f"{format_number(car_queue_range[0])}-{format_number(car_queue_range[1])}" if car_queue_range[0] is not None else '-'
            queue_avg_text = format_number(worker['queue_avg'])
            car_queue_avg_text = format_number(worker['car_queue_avg'])
            total_hours_text = format_number(worker['total_hours'], digits=2)
            car_hours_text = format_number(worker['car_hours'], digits=2)
            can_hours_text = format_number(worker['can_hours'], digits=2)
            print(
                f"- {label}: records={worker['records']}, iter={iter_range}, "
                f"queue_avg={queue_avg_text}, car_queue_avg={car_queue_avg_text}, "
                f"queue_range={queue_range_text}, car_queue_range={car_range_text}, "
                f"h_tot={total_hours_text}, h_car={car_hours_text}, h_can={can_hours_text}"
            )
            print(
                "  "
                + " | ".join([
                    f"car_results[{format_counter(worker['car_results'])}]",
                    f"can_results[{format_counter(worker['can_results'])}]",
                    f"outcomes[{format_counter(worker['outcomes'])}]",
                ])
            )


def _select_worker_id(worker_stats):
    if not worker_stats:
        return None
    workers = sorted(worker_stats)
    label_map = _build_worker_label_map(workers)
    selected = globals().get('SELECTED_DB10_WORKER')
    env_selected = os.environ.get('RESULTS_SELECTED_WORKER')
    if env_selected in worker_stats:
        selected = env_selected
    if selected not in worker_stats:
        selected = workers[0]
    print("Workers found in DB 10:")
    for idx, wid in enumerate(workers):
        marker = "*" if wid == selected else " "
        print(f"{marker}[{idx}] {label_map[wid]}")
    user_choice = input("Select worker by index or label (press Enter to keep the current selection): ").strip()
    if user_choice:
        resolved = None
        if user_choice.isdigit():
            index = int(user_choice)
            if 0 <= index < len(workers):
                resolved = workers[index]
        elif user_choice.upper() in label_map.values():
            label_to_id = {label: wid for wid, label in label_map.items()}
            resolved = label_to_id.get(user_choice.upper())
        if resolved is None and user_choice in worker_stats:
            resolved = user_choice
        if resolved is None:
            print("Invalid selection, keeping the current worker.")
        else:
            selected = resolved
    globals()['SELECTED_DB10_WORKER'] = selected
    return selected


def inspect_db10_worker(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, worker_id=None, *, max_events=5, sort_by='iteration', interactive=False):
    cache = globals().get('DB10_WORKER_CACHE')
    if not cache or cache.get('zip_name') != selected_zip_name:
        summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, show_summary=False)
        cache = globals().get('DB10_WORKER_CACHE')
    if not cache:
        print("DB 10 statistics are not available.")
        return
    worker_stats = cache.get('worker_stats') or {}
    if not worker_stats:
        print("No workers found in DB 10.")
        return
    label_map = _build_worker_label_map(worker_stats.keys())
    if worker_id and worker_id in worker_stats:
        globals()['SELECTED_DB10_WORKER'] = worker_id
    selected_worker = worker_id or globals().get('SELECTED_DB10_WORKER')
    if selected_worker not in worker_stats:
        if interactive:
            selected_worker = _select_worker_id(worker_stats)
        else:
            print('Specify a worker label (e.g., W01) or call the function with interactive=True to pick one interactively.')
            print('Available workers:')
            for wid in sorted(worker_stats):
                print(f" - {label_map.get(wid, wid)}")
            return
    elif interactive:
        change = input("Press Enter to keep the current worker or type 'c' to choose a different worker: ").strip().lower()
        if change == 'c':
            selected_worker = _select_worker_id(worker_stats)
        else:
            globals()['SELECTED_DB10_WORKER'] = selected_worker
    else:
        globals()['SELECTED_DB10_WORKER'] = selected_worker
    if selected_worker not in worker_stats:
        print('No worker selected.')
        return
    stats = worker_stats[selected_worker]
    worker_label = label_map.get(selected_worker, selected_worker)
    def fmt(value, digits=2):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else '-'
    def safe_mean(values):
        return statistics.mean(values) if values else None
    def prettify_metric(name: str) -> str:
        parts = name.replace('-', '_').split('_')
        return ' '.join(part.upper() if len(part) <= 3 and part.isupper() else part.capitalize() for part in parts if part)
    def metric_label(metric_key: str, category: str) -> str:
        if metric_key.startswith('car_'):
            return f"{CANDIDATE_ANTI_REASONS_NAME} {category.lower()} {prettify_metric(metric_key[4:])}"
        if metric_key.startswith('can_'):
            return f"{CANDIDATE_REASONS_NAME} {category.lower()} {prettify_metric(metric_key[4:])}"
        return f"{category} {prettify_metric(metric_key)}"
    iterations = stats.get('iterations', [])
    queue_sizes = stats.get('queue_sizes', [])
    car_queue_sizes = stats.get('car_queue_sizes', [])
    total_seconds = stats.get('total_seconds', [])
    car_seconds = stats.get('car_seconds', [])
    can_seconds = stats.get('can_seconds', [])
    print(f"Worker {worker_label} in DB 10 ({cache.get('zip_name')})")
    print(f"- recorded events: {stats.get('records', 0)}")
    if iterations:
        print(f"- iteration range: {min(iterations)}–{max(iterations)}")
    else:
        print('- iteration range: -')
    if queue_sizes:
        print(
            f"- queue size: avg {fmt(safe_mean(queue_sizes), 1)}, "
            f"min {fmt(min(queue_sizes), 0)}, max {fmt(max(queue_sizes), 0)}"
        )
    else:
        print('- queue size: no data')
    if car_queue_sizes:
        print(
            f"- CAR queue size: avg {fmt(safe_mean(car_queue_sizes), 1)}, "
            f"min {fmt(min(car_queue_sizes), 0)}, max {fmt(max(car_queue_sizes), 0)}"
        )
    else:
        print('- CAR queue size: no data')
    def sum_hours(values):
        return fmt(sum(values) / 3600 if values else None, 2)
    print(f"- total hours: {sum_hours(total_seconds)}")
    print(f"  - CAR: {sum_hours(car_seconds)}")
    print(f"  - CAN: {sum_hours(can_seconds)}")
    def format_counter(mapping):
        if not mapping:
            return '-'
        items = sorted(dict(mapping).items())
        return ', '.join(f"{key}:{value}" for key, value in items)
    def format_metric_map(metric_map, category):
        if not metric_map:
            return '-'
        pairs = []
        for key, value in sorted(metric_map.items()):
            label = metric_label(key, category)
            if isinstance(value, float) and value.is_integer():
                value_text = str(int(value))
            else:
                value_text = fmt(value, 1)
            pairs.append(f"{label}:{value_text}")
        return ', '.join(pairs)
    print(f"- CAR outcomes: {format_counter(stats.get('car_results'))}")
    print(f"- CAN outcomes: {format_counter(stats.get('can_results'))}")
    print(f"- outcomes: {format_counter(stats.get('outcomes'))}")
    print(f"- Raw info totals: {format_metric_map(stats.get('raw_info_totals') or {}, 'Raw info')}")
    print(f"- Extensions totals: {format_metric_map(stats.get('extensions_totals') or {}, 'Extensions')}")
    events = list(stats.get('events') or [])
    if not events:
        print('No event details stored.')
        return
    def event_key(event):
        if sort_by == 'timestamp':
            return (event.get('timestamp_start') or '', event.get('iteration') or float('inf'))
        iteration = event.get('iteration')
        return (iteration if iteration is not None else float('inf'), event.get('timestamp_start') or '')
    events_sorted = sorted(events, key=event_key)
    selected_events = events_sorted[-max_events:] if max_events and max_events > 0 else events_sorted
    print(f"Showing {len(selected_events)} events out of {len(events_sorted)} (sorted by {sort_by}).")
    for event in selected_events:
        iteration = event.get('iteration')
        timestamp_start = event.get('timestamp_start')
        timestamp_end = event.get('timestamp_end')
        queue_size = event.get('queue_size')
        car_queue_size = event.get('car_queue_size')
        timings = event.get('timings') or {}
        car_processing = event.get('car_processing') or {}
        can_processing = event.get('can_processing') or {}
        outcomes = event.get('outcomes') or {}
        line = (
            f"- iter={iteration}, start={timestamp_start}, end={timestamp_end}, "
            f"queue={queue_size}, car_queue={car_queue_size}, "
            f"tot_s={fmt(timings.get('total_seconds'), 1)}, "
            f"car={car_processing.get('result') or '-'} ({fmt(timings.get('car_seconds'), 1)}s), "
            f"can={can_processing.get('result') or '-'} ({fmt(timings.get('can_seconds'), 1)}s)"
        )
        print(line)
        extra_parts = []
        if car_processing.get('time_seconds'):
            extra_parts.append(f"car_step={fmt(car_processing.get('time_seconds'), 1)}s")
        if can_processing.get('time_seconds'):
            extra_parts.append(f"can_step={fmt(can_processing.get('time_seconds'), 1)}s")
        for prefix, processing in (('car', car_processing), ('can', can_processing)):
            proc_raw_info = processing.get('raw_info')
            if isinstance(proc_raw_info, dict) and proc_raw_info:
                summary = ', '.join(f"{key}:{value}" for key, value in sorted(proc_raw_info.items()))
                extra_parts.append(f"{prefix}_raw_info[{summary}]")
            proc_extensions = processing.get('extensions')
            if isinstance(proc_extensions, dict) and proc_extensions:
                summary = ', '.join(f"{key}:{value}" for key, value in sorted(proc_extensions.items()))
                extra_parts.append(f"{prefix}_extensions[{summary}]")
        extensions = event.get('extensions')
        if isinstance(extensions, dict) and extensions:
            ext_summary = ', '.join(f"{k}:{v}" for k, v in sorted(extensions.items()))
            extra_parts.append(f"extensions[{ext_summary}]")
        raw_info = event.get('raw_info')
        if isinstance(raw_info, dict) and raw_info:
            info_summary = ', '.join(f"{k}:{v}" for k, v in sorted(raw_info.items()))
            extra_parts.append(f"raw_info[{info_summary}]")
        if outcomes:
            outcomes_summary = ', '.join(
                f"{key}={'T' if bool(value) else 'F'}" for key, value in sorted(outcomes.items())
            )
            extra_parts.append(f"outcomes[{outcomes_summary}]")
        if extra_parts:
            print("  " + " | ".join(extra_parts))


def build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, *, max_events=None, sort_by='iteration'):
    cache = globals().get('DB10_WORKER_CACHE')
    if not cache or cache.get('zip_name') != selected_zip_name:
        #summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, show_summary=False)
        cache = globals().get('DB10_WORKER_CACHE')
    if not cache:
        print('DB 10 statistics are not available.')
        return {}
    worker_stats = cache.get('worker_stats') or {}
    if not worker_stats:
        print('No workers found in DB 10.')
        return {}

    def safe_mean(values):
        return statistics.mean(values) if values else None

    def hours(values):
        return sum(values) / 3600 if values else 0.0

    def event_key(event):
        if sort_by == 'timestamp':
            return (event.get('timestamp_start') or '', event.get('iteration') or float('inf'))
        iteration = event.get('iteration')
        return (iteration if iteration is not None else float('inf'), event.get('timestamp_start') or '')

    def serialize_event(event):
        timings = event.get('timings') or {}
        car_processing = event.get('car_processing') or {}
        can_processing = event.get('can_processing') or {}
        return {
            'iteration': event.get('iteration'),
            'timestamp_start': event.get('timestamp_start'),
            'timestamp_end': event.get('timestamp_end'),
            'queue_size': event.get('queue_size'),
            'car_queue_size': event.get('car_queue_size'),
            'timings': {
                'total_seconds': timings.get('total_seconds'),
                'car_seconds': timings.get('car_seconds'),
                'can_seconds': timings.get('can_seconds'),
            },
            'car_processing': {
                'result': car_processing.get('result'),
                'time_seconds': car_processing.get('time_seconds'),
                'raw_info': car_processing.get('raw_info'),
                'extensions': car_processing.get('extensions'),
            },
            'can_processing': {
                'result': can_processing.get('result'),
                'time_seconds': can_processing.get('time_seconds'),
                'raw_info': can_processing.get('raw_info'),
                'extensions': can_processing.get('extensions'),
            },
            'outcomes': event.get('outcomes'),
        }

    report = {
        'zip_name': cache.get('zip_name'),
        'worker_count': len(worker_stats),
        'workers': {}
    }

    for worker_id in sorted(worker_stats):
        stats = worker_stats[worker_id]
        iterations = stats.get('iterations') or []
        queue_sizes = stats.get('queue_sizes') or []
        car_queue_sizes = stats.get('car_queue_sizes') or []
        summary = {
            'records': stats.get('records', 0),
            'iteration_range': [min(iterations), max(iterations)] if iterations else None,
            'queue': {
                'min': min(queue_sizes) if queue_sizes else None,
                'mean': safe_mean(queue_sizes),
                'max': max(queue_sizes) if queue_sizes else None,
            },
            'car_queue': {
                'min': min(car_queue_sizes) if car_queue_sizes else None,
                'mean': safe_mean(car_queue_sizes),
                'max': max(car_queue_sizes) if car_queue_sizes else None,
            },
            'timings_hours': {
                'total': hours(stats.get('total_seconds') or []),
                'car': hours(stats.get('car_seconds') or []),
                'can': hours(stats.get('can_seconds') or []),
            },
            'car_results': dict(stats.get('car_results') or {}),
            'can_results': dict(stats.get('can_results') or {}),
            'outcomes': dict(stats.get('outcomes') or {}),
            'raw_info_totals': dict(stats.get('raw_info_totals') or {}),
            'extensions_totals': dict(stats.get('extensions_totals') or {}),
        }
        events = list(stats.get('events') or [])
        events_sorted = sorted(events, key=event_key)
        if max_events is not None and max_events > 0:
            events_sorted = events_sorted[-max_events:]
        summary['events'] = [serialize_event(event) for event in events_sorted]
        report['workers'][worker_id] = summary

    globals()['DB10_WORKER_REPORT'] = report
    print(f"Report built for {report['worker_count']} workers.")
    return report


def _ensure_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix):
    report = globals().get('DB10_WORKER_REPORT')
    if not report or report.get('zip_name') != selected_zip_name:
        report = build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, max_events=None)
    return report


def _build_worker_label_map(worker_ids):
    worker_ids = sorted(worker_ids)
    return {worker_id: f"W{index:02d}" for index, worker_id in enumerate(worker_ids, start=1)}


def _flatten_worker_summary(report):
    rows = []
    car_keys = set()
    can_keys = set()
    outcome_keys = set()
    raw_info_keys = set()
    extensions_keys = set()
    workers = report.get('workers') or {}
    for summary in workers.values():
        car_keys.update((summary.get('car_results') or {}).keys())
        can_keys.update((summary.get('can_results') or {}).keys())
        outcome_keys.update((summary.get('outcomes') or {}).keys())
        raw_info_keys.update((summary.get('raw_info_totals') or {}).keys())
        extensions_keys.update((summary.get('extensions_totals') or {}).keys())
    car_keys = sorted(car_keys)
    can_keys = sorted(can_keys)
    outcome_keys = sorted(outcome_keys)
    raw_info_keys = sorted(raw_info_keys)
    extensions_keys = sorted(extensions_keys)
    sorted_workers = sorted(workers.items())
    for index, (worker_id, summary) in enumerate(sorted_workers, start=1):
        queue = summary.get('queue') or {}
        car_queue = summary.get('car_queue') or {}
        timings = summary.get('timings_hours') or {}
        iteration_range = summary.get('iteration_range') or [None, None]
        short_name = worker_id.split(':')[-1]
        label = f"W{index:02d}"
        row = OrderedDict(
            worker_id=worker_id,
            worker_index=index,
            worker_label=label,
            worker_short_name=short_name,
            records=summary.get('records'),
            iter_min=iteration_range[0],
            iter_max=iteration_range[1],
            queue_min=queue.get('min'),
            queue_mean=queue.get('mean'),
            queue_max=queue.get('max'),
            car_queue_min=car_queue.get('min'),
            car_queue_mean=car_queue.get('mean'),
            car_queue_max=car_queue.get('max'),
            hours_total=timings.get('total'),
            hours_car=timings.get('car'),
            hours_can=timings.get('can'),
        )
        car_results = summary.get('car_results') or {}
        can_results = summary.get('can_results') or {}
        outcomes = summary.get('outcomes') or {}
        for key in car_keys:
            row[f'car_result_{key}'] = car_results.get(key, 0)
        for key in can_keys:
            row[f'can_result_{key}'] = can_results.get(key, 0)
        for key in outcome_keys:
            row[f'outcome_{key}'] = outcomes.get(key, 0)
        raw_info_totals = summary.get('raw_info_totals') or {}
        extensions_totals = summary.get('extensions_totals') or {}
        for key in raw_info_keys:
            value = raw_info_totals.get(key, 0.0)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            row[f'raw_info_{key}'] = value
        for key in extensions_keys:
            value = extensions_totals.get(key, 0.0)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            row[f'extensions_{key}'] = value
        rows.append(row)
    return rows


def _filter_valid_points(df: pd.DataFrame, x_key: str, y_key: str):
    subset = df.dropna(subset=[x_key, y_key])
    subset = subset[subset[[x_key, y_key]].applymap(lambda v: isinstance(v, (int, float))).all(axis=1)]
    if subset.empty:
        return []
    return list(zip(subset[x_key], subset[y_key], subset['worker_label']))


def _token_to_display(token: str) -> str:
    upper = token.upper()
    if upper in DB_DISPLAY_NAMES:
        return DB_DISPLAY_NAMES[upper]
    lookup = {
        'CAR': CANDIDATE_ANTI_REASONS_NAME,
        'CAN': CANDIDATE_REASONS_NAME,
    }
    if upper in lookup:
        return lookup[upper]
    if token.isupper():
        return token.capitalize()
    return token.replace('_', ' ').capitalize()


def _format_result_label(prefix: str, raw: str) -> str:
    parts = raw.split('_') if raw else []
    if parts and parts[-1].upper() in DB_DISPLAY_NAMES:
        parts = parts[:-1]
    descriptor = ' '.join(_token_to_display(part.lower()) for part in parts) if parts else 'Total'
    return f"{prefix} result: {descriptor}"


def _format_outcome_label(raw: str) -> str:
    key, sep, value = raw.partition('=')
    tokens = key.split('_') if key else []
    words = [_token_to_display(tok) for tok in tokens]
    label = ' '.join(words) if words else 'Outcome'
    if sep:
        value_text = {'T': 'True', 'F': 'False', 'TRUE': 'True', 'FALSE': 'False'}.get(value.upper(), value)
        return f"Outcome: {label} ({value_text})"
    return f"Outcome: {label}"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'worker_label': 'Worker',
        'records': 'Records',
        'iter_min': 'Iteration min',
        'iter_max': 'Iteration max',
        'queue_min': 'Queue min',
        'queue_mean': 'Queue mean',
        'queue_max': 'Queue max',
        'car_queue_min': f"{CANDIDATE_ANTI_REASONS_NAME} queue min",
        'car_queue_mean': f"{CANDIDATE_ANTI_REASONS_NAME} queue mean",
        'car_queue_max': f"{CANDIDATE_ANTI_REASONS_NAME} queue max",
        'hours_total': 'Total hours',
        'hours_car': f"{CANDIDATE_ANTI_REASONS_NAME} hours",
        'hours_can': f"{CANDIDATE_REASONS_NAME} hours",
        'event_order': 'Event order',
        'timestamp_start': 'Start time',
        'timestamp_end': 'End time',
        'car_queue_size': f"{CANDIDATE_ANTI_REASONS_NAME} queue size",
        'total_seconds': 'Total seconds',
        'car_seconds': f"{CANDIDATE_ANTI_REASONS_NAME} seconds",
        'can_seconds': f"{CANDIDATE_REASONS_NAME} seconds",
        'car_result': f"{CANDIDATE_ANTI_REASONS_NAME} result",
        'can_result': f"{CANDIDATE_REASONS_NAME} result",
        'outcomes': 'Outcomes',
    }
    def _prettify_metric(name: str) -> str:
        parts = name.replace('-', '_').split('_')
        return ' '.join(part.upper() if len(part) <= 3 and part.isupper() else part.capitalize() for part in parts if part)
    def _format_metric(metric_key: str, category: str) -> str:
        prefix_map = {
            'car': CANDIDATE_ANTI_REASONS_NAME,
            'can': CANDIDATE_REASONS_NAME,
        }
        if '_' in metric_key:
            prefix, remainder = metric_key.split('_', 1)
            if prefix in prefix_map:
                return f"{prefix_map[prefix]} {category.lower()} {_prettify_metric(remainder)}"
        return f"{category} {_prettify_metric(metric_key)}"
    computed = {}
    for col in df.columns:
        if col.startswith('car_result_'):
            computed[col] = _format_result_label(CANDIDATE_ANTI_REASONS_NAME, col[len('car_result_'):])
        elif col.startswith('can_result_'):
            computed[col] = _format_result_label(CANDIDATE_REASONS_NAME, col[len('can_result_'):])
        elif col.startswith('outcome_'):
            computed[col] = _format_outcome_label(col[len('outcome_'):])
        elif col.startswith('raw_info_'):
            metric = col[len('raw_info_'):]
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('extensions_'):
            metric = col[len('extensions_'):]
            computed[col] = _format_metric(metric, 'Extensions')
        elif col.startswith('car_raw_info_'):
            metric = f"car_{col[len('car_raw_info_'):]}"
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('can_raw_info_'):
            metric = f"can_{col[len('can_raw_info_'):]}"
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('car_extensions_'):
            metric = f"car_{col[len('car_extensions_'):]}"
            computed[col] = _format_metric(metric, 'Extensions')
        elif col.startswith('can_extensions_'):
            metric = f"can_{col[len('can_extensions_'):]}"
            computed[col] = _format_metric(metric, 'Extensions')
        elif col in rename_map:
            computed[col] = rename_map[col]
        else:
            computed[col] = _prettify_metric(col)
    return df.rename(columns=computed)


def _column_display_name(column: str) -> str:
    temp = _rename_columns(pd.DataFrame(columns=[column]))
    return temp.columns[0]


def _plot_scatter(df: pd.DataFrame, config: dict):
    points = _filter_valid_points(df, config['x'], config['y'])
    if not points:
        print(f"Plot '{config['title']}' skipped: insufficient data.")
        return
    plt.figure(figsize=(8, 5))
    xs, ys, labels = zip(*points)
    plt.scatter(xs, ys, alpha=0.7)
    for x, y, label in points:
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(5, 3), fontsize=8)
    xlabel = config.get('xlabel') or _column_display_name(config['x'])
    ylabel = config.get('ylabel') or _column_display_name(config['y'])
    plt.title(config['title'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_bar(df: pd.DataFrame, *, columns: Iterable[str], title: str, ylabel: str, stacked: bool = False, sort_by: str | None = None):
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        print(f"Plot '{title}' skipped: columns not available.")
        return
    subset = df[['worker_label'] + available_columns].copy()
    subset = subset.dropna(how='all', subset=available_columns)
    if subset.empty:
        print(f"Plot '{title}' skipped: insufficient data.")
        return
    subset[available_columns] = subset[available_columns].apply(pd.to_numeric, errors='coerce')
    subset = subset.dropna()
    if subset.empty:
        print(f"Plot '{title}' skipped: non-numeric data.")
        return
    renamed_subset = _rename_columns(subset)
    plot_df = renamed_subset.set_index('Worker')
    value_columns = []
    for original in available_columns:
        display_name = _column_display_name(original)
        if display_name in plot_df.columns:
            value_columns.append(display_name)
    if not value_columns:
        print(f"Plot '{title}' skipped: no numeric columns available after renaming.")
        return
    sort_column = _column_display_name(sort_by) if sort_by else None
    if sort_column and sort_column in plot_df.columns:
        plot_df = plot_df.sort_values(sort_column, ascending=False)
    plt.figure(figsize=(max(8, len(plot_df) * 0.5), 5))
    plot_df[value_columns].plot(kind='bar', stacked=stacked, alpha=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Worker')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _collect_heatmap_rows(entries):
    extracted = []
    for entry in entries or []:
        try:
            key_bytes = decode_key(entry)
            key_text = key_bytes.decode('utf-8', errors='replace')
        except Exception:
            continue
        preview, details = try_decode_value(entry)
        timestamp = None
        if isinstance(preview, bytes):
            preview = preview.decode('utf-8', errors='replace')
        if isinstance(preview, str):
            candidate = preview.strip()
            if candidate:
                timestamp = candidate
        if timestamp is None and isinstance(details, dict):
            decoded_bytes = details.get('decoded_bytes')
            if isinstance(decoded_bytes, (bytes, bytearray)):
                candidate = decoded_bytes.decode('utf-8', errors='replace').strip()
                if candidate:
                    timestamp = candidate
        extracted.append((timestamp, key_text))
    if not extracted:
        return np.empty((0, 0), dtype=int), []
    def _sort_key(item):
        ts, _ = item
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return datetime.max
    extracted.sort(key=_sort_key)
    width = max(len(key) for _, key in extracted)
    matrix = np.full((len(extracted), width), np.nan, dtype=float)
    labels = []
    for row_idx, (ts, key_text) in enumerate(extracted):
        if len(key_text) != width:
            key_text = key_text.ljust(width, '0')
        matrix[row_idx, :] = [1 if ch == '1' else 0 for ch in key_text]
        labels.append(ts or 'n/a')
    return matrix, labels


def render_bitmap_heatmaps(selected_manifest, selected_backups):
    heatmap_dbs = [
        (2, DB_DISPLAY_NAMES.get('R', 'Reasons')),
        (3, DB_DISPLAY_NAMES.get('NR', 'Non-reasons')),
        (5, DB_DISPLAY_NAMES.get('AR', 'Anti-reasons')),
        (6, DB_DISPLAY_NAMES.get('GP', 'Good profiles')),
        (7, DB_DISPLAY_NAMES.get('BP', 'Bad profiles')),
        (8, DB_DISPLAY_NAMES.get('PR', 'Preferred reasons')),
        (9, DB_DISPLAY_NAMES.get('AP', 'Anti-reason profiles')),
    ]
    if not selected_manifest or not selected_backups:
        print('Bitmap heatmaps unavailable: no backups loaded.')
        return
    files_map = selected_manifest.get('files') or {}
    generated_any = False
    for db_index, label in heatmap_dbs:
        file_name = files_map.get(str(db_index))
        if not file_name:
            continue
        data = selected_backups.get(file_name) if isinstance(selected_backups, dict) else None
        if not isinstance(data, dict):
            continue
        matrix, labels = _collect_heatmap_rows(data.get('entries'))
        if matrix.size == 0:
            continue
        generated_any = True
        fig_height = max(4, matrix.shape[0] * 0.25)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        ax.set_title(f"{label} bitmap heatmap (timestamp order)")
        ax.set_xlabel('Bitmap position')
        ax.set_ylabel('Timestamp')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bit value')
        plt.tight_layout()
        plt.show()
    if not generated_any:
        print('No bitmap heatmaps generated (missing or empty data).')


def render_db0_eu_analysis():
    entry_map = globals().get('selected_db0_values_by_key') or {}
    eu_entry = entry_map.get('EU') if isinstance(entry_map, dict) else None
    if not eu_entry:
        print('DB 0 EU entry not available for the current selection.')
        return
    series_map = eu_entry.get('value_json')
    if not isinstance(series_map, dict) or not series_map:
        print('DB 0 EU entry does not contain a feature map.')
        return
    feature_names = sorted(series_map)
    cleaned_series = []
    lengths = []
    finite_lengths = []
    max_len = 0
    feature_stats = []
    global_min = float('inf')
    global_max = float('-inf')
    for name in feature_names:
        values = series_map.get(name)
        numeric_values = []
        inf_positions = []
        finite_values = []
        if isinstance(values, list):
            for idx, value in enumerate(values):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = float('nan')
                if math.isfinite(numeric):
                    finite_values.append(numeric)
                    global_min = min(global_min, numeric)
                    global_max = max(global_max, numeric)
                    numeric_values.append(numeric)
                else:
                    inf_positions.append((idx, numeric))
                    numeric_values.append(float('nan'))
        finite_min = min(finite_values) if finite_values else 0.0
        finite_max = max(finite_values) if finite_values else 0.0
        if not finite_values and inf_positions:
            finite_min = finite_max = 0.0
        for idx, numeric in inf_positions:
            replacement = finite_max if numeric > 0 else finite_min
            numeric_values[idx] = replacement
            finite_values.append(replacement)
        cleaned_series.append(numeric_values)
        lengths.append(len(numeric_values))
        finite_lengths.append(len(finite_values))
        max_len = max(max_len, len(numeric_values))
        if finite_values:
            mean_val = float(np.mean(finite_values))
            std_val = float(np.std(finite_values))
            min_val = float(np.min(finite_values))
            max_val = float(np.max(finite_values))
        else:
            mean_val = std_val = min_val = max_val = float('nan')
        feature_stats.append({
            'feature': name,
            'length': len(numeric_values),
            'finite_samples': len(finite_values),
            'mean': mean_val,
            'std_dev': std_val,
            'min': min_val,
            'max': max_val,
        })
    if max_len == 0:
        print('DB 0 EU entry does not contain numeric data.')
        return
    matrix = np.full((len(cleaned_series), max_len), np.nan, dtype=float)
    for idx, series in enumerate(cleaned_series):
        if series:
            matrix[idx, : len(series)] = np.array(series, dtype=float)
    lengths = np.array(lengths)
    finite_lengths = np.array(finite_lengths)
    print('DB 0 EU summary:')
    print(f'  features: {len(feature_names)}')
    if lengths.size:
        print(f'  min length: {int(np.nanmin(lengths))}')
        print(f'  max length: {int(np.nanmax(lengths))}')
        print(f'  average length: {np.nanmean(lengths):.2f}')
        print(f'  length std dev: {np.nanstd(lengths):.2f}')
    if finite_lengths.size:
        print(f'  finite values min: {int(np.nanmin(finite_lengths))}')
        print(f'  finite values max: {int(np.nanmax(finite_lengths))}')
        print(f'  finite values avg: {np.nanmean(finite_lengths):.2f}')
    mean_values = [stat['mean'] for stat in feature_stats if math.isfinite(stat['mean'])]
    overall_mean = float(np.mean(mean_values)) if mean_values else float('nan')
    if math.isfinite(overall_mean):
        print(f'  mean across features: {overall_mean:.4f}')
    stats_df = pd.DataFrame(feature_stats)
    stats_display = stats_df.rename(columns={
        'feature': 'Feature',
        'length': 'Length',
        'finite_samples': 'Finite samples',
        'mean': 'Mean',
        'std_dev': 'Std dev',
        'min': 'Min',
        'max': 'Max',
    })
    display(stats_display)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    axes[0].bar(range(len(feature_names)), finite_lengths, color='tab:blue')
    axes[0].set_ylabel('Finite samples')
    axes[0].set_title('EU series finite counts per feature')
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=90, fontsize=6)
    box_data = []
    box_labels = []
    for name, series in zip(feature_names, cleaned_series):
        arr = np.array(series, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            box_data.append(finite)
            box_labels.append(name)
    axes[1].set_title('EU feature distribution (box plot)')
    axes[1].set_ylabel('Value')
    if box_data:
        axes[1].boxplot(box_data, vert=True, patch_artist=True)
        axes[1].set_xticks(range(1, len(box_data) + 1))
        axes[1].set_xticklabels(box_labels, rotation=90, fontsize=6)
    else:
        axes[1].text(0.5, 0.5, 'No finite data', ha='center', va='center')
    axes[1].set_xlabel('Feature index')
    axes[2].set_title('EU series heatmap')
    vmin = np.nanmin(matrix) if np.isfinite(matrix).any() else -1.0
    vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
    axes[2].imshow(matrix, aspect='auto', interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_yticks(range(len(feature_names)))
    axes[2].set_yticklabels(feature_names, fontsize=6)
    axes[2].set_xlabel('Time index')
    axes[2].set_ylabel('Feature')
    axes[2].grid(False)
    plt.tight_layout()
    plt.show()


def _coerce_numeric_series(data):
    if data is None:
        return []
    if isinstance(data, bool):
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, str):
        trimmed = data.strip()
        if not trimmed:
            return []
        try:
            return [float(trimmed)]
        except ValueError:
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                tokens = []
                for token in trimmed.replace(',', ' ').split():
                    try:
                        tokens.append(float(token))
                    except ValueError:
                        continue
                return tokens
            else:
                return _coerce_numeric_series(parsed)
    if isinstance(data, (list, tuple)):
        collected = []
        for item in data:
            if isinstance(item, dict):
                handled = False
                for candidate in ('value', 'y', 'val', 'score'):
                    if candidate in item:
                        nested = _coerce_numeric_series(item[candidate])
                        if nested:
                            collected.extend(nested)
                            handled = True
                        break
                if handled:
                    continue
                nested = _coerce_numeric_series(list(item.values()))
                if nested:
                    collected.extend(nested)
            else:
                nested = _coerce_numeric_series(item)
                if nested:
                    collected.extend(nested)
        return collected
    if isinstance(data, dict):
        for candidate in (
                'series',
                'values',
                'data',
                'points',
                'samples',
                'sample',
                'payload',
                'entries',
                'items',
                'measurements',
                'sample_dict',
        ):
            if candidate in data:
                nested = _coerce_numeric_series(data[candidate])
                if nested:
                    return nested
        numeric_items = []
        for idx, (key, value) in enumerate(data.items()):
            floats = _coerce_numeric_series(value)
            if not floats:
                continue
            if len(floats) == 1:
                numeric_items.append(((idx, 0), key, floats[0]))
            else:
                for offset, val in enumerate(floats):
                    numeric_items.append(((idx, offset), f"{key}[{offset}]", val))
        if not numeric_items:
            return []
        def _dict_sort_key(order, raw_key):
            order_token = tuple(order)
            if isinstance(raw_key, (int, float)):
                return (0, float(raw_key), order_token)
            if isinstance(raw_key, str):
                stripped = raw_key.strip()
                try:
                    return (0, float(stripped), order_token)
                except ValueError:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', stripped)
                    if numbers:
                        return (1, tuple(float(num) for num in numbers), order_token)
                    return (2, stripped.lower(), order_token)
            return (3, str(raw_key), order_token)
        numeric_items.sort(key=lambda item: _dict_sort_key(item[0], item[1]))
        return [value for _, _, value in numeric_items]
    return []


def _extract_sample_series(entry):
    if not isinstance(entry, dict):
        return []
    value_json = entry.get('value_json')
    series = _coerce_numeric_series(value_json)
    if series:
        return series
    value_text = entry.get('value_text')
    if isinstance(value_text, str):
        series = _coerce_numeric_series(value_text)
        if series:
            return series
    value_bytes = entry.get('value_bytes')
    if isinstance(value_bytes, (bytes, bytearray)):
        try:
            decoded = value_bytes.decode('utf-8', errors='replace')
        except Exception:
            decoded = ''
        if decoded:
            series = _coerce_numeric_series(decoded)
            if series:
                return series
    details = entry.get('details')
    if isinstance(details, dict):
        decoded_bytes = details.get('decoded_bytes')
        if isinstance(decoded_bytes, (bytes, bytearray)):
            try:
                decoded = decoded_bytes.decode('utf-8', errors='replace')
            except Exception:
                decoded = ''
            if decoded:
                series = _coerce_numeric_series(decoded)
                if series:
                    return series
    return []


def _extract_sample_timestamp(meta_entry):
    if not isinstance(meta_entry, dict):
        return None
    value_json = meta_entry.get('value_json')
    if isinstance(value_json, dict):
        for key in ('timestamp', 'created_at', 'created'):
            ts = value_json.get(key)
            if isinstance(ts, str) and ts:
                return ts
    value_text = meta_entry.get('value_text') if isinstance(meta_entry, dict) else None
    if isinstance(value_text, str):
        trimmed = value_text.strip()
        if trimmed.startswith('{') and trimmed.endswith('}'):
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                for key in ('timestamp', 'created_at', 'created'):
                    ts = parsed.get(key)
                    if isinstance(ts, str) and ts:
                        return ts
        else:
            return trimmed or None
    return None


def render_db0_sample_features():
    entry_map = globals().get('selected_db0_values_by_key') or {}
    if not isinstance(entry_map, dict):
        print('DB 0 sample entries not available for the current selection.')
        return
    collected = []
    for key, entry in entry_map.items():
        if not (key.startswith('sample_') and not key.endswith('_meta')):
            continue
        series = _extract_sample_series(entry)
        if not series:
            continue
        meta_entry = entry_map.get(f"{key}_meta")
        timestamp = _extract_sample_timestamp(meta_entry) if isinstance(meta_entry, dict) else None
        collected.append((timestamp, key, series))
    if not collected:
        print('No DB 0 sample data available.')
        return
    def _sort_key(item):
        ts, key, _ = item
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return key
    collected.sort(key=_sort_key)
    plt.figure(figsize=(12, 4))
    for timestamp, key, series in collected:
        xs = list(range(len(series)))
        label = f"{key} ({timestamp})" if timestamp else key
        plt.plot(xs, series, linewidth=1, label=label)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('DB 0 sample features overview')
    if len(collected) <= 12:
        legend_cols = 1 if len(collected) <= 6 else 2
        plt.legend(fontsize=6, ncol=legend_cols, frameon=False)
    elif len(collected) <= 24:
        plt.legend(fontsize=6, ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()
    cols = 2
    rows = (len(collected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), squeeze=False)
    flat_axes = axes.flatten()
    for ax in flat_axes[len(collected):]:
        ax.axis('off')
    for ax, (timestamp, key, series) in zip(flat_axes, collected):
        ax.plot(range(len(series)), series, marker='o', linewidth=1)
        title = f"{key} ({timestamp})" if timestamp else key
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
    plt.tight_layout()
    plt.show()


def _parse_worker_event_timestamp(event):
    candidates = [
        event.get('timestamp_start'),
        event.get('start'),
        event.get('started_at'),
        event.get('start_time'),
        event.get('timestamp'),
    ]
    for ts in candidates:
        if isinstance(ts, str) and ts:
            parsed = pd.to_datetime(ts, utc=False, errors='coerce')
            if pd.notna(parsed):
                return parsed.to_pydatetime()
    return None


def _queue_event_sort_key(event):
    iteration = event.get('iteration')
    timestamp = event.get('timestamp_start') or event.get('start') or ''
    return (
        iteration if isinstance(iteration, (int, float)) else float('inf'),
        timestamp,
    )


def _extract_numeric_value(*candidates):
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _render_combined_chart(series_data, *, title: str, ylabel: str):
    if not series_data:
        print(f"Plot '{title}' skipped: insufficient data.")
        return
    plt.figure(figsize=(12, 4))
    for label, items in series_data:
        timestamps = [entry['timestamp'] for entry in items if entry['timestamp'] is not None]
        if timestamps:
            xs = []
            for ts in timestamps:
                if hasattr(ts, 'to_pydatetime'):
                    xs.append(ts.to_pydatetime())
                else:
                    xs.append(ts)
            ys = [entry['value'] for entry in items if entry['timestamp'] is not None]
            plt.plot(xs, ys, label=label, linewidth=1)
        else:
            iterations = [entry['iteration'] for entry in items if entry['iteration'] is not None]
            if iterations:
                xs = iterations
                ys = [entry['value'] for entry in items if entry['iteration'] is not None]
                plt.plot(xs, ys, label=label, linewidth=1)
            else:
                xs = [entry['order'] for entry in items]
                ys = [entry['value'] for entry in items]
                plt.plot(xs, ys, label=label, linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    has_time = any(entry['timestamp'] is not None for _, items in series_data for entry in items)
    plt.xlabel('Time' if has_time else 'Iteration/order')
    if len(series_data) <= 12:
        plt.legend(fontsize=7)
    else:
        plt.legend(fontsize=6, ncol=2, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _draw_worker_chart(ax, items, ylabel: str):
    timestamp_points = [(entry['timestamp'], entry['value']) for entry in items if entry['timestamp'] is not None]
    if timestamp_points:
        xs = []
        for ts, _ in timestamp_points:
            if hasattr(ts, 'to_pydatetime'):
                xs.append(ts.to_pydatetime())
            else:
                xs.append(ts)
        ys = [value for _, value in timestamp_points]
        ax.plot(xs, ys, marker='o', linewidth=1)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_xlabel('Start time')
    else:
        iteration_points = [(entry['iteration'], entry['value']) for entry in items if entry['iteration'] is not None]
        if iteration_points:
            xs = [float(it) for it, _ in iteration_points]
            ys = [value for _, value in iteration_points]
            ax.plot(xs, ys, marker='o', linewidth=1)
            ax.set_xlabel('Iteration')
        else:
            xs = [entry['order'] for entry in items]
            ys = [entry['value'] for entry in items]
            ax.plot(xs, ys, marker='o', linewidth=1)
            ax.set_xlabel('Event order')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _render_worker_chart_grid(series_data, *, title: str, ylabel: str):
    if not series_data:
        print(f"Plot '{title}' skipped: insufficient data.")
        return
    if widgets is None:
        cols = 2
        rows = (len(series_data) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), squeeze=False)
        flat_axes = axes.flatten()
        for ax in flat_axes[len(series_data):]:
            ax.axis('off')
        for ax, (worker_label, items) in zip(flat_axes, series_data):
            _draw_worker_chart(ax, items, ylabel)
            ax.set_title(worker_label, fontsize=9)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
        return

    series_map = {label: items for label, items in series_data}
    labels = [label for label, _ in series_data]

    def _render_single_worker(worker_label: str):
        fig, ax = plt.subplots(figsize=(10, 4))
        _draw_worker_chart(ax, series_map[worker_label], ylabel)
        ax.set_title(worker_label, fontsize=10)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    if len(labels) == 1:
        _render_single_worker(labels[0])
        return

    dropdown = widgets.Dropdown(options=labels, value=labels[0], description='Worker')
    output = widgets.Output()

    def _update_plot(label):
        if label is None:
            return
        output.clear_output(wait=True)
        with output:
            _render_single_worker(label)

    def _on_change(change):
        if change['name'] == 'value':
            _update_plot(change['new'])

    dropdown.observe(_on_change, names='value')
    _update_plot(labels[0])
    display(widgets.VBox([widgets.HTML(f"<b>{title}</b>"), dropdown, output]))


def _plot_queue_chart(report, df: pd.DataFrame):
    workers_data = (report or {}).get('../workers') or {}
    cache = globals().get('DB10_WORKER_CACHE') or {}
    worker_stats = cache.get('worker_stats') or {}
    if not workers_data and not worker_stats:
        print("Queue chart skipped: worker events unavailable.")
        return
    union_worker_ids = set(workers_data.keys()) | set(worker_stats.keys())
    label_map = _build_worker_label_map(union_worker_ids)
    queue_series = []
    car_queue_series = []
    for worker_id in sorted(union_worker_ids):
        summary = workers_data.get(worker_id, {})
        events = list(summary.get('events') or [])
        if not events and worker_stats:
            events = list((worker_stats.get(worker_id) or {}).get('events') or [])
        if not events:
            continue
        events_sorted = sorted(events, key=_queue_event_sort_key)
        queue_points = []
        car_points = []
        label = label_map.get(worker_id)
        for order, event in enumerate(events_sorted):
            timestamp = _parse_worker_event_timestamp(event)
            iteration = event.get('iteration')
            iteration_value = int(iteration) if isinstance(iteration, (int, float)) else None
            queue_value = _extract_numeric_value(event.get('queue_size'), event.get('queue'))
            if queue_value is not None:
                queue_points.append({
                    'timestamp': timestamp,
                    'iteration': iteration_value,
                    'order': order + 1,
                    'value': queue_value,
                })
            car_queue_value = _extract_numeric_value(event.get('car_queue_size'), event.get('car_queue'))
            if car_queue_value is not None:
                car_points.append({
                    'timestamp': timestamp,
                    'iteration': iteration_value,
                    'order': order + 1,
                    'value': car_queue_value,
                })
        if queue_points:
            queue_series.append((label, queue_points))
        if car_points:
            car_queue_series.append((label, car_points))
    _render_combined_chart(queue_series, title='Queue size overview', ylabel='Queue size')
    _render_combined_chart(car_queue_series, title=f"{CANDIDATE_ANTI_REASONS_NAME} queue overview", ylabel=f"{CANDIDATE_ANTI_REASONS_NAME} queue size")


def _collect_worker_iteration_rows(report, df: pd.DataFrame):
    cache = globals().get('DB10_WORKER_CACHE') or {}
    worker_stats = cache.get('worker_stats') or {}
    workers = (report or {}).get('../workers') or {}
    union_worker_ids = set(workers.keys()) | set(worker_stats.keys())
    label_map = _build_worker_label_map(union_worker_ids)
    rows = []
    for worker_id in sorted(union_worker_ids):
        summary = workers.get(worker_id) or {}
        events = list(summary.get('events') or [])
        if not events and worker_stats:
            events = list((worker_stats.get(worker_id) or {}).get('events') or [])
        if not events:
            continue
        events_sorted = sorted(events, key=_queue_event_sort_key)
        label = label_map.get(worker_id)
        for order, event in enumerate(events_sorted, start=1):
            timings = event.get('timings') or {}
            car_processing = event.get('car_processing') or {}
            can_processing = event.get('can_processing') or {}
            outcomes = event.get('outcomes')
            if isinstance(outcomes, dict):
                outcome_text = ', '.join(
                    f"{key}={'T' if value else 'F'}" for key, value in sorted(outcomes.items())
                )
            else:
                outcome_text = None
            row = {
                'worker_id': worker_id,
                'worker_label': label,
                'event_order': order,
                'iteration': event.get('iteration'),
                'timestamp_start': event.get('timestamp_start'),
                'timestamp_end': event.get('timestamp_end'),
                'queue_size': _extract_numeric_value(event.get('queue_size'), event.get('queue')),
                'car_queue_size': _extract_numeric_value(event.get('car_queue_size'), event.get('car_queue')),
                'total_seconds': timings.get('total_seconds'),
                'car_seconds': timings.get('car_seconds'),
                'can_seconds': timings.get('can_seconds'),
                'car_result': car_processing.get('result'),
                'can_result': can_processing.get('result'),
                'outcomes': outcome_text,
            }
            for prefix, processing in (( 'car', car_processing), ('can', can_processing)):
                raw_info = processing.get('raw_info')
                if isinstance(raw_info, dict):
                    for key, value in raw_info.items():
                        numeric = _extract_numeric_value(value)
                        if numeric is None:
                            continue
                        if isinstance(numeric, float) and numeric.is_integer():
                            numeric = int(numeric)
                        row[f"{prefix}_raw_info_{key}"] = numeric
                extensions = processing.get('extensions')
                if isinstance(extensions, dict):
                    for key, value in extensions.items():
                        numeric = _extract_numeric_value(value)
                        if numeric is None:
                            continue
                        if isinstance(numeric, float) and numeric.is_integer():
                            numeric = int(numeric)
                        row[f"{prefix}_extensions_{key}"] = numeric
            rows.append(row)
    return rows


def _render_worker_iteration_table(report, df: pd.DataFrame, *, max_rows=5000):
    rows = _collect_worker_iteration_rows(report, df)
    if not rows:
        print('Worker iteration details are not available for the current selection.')
        return
    table = pd.DataFrame(rows)
    if table.empty:
        print('Worker iteration details are not available for the current selection.')
        return
    table['timestamp_start'] = pd.to_datetime(table['timestamp_start'], errors='coerce')
    table['timestamp_end'] = pd.to_datetime(table['timestamp_end'], errors='coerce')
    sort_columns = ['worker_label', 'event_order']
    if table['timestamp_start'].notna().any():
        sort_columns = ['worker_label', 'timestamp_start', 'event_order']
    elif table['iteration'].notna().any():
        sort_columns = ['worker_label', 'iteration', 'event_order']
    table = table.sort_values(sort_columns)
    total_rows = len(table)
    worker_count = table['worker_label'].nunique() if 'worker_label' in table.columns else None
    if max_rows is not None and total_rows > max_rows:
        summary = f"Showing first {max_rows} of {total_rows} worker iterations"
        if worker_count is not None:
            summary += f" across {worker_count} workers"
        print(summary + '.')
        display_df = table.head(max_rows)
    else:
        summary = f"Worker iterations: {total_rows} rows"
        if worker_count is not None:
            summary += f" across {worker_count} workers"
        print(summary + '.')
        display_df = table
    if 'worker_id' in display_df.columns:
        display_df = display_df.drop(columns=['worker_id'])
    display(_rename_columns(display_df))


def _plot_histogram(df: pd.DataFrame, *, column: str, title: str, xlabel: str):
    if column not in df.columns:
        print(f"Histogram '{title}' skipped: column not available.")
        return
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        print(f"Histogram '{title}' skipped: insufficient data.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=min(25, len(series)), alpha=0.7)
    label = xlabel or _column_display_name(column)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Number of workers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def render_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix):
    report = _ensure_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix)
    rows = _flatten_worker_summary(report)
    if not rows:
        print('No data available for the worker report.')
        return
    df = pd.DataFrame(rows)
    df = df.sort_values('worker_index')
    base_columns = [col for col in BASE_COLUMNS if col in df.columns]
    extra_columns = sorted(
        col for col in df.columns if col.startswith('car_result_') or col.startswith('can_result_') or col.startswith('outcome_')
    )
    base_table_columns = ['worker_label'] + [col for col in BASE_COLUMNS if col not in {'worker_id'}]
    base_table_columns = [col for col in base_table_columns if col in df.columns]
    display(_rename_columns(df[base_table_columns]))
    print('Detailed results per worker:')
    raw_info_columns = sorted(col for col in df.columns if col.startswith('raw_info_'))
    extensions_columns = sorted(col for col in df.columns if col.startswith('extensions_'))
    detail_columns = ['worker_label'] + extra_columns + raw_info_columns + extensions_columns
    detail_columns = [col for col in detail_columns if col in df.columns]
    display(_rename_columns(df[detail_columns]))
    for config in SCATTER_PLOTS:
        _plot_scatter(df, config)
    for x_key, y_key, title in ADDITIONAL_SCATTER_PREFIX_PAIRS:
        if x_key in df.columns and y_key in df.columns:
            _plot_scatter(
                df,
                {
                    'title': title,
                    'x': x_key,
                    'y': y_key,
                    'xlabel': _column_display_name(x_key),
                    'ylabel': _column_display_name(y_key),
                },
            )
    for config in BAR_PLOTS:
        _plot_bar(
            df,
            columns=config['columns'],
            title=config['title'],
            ylabel=config['ylabel'],
            stacked=False,
            sort_by=config.get('sort_by'),
        )
    for config in STACKED_BAR_CONFIG:
        columns = sorted(col for col in df.columns if col.startswith(config['prefix']))
        if columns:
            _plot_bar(
                df,
                columns=columns,
                title=config['title'],
                ylabel=config['ylabel'],
                stacked=True,
            )
        else:
            print(f"Plot '{config['title']}' skipped: no columns with prefix {config['prefix']!r}.")
    for config in HISTOGRAMS:
        _plot_histogram(df, column=config['column'], title=config['title'], xlabel=config['xlabel'])
    if extensions_columns:
        _plot_bar(df, columns=extensions_columns, title='Extensions totals per worker', ylabel='Count', stacked=False)
    _plot_queue_chart(report, df)
    _render_worker_iteration_table(report, df)
