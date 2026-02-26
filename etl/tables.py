from __future__ import annotations

import json
import os
from collections import defaultdict
import base64
import binascii
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# Optional styling (if unavailable, neutral fallback)
try:
	from etl.table_styling import style_summary_table, print_color_legend, COLUMN_COLORMAPS, DEFAULT_CMAP
except Exception:
	COLUMN_COLORMAPS = {}
	DEFAULT_CMAP = 'Greys'
	def style_summary_table(df: pd.DataFrame):
		return df.style
	def print_color_legend():
		pass

try:
	from etl.mcr_results import (
		compute_counts_from_results,
		load_analyzed_df,
		cast_dataset_str,
		DISPLAY_CATEGORIES,
		DISPLAY_NAMES,
		DISPLAY_LABELS,
	)
except Exception as exc:
	compute_counts_from_results = None
	load_analyzed_df = None
	cast_dataset_str = None
	DISPLAY_CATEGORIES = []
	DISPLAY_NAMES = {}
	DISPLAY_LABELS = {}
	print(f"Unable to import etl.mcr_results: {exc}")

CACHE_REFRESH_ENV = 'FORCE_RESULTS_REFRESH'
SUMMARY_FILENAME = 'redis_reason_counts.csv'
COUNTS_CACHE_FILENAME = '_counts_cache.csv'
META_FILENAME = 'redis_counts_meta.json'

REFRESH_TRUE = {'1', 'true', 'yes', 'y', 'on'}

RESULT_COUNT_COLUMNS = [
	'Total time (s) max',
	'Total time (s) mean',
	'ICF checks',
	'Reason check iteration total',
	'IterGoodRatio',
	'IterBadRatio',
	'Early Stop Good total',
	'Early Stop from Good',
	'Early Stop from Bad',
	'Filtrered rate',
]
WORKER_CAN_COLUMN_MAP = {
	'total_time_max': 'Total time (s) max',
	'total_time_mean': 'Total time (s) mean',
	'icf_checks': 'ICF checks',
	'reason_iterations_total': 'Reason check iteration total',
	'iter_good_ratio': 'IterGoodRatio',
	'iter_bad_ratio': 'IterBadRatio',
	'Early Stop_good_total': 'Early Stop Good total',
	'Early Stop from Good': 'Early Stop from Good',
	'Early Stop from Bad': 'Early Stop from Bad',
	'filtrered_rate': 'Filtrered rate',
}
SUMMARY_INT_COLUMNS = ['train_size', 'test_size', 'n_estimators']
COUNTS_INT_COLUMNS = ['ICF checks', 'Reason check iteration total', 'Early Stop Good total', 'Early Stop from Good', 'Early Stop from Bad']
COUNTS_FLOAT_COLUMNS = ['Total time (s) max', 'Total time (s) mean', 'IterGoodRatio', 'IterBadRatio', 'Filtrered rate']
COMBINED_INT_ROWS = [
	'Train Size',
	'Test Size',
	'N Estimators',
	'Total Time (ms)',
	'ICF Checks',
	'Reason Check Iteration',
	'Early Stop Good',
	'Early Stop from Good',
	'Early Stop from Bad',
]
COMBINED_FLOAT_ROWS = [
	'Mean EU Features',
	'EU Std',
]
COMBINED_PERCENT_ROWS = [
	'IterGoodRadio %',
	'IterBadRadio %',
	'Filtrered Rate %',
]


def _apply_int_format(df: pd.DataFrame, columns: Iterable[str]) -> None:
	for col in columns:
		if col in df.columns:
			numeric_col = pd.to_numeric(df[col], errors='coerce')
			df[col] = numeric_col.round().astype('Int64')


def _apply_float_format(df: pd.DataFrame, columns: Iterable[str], *, decimals: int = 3) -> None:
	for col in columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce').round(decimals)


def _format_int_value(value: Any) -> str:
	if pd.isna(value):
		return ''
	return f'{int(round(float(value))):d}'


def _format_float_value(value: Any, *, decimals: int = 3) -> str:
	if pd.isna(value):
		return ''
	return f'{float(value):.{decimals}f}'


def _format_percent_value(value: Any, *, decimals: int = 1) -> str:
	if pd.isna(value):
		return ''
	return f'{float(value):.{decimals}f}%'


@dataclass
class FirstTableArtifacts:
	summary_styler: Any
	summary_df: pd.DataFrame
	counts_summary_df: pd.DataFrame
	eu_only_df: pd.DataFrame
	result_count_columns: list[str]
	worker_can_column_map: dict[str, str]
	worker_can_aggregate_row: dict[str, Any] | None
	worker_can_metrics: dict[str, dict[str, Any]]


@dataclass
class ModelsAnalysisArtifacts:
	base_dir: Path
	results_dir: Path
	forest_json: Path
	forest_csv: Path
	report_data: list[dict[str, Any]]
	results_artifacts: dict[str, Any]
	summary: pd.DataFrame
	counts_df: pd.DataFrame
	analyzed_df: pd.DataFrame
	results_datasets: set[str]
	zip_dataset_prefixes: set[str]
	missing_zip_manifests: list[tuple[str, Any]]
	log_summary: dict[str, Any]
	saved_summary_path: Path | None
	used_cache: bool


@dataclass
class ModelsAnalysisContext:
	artifacts: ModelsAnalysisArtifacts
	first_table: FirstTableArtifacts
	analyzed_counts_df: pd.DataFrame
	analyzed_counts_styler: Any
	combined_analyzed_df: pd.DataFrame
	combined_analyzed_styler: Any
	summary_to_show: pd.DataFrame
	summary_styler: Any

def detect_base_dir() -> Path:
	try:
		from IPython import get_ipython
		ip = get_ipython()
		if ip is not None:
			pwd = ip.run_line_magic('pwd', '')
			if pwd:
				return Path(pwd).resolve()
	except Exception:
		pass
	return Path.cwd().resolve()

def _ensure_cache_dir(results_dir: Path) -> Path:
	base = results_dir / '_cache' if results_dir.exists() else (Path.cwd() / '_results_cache')
	base.mkdir(parents=True, exist_ok=True)
	return base

def _latest_results_mtime(results_dir: Path) -> float:
	if not results_dir.exists():
		return 0.0
	mtimes: list[float] = []
	for entry in results_dir.iterdir():
		try:
			mtimes.append(entry.stat().st_mtime)
		except OSError:
			continue
	return max(mtimes, default=0.0)

def _should_use_cache(summary_path: Path, counts_cache: Path, meta_path: Path, results_dir: Path, refresh_flag: bool) -> tuple[bool, dict[str, Any]]:
	if refresh_flag or not summary_path.exists() or not meta_path.exists() or not counts_cache.exists():
		return False, {}
	try:
		meta = json.loads(meta_path.read_text(encoding='utf-8'))
	except Exception:
		return False, {}
	latest_input = _latest_results_mtime(results_dir)
	cached_source = meta.get('source_mtime', 0.0)
	if latest_input and latest_input > cached_source:
		return False, {}
	return True, meta

def load_forest_report(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		raise FileNotFoundError(f'File not found: {path}')
	with path.open('r', encoding='utf-8') as handle:
		return json.load(handle)

def _load_cached_summary(summary_path: Path) -> pd.DataFrame:
	summary = pd.read_csv(summary_path)
	summary['dataset'] = summary['dataset'].astype(str)
	return summary

def _build_counts_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
	if summary.empty:
		return pd.DataFrame()
	label_map = {cat: DISPLAY_LABELS.get(cat, DISPLAY_NAMES.get(cat, cat)) for cat in DISPLAY_CATEGORIES}
	label_to_cat = {label: cat for cat, label in label_map.items()}
	cols = [col for col in summary.columns if col in label_to_cat]
	if not cols:
		return pd.DataFrame()
	counts = summary[['dataset', *cols]].rename(columns=label_to_cat)
	return counts


class RedisDumpDecodeError(RuntimeError):
	"""Raised when a Redis DUMP payload cannot be decoded."""

_RDB_ENCODING_INT8 = 0
_RDB_ENCODING_INT16 = 1
_RDB_ENCODING_INT32 = 2
_RDB_ENCODING_LZF = 3

def _split_dump_sections(raw: bytes) -> tuple[bytes, int, bytes]:
	if len(raw) < 10:
		raise RedisDumpDecodeError('DUMP payload is too short')
	checksum = raw[-8:]
	version = int.from_bytes(raw[-10:-8], 'little', signed=False)
	payload = raw[:-10]
	return payload, version, checksum

def _read_length_info(buffer: bytes, offset: int) -> tuple[int | None, int | None, int]:
	if offset >= len(buffer):
		raise RedisDumpDecodeError('Offset out of range while reading length')
	first = buffer[offset]
	prefix = first >> 6
	if prefix == 0:
		return first & 0x3F, None, offset + 1
	if prefix == 1:
		if offset + 1 >= len(buffer):
			raise RedisDumpDecodeError('Truncated 14-bit encoded length')
		second = buffer[offset + 1]
		length = ((first & 0x3F) << 8) | second
		return length, None, offset + 2
	if prefix == 2:
		if offset + 4 >= len(buffer):
			raise RedisDumpDecodeError('Truncated 32-bit encoded length')
		length = int.from_bytes(buffer[offset + 1 : offset + 5], 'big', signed=False)
		return length, None, offset + 5
	return None, first & 0x3F, offset + 1

def _lzf_decompress(data: bytes, expected_length: int) -> bytes:
	output = bytearray()
	idx = 0
	data_len = len(data)
	while idx < data_len:
		ctrl = data[idx]
		idx += 1
		if ctrl < 32:
			literal_len = ctrl + 1
			if idx + literal_len > data_len:
				raise RedisDumpDecodeError('Truncated literal LZF sequence')
			output.extend(data[idx : idx + literal_len])
			idx += literal_len
			continue
		length = ctrl >> 5
		ref_offset = len(output) - ((ctrl & 0x1F) << 8) - 1
		if length == 7:
			if idx >= data_len:
				raise RedisDumpDecodeError('Truncated LZF sequence while extending length')
			length += data[idx]
			idx += 1
		if idx >= data_len:
			raise RedisDumpDecodeError('Truncated LZF sequence while resolving reference')
		ref_offset -= data[idx]
		idx += 1
		length += 2
		if ref_offset < 0:
			raise RedisDumpDecodeError('Negative LZF reference')
		for _ in range(length):
			if ref_offset >= len(output):
				raise RedisDumpDecodeError('LZF reference out of range')
			output.append(output[ref_offset])
			ref_offset += 1
	if len(output) != expected_length:
		raise RedisDumpDecodeError('Unexpected decompressed length')
	return bytes(output)

def _decode_special_encoding(buffer: bytes, offset: int, encoding: int) -> tuple[bytes, int]:
	if encoding == _RDB_ENCODING_INT8:
		if offset >= len(buffer):
			raise RedisDumpDecodeError('Truncated 8-bit encoded integer')
		value = int.from_bytes(buffer[offset : offset + 1], 'little', signed=True)
		return str(value).encode('ascii'), offset + 1
	if encoding == _RDB_ENCODING_INT16:
		if offset + 2 > len(buffer):
			raise RedisDumpDecodeError('Truncated 16-bit encoded integer')
		value = int.from_bytes(buffer[offset : offset + 2], 'little', signed=True)
		return str(value).encode('ascii'), offset + 2
	if encoding == _RDB_ENCODING_INT32:
		if offset + 4 > len(buffer):
			raise RedisDumpDecodeError('Truncated 32-bit encoded integer')
		value = int.from_bytes(buffer[offset : offset + 4], 'little', signed=True)
		return str(value).encode('ascii'), offset + 4
	if encoding == _RDB_ENCODING_LZF:
		compressed_len, enc, next_offset = _read_length_info(buffer, offset)
		if enc is not None:
			raise RedisDumpDecodeError('Unexpected encoding for LZF length')
		data_len, enc, data_offset = _read_length_info(buffer, next_offset)
		if enc is not None:
			raise RedisDumpDecodeError('Unexpected encoding for LZF payload length')
		if compressed_len is None or data_len is None:
			raise RedisDumpDecodeError('Invalid LZF length encoding')
		end = data_offset + compressed_len
		if end > len(buffer):
			raise RedisDumpDecodeError('Truncated encoded string payload')
		compressed = buffer[data_offset:end]
		decompressed = _lzf_decompress(compressed, data_len)
		return decompressed, end
	raise RedisDumpDecodeError('Unknown string encoding')

def _read_encoded_string(buffer: bytes, offset: int) -> tuple[bytes, int]:
	length, encoding, next_offset = _read_length_info(buffer, offset)
	if encoding is None:
		if length is None:
			raise RedisDumpDecodeError('Missing length for raw string')
		end = next_offset + length
		if end > len(buffer):
			raise RedisDumpDecodeError('Truncated encoded string payload')
		return buffer[next_offset:end], end
	return _decode_special_encoding(buffer, next_offset, encoding)

def _decode_dump_string(entry: Mapping[str, Any]) -> bytes | None:
	value = entry.get('value')
	if not isinstance(value, Mapping):
		return None
	data_b64 = value.get('data')
	if not isinstance(data_b64, str):
		return None
	try:
		raw = base64.b64decode(data_b64.encode('ascii'))
	except (binascii.Error, UnicodeEncodeError):
		return None
	try:
		payload, _, _ = _split_dump_sections(raw)
		if not payload or payload[0] != 0:
			return None
		decoded, _ = _read_encoded_string(payload, 1)
		return decoded
	except RedisDumpDecodeError:
		return None

def _decode_worker_id(entry: Mapping[str, Any]) -> str | None:
	key_b64 = entry.get('key')
	if not isinstance(key_b64, str):
		return None
	try:
		raw = base64.b64decode(key_b64.encode('ascii'))
	except (binascii.Error, UnicodeEncodeError):
		return None
	text = raw.decode('utf-8', errors='replace')
	if not text:
		return None
	if ':' in text:
		return text.rsplit(':', 1)[0]
	return text

def _load_db10_entries_from_zip(zip_path: Path) -> list[Mapping[str, Any]]:
	try:
		with zipfile.ZipFile(zip_path, 'r') as archive:
			names = [name for name in archive.namelist() if 'redis_backup_db10' in name]
			entries: list[Mapping[str, Any]] = []
			for name in names:
				try:
					payload = json.loads(archive.read(name).decode('utf-8'))
				except Exception:
					continue
				entries.extend(payload.get('entries') or [])
			return entries
	except (FileNotFoundError, zipfile.BadZipFile):
		return []

def _load_db10_entries_from_dir(directory: Path) -> list[Mapping[str, Any]]:
	entries: list[Mapping[str, Any]] = []
	for path in directory.rglob('redis_backup_db10.json'):
		try:
			payload = json.loads(path.read_text(encoding='utf-8'))
		except Exception:
			continue
		entries.extend(payload.get('entries') or [])
	return entries

def _aggregate_worker_can_times(entries: list[Mapping[str, Any]]) -> dict[str, float]:
	totals: dict[str, float] = {}
	for entry in entries:
		decoded = _decode_dump_string(entry)
		if decoded is None:
			continue
		try:
			payload = json.loads(decoded.decode('utf-8', errors='replace'))
		except json.JSONDecodeError:
			continue
		worker_id = payload.get('worker_id') or _decode_worker_id(entry)
		if not worker_id:
			continue
		can_processing = payload.get('can_processing') or {}
		if not isinstance(can_processing, Mapping):
			continue
		time_value = can_processing.get('time')
		if time_value is None:
			time_value = can_processing.get('time_seconds')
		if time_value is None:
			continue
		try:
			seconds = float(time_value)
		except (TypeError, ValueError):
			continue
		totals[worker_id] = totals.get(worker_id, 0.0) + seconds
	return totals

def _safe_float(value: Any | None) -> float | None:
	try:
		return float(value)
	except (TypeError, ValueError):
		return None

def _safe_number(value: Any | None) -> float:
	result = _safe_float(value)
	return result if result is not None else 0.0


def _init_worker_metrics() -> dict[str, float]:
	return {
		'time_seconds': 0.0,
		'can_checks': 0.0,
		'iterations_total': 0.0,
		'iterations_good': 0.0,
		'iterations_bad': 0.0,
		'early_stop_good_total': 0.0,
		'early_stop_good_result_good': 0.0,
		'early_stop_good_result_bad': 0.0,
		'extensions_total_good': 0.0,
		'extensions_filtered_good': 0.0,
	}

def _iter_db10_entries(entry: Path) -> list[Mapping[str, Any]]:
	if entry.is_file() and entry.suffix.lower() == '.zip':
		try:
			with zipfile.ZipFile(entry, 'r') as archive:
				items: list[Mapping[str, Any]] = []
				for name in archive.namelist():
					if 'redis_backup_db10' not in name:
						continue
					try:
						payload = json.loads(archive.read(name).decode('utf-8'))
					except Exception:
						continue
					items.extend(payload.get('entries') or [])
				return items
		except (FileNotFoundError, zipfile.BadZipFile):
			return []
	if entry.is_dir():
		items: list[Mapping[str, Any]] = []
		for path in entry.rglob('redis_backup_db10.json'):
			try:
				payload = json.loads(path.read_text(encoding='utf-8'))
			except Exception:
				continue
			items.extend(payload.get('entries') or [])
		return items
	return []

def _decode_db10_entry(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
	decoded = _decode_dump_string(entry)
	if decoded is None:
		return None
	try:
		return json.loads(decoded.decode('utf-8', errors='replace'))
	except json.JSONDecodeError:
		return None

def compute_worker_can_metrics(results_dir: Path) -> dict[str, dict[str, Any]]:
	dataset_stats: dict[str, dict[str, Any]] = {}
	all_time_values: list[float] = []
	overall_totals = defaultdict(float)
	overall_counts = {
		'worker_count': 0,
	}
	if not results_dir.exists():
		return {}
	for entry in sorted(results_dir.iterdir()):
		if entry.name.startswith('_'):
			continue
		dataset = entry.stem.split('_')[0] if entry.is_file() else entry.name.split('_')[0]
		if not dataset:
			continue
		db10_entries = _iter_db10_entries(entry)
		if not db10_entries:
			continue
		worker_metrics: dict[str, dict[str, float]] = {}
		for raw_entry in db10_entries:
			payload = _decode_db10_entry(raw_entry)
			if not isinstance(payload, Mapping):
				continue
			worker_id = payload.get('worker_id') or _decode_worker_id(raw_entry)
			if not worker_id:
				continue
			can_processing = payload.get('can_processing') or {}
			if not isinstance(can_processing, Mapping):
				continue
			metrics = worker_metrics.setdefault(worker_id, _init_worker_metrics())
			metrics['can_checks'] += 1.0
			time_value = can_processing.get('time')
			if time_value is None:
				time_value = can_processing.get('time_seconds')
			time_seconds = _safe_float(time_value)
			if time_seconds is not None:
				metrics['time_seconds'] += time_seconds
			result = str(can_processing.get('result') or '').upper()
			raw_info = can_processing.get('raw_info') or {}
			if isinstance(raw_info, Mapping):
				iterations_value = _safe_float(raw_info.get('iterations'))
				if iterations_value is not None:
					metrics['iterations_total'] += iterations_value
					if result == 'GOOD':
						metrics['iterations_good'] += iterations_value
					elif result == 'BAD':
						metrics['iterations_bad'] += iterations_value
				early_stop_good = _safe_float(raw_info.get('early_stop_good'))
				if early_stop_good is not None and early_stop_good:
					metrics['early_stop_good_total'] += early_stop_good
					if result == 'GOOD':
						metrics['early_stop_good_result_good'] += early_stop_good
					elif result == 'BAD':
						metrics['early_stop_good_result_bad'] += early_stop_good
			extensions = can_processing.get('extensions') or {}
			if result == 'GOOD' and isinstance(extensions, Mapping):
				metrics['extensions_total_good'] += _safe_number(extensions.get('total'))
				metrics['extensions_filtered_good'] += _safe_number(extensions.get('filtered'))
		if not worker_metrics:
			continue
		worker_count = len(worker_metrics)
		time_values = [m['time_seconds'] for m in worker_metrics.values() if m['time_seconds'] > 0.0]
		if time_values:
			all_time_values.extend(time_values)
		aggregate_totals = defaultdict(float)
		for metrics in worker_metrics.values():
			for key, value in metrics.items():
				aggregate_totals[key] += float(value)
		total_iterations = aggregate_totals['iterations_total']
		aggregate = {
			'worker_count': worker_count,
			'total_time_max': float(max(time_values)) if time_values else None,
			'total_time_mean': float(np.mean(time_values)) if time_values else None,
			'icf_checks': float(aggregate_totals['can_checks']),
			'reason_iterations_total': float(total_iterations),
			'reason_iterations_good': float(aggregate_totals['iterations_good']),
			'reason_iterations_bad': float(aggregate_totals['iterations_bad']),
			'iter_good_ratio': float(aggregate_totals['iterations_good'] / total_iterations) if total_iterations else None,
			'iter_bad_ratio': float(aggregate_totals['iterations_bad'] / total_iterations) if total_iterations else None,
			'Early Stop_good_total': float(aggregate_totals['early_stop_good_total']),
			'Early Stop from Good': float(aggregate_totals['early_stop_good_result_good']),
			'Early Stop from Bad': float(aggregate_totals['early_stop_good_result_bad']),
			'filtrered_total': float(aggregate_totals['extensions_total_good']),
			'filtrered_filtered': float(aggregate_totals['extensions_filtered_good']),
			'filtrered_rate': float(aggregate_totals['extensions_filtered_good'] / aggregate_totals['extensions_total_good']) if aggregate_totals['extensions_total_good'] else None,
		}
		dataset_stats[dataset] = {
			'workers': worker_metrics,
			'aggregate': aggregate,
		}
		overall_counts['worker_count'] += worker_count
		for key in (
				'can_checks',
				'iterations_total',
				'iterations_good',
				'iterations_bad',
				'early_stop_good_total',
				'early_stop_good_result_good',
				'early_stop_good_result_bad',
				'extensions_total_good',
				'extensions_filtered_good',
		):
			overall_totals[key] += aggregate_totals[key]
	if dataset_stats:
		total_iterations = overall_totals['iterations_total']
		overall_aggregate = {
			'worker_count': overall_counts['worker_count'],
			'total_time_max': float(max(all_time_values)) if all_time_values else None,
			'total_time_mean': float(np.mean(all_time_values)) if all_time_values else None,
			'icf_checks': float(overall_totals['can_checks']),
			'reason_iterations_total': float(total_iterations),
			'reason_iterations_good': float(overall_totals['iterations_good']),
			'reason_iterations_bad': float(overall_totals['iterations_bad']),
			'iter_good_ratio': float(overall_totals['iterations_good'] / total_iterations) if total_iterations else None,
			'iter_bad_ratio': float(overall_totals['iterations_bad'] / total_iterations) if total_iterations else None,
			'Early Stop_good_total': float(overall_totals['early_stop_good_total']),
			'Early Stop from Good': float(overall_totals['early_stop_good_result_good']),
			'Early Stop from Bad': float(overall_totals['early_stop_good_result_bad']),
			'filtrered_total': float(overall_totals['extensions_total_good']),
			'filtrered_filtered': float(overall_totals['extensions_filtered_good']),
			'filtrered_rate': float(overall_totals['extensions_filtered_good'] / overall_totals['extensions_total_good']) if overall_totals['extensions_total_good'] else None,
		}
		dataset_stats['__overall__'] = {
			'workers': {},
			'aggregate': overall_aggregate,
		}
	return dataset_stats
def load_results_artifacts(
		results_dir: Path,
		forest_csv: Path,
		*,
		verbose: bool = True,
		refresh: bool | None = None,
		cache_dir: Path | None = None,
) -> dict[str, Any]:
	if cache_dir is None:
		cache_dir = _ensure_cache_dir(results_dir)
	else:
		cache_dir.mkdir(parents=True, exist_ok=True)

	summary_path = results_dir / SUMMARY_FILENAME
	counts_cache = cache_dir / COUNTS_CACHE_FILENAME
	meta_path = cache_dir / META_FILENAME

	env_flag = os.environ.get(CACHE_REFRESH_ENV, '').strip().lower()
	refresh_flag = refresh if refresh is not None else env_flag in REFRESH_TRUE

	use_cache, meta = _should_use_cache(summary_path, counts_cache, meta_path, results_dir, refresh_flag)

	if use_cache:
		if verbose:
			cached_at = meta.get('cached_at')
			print(f"Using cached redis summary (cached_at={cached_at})")
		summary = _load_cached_summary(summary_path)
		counts_df = _build_counts_from_summary(summary)
		if not counts_df.empty and isinstance(meta.get('log_summary'), dict):
			counts_df.attrs['log_summary'] = meta['log_summary']
		analyzed_df = load_analyzed_df(forest_csv) if load_analyzed_df else pd.DataFrame()
		return {
			'counts_df': counts_df,
			'analyzed_df': analyzed_df,
			'summary': summary,
			'results_datasets': set(meta.get('results_datasets', [])),
			'missing_zip_manifests': meta.get('missing_zip_manifests', []),
			'zip_dataset_prefixes': set(meta.get('zip_dataset_prefixes', [])),
			'log_summary': meta.get('log_summary', {}),
			'summary_path': summary_path,
			'used_cache': True,
		}

	if not compute_counts_from_results or not load_analyzed_df or not cast_dataset_str:
		empty = pd.DataFrame()
		return {
			'counts_df': empty,
			'analyzed_df': empty,
			'summary': empty,
			'results_datasets': [],
			'missing_zip_manifests': [],
			'zip_dataset_prefixes': [],
			'log_summary': {},
			'summary_path': summary_path,
			'used_cache': False,
		}

	counts_df = compute_counts_from_results(results_dir, verbose=verbose)
	analyzed_df = load_analyzed_df(forest_csv)

	counts_df = cast_dataset_str(counts_df)
	analyzed_df = cast_dataset_str(analyzed_df)

	if counts_df.empty:
		results_datasets: list[str] = []
	else:
		results_datasets = sorted(counts_df['dataset'].astype(str).unique())

	if results_datasets:
		analyzed_df_res = analyzed_df[analyzed_df['dataset'].isin(results_datasets)].copy()
		merged_results_only = analyzed_df_res.merge(counts_df, on='dataset', how='inner')
	else:
		merged_results_only = pd.DataFrame(columns=['dataset', *DISPLAY_CATEGORIES])

	if not merged_results_only.empty:
		for cat in DISPLAY_CATEGORIES:
			if cat not in merged_results_only.columns:
				merged_results_only[cat] = 0
		merged_results_only[DISPLAY_CATEGORIES] = (
			merged_results_only[DISPLAY_CATEGORIES]
			.fillna(0)
			.astype(int)
		)
		merged_results_only['TOT'] = merged_results_only[DISPLAY_CATEGORIES].sum(axis=1)
		merged_results_only = merged_results_only.sort_values('R', ascending=False)
		summary_cols = ['dataset', *DISPLAY_CATEGORIES, 'TOT']
		summary_cols = [c for c in summary_cols if c in merged_results_only.columns]
		summary = merged_results_only[summary_cols].copy()
		rename_map = {c: DISPLAY_LABELS.get(c, DISPLAY_NAMES.get(c, c)) for c in DISPLAY_CATEGORIES}
		rename_map['TOT'] = DISPLAY_LABELS.get('TOT', 'Total')
		summary = summary.rename(columns=rename_map)
	else:
		summary = pd.DataFrame(columns=['dataset', *DISPLAY_CATEGORIES])

	log_summary = counts_df.attrs.get('log_summary', {}) if hasattr(counts_df, 'attrs') else {}
	if not summary.empty and log_summary:
		summary['Worker start (min)'] = summary['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_start_min'))
		summary['Worker end (max)'] = summary['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_end_max'))
		summary['Worker span (s)'] = summary['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_duration_seconds'))
		if 'Worker span (s)' in summary.columns:
			summary['Worker span (s)'] = pd.to_numeric(summary['Worker span (s)'], errors='coerce').round(3)

	import zipfile

	missing: list[tuple[str, object]] = []
	zip_dataset_prefixes: set[str] = set()
	if results_dir.exists():
		for z in sorted(results_dir.glob('*.zip')):
			zip_dataset_prefixes.add(z.name.split('_')[0])
			try:
				with zipfile.ZipFile(z, 'r') as archive:
					names = archive.namelist()
					has_manifest = any('redis_backup_db' in name for name in names)
					if not has_manifest:
						missing.append((z.name, names[:10]))
			except Exception as exc:
				missing.append((z.name, f'error: {exc}'))

	cache_dir.mkdir(parents=True, exist_ok=True)
	try:
		summary.to_csv(summary_path, index=False)
		counts_df.to_csv(counts_cache, index=False)
		meta_payload = {
			'cached_at': datetime.utcnow().isoformat(),
			'source_mtime': _latest_results_mtime(results_dir),
			'results_datasets': results_datasets,
			'zip_dataset_prefixes': sorted(zip_dataset_prefixes),
			'missing_zip_manifests': missing,
			'log_summary': log_summary,
		}
		meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding='utf-8')
		if verbose:
			print(f"Saved redis summary cache at {summary_path}")
	except Exception as exc:
		if verbose:
			print(f"Warning: unable to persist redis summary caches ({exc})")

	return {
		'counts_df': counts_df,
		'analyzed_df': analyzed_df,
		'summary': summary,
		'results_datasets': results_datasets,
		'missing_zip_manifests': missing,
		'zip_dataset_prefixes': zip_dataset_prefixes,
		'log_summary': log_summary,
		'summary_path': summary_path,
		'used_cache': False,
	}

def get_first_table(
	report_data: Iterable[Mapping[str, Any]] | None,
	*,
	counts_df: pd.DataFrame | None = None,
	results_datasets: Iterable[str] | None = None,
	zip_dataset_prefixes: Iterable[str] | None = None,
	results_dir: Path | None = None,
	summary: pd.DataFrame | None = None,
) -> FirstTableArtifacts:
	def to_int(value: Any | None) -> int | None:
		try:
			return int(value)
		except (TypeError, ValueError):
			return None

	def to_float(value: Any | None) -> float | None:
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def extract_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
		metadata = entry.get('metadata') if isinstance(entry.get('metadata'), Mapping) else {}
		statistics = entry.get('forest_statistics') if isinstance(entry.get('forest_statistics'), Mapping) else {}
		dataset = str(entry.get('dataset', '') or '').strip() or '<unknown>'
		return {
			'dataset': dataset,
			'n_estimators': to_int(statistics.get('n_estimators')),
			'train_size': to_int(metadata.get('train_size')),
			'test_size': to_int(metadata.get('test_size')),
			'avg_depth': to_float(statistics.get('avg_depth')),
			'avg_leaves': to_float(statistics.get('avg_leaves')),
			'avg_nodes': to_float(statistics.get('avg_nodes')),
		}

	def build_eu_metrics(entries: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
		eu: dict[str, dict[str, Any]] = {}
		for e in entries:
			dataset = str(e.get('dataset', '') or '').strip()
			if not dataset:
				continue
			n_features = e.get('n_features') if isinstance(e.get('n_features'), (int, float)) else None
			mean_eu = e.get('mean_eu') if isinstance(e.get('mean_eu'), (int, float)) else None
			eu_complexity = e.get('eu_complexity') if isinstance(e.get('eu_complexity'), (int, float)) else None
			eu_obj = e.get('endpoints_universe') or e.get('endpoints') or e.get('endpoints_universe_summary')
			lengths = None
			eu_min = None
			eu_max = None
			eu_std_dev = None
			if isinstance(eu_obj, Mapping):
				lens: list[int] = []
				for _, endpoints in eu_obj.items():
					if isinstance(endpoints, (list, tuple)):
						lens.append(len(endpoints))
				if lens:
					lengths = lens
					comp_n = len(lens)
					comp_mean = float(np.mean(lens))
					comp_cplx = comp_mean * comp_n
					comp_min = float(np.min(lens))
					comp_max = float(np.max(lens))
					comp_std = float(np.std(lens)) if len(lens) > 1 else 0.0
					if n_features is None:
						n_features = comp_n
					if mean_eu is None:
						mean_eu = comp_mean
					if eu_complexity is None:
						eu_complexity = comp_cplx
					eu_min = comp_min
					eu_max = comp_max
					eu_std_dev = comp_std
			if n_features is not None and mean_eu is not None:
				entry = {
					'n_features': int(n_features),
					'mean_eu': float(mean_eu),
					'eu_complexity': float(eu_complexity) if eu_complexity is not None else float(mean_eu) * int(n_features),
					'lengths': lengths,
				}
				if eu_min is not None:
					entry['eu_min'] = float(eu_min)
				if eu_max is not None:
					entry['eu_max'] = float(eu_max)
				if eu_std_dev is not None:
					entry['eu_std_dev'] = float(eu_std_dev)
				eu[dataset] = entry
		return eu

	results_entries = list(report_data or [])
	summary_df = pd.DataFrame([extract_metadata(e) for e in results_entries])
	eu_metrics = build_eu_metrics(results_entries)

	for column_name, metric_key in (
		('n_features', 'n_features'),
		('mean_eu', 'mean_eu'),
		('eu_complexity', 'eu_complexity'),
		('eu_min', 'eu_min'),
		('eu_max', 'eu_max'),
		('eu_std_dev', 'eu_std_dev'),
	):
		summary_df[column_name] = summary_df['dataset'].apply(lambda d: eu_metrics.get(d, {}).get(metric_key))

	summary_df = summary_df.rename(columns={'mean_eu': 'mean eu features', 'eu_std_dev': 'eu std'})

	analyzed_sources: set[str] = {str(item).strip() for item in (results_datasets or []) if str(item).strip()}
	analyzed_sources.update(str(item).strip() for item in (zip_dataset_prefixes or []) if str(item).strip())

	results_dir_path = Path(results_dir) if results_dir is not None else None
	if not analyzed_sources and results_dir_path and results_dir_path.exists():
		analyzed_sources.update(p.name.split('_')[0] for p in results_dir_path.glob('*.zip'))

	if analyzed_sources:
		summary_df['analyzed'] = summary_df['dataset'].apply(lambda d: 'YES' if d in analyzed_sources else 'NO')
	else:
		summary_df['analyzed'] = 'N/A'

	INF = float('inf')
	summary_df['_sort_n_estimators'] = summary_df['n_estimators'].fillna(INF)
	summary_df['_sort_eu'] = summary_df.apply(
		lambda r: r['eu_complexity'] if pd.notna(r.get('eu_complexity')) else INF,
		axis=1,
	)
	summary_df = (
		summary_df
		.sort_values(['_sort_n_estimators', '_sort_eu', 'dataset'], ascending=[True, True, True])
		.drop(columns=['_sort_n_estimators', '_sort_eu'])
	)
	summary_df = summary_df.reset_index(drop=True)
	eu_only_df = summary_df[summary_df['eu_complexity'].notna()].copy()
	counts_df_local = counts_df.copy() if isinstance(counts_df, pd.DataFrame) else pd.DataFrame()
	log_summary = counts_df_local.attrs.get('log_summary', {}) if hasattr(counts_df_local, 'attrs') else {}

	if not summary_df.empty and log_summary:
		summary_df['Worker start (min)'] = summary_df['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_start_min'))
		summary_df['Worker end (max)'] = summary_df['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_end_max'))
		summary_df['Worker span (s)'] = summary_df['dataset'].map(lambda ds: log_summary.get(ds, {}).get('log_duration_seconds'))
		if 'Worker span (s)' in summary_df.columns:
			summary_df['Worker span (s)'] = pd.to_numeric(summary_df['Worker span (s)'], errors='coerce').round(3)

	columns_to_remove = [
		'avg_depth',
		'avg_leaves',
		'avg_nodes',
		'n_features',
		'eu_complexity',
		'Worker start (min)',
		'Worker end (max)',
		'Worker span (s)',
	]
	summary_df = summary_df.drop(columns=columns_to_remove, errors='ignore')

	_apply_int_format(summary_df, SUMMARY_INT_COLUMNS)
	numeric_cols = summary_df.select_dtypes(include='number').columns.tolist()
	float_cols = [col for col in numeric_cols if col not in SUMMARY_INT_COLUMNS]
	_apply_float_format(summary_df, float_cols, decimals=3)
	summary_df.attrs['format_int_columns'] = [col for col in SUMMARY_INT_COLUMNS if col in summary_df.columns]
	summary_df.attrs['format_float_columns'] = float_cols
	summary_df.attrs['format_float_decimals'] = 3

	summary = summary if isinstance(summary, pd.DataFrame) else pd.DataFrame()
	summary_counts = pd.DataFrame(columns=['dataset', *RESULT_COUNT_COLUMNS])
	if not summary.empty:
		summary_counts = summary.reindex(columns=['dataset', *RESULT_COUNT_COLUMNS]).copy()
		for col in RESULT_COUNT_COLUMNS:
			if col in summary_counts.columns:
				summary_counts[col] = pd.to_numeric(summary_counts[col], errors='coerce')

	worker_can_metrics = compute_worker_can_metrics(results_dir_path) if results_dir_path else {}
	per_dataset_rows: list[dict[str, Any]] = []
	overall_aggregate: dict[str, Any] | None = None
	if worker_can_metrics:
		for dataset_key, payload in worker_can_metrics.items():
			if dataset_key == '__overall__':
				aggregate = payload.get('aggregate') if isinstance(payload, Mapping) else {}
				if isinstance(aggregate, Mapping):
					overall_aggregate = {key: aggregate.get(key) for key in WORKER_CAN_COLUMN_MAP}
				continue
			aggregate = payload.get('aggregate') if isinstance(payload, Mapping) else {}
			if not isinstance(aggregate, Mapping):
				continue
			row = {'dataset': str(dataset_key)}
			for agg_key, column_name in WORKER_CAN_COLUMN_MAP.items():
				row[column_name] = aggregate.get(agg_key)
			per_dataset_rows.append(row)

	if per_dataset_rows:
		metrics_df = pd.DataFrame(per_dataset_rows).set_index('dataset')
		summary_counts = summary_counts.set_index('dataset')
		summary_counts = summary_counts.combine_first(metrics_df)
		summary_counts.update(metrics_df)
		summary_counts = summary_counts.reset_index()

	if overall_aggregate:
		rounded_overall = overall_aggregate.copy()
		for agg_key, column_name in WORKER_CAN_COLUMN_MAP.items():
			value = rounded_overall.get(agg_key)
			if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
				continue
			if column_name in COUNTS_INT_COLUMNS:
				rounded_overall[agg_key] = int(round(float(value)))
			else:
				rounded_overall[agg_key] = round(float(value), 3)
		worker_can_aggregate_row: dict[str, Any] | None = rounded_overall
	else:
		worker_can_aggregate_row = None

	_apply_int_format(summary_counts, COUNTS_INT_COLUMNS)
	_apply_float_format(summary_counts, COUNTS_FLOAT_COLUMNS, decimals=3)
	summary_counts.attrs['format_int_columns'] = [col for col in COUNTS_INT_COLUMNS if col in summary_counts.columns]
	summary_counts.attrs['format_float_columns'] = [col for col in COUNTS_FLOAT_COLUMNS if col in summary_counts.columns]
	summary_counts.attrs['format_float_decimals'] = 3

	primary_columns = [
		'dataset',
		'analyzed',
		'train_size',
		'test_size',
		'n_estimators',
		'mean eu features',
		'eu std',
	]
	available_primary = [col for col in primary_columns if col in summary_df.columns]
	remaining_columns = [col for col in summary_df.columns if col not in available_primary]
	summary_df = summary_df[available_primary + remaining_columns]

	return FirstTableArtifacts(
		summary_styler=style_summary_table(summary_df),
		summary_df=summary_df,
		counts_summary_df=summary_counts.copy(),
		eu_only_df=eu_only_df,
		result_count_columns=list(RESULT_COUNT_COLUMNS),
		worker_can_column_map=dict(WORKER_CAN_COLUMN_MAP),
		worker_can_aggregate_row=worker_can_aggregate_row,
		worker_can_metrics=worker_can_metrics,
	)


def _extract_artifacts(payload: ModelsAnalysisContext | ModelsAnalysisArtifacts) -> ModelsAnalysisArtifacts:
	if isinstance(payload, ModelsAnalysisContext):
		return payload.artifacts
	if isinstance(payload, ModelsAnalysisArtifacts):
		return payload
	raise TypeError('Expected ModelsAnalysisContext or ModelsAnalysisArtifacts')


def load_models_analysis_artifacts(
	base_dir: Path | str | None = None,
	*,
	verbose: bool = True,
) -> ModelsAnalysisArtifacts:
	base_dir_path = Path(base_dir).resolve() if base_dir else detect_base_dir()
	results_dir = base_dir_path / 'results'
	results_dir.mkdir(parents=True, exist_ok=True)
	forest_json = base_dir_path / 'forest_report.json'
	forest_csv = base_dir_path / 'forest_report.csv'

	report_data_raw = load_forest_report(forest_json)
	report_data = list(report_data_raw) if isinstance(report_data_raw, Iterable) else []

	results = load_results_artifacts(results_dir, forest_csv, verbose=verbose)

	summary = results.get('summary')
	summary_df = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame()
	counts = results.get('counts_df')
	counts_df = counts.copy() if isinstance(counts, pd.DataFrame) else pd.DataFrame()
	analyzed = results.get('analyzed_df')
	analyzed_df = analyzed.copy() if isinstance(analyzed, pd.DataFrame) else pd.DataFrame()

	saved_summary_path = results.get('summary_path')
	if isinstance(saved_summary_path, (str, os.PathLike)):
		saved_summary = Path(saved_summary_path)
	elif isinstance(saved_summary_path, Path):
		saved_summary = saved_summary_path
	else:
		saved_summary = None

	return ModelsAnalysisArtifacts(
		base_dir=base_dir_path,
		results_dir=results_dir,
		forest_json=forest_json,
		forest_csv=forest_csv,
		report_data=report_data,
		results_artifacts=dict(results),
		summary=summary_df,
		counts_df=counts_df,
		analyzed_df=analyzed_df,
		results_datasets=set(results.get('results_datasets', [])),
		zip_dataset_prefixes=set(results.get('zip_dataset_prefixes', [])),
		missing_zip_manifests=list(results.get('missing_zip_manifests', [])),
		log_summary=dict(results.get('log_summary', {})),
		saved_summary_path=saved_summary,
		used_cache=bool(results.get('used_cache')),
	)


def print_models_analysis_overview(payload: ModelsAnalysisContext | ModelsAnalysisArtifacts) -> None:
	artifacts = _extract_artifacts(payload)
	print(f'Base dir           : {artifacts.base_dir}')
	print(f'Loaded {len(artifacts.report_data)} rows from {artifacts.forest_json}')
	print(f'Results directory  : {artifacts.results_dir} (exists={artifacts.results_dir.exists()})')
	if artifacts.saved_summary_path:
		print(f'Summary cache path : {artifacts.saved_summary_path}')
	print(f'Using cached counts: {artifacts.used_cache}')


def print_models_analysis_diagnostics(payload: ModelsAnalysisContext | ModelsAnalysisArtifacts) -> None:
	artifacts = _extract_artifacts(payload)
	print(f'? BASE_DIR     : {artifacts.base_dir}')
	print(f'? RESULTS_DIR  : {artifacts.results_dir}')
	print(f'? FOREST_REPORT: {artifacts.forest_csv.exists()}')

	if artifacts.missing_zip_manifests:
		print('ZIP without redis manifest:')
		for entry in artifacts.missing_zip_manifests:
			if isinstance(entry, (list, tuple)) and entry:
				name = entry[0]
				sample = entry[1] if len(entry) > 1 else ''
			else:
				name = entry
				sample = ''
			print('-', name, '->', sample)


	if artifacts.results_datasets:
		print('Dataset:', ', '.join(sorted(artifacts.results_datasets)))


def build_analyzed_counts_table(first_table: FirstTableArtifacts) -> tuple[pd.DataFrame, Any]:
	summary_df = first_table.summary_df if isinstance(first_table.summary_df, pd.DataFrame) else pd.DataFrame()
	counts_summary_df = first_table.counts_summary_df if isinstance(first_table.counts_summary_df, pd.DataFrame) else pd.DataFrame()
	result_columns = list(first_table.result_count_columns or RESULT_COUNT_COLUMNS)
	counts_display_cols = ['dataset', *result_columns]

	if not summary_df.empty and 'dataset' in summary_df.columns:
		analyzed_mask = summary_df.get('analyzed') == 'YES'
		analyzed_datasets = summary_df.loc[analyzed_mask.fillna(False), 'dataset'].astype(str)
	else:
		analyzed_datasets = pd.Series(dtype=str)

	if not counts_summary_df.empty:
		working_counts = counts_summary_df.copy()
		if 'dataset' in working_counts.columns:
			working_counts['dataset'] = working_counts['dataset'].astype(str)
			analyzed_counts_df = working_counts[working_counts['dataset'].isin(analyzed_datasets)].copy()
		else:
			analyzed_counts_df = pd.DataFrame(columns=counts_display_cols)
	else:
		analyzed_counts_df = pd.DataFrame(columns=counts_display_cols)

	if analyzed_counts_df.empty:
		analyzed_counts_df = pd.DataFrame(columns=counts_display_cols)

	for col in counts_display_cols:
		if col not in analyzed_counts_df.columns:
			analyzed_counts_df[col] = pd.NA
	analyzed_counts_df = analyzed_counts_df[counts_display_cols]

	worker_map = first_table.worker_can_column_map or {}
	aggregate_row = first_table.worker_can_aggregate_row or {}
	if worker_map and aggregate_row:
		overall_row = {'dataset': 'All workers'}
		for agg_key, column_name in worker_map.items():
			overall_row[column_name] = aggregate_row.get(agg_key)
		overall_df = pd.DataFrame([overall_row])
		for col in counts_display_cols:
			if col not in overall_df.columns:
				overall_df[col] = pd.NA
		overall_df = overall_df[counts_display_cols]
		analyzed_counts_df = pd.concat([analyzed_counts_df, overall_df], ignore_index=True, sort=False)

	_apply_int_format(analyzed_counts_df, COUNTS_INT_COLUMNS)
	_apply_float_format(analyzed_counts_df, COUNTS_FLOAT_COLUMNS, decimals=3)
	analyzed_counts_df.attrs['format_int_columns'] = [col for col in COUNTS_INT_COLUMNS if col in analyzed_counts_df.columns]
	analyzed_counts_df.attrs['format_float_columns'] = [col for col in COUNTS_FLOAT_COLUMNS if col in analyzed_counts_df.columns]
	analyzed_counts_df.attrs['format_float_decimals'] = 3

	styled_counts = style_summary_table(analyzed_counts_df)
	return analyzed_counts_df, styled_counts


def build_combined_analyzed_table(
	summary_df: pd.DataFrame | None,
	analyzed_counts_df: pd.DataFrame | None,
	*,
	column_colormaps: Mapping[str, str] | None = None,
	default_cmap: str | None = None,
) -> tuple[pd.DataFrame, Any]:
	summary_df = summary_df.copy() if isinstance(summary_df, pd.DataFrame) else pd.DataFrame()
	counts_df = analyzed_counts_df.copy() if isinstance(analyzed_counts_df, pd.DataFrame) else pd.DataFrame()

	if not summary_df.empty and 'analyzed' in summary_df.columns:
		analyzed_summary = summary_df[summary_df['analyzed'] == 'YES'].copy()
	else:
		analyzed_summary = summary_df.copy()

	if not counts_df.empty:
		counts_df = counts_df.copy()
		if 'dataset' in counts_df.columns:
			counts_df['dataset'] = counts_df['dataset'].astype(str)
			counts_df = counts_df[counts_df['dataset'] != 'All workers']
	else:
		counts_df = pd.DataFrame()

	if not analyzed_summary.empty and 'dataset' in analyzed_summary.columns:
		analyzed_summary['dataset'] = analyzed_summary['dataset'].astype(str)
		combined_df = analyzed_summary.merge(counts_df, on='dataset', how='left', suffixes=('', '_worker_can'))
	else:
		combined_df = counts_df.copy()

	first_columns = [col for col in analyzed_summary.columns if col != 'analyzed'] if not analyzed_summary.empty else ['dataset']
	if 'dataset' not in first_columns:
		first_columns = ['dataset', *first_columns]
	second_columns = [col for col in counts_df.columns if col != 'dataset']
	combined_columns: list[str] = []
	for col in first_columns:
		if col not in combined_columns:
			combined_columns.append(col)
	for col in ['dataset', *second_columns]:
		if col not in combined_columns:
			combined_columns.append(col)

	for col in combined_columns:
		if col not in combined_df.columns:
			combined_df[col] = pd.NA
	combined_df = combined_df[combined_columns]

	if 'analyzed' in combined_df.columns:
		combined_df = combined_df.drop(columns=['analyzed'])

	if 'dataset' in combined_df.columns:
		combined_df = combined_df.set_index('dataset')

	combined_analyzed_df = combined_df.transpose()

	index_renames = {
		'train_size': 'Train Size',
		'test_size': 'Test Size',
		'n_estimators': 'N Estimators',
		'mean eu features': 'Mean EU Features',
		'eu std': 'EU Std',
		'Total time (s) mean': 'Total Time (ms)',
		'ICF checks': 'ICF Checks',
		'Reason check iteration total': 'Reason Check Iteration',
		'IterGoodRatio': 'IterGoodRadio %',
		'IterBadRatio': 'IterBadRadio %',
		'Early Stop Good total': 'Early Stop Good',
		'Filtrered rate': 'Filtrered Rate %',
	}
	combined_analyzed_df = combined_analyzed_df.rename(index=index_renames)

	target_order = [
		'Train Size',
		'Test Size',
		'N Estimators',
		'Mean EU Features',
		'EU Std',
		'Total Time (ms)',
		'ICF Checks',
		'Reason Check Iteration',
		'IterGoodRadio %',
		'IterBadRadio %',
		'Early Stop Good',
		'Early Stop from Good',
		'Early Stop from Bad',
		'Filtrered Rate %',
	]
	combined_analyzed_df = combined_analyzed_df.reindex(target_order)

	if 'Total Time (ms)' in combined_analyzed_df.index:
		total_ms = pd.to_numeric(combined_analyzed_df.loc['Total Time (ms)'], errors='coerce') * 1000.0
		combined_analyzed_df.loc['Total Time (ms)'] = total_ms

	percent_rows_present = [row for row in COMBINED_PERCENT_ROWS if row in combined_analyzed_df.index]
	for row in percent_rows_present:
		percent_series = pd.to_numeric(combined_analyzed_df.loc[row], errors='coerce') * 100.0
		combined_analyzed_df.loc[row] = percent_series

	float_rows_present = [row for row in COMBINED_FLOAT_ROWS if row in combined_analyzed_df.index]
	for row in float_rows_present:
		combined_analyzed_df.loc[row] = pd.to_numeric(combined_analyzed_df.loc[row], errors='coerce').round(3)

	combined_analyzed_df.attrs['format_int_rows'] = [row for row in COMBINED_INT_ROWS if row in combined_analyzed_df.index]
	combined_analyzed_df.attrs['format_float_rows'] = float_rows_present
	combined_analyzed_df.attrs['format_float_decimals'] = 3
	combined_analyzed_df.attrs['format_percent_rows'] = percent_rows_present
	combined_analyzed_df.attrs['format_percent_decimals'] = 1

	for col in combined_analyzed_df.columns:
		try:
			converted = pd.to_numeric(combined_analyzed_df[col])
		except (ValueError, TypeError):
			continue
		else:
			combined_analyzed_df[col] = converted

	color_map = column_colormaps or COLUMN_COLORMAPS
	default = default_cmap or DEFAULT_CMAP

	styled = combined_analyzed_df.style
	for metric in combined_analyzed_df.index:
		row_numeric = pd.to_numeric(combined_analyzed_df.loc[metric], errors='coerce')
		if not row_numeric.notna().any():
			continue
		''' TODO coloration disabled
		cmap = color_map.get(metric, default)
		styled = styled.background_gradient(subset=pd.IndexSlice[[metric], :], cmap=cmap, axis=1)
		'''

	int_rows = combined_analyzed_df.attrs.get('format_int_rows', [])
	if int_rows:
		styled = styled.format(_format_int_value, subset=pd.IndexSlice[int_rows, :])

	float_rows = combined_analyzed_df.attrs.get('format_float_rows', [])
	if float_rows:
		decimals = int(combined_analyzed_df.attrs.get('format_float_decimals', 3))
		styled = styled.format(lambda v, d=decimals: _format_float_value(v, decimals=d), subset=pd.IndexSlice[float_rows, :])

	percent_rows = combined_analyzed_df.attrs.get('format_percent_rows', [])
	if percent_rows:
		decimals = int(combined_analyzed_df.attrs.get('format_percent_decimals', 1))
		styled = styled.format(lambda v, d=decimals: _format_percent_value(v, decimals=d), subset=pd.IndexSlice[percent_rows, :])

	return combined_analyzed_df, styled


def prepare_summary_display(
	summary: pd.DataFrame | None,
	*,
	display_categories: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, Any]:
	display_categories = list(display_categories) if display_categories is not None else list(DISPLAY_CATEGORIES)
	if isinstance(summary, pd.DataFrame) and not summary.empty:
		summary_to_show = summary.copy()
	else:
		columns = ['dataset', *display_categories] if display_categories else ['dataset']
		summary_to_show = pd.DataFrame(columns=columns)
	try:
		styler = style_summary_table(summary_to_show)
	except Exception:
		styler = summary_to_show
	return summary_to_show, styler


def prepare_models_analysis(
	base_dir: Path | str | None = None,
	*,
	verbose: bool = True,
) -> ModelsAnalysisContext:
	artifacts = load_models_analysis_artifacts(base_dir=base_dir, verbose=verbose)
	first_table = get_first_table(
		artifacts.report_data,
		counts_df=artifacts.counts_df,
		results_datasets=artifacts.results_datasets,
		zip_dataset_prefixes=artifacts.zip_dataset_prefixes,
		results_dir=artifacts.results_dir,
		summary=artifacts.summary,
	)
	analyzed_counts_df, analyzed_counts_styler = build_analyzed_counts_table(first_table)
	combined_analyzed_df, combined_analyzed_styler = build_combined_analyzed_table(first_table.summary_df, analyzed_counts_df)
	summary_to_show, summary_styler = prepare_summary_display(artifacts.summary)

	return ModelsAnalysisContext(
		artifacts=artifacts,
		first_table=first_table,
		analyzed_counts_df=analyzed_counts_df,
		analyzed_counts_styler=analyzed_counts_styler,
		combined_analyzed_df=combined_analyzed_df,
		combined_analyzed_styler=combined_analyzed_styler,
		summary_to_show=summary_to_show,
		summary_styler=summary_styler,
	)
