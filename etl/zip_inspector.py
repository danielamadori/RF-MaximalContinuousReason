import zipfile
from dataclasses import dataclass
import os
import base64
import binascii
import json
import math
import seaborn as sns

HOST_BASE = 9
DB_LABELS = {
    0: "DATA",
    1: "CAN",
    2: "R",
    3: "NR",
    4: "CAR",
    5: "AR",
    6: "GP",
    7: "BP",
    8: "PR",
    9: "AP",
    10: "LOGS",
}
MAX_FULL_BYTES = 999_999
MAX_PREVIEW_BYTES = 10_000


def scan_and_load(zip_paths, results_dir):

    zip_names = [path.name for path in zip_paths]
    zip_inventory = {
        "results_dir": str(results_dir),
        "count": len(zip_paths),
        "found": bool(zip_paths),
        "paths": [str(path) for path in zip_paths],
        "names": zip_names,
    }


    env_selected_index = os.environ.get("RESULTS_SELECTED_ZIP_INDEX")
    env_selected_zip = os.environ.get("RESULTS_SELECTED_ZIP")
    selected_zip_index = None
    selected_zip_name = None

    if env_selected_index is not None:
        try:
            candidate_index = int(env_selected_index)
        except ValueError:
            candidate_index = None
        if isinstance(candidate_index, int) and 0 <= candidate_index < len(zip_paths):
            selected_zip_index = candidate_index
            selected_zip_name = zip_names[selected_zip_index]
    if selected_zip_name is None and env_selected_zip in zip_names:
        selected_zip_name = env_selected_zip
        selected_zip_index = zip_names.index(selected_zip_name)
    if selected_zip_name is None and zip_names:
        selected_zip_index = 0
        selected_zip_name = zip_names[0]

    selected_zip_path = zip_paths[selected_zip_index] if selected_zip_index is not None else None

    if zip_names:
        print("Available ZIP archives:")
        for index, name in enumerate(zip_names):
            print(f"[{index}] {name}")
        user_choice = input("Select ZIP by index or name (press Enter to keep current selection): ").strip()
        if user_choice:
            resolved_index = None
            try:
                resolved_index = int(user_choice)
            except ValueError:
                resolved_index = None
            if resolved_index is not None and 0 <= resolved_index < len(zip_paths):
                selected_zip_index = resolved_index
                selected_zip_name = zip_names[selected_zip_index]
            elif user_choice in zip_names:
                selected_zip_name = user_choice
                selected_zip_index = zip_names.index(selected_zip_name)
            else:
                print("Invalid selection, keeping previous choice.")
        selected_zip_path = zip_paths[selected_zip_index] if selected_zip_index is not None else None
        if selected_zip_name is not None and selected_zip_path is not None:
            print(f"Current selection: [{selected_zip_index}] {selected_zip_name}")
        else:
            print("Current selection: none")
    else:
        print("No ZIP archives found in results directory.")

    zip_inventory["selection"] = {
        "name": selected_zip_name,
        "index": selected_zip_index,
        "path": str(selected_zip_path) if selected_zip_path else None,
    }

    return selected_zip_name

def format_bytes(size):
    """Return a human-readable representation of a file size."""
    if size is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} B"

def parse_zip_metadata(zip_path):
    """Extract dataset, class, completion flag, and host numbers from the archive name."""
    dataset, class_name, completion_flag, host_fragment = zip_path.stem.rsplit("_", 3)
    try:
        host_offset = int(host_fragment)
        host_id = host_offset + HOST_BASE
    except ValueError:
        host_offset = None
        host_id = None
    flag_lower = completion_flag.lower()
    if flag_lower in {"true", "false"}:
        is_completed = flag_lower == "true"
    else:
        is_completed = None
    size_bytes = zip_path.stat().st_size
    return {
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "dataset": dataset,
        "class": class_name,
        "completion_raw": completion_flag,
        "is_completed": is_completed,
        "size_bytes": size_bytes,
        "size_text": format_bytes(size_bytes),
        "host_offset": host_offset,
        "host_id": host_id,
    }

def detect_root_prefix(archive, zip_path):
    """Guess the common directory prefix used inside the archive."""
    stem_prefix = f"{zip_path.stem}/"
    has_stem = any(
        info.filename.startswith(stem_prefix)
        for info in archive.infolist()
        if not info.is_dir()
    )
    if has_stem:
        return stem_prefix
    return ""

def resolve_manifest(archive, zip_path):
    """Return the manifest data together with the prefix used inside the archive."""
    candidates = []
    stem_prefix = f"{zip_path.stem}/"
    candidates.append(stem_prefix)
    for info in archive.infolist():
        if info.is_dir():
            dirname = info.filename
            if dirname.startswith("__MACOSX/"):
                continue
            if not dirname.endswith("/"):
                dirname += "/"
            candidates.append(dirname)
    candidates.append("")
    seen = set()
    for prefix in candidates:
        if prefix in seen:
            continue
        seen.add(prefix)
        manifest_path = f"{prefix}manifest.json"
        try:
            with archive.open(manifest_path) as manifest_file:
                manifest = json.load(manifest_file)
        except KeyError:
            continue
        else:
            return prefix, manifest
    raise KeyError("manifest.json not found")

class DumpDecodeError(RuntimeError):
    """Generic error raised while decoding a Redis DUMP payload."""

@dataclass
class DumpSections:
    payload: bytes
    version: int
    checksum: bytes

class _LengthEncoding:
    __slots__ = ("value", "encoding")

    def __init__(self, value=None, encoding=None):
        self.value = value
        self.encoding = encoding


RDB_ENCODING_INT8 = 0
RDB_ENCODING_INT16 = 1
RDB_ENCODING_INT32 = 2
RDB_ENCODING_LZF = 3

def split_dump_sections(raw: bytes) -> DumpSections:
    """Split payload, RDB version, and checksum from a Redis dump."""
    if len(raw) < 10:
        raise DumpDecodeError("DUMP payload is too short to contain metadata")
    checksum = raw[-8:]
    version_bytes = raw[-10:-8]
    version = int.from_bytes(version_bytes, "little", signed=False)
    payload = raw[:-10]
    return DumpSections(payload=payload, version=version, checksum=checksum)


def _read_length_info(buffer: bytes, offset: int):
    if offset >= len(buffer):
        raise DumpDecodeError("Offset out of range while reading length")
    first = buffer[offset]
    prefix = first >> 6
    if prefix == 0:
        length = first & 0x3F
        return _LengthEncoding(length), offset + 1
    if prefix == 1:
        if offset + 1 >= len(buffer):
            raise DumpDecodeError("Truncated 14-bit encoded length")
        second = buffer[offset + 1]
        length = ((first & 0x3F) << 8) | second
        return _LengthEncoding(length), offset + 2
    if prefix == 2:
        if offset + 4 >= len(buffer):
            raise DumpDecodeError("Truncated 32-bit encoded length")
        length = int.from_bytes(buffer[offset + 1 : offset + 5], "big", signed=False)
        return _LengthEncoding(length), offset + 5
    return _LengthEncoding(None, first & 0x3F), offset + 1


def lzf_decompress(data: bytes, expected_length: int) -> bytes:
    """Minimal implementation of the LZF decompression used by Redis."""
    output = bytearray()
    idx = 0
    data_len = len(data)
    while idx < data_len:
        ctrl = data[idx]
        idx += 1
        if ctrl < 32:
            literal_len = ctrl + 1
            if idx + literal_len > data_len:
                raise DumpDecodeError("Truncated literal LZF sequence")
            output.extend(data[idx : idx + literal_len])
            idx += literal_len
        else:
            length = ctrl >> 5
            ref_offset = len(output) - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                if idx >= data_len:
                    raise DumpDecodeError("Truncated LZF sequence while extending length")
                length += data[idx]
                idx += 1
            if idx >= data_len:
                raise DumpDecodeError("Truncated LZF sequence while resolving reference")
            ref_offset -= data[idx]
            idx += 1
            length += 2
            if ref_offset < 0:
                raise DumpDecodeError("Negative LZF reference")
            for _ in range(length):
                if ref_offset >= len(output):
                    raise DumpDecodeError("LZF reference out of range")
                output.append(output[ref_offset])
                ref_offset += 1
    if len(output) != expected_length:
        raise DumpDecodeError(
            f"Unexpected decompressed length: expected {expected_length}, got {len(output)}"
        )
    return bytes(output)


def _decode_special_encoding(buffer: bytes, offset: int, encoding: int):
    if encoding == RDB_ENCODING_INT8:
        if offset >= len(buffer):
            raise DumpDecodeError("Truncated 8-bit encoded integer")
        value = int.from_bytes(buffer[offset : offset + 1], "little", signed=True)
        return str(value).encode("ascii"), offset + 1
    if encoding == RDB_ENCODING_INT16:
        if offset + 2 > len(buffer):
            raise DumpDecodeError("Truncated 16-bit encoded integer")
        value = int.from_bytes(buffer[offset : offset + 2], "little", signed=True)
        return str(value).encode("ascii"), offset + 2
    if encoding == RDB_ENCODING_INT32:
        if offset + 4 > len(buffer):
            raise DumpDecodeError("Truncated 32-bit encoded integer")
        value = int.from_bytes(buffer[offset : offset + 4], "little", signed=True)
        return str(value).encode("ascii"), offset + 4
    if encoding == RDB_ENCODING_LZF:
        compressed_len_info, next_offset = _read_length_info(buffer, offset)
        data_len_info, data_offset = _read_length_info(buffer, next_offset)
        if compressed_len_info.value is None or data_len_info.value is None:
            raise DumpDecodeError("Invalid LZF length encoding")
        end = data_offset + compressed_len_info.value
        if end > len(buffer):
            raise DumpDecodeError("Truncated encoded string")
        compressed = buffer[data_offset:end]
        decompressed = lzf_decompress(compressed, data_len_info.value)
        return decompressed, end
    raise DumpDecodeError("Unknown string encoding")


def _read_encoded_string(buffer: bytes, offset: int):
    length_info, next_offset = _read_length_info(buffer, offset)
    if length_info.encoding is None:
        end = next_offset + length_info.value
        if end > len(buffer):
            raise DumpDecodeError("Truncated encoded string")
        return buffer[next_offset:end], end
    return _decode_special_encoding(buffer, next_offset, length_info.encoding)


def decode_string_from_dump(raw: bytes) -> bytes:
    sections = split_dump_sections(raw)
    payload = sections.payload
    if not payload:
        raise DumpDecodeError("Empty payload")
    object_type = payload[0]
    if object_type != 0:
        raise DumpDecodeError(f"Non-string object type: {object_type}")
    value, _ = _read_encoded_string(payload, 1)
    return value


def decode_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise DumpDecodeError("Encoded value must be a string")
    try:
        return base64.b64decode(value.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise DumpDecodeError(f"Invalid base64 payload: {exc}") from exc


def decode_key(entry):
    return decode_bytes(entry["key"])


def text_preview(value: bytes, limit: int = 120) -> str:
    text = value.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[: limit - 1] + "."
    return text


def try_decode_value(entry):
    value_info = dict(entry.get("value") or {})
    data_b64 = value_info.get("data")
    if not data_b64:
        return "<no value>", value_info
    try:
        raw = decode_bytes(data_b64)
    except DumpDecodeError as exc:
        value_info["decode_error"] = str(exc)
        return "<invalid base64>", value_info
    details = {
        "dump_size": len(raw),
    }
    try:
        sections = split_dump_sections(raw)
        details["rdb_version"] = sections.version
        details["checksum"] = sections.checksum.hex()
    except DumpDecodeError as exc:
        details["dump_error"] = str(exc)
        return "<invalid dump>", details
    if entry.get("type") == "string":
        try:
            decoded = decode_string_from_dump(raw)
        except DumpDecodeError as exc:
            details["decode_error"] = str(exc)
            return "<string not decoded>", details
        details["decoded_bytes"] = decoded
        preview = text_preview(decoded)
        return preview, details
    return f"<{entry.get('type')} - {len(sections.payload)} bytes>", details


def shorten_text(text: str, limit: int = 600) -> str:
    sanitized = text.replace("````", "``` `")
    if len(sanitized) > limit:
        return sanitized[: limit - 1] + "."
    return sanitized


def summarise_backup_entries(entries, limit: int = 3):
    if not entries:
        return ["> No entries stored in this backup."]
    lines = []
    for index, entry in enumerate(entries[:limit], start=1):
        try:
            key_bytes = decode_key(entry)
            key_text = key_bytes.decode("utf-8", errors="replace") or "<empty key>"
        except (KeyError, DumpDecodeError) as exc:
            key_text = f"<unable to decode key: {exc}>"
        preview, details = try_decode_value(entry)
        entry_type = entry.get("type", "unknown")
        ttl = entry.get("pttl")
        ttl_text = f"{ttl}" if isinstance(ttl, int) else "persistent"
        lines.append(f"Entry {index}: key `{key_text}`")
        lines.append(f"Type: `{entry_type}`; TTL (ms): `{ttl_text}`")
        decoded_bytes = details.get("decoded_bytes")
        error = details.get("decode_error") or details.get("dump_error")
        if isinstance(decoded_bytes, (bytes, bytearray)):
            text_value = decoded_bytes.decode("utf-8", errors="replace")
            lines.append(shorten_text(text_value))
        else:
            lines.append(shorten_text(str(preview)))
        if error:
            lines.append(f"Warning: {error}")
    if len(entries) > limit:
        lines.append(f"Additional entries not shown: {len(entries) - limit}")
    return lines


def build_backup_preview(data):
    entries = data.get("entries") or []
    metadata = data.get("metadata") or {}
    return {
        "key_count": metadata.get("key_count", len(entries)),
        "created_at": metadata.get("created_at_utc"),
        "source": metadata.get("source") or {},
        "type_summary": metadata.get("type_summary") or {},
        "sample_entries": summarise_backup_entries(entries),
    }


def try_render_backup_preview(relative_name: str, payload: bytes):
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "entries" not in data or "metadata" not in data:
        return None
    return build_backup_preview(data)


def get_relative_member_name(info, prefix):
    member_name = info.filename
    if prefix and member_name.startswith(prefix):
        return member_name[len(prefix):]
    return member_name


def is_logs_entry(relative_name):
    normalized = relative_name.replace('\\', '/').lstrip('./')
    return normalized == 'logs' or normalized.startswith('logs/')


READ_JSON_LIMIT_BYTES = 9_000_000

def collect_archive_data(zip_path):
    meta = parse_zip_metadata(zip_path)
    result = {
        'zip_name': zip_path.name,
        'zip_path': str(zip_path),
        'metadata': meta,
        'manifest': None,
        'manifest_prefix': '',
        'db_overview': [],
        'members': [],
        'backups': {},
    }
    with zipfile.ZipFile(zip_path) as archive:
        try:
            prefix, manifest = resolve_manifest(archive, zip_path)
            result['manifest'] = manifest
            result['manifest_prefix'] = prefix
        except Exception:
            prefix = detect_root_prefix(archive, zip_path)
            manifest = None
            result['manifest_prefix'] = prefix
        if manifest:
            files_map = manifest.get('files', {})
            dbs = manifest.get('databases', [])
            for db_index in dbs:
                file_name = files_map.get(str(db_index))
                if not file_name:
                    continue
                archive_name = f"{prefix}{file_name}"
                try:
                    size = archive.getinfo(archive_name).file_size
                except KeyError:
                    size = None
                result['db_overview'].append({
                    'db_index': db_index,
                    'label': DB_LABELS.get(db_index, 'Unknown'),
                    'json_file': file_name,
                    'size_bytes': size,
                    'size_text': format_bytes(size) if size is not None else None,
                })
        members = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: info.filename)
        for info in members:
            relative = get_relative_member_name(info, prefix)
            if is_logs_entry(relative):
                continue
            size = info.file_size
            entry = {
                'relative_name': relative,
                'size_bytes': size,
                'size_text': format_bytes(size),
                'json_data': None,
                'json_truncated': False,
                'text_preview': None,
                'backup_preview': None,
            }
            read_entire = size <= MAX_FULL_BYTES or relative.endswith('.json')
            with archive.open(info.filename) as handle:
                payload = handle.read() if read_entire else handle.read(MAX_PREVIEW_BYTES)
            if relative.endswith('.json') and (size is None or size <= READ_JSON_LIMIT_BYTES):
                try:
                    text = payload.decode('utf-8')
                    data = json.loads(text)
                except Exception:
                    data = None
                if data is not None:
                    entry['json_data'] = data
                    preview = try_render_backup_preview(relative, payload)
                    if preview is not None:
                        entry['backup_preview'] = preview
                        result['backups'][relative] = data
                else:
                    entry['text_preview'] = payload.decode('utf-8', errors='replace')[:1000]
            else:
                entry['json_truncated'] = relative.endswith('.json') and (size is not None and size > READ_JSON_LIMIT_BYTES)
                try:
                    entry['text_preview'] = payload.decode('utf-8', errors='replace')[:1000]
                except Exception:
                    entry['text_preview'] = None
            result['members'].append(entry)
    return result

def _format_number(value: float) -> str:
    if math.isinf(value):
        return '∞' if value > 0 else '-∞'
    if math.isnan(value):
        return 'NaN'
    if abs(value) >= 1_000:
        return f"{value:.3g}"
    return f"{value:.6g}"

def _short_text(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + '…'

def _inline_summary(value, depth: int = 0) -> str:
    if isinstance(value, dict):
        return f"object({len(value)})"
    if isinstance(value, list):
        return f"array({len(value)})"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        return repr(_short_text(value))
    if value is None:
        return 'null'
    return repr(value)

def _extract_metadata_from_chunk(text: str):
    key = '"metadata"'
    idx = text.find(key)
    if idx == -1:
        return {}
    brace_start = text.find('{', idx)
    if brace_start == -1:
        return {}
    depth = 0
    for pos in range(brace_start, len(text)):
        char = text[pos]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                snippet = text[brace_start : pos + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return {}
    return {}

def _fetch_backup_metadata(zip_path, member_name, size_limit=2_000_000):
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(member_name) as handle:
                chunk = handle.read(size_limit)
    except KeyError:
        return {}
    text = chunk.decode('utf-8', errors='replace')
    return _extract_metadata_from_chunk(text)

def _describe_json(value, depth: int = 0):
    if isinstance(value, dict):
        keys = list(value.keys())
        preview_keys = keys[:5]
        headline = f"object with {len(keys)} keys: {', '.join(preview_keys)}" + ("…" if len(keys) > 5 else '')
        detail_lines = []
        for key in keys[:3]:
            detail_lines.append(f"{key}: {_inline_summary(value[key], depth + 1)}")
        if len(keys) > 3:
            detail_lines.append('…')
        return headline, detail_lines
    if isinstance(value, list):
        length = len(value)
        headline = f"array with {length} items"
        sample = [_inline_summary(item, depth + 1) for item in value[:3]]
        detail_lines = []
        if sample:
            detail = ', '.join(sample)
            detail_lines.append(f"sample: {detail}{'…' if length > 3 else ''}")
        return headline, detail_lines
    return _inline_summary(value, depth), []

def summarise_endpoint_univers(series_map):
    if not isinstance(series_map, dict):
        return 'feature map (unexpected structure)', []
    feature_names = sorted(series_map.keys())
    lengths = []
    details = []
    for name in feature_names[:5]:
        points = series_map.get(name)
        if isinstance(points, list):
            length = len(points)
            lengths.append(length)
            sample = []
            for value in points[:3]:
                if isinstance(value, (int, float)):
                    sample.append(_format_number(value))
                else:
                    sample.append(str(value))
            preview = ', '.join(sample)
            details.append(f"{name}: len={length}, sample=[{preview}{'…' if len(points) > 3 else ''}]")
        else:
            details.append(f"{name}: unexpected {type(points).__name__}")
    if len(series_map) > 5:
        details.append('…')
    if lengths:
        distinct_lengths = sorted(set(lengths))
        if len(distinct_lengths) == 1:
            headline = f"feature map (Endpoint Univers) with {len(series_map)} features; length {distinct_lengths[0]}"
        else:
            headline = (
                    f"feature map (Endpoint Univers) with {len(series_map)} features; "
                    + f"lengths {', '.join(str(l) for l in distinct_lengths)}"
            )
    else:
        headline = f"feature map (Endpoint Univers) with {len(series_map)} features"
    return headline, details

def summarise_random_forest(trees):
    if not isinstance(trees, list):
        return 'random forest (unexpected structure)', []
    count = len(trees)
    details = []
    for tree in trees[:5]:
        if isinstance(tree, dict):
            tree_id = tree.get('tree_id')
            feature = tree.get('feature')
            value = tree.get('value')
            feature_text = feature if feature is not None else '?'
            threshold = _format_number(value) if isinstance(value, (int, float)) else str(value)
            prefix = f"tree {tree_id}" if tree_id is not None else 'tree'
            details.append(f"{prefix}: root feature {feature_text}, threshold {threshold}")
        else:
            details.append(f"tree: unexpected {type(tree).__name__}")
    if count > 5:
        details.append('…')
    return f"random forest with {count} trees", details

def summarise_rf_optimization(result):
    if not isinstance(result, dict):
        return 'RF optimisation summary (unexpected structure)', []
    best = result.get('best_params') or {}
    best_keys = list(best.keys())
    headline_parts = []
    cv_score = result.get('best_cv_score')
    test_score = result.get('test_score')
    if isinstance(cv_score, (int, float)):
        headline_parts.append(f"best CV {cv_score:.3f}")
    if isinstance(test_score, (int, float)):
        headline_parts.append(f"test {test_score:.3f}")
    iter_count = result.get('n_iter')
    if isinstance(iter_count, int):
        headline_parts.append(f"n_iter {iter_count}")
    headline = "RF optimisation results"
    if headline_parts:
        headline += " (" + ', '.join(headline_parts) + ")"
    details = []
    for param in best_keys[:5]:
        details.append(f"best_params.{param} = {best[param]}")
    if len(best_keys) > 5:
        details.append('…')
    used_test = result.get('used_test_for_validation')
    if isinstance(used_test, bool):
        details.append(f"used_test_for_validation: {used_test}")
    timestamp = result.get('timestamp')
    if timestamp:
        details.append(f"timestamp: {timestamp}")
    return headline, details

def summarise_entry(key, value_json, value_text, skip_sample_keys=True):
    if skip_sample_keys and key.startswith('sample_'):
        return None, None
    if key == 'EU' and value_json is not None:
        return summarise_endpoint_univers(value_json)
    if key == 'RF' and value_json is not None:
        return summarise_random_forest(value_json)
    if key == 'RF_OPTIMIZATION_RESULTS' and value_json is not None:
        return summarise_rf_optimization(value_json)
    if value_json is not None:
        return _describe_json(value_json)
    return _short_text(value_text), []

def summarise_entry_generic(key, value_json, value_text):
    if key == 'EU' and value_json is not None:
        return summarise_endpoint_univers(value_json)
    if key == 'RF' and value_json is not None:
        return summarise_random_forest(value_json)
    if key == 'RF_OPTIMIZATION_RESULTS' and value_json is not None:
        return summarise_rf_optimization(value_json)
    if value_json is not None:
        return _describe_json(value_json)
    return _short_text(value_text), []


# Set seaborn style for better visualizations
sns.set_palette("husl")
sns.set_style("whitegrid")

DB_DISPLAY_NAMES = {
    'DATA': 'Data',
    'CAN': 'Candidate reasons',
    'R': 'Reasons',
    'NR': 'Non-reasons',
    'CAR': 'Candidate anti-reasons',
    'AR': 'Anti-reasons',
    'GP': 'Good profiles',
    'BP': 'Bad profiles',
    'PR': 'Preferred reasons',
    'AP': 'Anti-reason profiles',
    'LOGS': 'Worker iteration logs',
}
CANDIDATE_REASONS_NAME = DB_DISPLAY_NAMES['CAN']
CANDIDATE_ANTI_REASONS_NAME = DB_DISPLAY_NAMES['CAR']
BASE_COLUMNS = [
    'worker_id',
    'records',
    'iter_min',
    'iter_max',
    'queue_min',
    'queue_mean',
    'queue_max',
    'car_queue_min',
    'car_queue_mean',
    'car_queue_max',
    'hours_total',
    'hours_car',
    'hours_can',
]
RAW_INFO_KEYS = ['iterations', 'early_stop_good', 'deleted_from_R', 'deleted_from_GP', 'deleted_from_CAN']
EXTENSIONS_KEYS = ['total', 'added', 'filtered']
SCATTER_PLOTS = [
    {
        'title': 'Queue size vs iteration range',
        'x': 'iter_max',
        'y': 'queue_mean',
        'xlabel': None,
        'ylabel': None,
    },
    {
        'title': f'{CANDIDATE_ANTI_REASONS_NAME} queue vs queue size',
        'x': 'queue_mean',
        'y': 'car_queue_mean',
        'xlabel': None,
        'ylabel': None,
    },
    {
        'title': 'Total hours vs records',
        'x': 'records',
        'y': 'hours_total',
        'xlabel': None,
        'ylabel': None,
    },
    {
        'title': f'Processing hours ({CANDIDATE_ANTI_REASONS_NAME} vs {CANDIDATE_REASONS_NAME})',
        'x': 'hours_car',
        'y': 'hours_can',
        'xlabel': None,
        'ylabel': None,
    },
]
BAR_PLOTS = [
    {
        'columns': ['records'],
        'title': 'Events processed per worker',
        'ylabel': 'Events',
        'sort_by': 'records',
    },
]
STACKED_BAR_CONFIG = [
    {
        'prefix': 'car_result_',
        'title': f'{CANDIDATE_ANTI_REASONS_NAME} results per worker',
        'ylabel': 'Count',
    },
    {
        'prefix': 'can_result_',
        'title': f'{CANDIDATE_REASONS_NAME} results per worker',
        'ylabel': 'Count',
    },
    {
        'prefix': 'outcome_',
        'title': 'Outcome flags per worker',
        'ylabel': 'Count',
    },
]
HISTOGRAMS = [
    {
        'column': 'queue_mean',
        'title': 'Distribution of average queue size',
        'xlabel': None,
    },
    {
        'column': 'hours_total',
        'title': 'Distribution of total processing hours',
        'xlabel': None,
    },
]
ADDITIONAL_SCATTER_PREFIX_PAIRS = [
    ('car_result_CONFIRMED_AR', 'car_result_NOT_AR', f'{CANDIDATE_ANTI_REASONS_NAME} confirmed vs not'),
    ('can_result_GOOD', 'hours_total', f"{CANDIDATE_REASONS_NAME} GOOD vs total hours"),
]
