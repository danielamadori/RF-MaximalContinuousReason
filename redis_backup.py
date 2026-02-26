"""Utility functions for exporting and restoring Redis databases."""
from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import redis
from redis.exceptions import ResponseError

RedisConfig = Dict[str, Any]
BackupEntry = Dict[str, Any]
BackupPayload = Dict[str, Any]
SerializedValue = Dict[str, Any]

DEFAULT_REDIS_DATABASES: tuple[int, ...] = tuple(range(0, 11))


class _BinaryRedis(redis.Redis):
    """Redis client with backwards-compatible binary-safe helpers."""

    # ``redis-py`` 5.0 tightened the ``hset`` signature so that passing a
    # mapping as the second positional argument now raises ``DataError``.  Our
    # helpers – and the tests that exercise them – still rely on the historic
    # behaviour where ``client.hset(name, {b"field": b"value"})`` worked.  To
    # keep that ergonomic API we intercept the positional mapping and forward
    # it via the keyword parameter expected by modern versions of the client.
    def hset(
        self,
        name,
        key=None,
        value=None,
        mapping: Mapping[object, object] | None = None,
        items=None,
    ):
        if mapping is None and isinstance(key, Mapping) and value is None:
            mapping = key
            key = None
        return super().hset(name, key=key, value=value, mapping=mapping, items=items)


def build_redis_client(config: RedisConfig) -> redis.Redis:
    """Return a Redis client configured for binary-safe operations."""
    params = dict(config)
    if "host" not in params:
        raise ValueError("Configuration must include the 'host' key.")
    params.setdefault("port", 6379)
    params.setdefault("db", 0)
    if params.get("username") in (None, ""):
        params.pop("username", None)
    if params.get("password") in (None, ""):
        params.pop("password", None)
    params["decode_responses"] = False
    return _BinaryRedis(**params)


def encode_bytes(value: bytes) -> str:
    """Encode raw bytes as a base64 ASCII string."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError("The value to encode must be bytes-like.")
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    """Decode a base64 ASCII string back to bytes."""
    return base64.b64decode(value.encode("ascii"))


def utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO-8601."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_type(key_type: Any) -> str:
    if isinstance(key_type, bytes):
        return key_type.decode()
    return str(key_type)


def _serialise_value(
    client: redis.Redis, key: bytes, key_type: str
) -> SerializedValue | None:
    """Return a serialisable payload representing the key's value."""

    def key_still_exists() -> bool:
        try:
            return bool(client.exists(key))
        except AttributeError:
            # Older redis client versions may expose ``exists`` as a command on pipelines
            # only.  Falling back to ``dump`` ensures we fail safe.
            dumped = client.dump(key)
            return dumped is not None
    dumped_value = client.dump(key)
    if dumped_value is None:
        if not key_still_exists():
            return None
        raise RuntimeError(f"Failed to serialise key {key!r}: empty dump result")
    return {"encoding": "dump", "data": encode_bytes(dumped_value)}


def create_redis_backup(config: RedisConfig, scan_count: int = 1000) -> BackupPayload:
    """Export the full content of a Redis instance into a serialisable payload."""
    client = build_redis_client(config)
    entries: List[BackupEntry] = []
    type_counter: Counter[str] = Counter()

    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, count=scan_count)
        if not keys and cursor == 0:
            break
        for key in keys:
            key_type = _normalise_type(client.type(key))
            if key_type == "none":
                continue
            ttl_ms = client.pttl(key)
            if ttl_ms == -2:
                # Key removed between SCAN and TTL read
                continue
            ttl_field = int(ttl_ms) if ttl_ms is not None and ttl_ms > 0 else None
            serialized_value = _serialise_value(client, key, key_type)
            if serialized_value is None:
                continue
            entries.append(
                {
                    "key": encode_bytes(key),
                    "value": serialized_value,
                    "pttl": ttl_field,
                    "type": key_type,
                }
            )
            type_counter[key_type] += 1
        if cursor == 0:
            break

    entries.sort(key=lambda item: item["key"])

    metadata = {
        "created_at_utc": utc_now_iso(),
        "source": {
            "host": config.get("host"),
            "port": config.get("port", 6379),
            "db": config.get("db", 0),
        },
        "key_count": len(entries),
        "type_summary": dict(type_counter),
        "scan_count": scan_count,
    }
    username = config.get("username")
    if username not in (None, ""):
        metadata["source"]["username"] = username
    return {"metadata": metadata, "entries": entries}


def create_multi_database_backup(
    base_config: RedisConfig,
    databases: Iterable[int] | None = None,
    scan_count: int = 1000,
) -> Dict[int, BackupPayload]:
    """Create backups for multiple Redis databases using shared connection options.

    Args:
        base_config: Common Redis connection parameters (host, port, auth, etc.).
        databases: Iterable of database numbers to back up.  If ``None`` the
            :data:`DEFAULT_REDIS_DATABASES` interval (0 through 10) is used.
        scan_count: Hint for SCAN operations (forwarded to :func:`create_redis_backup`).

    Returns:
        Dictionary mapping each database number to its backup payload.
    """

    backups: Dict[int, BackupPayload] = {}
    shared_config = dict(base_config)
    db_iterable = DEFAULT_REDIS_DATABASES if databases is None else databases
    db_list = list(dict.fromkeys(int(db) for db in db_iterable))

    for db in db_list:
        db_config = dict(shared_config)
        db_config["db"] = db
        backups[db] = create_redis_backup(db_config, scan_count=scan_count)

    return backups


def save_multi_database_backup_to_directory(
    backups: Mapping[int, BackupPayload],
    directory: Path,
    *,
    file_prefix: str = "redis_backup",
) -> Dict[int, Path]:
    """Persist multi-database backups to a directory.

    Each database payload is stored in a dedicated JSON file and a manifest is
    generated to simplify loading the backups later on.

    Args:
        backups: Mapping of database number to backup payload.
        directory: Destination directory where files will be written.
        file_prefix: Prefix applied to every generated backup filename.

    Returns:
        Mapping from database number to the :class:`Path` of the generated file.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path_map: Dict[int, Path] = {}
    manifest = {
        "file_prefix": file_prefix,
        "databases": [],
        "files": {},
    }

    for db in sorted(int(key) for key in backups.keys()):
        filename = f"{file_prefix}_db{db}.json"
        path = directory / filename
        save_backup_to_file(backups[db], path)
        path_map[db] = path
        manifest["databases"].append(db)
        manifest["files"][str(db)] = filename

    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return path_map


def load_multi_database_backup_from_directory(directory: Path) -> Dict[int, BackupPayload]:
    """Load multi-database backups saved with :func:`save_multi_database_backup_to_directory`."""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found in '{directory}'. Did you save any backups yet?"
        )

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files") or {}
    if not files:
        raise ValueError(f"Manifest '{manifest_path}' does not list any backup files")

    backups: Dict[int, BackupPayload] = {}
    for db_str, filename in files.items():
        file_path = directory / filename
        db = int(db_str)
        backups[db] = load_backup_from_file(file_path)

    return backups


def save_backup_to_file(backup: BackupPayload, path: Path) -> Path:
    """Write a Redis backup payload to disk as JSON."""
    path = Path(path)
    path.write_text(json.dumps(backup, indent=2, sort_keys=True))
    return path


def load_backup_from_file(path: Path) -> BackupPayload:
    """Load a backup payload from a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_backup_summary(backup: BackupPayload) -> str:
    """Return a formatted summary string for a Redis backup payload."""

    metadata = backup.get("metadata", {})
    source = metadata.get("source", {})
    auth_fragment = ""
    username = source.get("username")
    if username not in (None, ""):
        auth_fragment = f", username={username}"

    lines = [
        "",
        "Backup summary",
        "-" * 40,
        f"Created at: {metadata.get('created_at_utc', 'n/a')}",
        "Source: host={host}, port={port}, db={db}{auth}".format(
            host=source.get("host", "n/a"),
            port=source.get("port", "n/a"),
            db=source.get("db", "n/a"),
            auth=auth_fragment,
        ),
        f"Number of keys: {metadata.get('key_count', 0)}",
    ]

    type_summary = metadata.get("type_summary") or {}
    if type_summary:
        lines.append("Type distribution:")
        for redis_type, count in sorted(type_summary.items()):
            lines.append(f"  - {redis_type}: {count}")
    else:
        lines.append("No keys found.")

    lines.append("-" * 40)
    return "\n".join(lines)


def display_backup_summary(backup: BackupPayload) -> None:
    """Pretty-print a short summary of a backup payload."""

    print(format_backup_summary(backup))


def get_database_overview(config: RedisConfig) -> Dict[str, Any]:
    """Return basic information about a Redis instance."""
    client = build_redis_client(config)
    info = client.info(section="server")
    return {
        "redis_version": info.get("redis_version", "?"),
        "redis_mode": info.get("redis_mode", "?"),
        "db": config.get("db", 0),
        "key_count": client.dbsize(),
    }


def restore_redis_backup(
    backup: BackupPayload, config: RedisConfig, *, flush_target: bool = False
) -> int:
    """Restore a backup payload into a Redis instance."""
    client = build_redis_client(config)
    if flush_target:
        client.flushdb()

    restored = 0
    pending_expiries: list[tuple[bytes, int]] = []
    for entry in backup.get("entries", []):
        key = decode_bytes(entry["key"])
        ttl_field = entry.get("pttl")
        ttl_ms = int(ttl_field) if ttl_field is not None and int(ttl_field) > 0 else None
        value_info: SerializedValue = entry.get("value", {})  # type: ignore[assignment]
        encoding = value_info.get("encoding")
        data = value_info.get("data")

        if encoding is None:
            encoding = "dump"
        if encoding != "dump":
            raise ValueError(f"Unsupported encoding '{encoding}' in backup entry")
        if data is None:
            raise ValueError("Missing raw dump payload for backup entry")

        raw_value = decode_bytes(data)
        try:
            client.restore(key, 0, raw_value, replace=True)
        except ResponseError as exc:
            message = str(exc)
            if "BUSYKEY" in message or "Target key name is busy" in message:
                client.delete(key)
                client.restore(key, 0, raw_value, replace=True)
            else:
                raise

        if ttl_ms is not None:
            pending_expiries.append((key, ttl_ms))
        restored += 1

    if pending_expiries:
        pipeline = client.pipeline(transaction=False)
        for expire_key, ttl_ms in pending_expiries:
            pipeline.pexpire(expire_key, ttl_ms)
        pipeline.execute()
    return restored


def restore_multi_database_backup(
    backups: Mapping[int, BackupPayload],
    base_config: RedisConfig,
    *,
    flush_each: bool = False,
) -> Dict[int, int]:
    """Restore multiple Redis databases from backup payloads.

    Args:
        backups: Mapping from database number to backup payload.
        base_config: Common Redis connection parameters (host, port, auth, etc.).
        flush_each: Whether to flush each database before restoring into it.

    Returns:
        Dictionary mapping database numbers to the number of restored keys.
    """

    restored_counts: Dict[int, int] = {}
    shared_config = dict(base_config)

    for db, payload in backups.items():
        db_config = dict(shared_config)
        db_config["db"] = db
        restored_counts[db] = restore_redis_backup(payload, db_config, flush_target=flush_each)

    return restored_counts


__all__ = [
    "BackupEntry",
    "BackupPayload",
    "RedisConfig",
    "DEFAULT_REDIS_DATABASES",
    "build_redis_client",
    "create_redis_backup",
    "decode_bytes",
    "display_backup_summary",
    "format_backup_summary",
    "create_multi_database_backup",
    "encode_bytes",
    "get_database_overview",
    "load_backup_from_file",
    "load_multi_database_backup_from_directory",
    "restore_multi_database_backup",
    "restore_redis_backup",
    "save_backup_to_file",
    "save_multi_database_backup_to_directory",
    "utc_now_iso",
]
