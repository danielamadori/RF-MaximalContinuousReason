#!/usr/bin/env python3
"""Utilities converted from redis_backup_migration.ipynb."""

from pathlib import Path
import os
from typing import Any, Dict, Iterable, Tuple

from redis.exceptions import ConnectionError, ResponseError

from redis_backup import (
    DEFAULT_REDIS_DATABASES,
    create_multi_database_backup,
    display_backup_summary,
    get_database_overview,
    load_multi_database_backup_from_directory,
    restore_multi_database_backup,
    save_multi_database_backup_to_directory,
)

# Redis instance configuration
SOURCE_REDIS_CONFIG: Dict[str, Any] = {
    "host": "127.0.0.1",  # Source instance IP or hostname
    "port": 6379,
    "db": 0,
    "username": None,  # Set a value if the server requires ACL authentication
    "password": os.environ.get("SOURCE_REDIS_PASSWORD", ""),
}

TARGET_REDIS_CONFIG: Dict[str, Any] = {
    "host": "127.0.0.1",  # Target instance IP or hostname
    "port": 6379,
    "db": 0,
    "username": None,
    "password": os.environ.get("TARGET_REDIS_PASSWORD", ""),
}

DATABASES_TO_MIGRATE = tuple(DEFAULT_REDIS_DATABASES)  # Databases 0 through 10
BACKUP_DIRECTORY = Path("redis_backups")  # Directory where backups will be stored
SCAN_COUNT = 1000  # Number of keys inspected per iteration during the SCAN
FLUSH_TARGET_BEFORE_RESTORE = False  # Set True to flush the target DB before restoring

# Helper functions are provided by the `redis_backup` module.
# You can open the `redis_backup.py` file to review the implementation or
# reuse them in other Python scripts.

# --- Command-line interface ------------------------------------------------------

import argparse
import logging
import sys


def _parse_databases(value: str | None) -> Tuple[int, ...]:
    if value is None:
        return DATABASES_TO_MIGRATE
    parts = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return tuple(sorted({int(part) for part in parts}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid database list {value!r}") from exc


def _config_repr(label: str, configuration: Dict[str, Any]) -> str:
    host = configuration.get("host", "localhost")
    port = configuration.get("port", 6379)
    db = configuration.get("db", 0)
    return f"{label} Redis at {host}:{port}/db{db}"


def _check_connection(label: str, configuration: Dict[str, Any]) -> bool:
    try:
        overview = get_database_overview(configuration)
    except ConnectionError as error:
        logging.error("%s – connection failed: %s", _config_repr(label, configuration), error)
        return False
    info = (
        f"{label}: Redis {overview['redis_version']} (mode {overview['redis_mode']}) – "
        f"{overview['key_count']} keys in db {overview['db']}"
    )
    logging.info(info)
    return True


def run_backup(
    source_config: Dict[str, Any],
    *,
    databases: Iterable[int],
    scan_count: int,
    backup_directory: Path,
) -> None:
    try:
        backup_payloads = create_multi_database_backup(
            source_config,
            databases=databases,
            scan_count=scan_count,
        )
    except ConnectionError as error:
        logging.error("Source Redis connection failed: %s", error)
        raise SystemExit(1) from error
    except ResponseError as error:
        logging.error("Error during export: %s", error)
        raise SystemExit(1) from error

    save_multi_database_backup_to_directory(
        backup_payloads,
        backup_directory,
        file_prefix="redis_backup",
    )
    total_keys = sum(payload["metadata"]["key_count"] for payload in backup_payloads.values())
    logging.info(
        "Backup completed: %s keys saved across %s databases (destination %s).",
        total_keys,
        len(backup_payloads),
        backup_directory.resolve(),
    )
    for db, payload in sorted(backup_payloads.items()):
        logging.info("Database %s:", db)
        display_backup_summary(payload)


def run_restore(
    target_config: Dict[str, Any],
    *,
    backup_directory: Path,
    flush_target: bool,
) -> None:
    try:
        backup_payloads = load_multi_database_backup_from_directory(backup_directory)
    except FileNotFoundError as error:
        logging.error("Backup directory not found: %s", error)
        raise SystemExit(1) from error
    try:
        restored_counts = restore_multi_database_backup(
            backup_payloads,
            target_config,
            flush_each=flush_target,
        )
    except ConnectionError as error:
        logging.error("Target Redis connection failed: %s", error)
        raise SystemExit(1) from error
    except ResponseError as error:
        logging.error("Error during restore: %s", error)
        raise SystemExit(1) from error

    total_restored = sum(restored_counts.values())
    logging.info(
        "Restore completed: %s keys imported across %s databases.",
        total_restored,
        len(restored_counts),
    )
    for db, restored in sorted(restored_counts.items()):
        logging.info("  Database %s: %s keys restored", db, restored)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or restore multi-database Redis backups.",
        epilog=(
            "Example usage:\n"
            "  python redis_backup_migration.py full "
            "--backup-dir ./redis_backups --verbose"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=("check", "backup", "restore", "full"),
        nargs="?",
        default="full",
        help="Operation to perform.",
    )
    parser.add_argument(
        "--databases",
        type=_parse_databases,
        default=None,
        help=(
            "Comma separated list of database numbers to process "
            f"(defaults to {','.join(map(str, DATABASES_TO_MIGRATE))})."
        ),
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=SCAN_COUNT,
        help="Number of keys inspected per SCAN iteration during backup.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=BACKUP_DIRECTORY,
        help="Directory where backup files are stored/read.",
    )
    parser.add_argument(
        "--flush-target",
        action="store_true",
        default=FLUSH_TARGET_BEFORE_RESTORE,
        help="Flush each target database before restoring.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip connection checks before running backup/restore.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    databases = args.databases or DATABASES_TO_MIGRATE
    action = args.action

    if not args.skip_check or action == "check":
        source_ok = _check_connection("Source", SOURCE_REDIS_CONFIG)
        target_ok = _check_connection("Target", TARGET_REDIS_CONFIG)
        if action == "check":
            return 0 if (source_ok and target_ok) else 1
        if not args.skip_check and (not source_ok or not target_ok):
            logging.error("Aborting due to failed connection check.")
            return 1

    if action in {"backup", "full"}:
        run_backup(
            SOURCE_REDIS_CONFIG,
            databases=databases,
            scan_count=args.scan_count,
            backup_directory=args.backup_dir,
        )

    if action in {"restore", "full"}:
        run_restore(
            TARGET_REDIS_CONFIG,
            backup_directory=args.backup_dir,
            flush_target=args.flush_target,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
