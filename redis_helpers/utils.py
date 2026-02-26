def clean_all_databases(connections, db_mapping):
    """Clean all Redis databases by flushing their contents"""
    cleaned = 0
    for name, conn in connections.items():
        try:
            conn.flushdb()
            print(f"Cleaned database {name} (DB {db_mapping[name]})")
            cleaned += 1
        except Exception as e:
            print(f"Error cleaning database {name}: {e}")

    print(f"\nCleaned {cleaned}/{len(connections)} databases")