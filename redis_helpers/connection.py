"""
Redis connection utilities.
"""
import redis


# Default database mapping
DB_MAPPING = {
    'DATA': 0,
    'CAN': 1,      # Candidate reasons (for positive samples)
    'R': 2,        # Reasons (confirmed positive ICFs)
    'NR': 3,       # Non-reasons (confirmed negative ICFs)
    'CAR': 4,      # Candidate Anti-Reasons
    'AR': 5,       # Anti-Reasons (confirmed ARs)
    'GP': 6,       # Good Profiles (forest profiles of reasons)
    'BP': 7,       # Bad Profiles (forest profiles of non-reasons)
    'PR': 8,       # Preferred Reasons
    'AP': 9,       # Anti-Reason Profiles
    'LOGS': 10     # Worker iteration logs
}


def connect_redis(port=6379, host='localhost', db_mapping=None):
    """
    Establish Redis connections to all databases.
    
    Args:
        port: Redis server port (default: 6379)
        host: Redis server host (default: 'localhost')
        db_mapping: Optional custom database mapping, uses DB_MAPPING if None
        
    Returns:
        tuple: (connections dict, db_mapping dict)
    """
    if db_mapping is None:
        db_mapping = DB_MAPPING
    
    connections = {}
    
    for name, db_id in db_mapping.items():
        try:
            conn = redis.Redis(host=host, port=port, db=db_id, decode_responses=True)
            conn.ping()
            connections[name] = conn
            print(f"Connected to Redis DB {db_id} ({name}) on port {port}")
        except redis.ConnectionError:
            raise Exception(f"Failed to connect to Redis DB {db_id} ({name}) on port {port}")

    print(f"Established {len(connections)} Redis connections")
    return connections, db_mapping
