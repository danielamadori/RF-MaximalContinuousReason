import json
from typing import Union, List, Dict, Any
from forest import Forest

def store_forest(redis_conn, key: str, forest: Forest) -> bool:
    """Store a Forest object in Redis"""
    if not isinstance(forest, Forest):
        raise TypeError("forest must be a Forest instance")

    if not isinstance(key, str):
        raise TypeError("key must be a string")

    try:
        # Extract the raw tree data from each DecisionTree in the forest
        trees_data = []
        for tree in forest.trees:
            trees_data.append(tree.root)

        # Convert to JSON string and store
        json_data = json.dumps(trees_data)
        redis_conn.set(key, json_data)
        print(f"Successfully stored forest with {len(trees_data)} trees in Redis key '{key}'")
        return True
    except Exception as e:
        print(f"Error storing forest: {e}")
        return False

def retrieve_forest(redis_conn, key: str) -> Union[Forest, None]:
    """Retrieve a Forest object from Redis"""
    try:
        json_data = redis_conn.get(key)
        if json_data is None:
            print(f"No forest found for key '{key}'")
            return None

        trees_data = json.loads(json_data)

        # Validate retrieved data
        if not isinstance(trees_data, list):
            raise ValueError("Retrieved data is not a list")

        # Create and return Forest object (this will validate the structure)
        forest = Forest(trees_data)
        print(f"Successfully retrieved forest with {len(forest)} trees from Redis key '{key}'")
        return forest

    except Exception as e:
        print(f"Error retrieving forest: {e}")
        return None

def store_forest_dict(redis_conn, key: str, trees_data: List[Dict[str, Any]]) -> bool:
    """Store a forest from raw dictionary data in Redis"""
    if not isinstance(trees_data, list):
        raise TypeError("trees_data must be a list")

    if not isinstance(key, str):
        raise TypeError("key must be a string")

    try:
        # Validate by creating a Forest object (this will check structure)
        forest = Forest(trees_data)

        # Convert to JSON string and store
        json_data = json.dumps(trees_data)
        redis_conn.set(key, json_data)
        print(f"Successfully stored forest dictionary with {len(trees_data)} trees in Redis key '{key}'")
        return True
    except Exception as e:
        print(f"Error storing forest dictionary: {e}")
        return False