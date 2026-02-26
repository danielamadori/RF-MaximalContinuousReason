"""
Load RandomForest from JSON format (created by convert_classifiers.py).

This allows loading classifiers across different sklearn versions.

Usage:
    from load_rf_from_json import load_rf_from_json

    rf = load_rf_from_json('resources/Classifiers-converted/iris/iris_nbestim_50_maxdepth_6.json')
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree


def reconstruct_tree(tree_dict):
    """
    Reconstruct a DecisionTree from dictionary.

    Args:
        tree_dict: Dictionary with tree structure

    Returns:
        DecisionTreeClassifier instance
    """
    # Create empty tree
    dt = DecisionTreeClassifier()

    # Extract tree data
    n_features = tree_dict['n_features']
    nodes = tree_dict['nodes']
    n_nodes = tree_dict['node_count']

    # Infer classes from value array
    n_classes = np.array([len(tree_dict['nodes']['value'][0][0])])
    n_outputs = 1

    # Create numpy arrays for tree structure
    children_left = np.array(nodes['left_child'], dtype=np.intp)
    children_right = np.array(nodes['right_child'], dtype=np.intp)
    feature = np.array(nodes['feature'], dtype=np.intp)
    threshold = np.array(nodes['threshold'], dtype=np.float64)
    impurity = np.array(nodes['impurity'], dtype=np.float64)
    n_node_samples = np.array(nodes['n_node_samples'], dtype=np.intp)
    weighted_n_node_samples = np.array(nodes['weighted_n_node_samples'], dtype=np.float64)
    value = np.array(nodes['value'], dtype=np.float64)

    # Initialize the tree structure
    dt.tree_ = Tree(n_features, n_classes, n_outputs)

    # Create missing_go_to_left array (default to False for old models)
    # This field was added in sklearn 1.4 for handling missing values
    missing_go_to_left = np.zeros(n_nodes, dtype=np.uint8)

    # Use __setstate__ to populate the tree (works across sklearn versions)
    state = {
        'max_depth': tree_dict['max_depth'],
        'node_count': n_nodes,
        'nodes': np.array(
            list(zip(
                children_left,
                children_right,
                feature,
                threshold,
                impurity,
                n_node_samples,
                weighted_n_node_samples,
                missing_go_to_left  # New field for sklearn 1.4+
            )),
            dtype=[
                ('left_child', '<i8'),
                ('right_child', '<i8'),
                ('feature', '<i8'),
                ('threshold', '<f8'),
                ('impurity', '<f8'),
                ('n_node_samples', '<i8'),
                ('weighted_n_node_samples', '<f8'),
                ('missing_go_to_left', 'u1')  # New field for sklearn 1.4+
            ]
        ),
        'values': value
    }

    dt.tree_.__setstate__(state)

    # Set DecisionTree attributes
    dt.n_features_in_ = n_features
    dt.n_outputs_ = n_outputs
    dt.n_classes_ = n_classes[0]
    dt.classes_ = np.arange(n_classes[0])
    dt.max_depth = tree_dict['max_depth']

    return dt


def load_rf_from_json(json_path):
    """
    Load RandomForestClassifier from JSON file.

    Args:
        json_path: Path to JSON file (str or Path)

    Returns:
        RandomForestClassifier instance
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load JSON
    with open(json_path, 'r') as f:
        rf_dict = json.load(f)

    if rf_dict['type'] != 'RandomForestClassifier':
        raise ValueError(f"Invalid classifier type: {rf_dict['type']}")

    # Create RandomForest with original parameters
    params = rf_dict['params']
    rf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        random_state=params['random_state'],
        criterion=params['criterion']
    )

    # Set fitted attributes
    fitted = rf_dict['fitted']
    rf.n_features_in_ = fitted['n_features_in_']
    rf.n_classes_ = fitted['n_classes_']
    rf.classes_ = np.array(fitted['classes_'])
    rf.n_outputs_ = fitted['n_outputs_']

    # Reconstruct trees
    rf.estimators_ = [reconstruct_tree(tree_dict) for tree_dict in rf_dict['trees']]

    # Note: feature_importances_ is computed automatically from trees in newer sklearn
    # No need to set it explicitly

    return rf


def list_available_classifiers(converted_dirs=None):
    """
    List all available converted classifiers.

    Args:
        converted_dirs: List of directories with converted classifiers.
                       If None, checks both Classifiers-converted and Classifiers-100-converted

    Returns:
        List of (dataset_name, json_path) tuples
    """
    if converted_dirs is None:
        converted_dirs = [
            'resources/Classifiers-converted',
            'resources/Classifiers-100-converted'
        ]
    elif isinstance(converted_dirs, str):
        converted_dirs = [converted_dirs]

    classifiers = []
    for converted_dir in converted_dirs:
        converted_path = Path(converted_dir)

        if not converted_path.exists():
            continue

        for json_file in converted_path.rglob('*.json'):
            dataset_name = json_file.parent.name
            # Add source directory info
            source = converted_path.name.replace('-converted', '')
            classifiers.append((dataset_name, json_file, source))

    return sorted(classifiers)


if __name__ == "__main__":
    """Test loading a converted classifier."""

    print("="*70)
    print("TESTING JSON CLASSIFIER LOADING")
    print("="*70)

    # List available classifiers
    classifiers = list_available_classifiers("baseline/Classifiers-100-converted")

    if not classifiers:
        print("\n✗ No converted classifiers found.")
        print("  Run convert_classifiers.py first in sklearn-old environment")
        exit(1)

    print(f"\nFound {len(classifiers)} converted classifiers:")
    for dataset, path, source in classifiers[:5]:  # Show first 5
        print(f"  - {dataset} ({source}): {path.name}")
    if len(classifiers) > 5:
        print(f"  ... and {len(classifiers) - 5} more")

    # Try loading iris classifier
    iris_classifiers = [p for d, p, s in classifiers if d == 'iris']

    if iris_classifiers:
        print(f"\nTesting with iris classifier...")
        json_path = iris_classifiers[0]
        print(f"Loading: {json_path}")

        rf = load_rf_from_json(json_path)

        print(f"\n✓ Successfully loaded RandomForest:")
        print(f"  Trees: {rf.n_estimators}")
        print(f"  Max depth: {rf.max_depth}")
        print(f"  Features: {rf.n_features_in_}")
        print(f"  Classes: {rf.classes_}")

        # Test prediction
        from sklearn.datasets import load_iris
        iris = load_iris()
        X_test = iris.data[:5]

        predictions = rf.predict(X_test)
        print(f"\n✓ Predictions work: {predictions}")
    else:
        print("\n✗ No iris classifier found")

    print("\n" + "="*70)