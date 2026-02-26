from tree import DecisionTree
from typing import List, Dict, Any

class Forest:
    def __init__(self, trees_data: List[Dict[str, Any]]):
        if not isinstance(trees_data, list):
            raise TypeError("trees_data must be a list")

        if len(trees_data) == 0:
            raise ValueError("trees_data cannot be empty")

        # Extract tree_ids and validate
        tree_ids = []
        for i, tree_dict in enumerate(trees_data):
            if not isinstance(tree_dict, dict):
                raise TypeError(f"Tree at index {i} must be a dictionary")

            if "tree_id" not in tree_dict:
                raise ValueError(f"Tree at index {i} must contain 'tree_id'")

            tree_id = tree_dict["tree_id"]
            if not isinstance(tree_id, int):
                raise TypeError(f"tree_id at index {i} must be an integer")

            tree_ids.append(tree_id)

        # Validate tree_ids start from 0 and have no holes
        tree_ids.sort()
        expected_ids = list(range(len(tree_ids)))

        if tree_ids != expected_ids:
            raise ValueError(f"tree_ids must start from 0 and have no holes. Expected {expected_ids}, got {tree_ids}")

        # Check for duplicates (should not happen after sorting check, but being extra safe)
        if len(set(tree_ids)) != len(tree_ids):
            raise ValueError("tree_ids must be unique")

        # Create DecisionTree objects and store them
        self.trees = []
        for tree_dict in trees_data:
            decision_tree = DecisionTree(tree_dict)
            self.trees.append(decision_tree)

        # Sort trees by tree_id to ensure consistent ordering
        self.trees.sort(key=lambda tree: tree.root["tree_id"])

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return self.trees[index]

    def get_tree_by_id(self, tree_id: int):
        """Get a tree by its tree_id"""
        if not isinstance(tree_id, int):
            raise TypeError("tree_id must be an integer")

        if tree_id < 0 or tree_id >= len(self.trees):
            raise IndexError(f"tree_id {tree_id} out of range [0, {len(self.trees)-1}]")

        return self.trees[tree_id]

    def extract_feature_thresholds(self):
        """
        Extract all threshold values for each feature across all trees in the forest.
        Merges thresholds from all trees, maintaining unique values and sorting.

        Returns:
            Dictionary mapping feature names to strictly monotonically increasing arrays
            starting with -inf and ending with +inf, containing all unique thresholds
            for that feature across all trees.
        """
        all_feature_thresholds = {}

        # Collect thresholds from all trees
        for tree in self.trees:
            tree_thresholds = tree.extract_feature_thresholds()

            for feature, thresholds in tree_thresholds.items():
                if feature not in all_feature_thresholds:
                    all_feature_thresholds[feature] = set()

                # Add all thresholds except the -inf and +inf endpoints
                # (we'll add these back at the end)
                for threshold in thresholds:
                    if threshold not in [float('-inf'), float('inf')]:
                        all_feature_thresholds[feature].add(threshold)

        # Convert sets to sorted arrays with -inf and +inf endpoints
        result = {}
        for feature, thresholds in all_feature_thresholds.items():
            # Sort unique thresholds and add endpoints
            sorted_thresholds = sorted(list(thresholds))
            monotonic_array = [float('-inf')] + sorted_thresholds + [float('inf')]
            result[feature] = monotonic_array

        return result

    def predict(self, sample_dict):
        """
        Predict the label for a given sample using majority voting across all trees.

        Args:
            sample_dict: Dictionary with feature_name: float_value pairs

        Returns:
            str: The predicted label from majority voting across all trees

        Raises:
            ValueError: If required features are missing from sample_dict
            TypeError: If sample_dict is not a dictionary
        """
        if not isinstance(sample_dict, dict):
            raise TypeError("sample_dict must be a dictionary")

        if len(self.trees) == 0:
            raise ValueError("Forest has no trees")

        # Get predictions from all trees
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(sample_dict)
            predictions.append(prediction)

        # Count votes for each label
        vote_counts = {}
        for prediction in predictions:
            if prediction in vote_counts:
                vote_counts[prediction] += 1
            else:
                vote_counts[prediction] = 1

        # Find the label with the most votes
        max_votes = max(vote_counts.values())

        # Get all labels with maximum votes (in case of ties)
        winning_labels = [label for label, count in vote_counts.items() if count == max_votes]

        # If there's a tie, return the first one (could be made more sophisticated)
        # In practice, ties are rare with enough trees
        winning_label = winning_labels[0]

        return winning_label

    def extract_icf(self, sample_dict):
        """
        Extract Interval Condition Features (ICF) for a given sample across the forest.

        Collects ICF from each tree and intersects the intervals for each feature
        to obtain the forest-level ICF.

        Args:
            sample_dict: Dictionary with feature_name: float_value pairs

        Returns:
            dict: Feature name -> (inf, sup) interval representing the intersection
                  of intervals across all trees in the forest

        Raises:
            ValueError: If required features are missing from sample_dict
            TypeError: If sample_dict is not a dictionary
        """
        if not isinstance(sample_dict, dict):
            raise TypeError("sample_dict must be a dictionary")

        if len(self.trees) == 0:
            raise ValueError("Forest has no trees")

        # Collect all features that appear in any tree
        all_features = set()
        tree_icfs = []

        # Extract ICF from each tree
        for tree in self.trees:
            tree_icf = tree.extract_icf(sample_dict)
            tree_icfs.append(tree_icf)
            all_features.update(tree_icf.keys())

        # Initialize forest ICF with full range for all features
        forest_icf = {}
        for feature in all_features:
            forest_icf[feature] = (float('-inf'), float('inf'))

        # Intersect intervals for each feature across all trees
        for feature in all_features:
            inf_bound = float('-inf')
            sup_bound = float('inf')

            for tree_icf in tree_icfs:
                if feature in tree_icf:
                    tree_inf, tree_sup = tree_icf[feature]
                    # Intersection: take maximum of lower bounds and minimum of upper bounds
                    inf_bound = max(inf_bound, tree_inf)
                    sup_bound = min(sup_bound, tree_sup)

            forest_icf[feature] = (inf_bound, sup_bound)

        return forest_icf

    def get_icf_profile(self, icf):
        """
        Compute the profile of an ICF in the forest.

        The profile is the set of tuples (tree_id, leaf_id) that can be reached
        within the ICF constraints across all trees in the forest.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals

        Returns:
            set: Set of tuples (tree_id, leaf_id) representing the forest profile
        """
        if not isinstance(icf, dict):
            raise TypeError("icf must be a dictionary")

        if len(self.trees) == 0:
            raise ValueError("Forest has no trees")

        forest_profile = set()

        # Get profile from each tree and add tree_id prefix
        for tree in self.trees:
            tree_id = tree.root["tree_id"]
            tree_profile = tree.get_icf_profile(icf)

            # Add (tree_id, leaf_id) tuples to forest profile
            for leaf_id in tree_profile:
                forest_profile.add((tree_id, leaf_id))

        return forest_profile

    def inflate_interval(self, icf, endpoints, feature_name, direction):
        """
        Inflate an ICF interval in a given direction until the forest profile changes.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals
            endpoints: Dictionary mapping feature names to monotonic arrays starting at -inf and ending at +inf
            feature_name: Name of the feature to inflate
            direction: "low" or "high" - direction to inflate

        Returns:
            dict or None: The inflated ICF, or None if already at infinity in the requested direction
        """
        if not isinstance(icf, dict):
            raise TypeError("icf must be a dictionary")
        if not isinstance(endpoints, dict):
            raise TypeError("endpoints must be a dictionary")
        if not isinstance(feature_name, str):
            raise TypeError("feature_name must be a string")
        if direction not in ["low", "high"]:
            raise ValueError("direction must be 'low' or 'high'")
        if feature_name not in endpoints:
            raise ValueError(f"Feature '{feature_name}' not found in endpoints")

        # Get sorted endpoints for the specific feature
        feature_endpoints = endpoints[feature_name]
        if not hasattr(feature_endpoints, '__iter__'):
            raise TypeError(f"endpoints['{feature_name}'] must be iterable")

        # Get current ICF bounds for the feature
        if feature_name in icf:
            icf_inf, icf_sup = icf[feature_name]
        else:
            icf_inf, icf_sup = float('-inf'), float('inf')

        # Check boundary conditions
        if direction == "low" and icf_inf == float('-inf'):
            return None
        if direction == "high" and icf_sup == float('inf'):
            return None

        # Get original forest profile
        original_profile = self.get_icf_profile(icf)

        # Start with current ICF bounds
        current_inf, current_sup = icf_inf, icf_sup

        while True:
            # Find next predecessor/successor to try
            if direction == "low":
                # Find predecessor of current_inf
                predecessors = [x for x in feature_endpoints if x < current_inf]
                if not predecessors:
                    # No predecessor available, return ICF with -infinity
                    result_icf = icf.copy()
                    result_icf[feature_name] = (float('-inf'), current_sup)
                    return result_icf

                new_bound = predecessors[-1]  # Largest predecessor

                # Create new ICF with expanded lower bound
                test_icf = icf.copy()
                test_icf[feature_name] = (new_bound, current_sup)

            else:  # direction == "high"
                # Find successor of current_sup
                successors = [x for x in feature_endpoints if x > current_sup]
                if not successors:
                    # No successor available, return ICF with +infinity
                    result_icf = icf.copy()
                    result_icf[feature_name] = (current_inf, float('inf'))
                    return result_icf

                new_bound = successors[0]  # Smallest successor

                # Create new ICF with expanded upper bound
                test_icf = icf.copy()
                test_icf[feature_name] = (current_inf, new_bound)

            # Compute new forest profile
            new_profile = self.get_icf_profile(test_icf)

            # If profile changed, return this ICF (first one that changes profile)
            if new_profile != original_profile:
                return test_icf

            # Profile didn't change, continue with this bound
            if direction == "low":
                current_inf = new_bound
            else:  # direction == "high"
                current_sup = new_bound

            # Special case: if we reached infinity and profile didn't change
            if new_bound == float('-inf') and direction == "low":
                result_icf = icf.copy()
                result_icf[feature_name] = (float('-inf'), current_sup)
                return result_icf
            elif new_bound == float('inf') and direction == "high":
                result_icf = icf.copy()
                result_icf[feature_name] = (current_inf, float('inf'))
                return result_icf

    def icf_profile_to_bitmap(self, icf):
        """
        Convert an ICF profile to a bitmap representation for the entire forest.

        The bitmap is the concatenation of individual tree bitmaps ordered by tree_id.
        Each tree's bitmap has length equal to that tree's number of leaves.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals

        Returns:
            list: Concatenated bitmap where each tree's portion has bitmap[i] = 1
                  if leaf i is in that tree's ICF profile, 0 otherwise
        """
        if not isinstance(icf, dict):
            raise TypeError("icf must be a dictionary")

        if len(self.trees) == 0:
            raise ValueError("Forest has no trees")

        forest_bitmap = []

        # Process trees in order by tree_id (trees are already sorted)
        for tree in self.trees:
            tree_bitmap = tree.icf_profile_to_bitmap(icf)
            forest_bitmap.extend(tree_bitmap)

        return forest_bitmap

    def deflate_interval(self, icf, endpoints, feature_name, direction):
        """
        Deflate an ICF interval in a given direction until it reaches consecutive endpoints
        or the forest profile changes.

        Returns None if the interval is already between two consecutive endpoints
        (endpoint_i <= b < e <= endpoint_(i+1)). Otherwise, deflates by moving the bound
        inward until the profile changes.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals
            endpoints: Dictionary mapping feature names to monotonic arrays starting at -inf and ending at +inf
            feature_name: Name of the feature to deflate
            direction: "low" or "high" - direction to deflate (inward)

        Returns:
            dict or None: The deflated ICF, or None if already between consecutive endpoints
        """
        if not isinstance(icf, dict):
            raise TypeError("icf must be a dictionary")
        if not isinstance(endpoints, dict):
            raise TypeError("endpoints must be a dictionary")
        if not isinstance(feature_name, str):
            raise TypeError("feature_name must be a string")
        if direction not in ["low", "high"]:
            raise ValueError("direction must be 'low' or 'high'")
        if feature_name not in endpoints:
            raise ValueError(f"Feature '{feature_name}' not found in endpoints")

        # Get sorted endpoints for the specific feature
        feature_endpoints = endpoints[feature_name]
        if not hasattr(feature_endpoints, '__iter__'):
            raise TypeError(f"endpoints['{feature_name}'] must be iterable")

        # Get current ICF bounds for the feature
        if feature_name in icf:
            icf_inf, icf_sup = icf[feature_name]
        else:
            icf_inf, icf_sup = float('-inf'), float('inf')

        # Check if interval is already between consecutive endpoints
        for i in range(len(feature_endpoints) - 1):
            endpoint_i = feature_endpoints[i]
            endpoint_i_plus_1 = feature_endpoints[i + 1]
            if endpoint_i <= icf_inf < icf_sup <= endpoint_i_plus_1:
                return None

        # Get original forest profile
        original_profile = self.get_icf_profile(icf)

        # Start with current ICF bounds
        current_inf, current_sup = icf_inf, icf_sup

        while True:
            # Find next successor/predecessor to try (opposite of inflate)
            if direction == "low":
                # Find successor of current_inf (move inward, increase lower bound)
                successors = [x for x in feature_endpoints if x > current_inf]
                if not successors or successors[0] >= current_sup:
                    # No valid successor or would make invalid interval
                    break

                new_bound = successors[0]  # Smallest successor

                # Create new ICF with contracted lower bound
                test_icf = icf.copy()
                test_icf[feature_name] = (new_bound, current_sup)

            else:  # direction == "high"
                # Find predecessor of current_sup (move inward, decrease upper bound)
                predecessors = [x for x in feature_endpoints if x < current_sup]
                if not predecessors or predecessors[-1] <= current_inf:
                    # No valid predecessor or would make invalid interval
                    break

                new_bound = predecessors[-1]  # Largest predecessor

                # Create new ICF with contracted upper bound
                test_icf = icf.copy()
                test_icf[feature_name] = (current_inf, new_bound)

            # Compute new forest profile
            new_profile = self.get_icf_profile(test_icf)

            # If profile changed, return this ICF (first one that changed profile)
            if new_profile != original_profile:
                return test_icf

            # Profile didn't change, continue with this bound
            if direction == "low":
                current_inf = new_bound
            else:  # direction == "high"
                current_sup = new_bound

            # Check if we're now between consecutive endpoints
            for i in range(len(feature_endpoints) - 1):
                endpoint_i = feature_endpoints[i]
                endpoint_i_plus_1 = feature_endpoints[i + 1]
                if endpoint_i <= current_inf < current_sup <= endpoint_i_plus_1:
                    result_icf = icf.copy()
                    result_icf[feature_name] = (current_inf, current_sup)
                    return result_icf

        # Return the current deflated ICF
        result_icf = icf.copy()
        result_icf[feature_name] = (current_inf, current_sup)
        return result_icf