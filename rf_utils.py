"""
Random Forest utilities shared between init scripts.

Functions for training, optimizing, and converting Random Forest models.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from skforest_to_forest import sklearn_forest_to_forest


def create_forest_params(args):
    """
    Create RandomForest parameters from command line arguments.
    
    Args:
        args: Parsed argparse arguments with RF parameter fields
        
    Returns:
        dict: Parameters dictionary for RandomForestClassifier
    """
    rf_params = {'random_state': args.random_state}

    # Basic parameters
    if args.n_estimators:
        rf_params['n_estimators'] = args.n_estimators
    if args.criterion:
        rf_params['criterion'] = args.criterion

    # Tree structure parameters
    if args.max_depth:
        rf_params['max_depth'] = args.max_depth
    if args.min_samples_split:
        rf_params['min_samples_split'] = args.min_samples_split
    if args.min_samples_leaf:
        rf_params['min_samples_leaf'] = args.min_samples_leaf
    if args.max_leaf_nodes:
        rf_params['max_leaf_nodes'] = args.max_leaf_nodes

    # Feature selection
    if args.max_features:
        rf_params['max_features'] = args.max_features

    # Split quality
    if args.min_impurity_decrease:
        rf_params['min_impurity_decrease'] = args.min_impurity_decrease

    # Sampling parameters
    if args.bootstrap:
        rf_params['bootstrap'] = (args.bootstrap == 'True')
    if args.max_samples:
        rf_params['max_samples'] = args.max_samples

    # Pruning
    if args.ccp_alpha:
        rf_params['ccp_alpha'] = args.ccp_alpha

    return rf_params


def get_rf_search_space(include_bootstrap=True):
    """
    Define the hyperparameter search space for Bayesian optimization.

    Args:
        include_bootstrap: If True, includes bootstrap and max_samples in search space.
                          If False, excludes them to avoid constraint violations (default: True)
                          
    Returns:
        dict: Search space dictionary for BayesSearchCV
    """
    search_space = {
        # Number of trees
        'n_estimators': Integer(10, 300, name='n_estimators'),

        # Tree structure
        'max_depth': Integer(2, 50, name='max_depth'),
        'min_samples_split': Integer(2, 20, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 10, name='min_samples_leaf'),
        'max_leaf_nodes': Categorical([None, 10, 20, 30, 50, 100], name='max_leaf_nodes'),

        # Feature selection
        'max_features': Categorical(['sqrt', 'log2', None], name='max_features'),

        # Split quality
        'criterion': Categorical(['gini', 'entropy'], name='criterion'),
        'min_impurity_decrease': Real(0.0, 0.1, prior='uniform', name='min_impurity_decrease'),

        # Pruning
        'ccp_alpha': Real(0.0, 0.05, prior='uniform', name='ccp_alpha'),
    }

    # Only include bootstrap-related parameters if requested
    if include_bootstrap:
        search_space['bootstrap'] = Categorical([True], name='bootstrap')
        search_space['max_samples'] = Categorical([None, 0.5, 0.7, 0.9], name='max_samples')

    return search_space


def optimize_rf_hyperparameters(X_train, y_train, search_space, n_iter=50, cv=5,
                                 n_jobs=-1, random_state=42, verbose=1,
                                 X_test=None, y_test=None, use_test_for_validation=False):
    """
    Perform Bayesian optimization to find best Random Forest hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels
        search_space: Dictionary defining the search space for hyperparameters
        n_iter: Number of iterations for optimization (default: 50)
        cv: Number of cross-validation folds (default: 5)
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        X_test: Test features (optional, for validation)
        y_test: Test labels (optional, for validation)
        use_test_for_validation: If True, uses test set for validation instead of CV (default: False)

    Returns:
        best_params: Dictionary of best hyperparameters found
        best_score: Best cross-validation/test score achieved
        test_score: Test set score (if test data provided)
        optimizer: The fitted BayesSearchCV object (if CV used) or best RF model
    """
    print(f"[OPTIMIZE] Starting Bayesian Optimization for Random Forest hyperparameters")
    print(f"   Search space: {len(search_space)} hyperparameters")
    print(f"   Iterations: {n_iter}")

    if use_test_for_validation:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided when use_test_for_validation=True")
        print(f"   Validation: Test set ({X_test.shape[0]} samples)")
        print(f"   WARNING: Using test set for validation may lead to overfitting on test data!")
    else:
        print(f"   Validation: {cv}-fold cross-validation")

    print(f"   This may take a while...")

    if use_test_for_validation:
        # Manual optimization using test set for validation
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        # Convert search space to list format for gp_minimize
        dimensions = list(search_space.values())

        best_score = -np.inf
        best_params = None
        best_model = None

        @use_named_args(dimensions)
        def objective(**params):
            nonlocal best_score, best_params, best_model

            # Handle constraint: max_samples can only be used with bootstrap=True
            if 'bootstrap' in params and 'max_samples' in params:
                if not params['bootstrap'] and params['max_samples'] is not None:
                    params['max_samples'] = None

            # Train model with current parameters
            rf = RandomForestClassifier(**params, random_state=random_state)
            rf.fit(X_train, y_train)

            # Evaluate on test set
            score = rf.score(X_test, y_test)

            # Track best model
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = rf

            # Return negative score (gp_minimize minimizes)
            return -score

        print("\nRunning Bayesian optimization with test set validation...")
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iter,
            random_state=random_state,
            verbose=verbose > 0
        )

        print(f"\n[OK] Optimization complete!")
        print(f"   Best test score: {best_score:.4f}")
        print(f"   Best parameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

        return best_params, best_score, best_score, best_model

    else:
        # Standard cross-validation approach
        rf = RandomForestClassifier(random_state=random_state)

        # Create Bayesian optimizer
        optimizer = BayesSearchCV(
            estimator=rf,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            scoring='accuracy',
            return_train_score=True,
            error_score='raise'
        )

        # Fit the optimizer
        print("\n[RUNNING] Running Bayesian optimization...")
        optimizer.fit(X_train, y_train)

        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Optionally evaluate on test set
        test_score = None
        if X_test is not None and y_test is not None:
            test_score = optimizer.best_estimator_.score(X_test, y_test)
            print(f"\n[OK] Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Test set score: {test_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")
        else:
            print(f"\n[OK] Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")

        return best_params, best_score, test_score, optimizer


def train_and_convert_forest(X_train, y_train, X_test, y_test, rf_params, feature_names,
                               random_state=42, sample_percentage=None):
    """
    Train Random Forest and convert to our custom Forest format.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        rf_params: Random Forest parameters
        feature_names: List of feature names
        random_state: Random seed for reproducibility
        sample_percentage: If provided (e.g., 5.0), uses only a percentage of training data

    Returns:
        sklearn_rf: Trained sklearn RandomForestClassifier
        our_forest: Converted Forest object
        X_train_used: The actual training data used
        y_train_used: The actual training labels used
    """
    print("Training Random Forest...")
    print(f"RF parameters: {rf_params}")

    X_train_final = X_train
    y_train_final = y_train

    # Apply sample percentage filtering if specified
    if sample_percentage is not None and sample_percentage < 100.0:
        print(f"Applying sample percentage filter: {sample_percentage}%")
        n_samples = len(X_train_final)
        n_keep = max(1, int(n_samples * sample_percentage / 100.0))
        
        indices = np.random.choice(n_samples, size=n_keep, replace=False)
        X_train_final = X_train_final[indices]
        y_train_final = y_train_final[indices]
        
        print(f"Reduced training from {n_samples} to {len(X_train_final)} samples ({sample_percentage}%)")

    print(f"Training set: {X_train_final.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train Random Forest
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_final, y_train_final)

    # Evaluate
    train_score = rf.score(X_train_final, y_train_final)
    test_score = rf.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

    # Convert to our Forest format
    print("Converting to custom Forest format...")
    our_forest = sklearn_forest_to_forest(rf, feature_names)
    print(f"Converted to Forest with {len(our_forest)} trees")

    return rf, our_forest, X_train_final, y_train_final
