#!/usr/bin/env python3
"""
Run experiments for all classes of a dataset with a fixed number of workers.

Part of new hierarchy: dataset → workers → classes
Preserves AR (DB5) across class transitions within the same worker count.
"""
import argparse
import sys
import subprocess
import numpy as np
import os
import pandas as pd
import csv
import importlib


def get_dataset_classes(dataset_name, init_type):
    """
    Retrieve unique class labels for a dataset based on initialization type.
    Uses optimized metadata fetching where possible.
    """
    print(f"[WORKER ORCHESTRATOR] Fetching classes for '{dataset_name}' (Type: {init_type})...")
    
    if init_type == 'baseline':
        return get_baseline_classes(dataset_name)
    
    elif init_type == 'uci':
        try:
            init_module = importlib.import_module('init_uci')
            info = init_module.get_dataset_info(dataset_name=dataset_name)
            if 'error' in info:
                raise ValueError(info['error'])
            return np.unique([str(c) for c in info['classes']])
        except ImportError:
            print("[ERROR] Could not import 'init_uci'.")
            sys.exit(1)
            
    elif init_type == 'openml':
        try:
            init_module = importlib.import_module('init_openml')
            X, y, _ = init_module.load_and_prepare_dataset(dataset_name)
            if y is None:
                raise ValueError("Dataset not found or failed to load")
            return np.unique(y.astype(str))
        except ImportError:
            print("[ERROR] Could not import 'init_openml'.")
            sys.exit(1)
            
    elif init_type == 'pmlb':
        try:
            init_module = importlib.import_module('init_pmlb')
            X, y, _ = init_module.load_and_prepare_dataset(dataset_name, shuffle=False)
            if y is None:
                raise ValueError("Dataset not found on PMLB")
            return np.unique(y.astype(str))
        except ImportError:
            print("[ERROR] Could not import 'init_pmlb'.")
            sys.exit(1)
            
    else:
        print(f"[ERROR] Unknown init-type: {init_type}")
        sys.exit(1)


def get_baseline_classes(dataset_name):
    """Retrieve classes for baseline dataset from CSV (match init_baseline label logic)."""
    dataset_dir = os.path.join('baseline', 'resources', 'datasets', dataset_name)
    # Handle underscore vs hyphen retry
    if not os.path.exists(dataset_dir) and '_' in dataset_name:
        dataset_dir = os.path.join('baseline', 'resources', 'datasets', dataset_name.replace('_', '-'))
    
    dirname = os.path.basename(dataset_dir)
    csv_path = os.path.join(dataset_dir, f"{dirname}.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Baseline dataset CSV not found at {csv_path}")
        sys.exit(1)
        
    try:
        # Read labels as raw strings to mirror baseline.xrf.Dataset behavior
        labels = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty")
            for row in reader:
                if not row:
                    continue
                labels.append(str(row[-1]).strip())

        if not labels:
            raise ValueError("No labels found in CSV")

        # baseline.xrf.Dataset uses str.isnumeric() over raw strings
        all_numeric = all(lbl.isnumeric() for lbl in labels)

        if all_numeric:
            # Use actual numeric labels (e.g., 1,2,3 or 2,3,4,6,...)
            unique_labels = sorted({int(lbl) for lbl in labels})
            return np.array([str(v) for v in unique_labels])

        # Non-numeric labels are LabelEncoded to 0..k-1 in init_baseline
        num_classes = len(set(labels))
        return np.array([str(i) for i in range(num_classes)])
    except Exception as e:
        print(f"[ERROR] Failed to read baseline dataset: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for all classes with a fixed worker count."
    )
    parser.add_argument('dataset_name', help='Name of the dataset')
    parser.add_argument(
        '--num-workers',
        type=int,
        required=True,
        help='Fixed number of workers to use for all classes'
    )
    parser.add_argument(
        '--init-type',
        default='pmlb',
        choices=['pmlb', 'baseline', 'openml', 'uci'],
        help='Init script type'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=-1,
        help='Number of samples to select per class. Use -1 for ALL samples (default: -1 = all)'
    )
    parser.add_argument(
        '--one-solution',
        action='store_true',
        help='Stop after finding one solution'
    )
    parser.add_argument(
        '--preserve',
        action='store_true',
        help='Preserve AR from previous classes',
        default=True
    )
    # Capture any unknown args to pass to the runner
    args, remaining_args = parser.parse_known_args()

    print(f"\n{'=' * 60}")
    print(f"[WORKER ORCHESTRATOR] Dataset: {args.dataset_name}")
    print(f"[WORKER ORCHESTRATOR] Workers: {args.num_workers}")
    print(f"[WORKER ORCHESTRATOR] Init type: {args.init_type}")
    print(f"{'=' * 60}\n")
    
    # Get classes using generic helper
    try:
        classes = get_dataset_classes(args.dataset_name, args.init_type)
        print(f"[WORKER ORCHESTRATOR] Found {len(classes)} classes: {classes}\n")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve dataset classes: {e}")
        sys.exit(1)
    # Determine if we should preserve AR (all classes except first)
    preserve_ar = args.preserve
    for class_idx, cls in enumerate(sorted(classes)):
        print(f"\n{'-' * 60}")
        print(f"[WORKER ORCHESTRATOR] Processing Class {class_idx + 1}/{len(classes)}: {cls}")
        print(f"{'-' * 60}")
        
        if preserve_ar:
            print(f"[WORKER ORCHESTRATOR] Preserving AR (DB5) from previous classes")
        else:
            print(f"[WORKER ORCHESTRATOR] Fresh start (cleaning all databases)")
        
        indices_str = "auto"
        
        # For PMLB, we can efficiently select all or a subset of samples
        if args.init_type == 'pmlb':
            init_module = importlib.import_module('init_pmlb')
            _, y_full, _ = init_module.load_and_prepare_dataset(args.dataset_name, shuffle=False)
            indices = np.where(y_full.astype(str) == cls)[0]
            
            # Select all or subset based on args.samples_per_class
            if args.samples_per_class == -1:
                selected = sorted(indices)
                print(f"[WORKER ORCHESTRATOR] Selected ALL {len(selected)} samples for class '{cls}'")
            else:
                selected = sorted(indices)[:args.samples_per_class]
                print(f"[WORKER ORCHESTRATOR] Selected {len(selected)} samples for class '{cls}'")
            
            indices_str = ",".join(map(str, selected))

        # Construct Command
        cmd = [
            sys.executable, 'run_experiments.py',
            '--init-type', args.init_type,
            '--dataset', args.dataset_name,
            '--class-label', str(cls),
            '--num-workers', str(args.num_workers),  # Fixed worker count
        ]
        
        # Add preserve-ar flag for all classes except the first
        if preserve_ar:
            cmd.append('--preserve-ar')
        
        # Pass explicit index if we calculated it (PMLB optimization)
        if indices_str != "auto" and args.init_type == 'pmlb':
            cmd.extend(['--test-sample-index', indices_str])

        if args.one_solution:
            cmd.append('--one-solution')
            
        cmd += remaining_args
        
        print(f"[WORKER ORCHESTRATOR] Launching: {' '.join(cmd)}\n")
        
        try:
            subprocess.check_call(cmd)
            print(f"\n[WORKER ORCHESTRATOR] Class {cls} finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Experiment failed for class {cls} with code {e.returncode}.")
            print("[WORKER ORCHESTRATOR] Continuing to next class...")
            with open("worker_orchestrator_failures.txt", "a") as f:
                f.write(f"{args.dataset_name} workers_{args.num_workers} class_{cls}\n")
    
    print(f"\n{'=' * 60}")
    print(f"[WORKER ORCHESTRATOR] All classes completed for {args.num_workers} workers!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
