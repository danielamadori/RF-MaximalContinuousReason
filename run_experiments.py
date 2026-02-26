import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import time
import json
import csv
import pandas as pd  
from experiments_utils import *
import datetime
import shutil

from init_baseline import load_dataset_from_baseline
from redis_helpers.connection import connect_redis


def get_experiments_order():

    exp_orders = pd.read_csv('experiment order.csv')
    exp_orders = exp_orders.rename(columns={'row_name': 'dataset', 'column_name': 'num_workers'})
    if 'num_workers' in exp_orders.columns:
        exp_orders['num_workers'] = exp_orders['num_workers'].astype(int)

    tps_wide = pd.read_csv('tps.csv', sep=';', decimal=',')
    tps_wide = tps_wide.rename(columns={'Unnamed: 0': 'dataset_id', 'Unnamed: 1': 'dataset'})
    tps = tps_wide.melt(
        id_vars=['dataset_id', 'dataset'],
        var_name='num_workers',
        value_name='tp'
    )
    tps['num_workers'] = tps['num_workers'].astype(int)
    return exp_orders, tps

combinations = {
    # all possible combinations of ablation parameters
    # R, NR, GP, BP
    0: [True, True, True, True],    # baseline already done 
    1: [False, True, True, True],   # no R
    2: [True, False, True, True],   # no NR
    3: [True, True, False, True],   # no GP
    4: [True, True, True, False],   # no BP
    5: [False, False, True, True],  # no R, no NR
    6: [False, True, False, True],  # no R, no GP
    7: [False, True, True, False],  # no R, no BP
    8: [True, False, False, True],  # no NR, no GP
    9: [True, False, True, False],  # no NR, no BP
    10: [True, True, False, False], # no GP, no BP
    11: [False, False, False, True], # no R, no NR, no GP
    12: [False, False, True, False], # no R, no NR, no BP
    13: [False, True, False, False], # no R, no GP, no BP
    14: [True, False, False, False], # no NR, no GP, no BP
    15: [False, False, False, False] # no R, no NR, no GP, no BP
}
list_datasets_ablation = ['ann-thyroid', ]
num_workers_ablation = [32]  # fixed for ablation

redis_config={'host': 'localhost', 'port': 6379, 'password': 'letsg0reas0n'}
def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with varying worker counts or run the ablation study.",
        add_help=False # Disable auto help to pass flags to sub-scripts
    )

    parser.add_argument("--ablation", action="store_true", help="Enable ablation study", default=False)
    parser.add_argument("--one-solution", action="store_true", help="Stop after finding one solution", default=True)
    args = parser.parse_args()
    connections, db_mapping = connect_redis(port=redis_config['port'])
    exp_orders, tps = get_experiments_order()
    if args.ablation:
        for dataset in list_datasets_ablation:            
            classes = get_classes(dataset, connections)
            for nw in num_workers_ablation:
                sample_timeout = float(tps[(tps['dataset'] == dataset) & (tps['num_workers'] == nw)]['tp'].values[0]*60)
                for comb_id, ablation_params in combinations.items():
                    file_log = f'results/checkpoints/{dataset}/{dataset}_{nw}_Ablation_log.txt'
                    os.makedirs(os.path.dirname(file_log), exist_ok=True)
                    with open(file_log, 'w') as f:
                        f.write(50*'=' + '\n')
                        f.write('Starting experiment with parameters:\n')
                        f.write('dataset: {}\nnum workers: {}\ntime per sample: {}\n'.format(dataset, nw, sample_timeout))
                        f.write('ablation params: R {}, NR {}, GP {}, BP {}\n'.format(
                            ablation_params[0], ablation_params[1], ablation_params[2], ablation_params[3]))
                        f.write(50*'=' + '\n')
                    for class_label in classes:
                        run_experiment_loop(dataset, class_label, nw, one_solution=args.one_solution, sample_timeout=None,
                                            use_R=ablation_params[0], use_NR=ablation_params[1],
                                            use_GP=ablation_params[2], use_BP=ablation_params[3], redis_config=redis_config)
    else:
        classes_datasets = {}
        for _, row in exp_orders.iterrows():
            dataset = row['dataset']
            num_workers = int(row['num_workers'])
            sample_timeout = float(tps[(tps['dataset'] == dataset) & (tps['num_workers'] == num_workers)]['tp'].values[0]*60)
            file_log = f'results/checkpoints/{dataset}/{dataset}_{num_workers}_log.txt'
            # file che abbia la lista dei contenuti delle reason da prendere direttamente da redis
            # 
            os.makedirs(os.path.dirname(file_log), exist_ok=True)
            with open(file_log, 'w') as f:
                f.write(50*'=' + '\n')
                f.write('Starting experiment with parameters:\n')
                f.write('dataset: {}\nnum workers: {}\ntime per sample: {}\n'.format(dataset, num_workers, sample_timeout))
                f.write(50*'=' + '\n')
            # or specific class labels if needed
            start_time = time.time()
            if dataset in list(classes_datasets.keys()):
                classes = classes_datasets[dataset] 
            else:
                classes = get_classes(dataset, connections)
            for class_label in classes:
                run_experiment_loop(dataset, class_label, num_workers, one_solution=args.one_solution, sample_timeout=sample_timeout, redis_config=redis_config)

            end_time = time.time()
            total_duration = end_time - start_time
            with open(file_log, 'a') as f:
                f.write(f"Total experiment duration: {total_duration:.2f} seconds\n")

if __name__ == "__main__":
    main()
