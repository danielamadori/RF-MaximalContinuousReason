# Baseline Experiments Setup Guide

## Prerequisites

1. **Redis server** running
   ```bash
   redis-server --protected-mode no
   ```

2. **Python environment** configured
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install python-sat
   ```

## Initial Repository State

After cloning the repository, you will find:

```
MCR/
├── baseline/
│   ├── Archive.zip              (COMMITTED - contains JSON models)
│   ├── resources.tar.gz         (COMMITTED - contains datasets)
│   ├── Classifiers-100-converted/   (EMPTY or NON-EXISTENT - to be populated)
│   └── resources/               (EMPTY or NON-EXISTENT - to be populated)
├── requirements.txt
├── run_experiments.py
├── init_baseline.py
└── ...
```

The `Classifiers-100-converted/` and `resources/` directories are in `.gitignore` because they contain large files.

## Complete Setup - Step by Step

### 1. Extract datasets from resources.tar.gz

```bash
cd baseline
tar -xzf resources.tar.gz
cd ..
```

Verify:
```bash
test -f baseline/resources/datasets/sonar/sonar.csv && echo "OK" || echo "ERROR"
# Expected output: OK
```

Expected result:
```
baseline/resources/
├── datasets/
│   ├── sonar/
│   │   ├── sonar.csv
│   │   └── sonar.samples
│   ├── ann-thyroid/
│   ├── appendicitis/
│   └── ... (54 total datasets)
└── Classifiers/
    └── ... (pickle models - not usable)
```

### 2. Extract JSON models from Archive.zip

```bash
cd baseline
unzip -q Archive.zip -d temp_extract
```

### 3. Copy models to the correct directory

```bash
mkdir -p Classifiers-100-converted
cp -r temp_extract/Classifiers-100-converted/* Classifiers-100-converted/
```

### 4. Cleanup (optional)

```bash
rm -rf temp_extract
cd ..
```

### 5. Verify complete setup

```bash
# Count available datasets
ls -1d baseline/resources/datasets/*/ | wc -l
# Expected: approximately 54

# Count available models
ls -1d baseline/Classifiers-100-converted/*/ | wc -l
# Expected: approximately 35

# Verify a specific model
test -f baseline/Classifiers-100-converted/sonar/sonar_nbestim_100_maxdepth_5.mod.json && echo "OK" || echo "ERROR"
# Expected: OK
```

## Final Structure After Setup

```
baseline/
├── Archive.zip                    (original, can be deleted after)
├── resources.tar.gz               (original, can be deleted after)
├── Classifiers-100-converted/     (EXTRACTED - required for experiments)
│   ├── ann-thyroid/
│   │   └── ann-thyroid_nbestim_100_maxdepth_4.mod.json
│   ├── sonar/
│   │   └── sonar_nbestim_100_maxdepth_5.mod.json
│   ├── appendicitis/
│   └── ... (35 datasets with JSON models)
└── resources/                     (EXTRACTED - required for experiments)
    └── datasets/
        ├── sonar/
        │   ├── sonar.csv          (complete data)
        │   └── sonar.samples      (test sample indices)
        └── ... (54 datasets)
```

## Testing the Setup

### Quick test with sonar dataset

```bash
python init_baseline.py sonar --class-label "1"
```

Expected output:
```
[START] Initializing Random Path Worker System (Baseline)
[INFO] Dataset: sonar
[INFO] Target Class Label: 1
Connected to Redis DB 0 (DATA) on port 6379
...
[SUCCESS] Successfully initialized sonar
```

### Run complete experiment

```bash
python run_experiments.py --init-type baseline --dataset sonar --class-label "1" --dataset-timeout 300 --max-workers 3
```

## Available Baseline Datasets

Only datasets with JSON models in `Classifiers-100-converted/` can be used (35 out of 54 total):

- ann-thyroid
- appendicitis
- banknote
- biodegradation
- ecoli
- glass2
- heart-c
- ionosphere
- karhunen
- kr_vs_kp
- letter
- magic
- mofn-3-7-10
- mofn_3_7_10
- new-thyroid
- parity5+5
- pendigits
- phoneme
- ring
- segmentation
- shuttle
- sonar
- spambase
- spectf
- texture
- threeOf9
- twonorm
- vowel
- waveform-21
- waveform-40
- wdbc
- wine-recog
- wpbc
- xd6

## Standard Experiment Command

```bash
python run_experiments.py --init-type baseline --dataset DATASET_NAME --class-label "CLASS" --dataset-timeout 300 --max-workers 5
```

Replace:
- `DATASET_NAME` with dataset name (e.g., sonar, ann-thyroid)
- `CLASS` with class value as string (e.g., "0", "1")

### Command Parameters

- `--init-type baseline` - Required, specifies use of baseline pre-trained models
- `--dataset NAME` - Dataset name (must match directory in resources/datasets/)
- `--class-label "VALUE"` - Class to analyze (as string)
- `--dataset-timeout SECONDS` - Total dataset time budget per worker; per-sample timeout is `max((dataset_timeout * workers) / dataset_total_samples, MIN_SAMPLE_TIME=2)` and is passed to workers as `--sample-timeout`
- `--sample-timeout SECONDS` - Per-sample timeout passed directly to workers (mutually exclusive with `--dataset-timeout`)
- `--max-workers N` - Maximum number of workers to test (tests 1, 2, 3, ..., N)
- `--test-sample-index "0,1,2"` - Optional, specific indices to test (default: all)
- `--one-solution` - Optional, stop after finding first solution per sample

## Results Directory

Results are saved in:
```
results/
├── experiments_log.csv
└── checkpoints/
    └── {dataset}/
        └── workers_{N}/
            └── class_{class}_sample_all/
                ├── experiment_metadata.json
                ├── manifest.json
                ├── redis_dump_readable.json
                ├── redis_backup_db*.json
                └── logs/
```

This directory is in `.gitignore` and will be created automatically.

## Automated Setup Script

You can create a `setup.sh` script to automate everything:

```bash
#!/bin/bash
set -e

echo "Setting up baseline experiments..."

# Extract datasets
echo "Extracting datasets..."
cd baseline
tar -xzf resources.tar.gz
echo "Datasets extracted."

# Extract models
echo "Extracting JSON models..."
unzip -q Archive.zip -d temp_extract
mkdir -p Classifiers-100-converted
cp -r temp_extract/Classifiers-100-converted/* Classifiers-100-converted/
rm -rf temp_extract
cd ..
echo "Models extracted."

# Verify
echo "Verifying setup..."
DATASETS=$(ls -1d baseline/resources/datasets/*/ 2>/dev/null | wc -l)
MODELS=$(ls -1d baseline/Classifiers-100-converted/*/ 2>/dev/null | wc -l)

echo "Available datasets: $DATASETS"
echo "Available models: $MODELS"

if [ -f baseline/Classifiers-100-converted/sonar/sonar_nbestim_100_maxdepth_5.mod.json ]; then
    echo "Setup completed successfully!"
else
    echo "ERROR: Models not found!"
    exit 1
fi
```

Make it executable and run:
```bash
chmod +x setup.sh
./setup.sh
```

## Important Notes

- **Do not commit** extracted directories (they are in .gitignore)
- **Redis must be running** before executing experiments
- **Python environment must be activated** before every command: `source .venv/bin/activate`
- **Only 35 datasets** have available JSON models (out of 54 total in datasets)
- Models are Random Forests with 100 trees and varying max_depth (3-5 depending on dataset)

## Troubleshooting

### FileNotFoundError for missing models

**Cause:** JSON model not found in Classifiers-100-converted/

**Solution:** Verify that Archive.zip was extracted correctly

### ModuleNotFoundError: No module named 'pysat'

**Cause:** python-sat dependency not installed

**Solution:** 
```bash
pip install python-sat
```

### Redis connection refused

**Cause:** Redis not running

**Solution:** Start Redis with:
```bash
redis-server --protected-mode no
```

### Class mismatch warnings in samples

**Cause:** Samples in .samples file belong to different classes

**Impact:** System processes all samples of target class anyway (expected behavior)

## System Architecture

### Initialization (init_baseline.py)

1. Loads dataset from `baseline/resources/datasets/{dataset}/{dataset}.csv`
2. Loads test sample indices from `baseline/resources/datasets/{dataset}/{dataset}.samples`
3. Searches for JSON model in `baseline/Classifiers-100-converted/{dataset}/*.json`
4. If model does not exist, raises FileNotFoundError
5. Populates Redis with forest, samples, and endpoints

### Experiment Execution (run_experiments.py)

1. Calls init_baseline.py with specified dataset and class
2. Creates snapshot of initial Redis state
3. For each number of workers (1 to max-workers):
   - Restores Redis snapshot
   - Launches N workers in parallel
   - Waits for completion
   - Extracts results from Redis
   - Saves JSON dump to `results/checkpoints/{dataset}/workers_{N}/class_{class}_sample_all/`
