# Design of Experiment: Reviewer Response

## Objective
To address the reviewer's critique regarding:
1.  **Baselines**: Quantify the performance improvement of the distributed architecture compared to a non-distributed (single-core) baseline.
2.  **Correctness**: Demonstrate that DRIFTS explanations are formally correct (valid AXps) compared to formal definitions (e.g., SAT-based AXp).

## 1. Datasets
The experimental evaluation uses **31 datasets** from PMLB (wpbc not available).
The table below lists the characteristics (Features, Classes, Instances) for each dataset.

| Dataset | Features | Classes | Instances (Total) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ann_thyroid** | 21 | 3 | 7200 | Thyroid disease (Medical) |
| **appendicitis** | 7 | 2 | 106 | Medical binary class |
| **analcatdata_bankruptcy** | 4 | 2 | 1372 | Bankruptcy data (was: banknote) |
| **biomed** | 41 | 2 | 1055 | Biomedical data (was: biodegradation) |
| **ecoli** | 7 | 8 | 336 | Protein localization sites |
| **glass2** | 9 | 2 | 163 | Glass identification (Binary) |
| **heart_c** | 13 | 5 | 303 | Cleveland Heart Disease |
| **ionosphere** | 34 | 2 | 351 | Radar data |
| **iris** | 4 | 3 | 150 | Fisher's Iris plant |
| **mfeat_karhunen** | 64 | 10 | 2000 | Karhunen-Loeve coefficients (was: karhunen) |
| **letter** | 16 | 26 | 20000 | Letter recognition |
| **magic** | 10 | 2 | 19020 | MAGIC Gamma Telescope |
| **mofn_3_7_10** | 10 | 2 | 1324 | Synthetic boolean |
| **new_thyroid** | 5 | 3 | 215 | Thyroid (small) |
| **pendigits** | 16 | 10 | 10992 | Pen-based digits |
| **phoneme** | 5 | 2 | 5404 | Speech processing |
| **ring** | 20 | 2 | 7400 | Synthetic (Ringnorm) |
| **segmentation** | 19 | 7 | 2310 | Image segmentation |
| **shuttle** | 9 | 7 | 58000 | Statlog Shuttle |
| **sonar** | 60 | 2 | 208 | Mines vs Rocks |
| **spambase** | 57 | 2 | 4601 | Email Spam |
| **spectf** | 44 | 2 | 267 | SPECT Heart Data |
| **texture** | 40 | 11 | 5500 | Texture analysis |
| **threeOf9** | 9 | 2 | 103+ | Synthetic |
| **twonorm** | 20 | 2 | 7400 | Synthetic (Twonorm) |
| **vowel** | 13 | 11 | 990 | Vowel recognition |
| **waveform_21** | 21 | 3 | 5000 | Waveform generator |
| **waveform_40** | 40 | 3 | 5000 | Waveform generator (noise) |
| **wdbc** | 30 | 2 | 569 | Breast Cancer Diagnostic |
| **wine_recognition** | 13 | 3 | 178 | Wine recognition |
| **xd6** | 9 | 2 | 500+ | Synthetic |

**Note:** wpbc (Breast Cancer Wisconsin Prognostic) non è disponibile in PMLB. Total: 31 datasets.

Datasets are initialized using the specific script corresponding to the `--init-type` argument (e.g., `init_pmlb.py`, `init_baseline.py`, `init_openml.py`).

## 2. Experiment Automation Scripts

The experiment suite is organized in a hierarchical structure with a **NEW** organization optimized for analyzing worker scaling effects:

### New Hierarchy (2026-01-21)

```
run_all_experiments.py              # Top-level: Iterates all datasets
    └── run_dataset_experiments.py      # Per-dataset: Iterates worker counts [32,16,8,4,2,1]
            └── run_worker_experiments.py   # Per-worker-count: Iterates all classes
                    └── run_experiments_drifts.py  # Per-class: Runs with fixed worker count
                            └── init_*.py + launch_workers.py  # Core execution
```

**Key Changes:**
- **Worker order**: Default descending [32, 16, 8, 4, 2, 1] to maximize Anti-Reason (AR) accumulation benefits (can be capped via `--max-workers`)
- **Directory structure**: `results/checkpoints/<dataset>/workers_<N>/class_<label>_sample_<idx>/`
- **AR Persistence**: Anti-reasons (DB5) are preserved across classes within the same worker count (controlled by `--preserve` flag)

---

### 2.1 `run_all_experiments.py`
**Purpose:** Execute the full benchmark suite on all datasets.

**Usage:**
```bash
python run_all_experiments.py --init-type baseline --one-solution --dataset-timeout 300
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--init-type` | string | **Yes** | - | Type of init script to use. Choices: `pmlb`, `openml`, `uci`, `baseline` |
| `--start-from` | string | No | - | Resume from a specific dataset name (useful after failures) |
| `--dry-run` | flag | No | False | Print commands without executing |
| `--one-solution` | flag | No | False | Stop after finding one solution (passed to child scripts) |
| `--max-workers` | int | No | - | Cap the worker counts used by `run_dataset_experiments.py` (optional) |
| `--dataset-timeout` | float | No | - | Worker-level dataset budget in seconds, converted per sample with `max((dataset_timeout * workers) / dataset_total_samples, MIN_SAMPLE_TIME=2)` using total dataset samples computed during init, and passed to workers as `--sample-timeout`. Mutually exclusive with `--sample-timeout`. |
| `--sample-timeout` | float | No | - | Worker-level per-sample timeout in seconds, passed directly to workers (no division). Mutually exclusive with `--dataset-timeout`. |

**Logs:**
- Failures are logged to `suite_failures_log.txt` and `dataset_failures_log.txt`

**NEW Behavior:** Calls `run_dataset_experiments.py` for each dataset instead of `run_class_experiments.py`

---

### 2.2 `run_dataset_experiments.py` (NEW)
**Purpose:** Orchestrate experiments for a single dataset across all worker counts.

**Usage:**
```bash
python run_dataset_experiments.py ann-thyroid --init-type baseline --one-solution
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `dataset_name` | string | **Yes** | - | Name of the dataset (positional argument) |
| `--init-type` | string | **Yes** | - | Init script type: `pmlb`, `openml`, `uci`, `baseline` |
| `--one-solution` | flag | No | False | Stop after finding one solution |
| `--max-workers` | int | No | - | Maximum worker count; defaults to `[32, 16, 8, 4, 2, 1]` when not set |
| Additional args | - | No | - | Passed through to `run_worker_experiments.py` (e.g., `--dataset-timeout`, `--sample-timeout`) |

**Behavior:**
1. Iterates through worker counts in **descending** order: default `[32, 16, 8, 4, 2, 1]`. If `--max-workers` is set, it keeps counts <= max and includes max if not in the default list.
2. For each worker count, calls `run_worker_experiments.py`
3. Logs failures to `dataset_failures_log.txt`

**Why Descending Order?**
Starting with 32 workers accumulates more anti-reasons (AR) faster, which then speeds up experiments with fewer workers due to AR reuse within the same worker configuration.

---

### 2.3 `run_worker_experiments.py` (NEW)
**Purpose:** Run experiments for all classes of a dataset with a fixed number of workers.

**Usage:**
```bash
python run_worker_experiments.py ann-thyroid \
    --num-workers 32 \
    --init-type baseline \
    --one-solution
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `dataset_name` | string | **Yes** | - | Name of the dataset (positional argument) |
| `--num-workers` | int | **Yes** | - | Fixed number of workers to use for all classes |
| `--init-type` | string | No | `pmlb` | Init script type |
| `--samples-per-class` | int | No | -1 | Number of samples per class (-1 = all) |
| `--one-solution` | flag | No | False | Stop after finding one solution |
| `--preserve` | flag | No | **True** | Preserve AR (DB5) across classes |
| Additional args | - | No | - | Passed to `run_experiments_drifts.py` (e.g., `--dataset-timeout`, `--sample-timeout`) |

**Behavior:**
1. Retrieves all classes for the dataset
2. Iterates through classes in sorted order
3. For each class:
   - If `--preserve=True` (default), passes `--preserve-ar` flag to preserve AR from previous classes
   - Calls `run_experiments_drifts.py` with `--num-workers` and `--preserve-ar` flags
4. Logs failures to `worker_orchestrator_failures.txt`

**AR Preservation Logic:**
- When `--preserve=True` (default): AR database (DB5) is preserved across all classes
- When `--preserve=False`: AR database is cleaned for every class (fresh start)

---

### 2.4 `run_experiments_drifts.py` (MODIFIED)
**Purpose:** Core experiment runner. For a given dataset/class, runs experiments with a specific worker count.

**Usage (Fixed Worker Mode):**
```bash
python run_experiments_drifts.py \
    --init-type baseline \
    --dataset ann-thyroid \
    --class-label 0 \
    --num-workers 32 \
    --preserve-ar \
    --one-solution
```

**Usage (Scalability Mode):**
```bash
python run_experiments_drifts.py \
    --init-type baseline \
    --dataset ann-thyroid \
    --class-label 0 \
    --dataset-timeout 60 \
    --max-workers 5
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--init-type` | string | **Yes** | - | Type of initialization script. Choices: `openml`, `pmlb`, `uci`, `baseline` |
| `--dataset` | string | **Yes** | - | Name of the dataset to use |
| `--class-label` | string | **Yes** | - | Target class label to test |
| `--num-workers` | int | No | - | Run with fixed worker count |
| `--preserve-ar` | flag | No | False | Preserve AR database (DB5) when restoring state |
| `--test-sample-index` | string | No | - | Sample indices to use. Supports comma-separated and ranges (e.g., `1,5,10` or `1-5`) |
| `--dataset-timeout` | float | No | - | Worker-level dataset budget in seconds; converted per sample with `max((dataset_timeout * workers) / dataset_total_samples, MIN_SAMPLE_TIME=2)` using total dataset sample count computed during init (same value across classes), and passed to workers as `--sample-timeout`. Mutually exclusive with `--sample-timeout` |
| `--sample-timeout` | float | No | - | Worker-level per-sample timeout in seconds (direct value, no division). Mutually exclusive with `--dataset-timeout` |
| `--max-workers` | int | No | CPU count | Limit maximum number of workers (scalability mode) |
| `--no-clean` | flag | No | - | Ignored/Deprecated (cleaning is enforced) |

**Modes:**

1. **Fixed Worker Mode** (when `--num-workers` is specified):
   - Runs with a fixed worker count
   - Supports AR preservation with `--preserve-ar` flag
   - Uses directory structure: `results/checkpoints/<dataset>/workers_<N>/class_<label>_sample_<idx>/`
   
2. **Scalability Mode** (when `--num-workers` is NOT specified):
   - Runs scalability test from 1 to N workers
   - Uses the same directory structure as above

**Features:**
- **Worker timeouts:** Per-sample timeouts are derived from `--dataset-timeout` using total dataset samples and worker count (minimum 2 seconds) before launch and passed to workers, or `--sample-timeout` is passed directly
- **Readable Dump:** Generates `redis_dump_readable.json` alongside binary backups
- **Log Archival:** Worker logs copied to checkpoint directory
- **Metadata:** Saves `experiment_metadata.json` with input parameters
- **AR Preservation:** When `--preserve-ar` is set, DB5 (anti-reasons) is not cleared during restore

**Timeout Semantics (important):**
- `--dataset-timeout` is a **worker-level dataset budget**, converted before worker launch to per-sample timeout:  
  `per_sample = max((dataset_timeout * workers) / dataset_total_samples, MIN_SAMPLE_TIME)` with `MIN_SAMPLE_TIME = 2`.  
  `dataset_total_samples` is derived during init from the dataset split and sample settings and represents the total dataset test samples, not per-class, so the derived per-sample timeout is **the same across all classes** of the dataset.
- `--sample-timeout` is a **worker-level per-sample cap** used as-is (no division).
- `--dataset-timeout` and `--sample-timeout` are mutually exclusive.

**Output Structure (New Hierarchy):**
```
results/checkpoints/<dataset>/
    ├── workers_32/
    │   ├── class_0_sample_all/
    │   │   ├── redis_backup_db0.json ... db10.json
    │   │   ├── redis_dump_readable.json
    │   │   ├── experiment_metadata.json
    │   │   └── logs/
    │   ├── class_1_sample_all/
    │   └── class_2_sample_all/
    ├── workers_16/
    ...
```

---

### 2.4 `init_baseline.py`
**Purpose:** Initialize Redis with pre-trained classifiers from baseline directory.

**Usage:**
```bash
python init_baseline.py --list-datasets
python init_baseline.py iris --class-label "0"
python init_baseline.py sonar --class-label "1" --test-sample-index "0,5,10"
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `dataset_name` | string | **Yes** | - | Name of the baseline dataset (positional argument) |
| `--list-datasets` | flag | No | False | List all available baseline datasets with pre-trained classifiers |
| `--class-label` | string | **Yes** | - | Target class label to process |
| `--test-sample-index` | string | No | - | Sample indices to use (e.g., `0,5,10` or `0-10`) |
| `--classifier-index` | int | No | 0 | Index of classifier to use if multiple exist |
| `--redis-port` | int | No | 6379 | Redis server port |
| `--no-clean` | flag | No | False | Do not clean Redis databases before initialization |

**Directory Structure:**
```
baseline/
├── Classifiers-100-converted/
│   └── <dataset_name>/
│       └── <dataset>_nbestim_<N>_maxdepth_<D>.mod.json
└── resources/
    └── datasets/
        └── <dataset_name>/
            ├── <dataset_name>.csv      # Dataset content (labels used to identify classes)
            └── <dataset_name>.samples  # Test samples (used instead of random selection)
```

**Key Differences from Standard Workflow:**
- **No Training:** Loads pre-trained Random Forest models from JSON files.
- **Fixed Models:** Uses existing classifiers from `baseline/Classifiers-100-converted/`.
- **Sample Files:** Test samples are loaded exclusively from `.samples` files. Random selection logic (`--test-sample-index`) is bypassed when running in batch mode; all samples matching the target class in the `.samples` file are processed.
- **Label Handling:** Enforces integer consistency for labels (e.g., "1.0" -> "1") across dataset loading, CLI arguments, and model predictions to ensure correct matching.
- **Dynamic Discovery:** `run_all_experiments.py` with `--init-type baseline` dynamically discovers all datasets in `baseline/resources/datasets`, ignoring the hardcoded PMLB list.

**Features:**
- Automatic interaction with `run_all_experiments.py` for full suite execution.
- Validation of CSV, samples, and classifier files.
- Smart label normalization to prevent type mismatches (float-string vs int-string).
- Compatible with standard experiment workflow (`run_experiments_drifts.py`).

---

## 3. Scalability Testing Methodology

The core objective is to measure **how computation time scales with the number of parallel workers**.

### 3.1 Experiment Flow (per sample)

For each sample, the script executes the following loop:

```
for k = 1 to max_workers:
    1. RESTORE initial Redis state (from snapshot taken after init)
    2. LAUNCH k worker processes
    3. WAIT for all workers to complete
    4. SAVE Redis state to results/checkpoints/<dataset>/workers_k/class_<label>_sample_<idx>/
    5. LOG duration to experiments_log.csv
```

### 3.2 Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INIT PHASE (once)                           │
│  Standard: init_pmlb.py → Train RF → Store in Redis → SNAPSHOT     │
│  Baseline: init_baseline.py → Load RF → Store in Redis → SNAPSHOT  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATION k=1 (1 Worker)                         │
│  RESTORE snapshot → Launch 1 worker → Wait → Save workers_1/       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATION k=2 (2 Workers)                        │
│  RESTORE snapshot → Launch 2 workers → Wait → Save workers_2/      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                                  ...
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATION k=N (N Workers)                        │
│  RESTORE snapshot → Launch N workers → Wait → Save workers_N/      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Points

1. **Same Initial State:** Each iteration starts from the SAME Redis snapshot, ensuring fair comparison.
2. **Independent Runs:** Workers in iteration k=2 are completely independent from k=1 (state is reset).
3. **Metrics Collected:**
   - **Duration:** Time from worker launch to completion
   - **Reasons Found:** Number of maximal reasons discovered
   - **Timeout Status:** Whether the iteration completed or was killed

### 3.4 Expected Results

| Workers | Expected Behavior |
|---|---|
| k=1 | Baseline (slowest, single-threaded) |
| k=2..N | Speedup due to parallel search |
| k=N (N=CPU cores) | Near-optimal parallelism |

The collected data allows plotting **speedup curves** (time vs. workers) to demonstrate the efficiency of the distributed architecture.
