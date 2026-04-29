# PyXAI Experiments on Random Forests

This file documents `run_pyxai_rf_experiments.py`, the runner used to measure
PyXAI explanation computation times on Random Forest models.

The default mode still reproduces the existing baseline setup: it reads
datasets and samples from `baseline/resources/datasets/` and loads converted
Random Forest JSON files from `baseline/Classifiers-100-converted/`. The runner
can also reuse the dataset loaders from `init_uci.py`, `init_openml.py`, and
`init_pmlb.py`; in those modes it trains a scikit-learn Random Forest in memory
and then imports that model into PyXAI. These modes do not initialize Redis.

## What It Does

The runner:

1. selects a source with `--init-type` (`baseline`, `uci`, `openml`, or `pmlb`);
2. loads the dataset and samples using the matching local conventions;
3. loads a pre-converted RF for `baseline`, or trains an RF in memory for
   `uci`, `openml`, and `pmlb`;
4. imports the scikit-learn RF into PyXAI with `Learning.import_models`;
5. for each selected sample, runs `Explainer.set_instance(...)` and the
   selected PyXAI explanation method;
6. writes per-sample timings, dataset-level aggregates, and a manifest.

By default, the runner uses `majoritary_reason`, because it is the most
practical RF-specific explanation method in PyXAI. The script also supports
`sufficient_reason`, `minimal_majoritary_reason`, `minimal_sufficient_reason`,
and `direct_reason`.

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

`pyxai` is required when explanations are actually run. The non-baseline
sources also require their loader dependencies (`ucimlrepo`, `openml`, `pmlb`,
and `scikit-optimize` where imported by the init modules). If one of those is
missing, the runner exits with the missing import in the error message.

## Main Commands

List baseline datasets that have all required files: CSV, samples, and RF
classifier.

```bash
python run_pyxai_rf_experiments.py --list-datasets
```

List datasets for another source. This delegates to the corresponding
`init_*` module.

```bash
python run_pyxai_rf_experiments.py --init-type pmlb --list-datasets
python run_pyxai_rf_experiments.py --init-type openml --list-datasets
python run_pyxai_rf_experiments.py --init-type uci --list-datasets
```

Run the full baseline experiment on every available converted dataset:

```bash
python run_pyxai_rf_experiments.py
```

Smoke test on a few baseline `iris` samples:

```bash
python run_pyxai_rf_experiments.py --datasets iris --max-samples 5 --overwrite
```

Run with a source backed by an `init_*` file:

```bash
python run_pyxai_rf_experiments.py --init-type pmlb --datasets iris --max-samples 5 --overwrite
python run_pyxai_rf_experiments.py --init-type openml --datasets credit-g --class-label good
python run_pyxai_rf_experiments.py --init-type uci --datasets Iris --max-samples 5
```

`--init-type` also accepts the script-like form:

```bash
python run_pyxai_rf_experiments.py --init-type init_pmlb.py --datasets iris
```

Use another PyXAI method or a per-sample time limit:

```bash
python run_pyxai_rf_experiments.py --reason-method sufficient_reason --time-limit 60 --overwrite
```

When changing the method, source, RF parameters, class filters, or other
important parameters, use `--overwrite` or a different `--output-dir` so
aggregate files do not mix configurations.

## Source Options

- `--init-type baseline`: default. Uses converted RF JSON files and `.samples`
  files from `baseline/`. RF training options are ignored in this mode.
- `--init-type uci`: uses `init_uci.load_and_prepare_dataset(...)`, then trains
  an RF in memory. `--uci-id` can be used instead of a dataset name.
- `--init-type openml`: uses `init_openml.load_and_prepare_dataset(...)`, then
  trains an RF in memory.
- `--init-type pmlb`: uses `init_pmlb.load_and_prepare_dataset(...)`, then
  trains an RF in memory.

For `uci`, `openml`, and `pmlb`, `--datasets` is required unless you are only
using `--list-datasets`.

## Sampling and Filtering

- `--max-samples N`: limit how many selected samples are timed by PyXAI.
- `--sample-index 0,3-5`: select sample rows after loading the test samples.
- `--test-sample-index 0,3-5`: for non-baseline sources, hold out those source
  rows as the test samples and train on the remaining rows.
- `--test-split 0.3`: train/test split used by non-baseline sources.
- `--sample-pct 25`: for OpenML/PMLB, keep a percentage of loaded rows before
  splitting, unless `--test-sample-index` is used.
- `--class-label LABEL`: only time samples matching the selected class.
- `--class-filter predicted|actual`: choose whether `--class-label` is matched
  against RF predictions or actual labels. Baseline has prediction filtering;
  actual-label filtering is available only when labels are loaded by the source.

For baseline runs, `sample_index` refers to the `.samples` row. For OpenML/PMLB
strict holdout runs, output indices are the original loaded row indices. For
UCI default split runs, output indices refer to the generated test set.

## RF Training Options

These options apply to `uci`, `openml`, and `pmlb` modes:

- `--n-estimators`
- `--criterion`
- `--max-depth`
- `--min-samples-split`
- `--min-samples-leaf`
- `--max-leaf-nodes`
- `--max-features`
- `--min-impurity-decrease`
- `--bootstrap`
- `--rf-max-samples`
- `--ccp-alpha`
- `--random-state`
- `--optimize --n-calls N`

`--rf-max-samples` is the Random Forest bootstrap parameter. It is separate
from PyXAI runner option `--max-samples`.

## Output

Default outputs are written to `results/pyxai_rf/`:

- `pyxai_rf_sample_times.csv`: one row per sample;
- `pyxai_rf_dataset_aggregates.csv`: aggregate statistics per dataset;
- `manifest_<run_id>.json`: run configuration, source metadata, and RF params.

Main columns in the per-sample CSV:

- `dataset`: dataset name, including source prefix for non-baseline runs;
- `sample_index`: selected sample index as described above;
- `reason_method`: measured PyXAI method;
- `status`: `ok`, `no_reason`, or `error`;
- `prediction`: RF prediction for the sample;
- `reason_length`: explanation length, when available;
- `set_instance_seconds`: time spent in `Explainer.set_instance(...)`;
- `explanation_seconds`: time spent in the PyXAI explanation method;
- `total_seconds`: end-to-end per-sample time, excluding model import;
- `classifier_path`, `dataset_path`, `samples_path`: populated for baseline
  file-backed runs when applicable;
- `error`: captured exception when the sample fails.

Main columns in the aggregate CSV:

- `n_samples`, `n_ok`, `n_no_reason`, `n_error`;
- sum, mean, median, minimum, maximum, and p95 of `total_seconds`;
- the same statistics for `explanation_seconds`.

Aggregates use the latest available row for each `(dataset, sample_index)` and
method, which is useful when resuming a run.

## Resume

Resume is enabled by default: if a row for `(dataset, sample_index,
reason_method)` already exists in the CSV, the runner skips it.

Useful options:

- `--no-resume`: recompute and append new rows;
- `--retry-errors`: with resume enabled, retry only samples that previously had
  `status=error`;
- `--overwrite`: delete previous PyXAI CSV files before starting.

## PyXAI References

- Repository: https://github.com/crillab/pyxai
- Importing scikit-learn models: https://www.cril.univ-artois.fr/pyxai/documentation/importing/
- RF `majoritary_reason` explanations: https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/majoritary/
- RF `sufficient_reason` explanations: https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/sufficient/
