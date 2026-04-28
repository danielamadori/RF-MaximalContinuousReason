# PyXAI Experiments on Random Forests

This file documents `run_pyxai_rf_experiments.py`, the runner used to measure
PyXAI explanation computation times on the Random Forests already stored in
this repository.

## What It Does

The runner:

1. reads complete datasets from `baseline/resources/datasets/<dataset>/`;
2. reads test samples from `<dataset>.samples`;
3. loads the converted Random Forest from
   `baseline/Classifiers-100-converted/<dataset>/*.json`;
4. imports the scikit-learn model into PyXAI with `Learning.import_models`;
5. for each selected sample, runs `Explainer.set_instance(...)` and the
   selected PyXAI explanation method;
6. writes both per-sample timings and dataset-level aggregate statistics.

By default, the runner uses `majoritary_reason`, because it is the most
practical RF-specific explanation method in PyXAI. The script also supports
`sufficient_reason`, `minimal_majoritary_reason`, `minimal_sufficient_reason`,
and `direct_reason`.

## Installation

The `pyxai` dependency has been added to `requirements.txt`.

```bash
pip install -r requirements.txt
```

If PyXAI is not installed, the script exits early with an explicit error
message.

## Main Commands

List datasets that have all required files: CSV, samples, and RF classifier.

```bash
python run_pyxai_rf_experiments.py --list-datasets
```

Run the full experiment on every available dataset:

```bash
python run_pyxai_rf_experiments.py
```

Smoke test on a few `iris` samples:

```bash
python run_pyxai_rf_experiments.py --datasets iris --max-samples 5 --overwrite
```

Run with a per-sample time limit:

```bash
python run_pyxai_rf_experiments.py --time-limit 60
```

Use another PyXAI method:

```bash
python run_pyxai_rf_experiments.py --reason-method sufficient_reason --time-limit 60 --overwrite
```

When changing the method or other important parameters, use `--overwrite` or a
different `--output-dir` so aggregate files do not mix different
configurations.

## Output

Default outputs are written to `results/pyxai_rf/`:

- `pyxai_rf_sample_times.csv`: one row per sample;
- `pyxai_rf_dataset_aggregates.csv`: aggregate statistics per dataset;
- `manifest_<run_id>.json`: run configuration.

Main columns in the per-sample CSV:

- `dataset`: dataset name;
- `sample_index`: index in the `.samples` file;
- `reason_method`: measured PyXAI method;
- `status`: `ok`, `no_reason`, or `error`;
- `prediction`: RF prediction for the sample;
- `reason_length`: explanation length, when available;
- `set_instance_seconds`: time spent in `Explainer.set_instance(...)`;
- `explanation_seconds`: time spent in the PyXAI explanation method;
- `total_seconds`: end-to-end per-sample time, excluding model import;
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
