# Time Series Experiment Runner Documentation

## Overview

This document describes the integration of a new experiment runner for **univariate time series datasets** based on the `aeon` library. The goal is to provide full compatibility with the existing tabular experiment pipeline (`run_experiments_baseline.py`) while enabling experiments on time series data.

The new script:

```
run_experiments_baseline_timeseries.py
```

extends the baseline framework to support:
- Aeon/UCR-style univariate time series datasets
- Flattened feature representation for compatibility with Random Forest explainers
- Seamless reuse of XRF and INFXP explainability backends

---

## Design Principles

The implementation follows these principles:

1. **Compatibility-first**
   - Reuses the same experiment structure, outputs, and explainers as the tabular runner

2. **Minimal transformation**
   - Time series are reshaped, not reinterpreted
   - Each time step becomes a feature

3. **Explainability preservation**
   - Explanation methods (AXp, INFXP) operate on transformed features
   - Interval explanations remain valid and interpretable

---

## Data Handling

### Dataset Source

Datasets are loaded using:

```python
from aeon.datasets import load_classification
```

Supported datasets include standard UCR/UEA univariate benchmarks (e.g., `ECG200`, `Coffee`, `GunPoint`).

### Data Shape Transformation

Original format:

```
(n_samples, n_channels, n_timepoints)
```

Transformed format:

```
(n_samples, n_timepoints)
```

Transformation:

```python
X_flat = X.reshape(n_samples, -1)
```

### Feature Naming

Each time step is mapped to a feature:

```
t_000, t_001, ..., t_N
```

This ensures:
- Deterministic ordering
- Compatibility with explanation frameworks

---

## Model Training

Random Forest models are trained using:

```python
sklearn.ensemble.RandomForestClassifier
```

Supported parameters:
- `n_estimators`
- `max_depth`
- Additional sklearn RF parameters (consistent with tabular runner)

Optional features:
- Custom train/test split
- Sample percentage filtering

---

## Explainability Backends

The runner supports:

- **XRF (baseline)**
- **INFXP (interval explanations)**

### Backend Selection

```bash
--explainer xrf
--explainer infxp
--explainer all
```

### Explanation Outputs

Each explanation includes:

- Feature indices
- Interval-based explanations (if available)
- Coverage metrics (INFXP)

Example structure:

```json
{
  "feature_indices": [1, 5, 10],
  "interval_explanation": [...],
  "infxp_coverage": 0.82,
  "axp_domain_coverage": 0.76
}
```

---

## Coverage Metrics

The following metrics are computed when available:

- `infxp_coverage`
- `axp_domain_coverage`

These are:
- Stored per explanation
- Aggregated across samples

Aggregated statistics include:

- Average coverage
- Minimum/maximum coverage

---

## Experiment Pipeline

The pipeline mirrors the tabular version:

1. Load dataset
2. Flatten time series
3. Train Random Forest
4. Generate explanations
5. Compute metrics
6. Save results

---

## Output Format

Results are saved as JSON:

```
baseline/resources/experiments/<dataset_name>_results.json
```

Structure:

```json
{
  "dataset": "ECG200",
  "explainer": "infxp",
  "experiments": [
    {
      "n_estimators": 100,
      "max_depth": 6,
      "train_accuracy": 0.95,
      "test_accuracy": 0.90,
      "explanations": {
        "avg_explanation_length": 4.2,
        "avg_infxp_coverage": 0.78
      }
    }
  ]
}
```

---

## Command Line Usage

### Basic Example

```bash
python run_experiments_baseline_timeseries.py ECG200 \
    --n-estimators 100 \
    --max-depth 6
```

### Multiple Configurations

```bash
python run_experiments_baseline_timeseries.py Coffee \
    --n-estimators 50,100 \
    --max-depth 4,6
```

### Using INFXP

```bash
python run_experiments_baseline_timeseries.py ECG200 \
    --explainer infxp
```

### Sample Subset

```bash
python run_experiments_baseline_timeseries.py ECG200 \
    --sample-percentage 10
```

---

## Differences from Tabular Runner

| Aspect | Tabular | Time Series |
|------|--------|------------|
| Input format | CSV | Aeon dataset |
| Feature space | Original | Flattened time steps |
| Feature semantics | Domain features | Temporal positions |
| Preprocessing | Dataset-dependent | Reshape only |

---

## Limitations

- Only **univariate** time series supported
- Flattening removes explicit temporal structure
- Feature independence assumption may not hold

---

## Future Extensions

Potential improvements:

- Multivariate time series support
- Shapelet-based feature extraction
- Temporal-aware explainability
- Native sequence models integration

---

## Conclusion

This extension enables:

- Direct comparison between tabular and time series experiments
- Reuse of explainability pipelines on temporal data
- Evaluation of coverage-based explanation metrics on time series

while maintaining full compatibility with the existing experimental framework.

