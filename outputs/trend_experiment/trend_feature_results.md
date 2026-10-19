# Causal Vital-Trend Feature Experiment

## What changed

I added three causal features for each of eight vital signs (24 features total): the current value minus the previous hourly value, the current value minus the prior six-hour mean, and the standard deviation across prior six-hour readings. The current hour can be used for the current-hour prediction; all rolling history features exclude future readings. The original trained model and bundle were left untouched.

I trained three HistGradientBoosting settings with the expanded features. Each model used the same patient split and the same **1-of-5** alert policy. The classification threshold was chosen on validation patients to maximize normalized challenge utility while keeping the non-sepsis patient alert rate no higher than the existing model's validation-selected rate.

## Validation results

| Candidate | Cutoff | Utility | Non-sepsis patient alert rate | Raw-score PR-AUC | Warning-window coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing model | 0.65 | 0.384 | 26.3% | 0.0964 | 76.1% |
| Trend features, default settings | 0.70 | 0.373 | 19.7% | 0.0936 | 67.7% |
| **Trend features, regularized settings (selected)** | **0.65** | **0.392** | **26.2%** | 0.0962 | **76.4%** |
| Trend features, higher capacity | 0.65 | 0.391 | 25.5% | 0.0944 | 74.8% |

The regularized trend model won on the validation objective, with almost the same false-alert rate as the existing model. It was selected without looking at test metrics.

## Exploratory held-out comparison

| Metric | Existing model | Selected trend model |
| --- | ---: | ---: |
| Raw-score PR-AUC | 0.1234 | **0.1361** |
| Raw-score ROC-AUC | 0.8308 | **0.8341** |
| Accuracy | 84.33% | **84.39%** |
| Hourly sensitivity | **66.02%** | 65.55% |
| Sepsis warning-window coverage | 78.41% | 78.18% |
| Non-sepsis patients with any alert | **25.98% (1,458/5,611)** | 26.39% (1,481/5,611) |
| False-positive hours | 35,291 | **35,121** |
| Normalized challenge utility | **0.427** | 0.421 |

This is a mixed result: the trend model ranked positive hours better by PR-AUC and ROC-AUC and produced slightly fewer false-positive hours, but it had marginally lower sepsis coverage and utility and alerted slightly more non-sepsis patients. The test split has already been examined in earlier project work, so these figures are exploratory rather than a final independent estimate. The evidence does **not** justify replacing the existing model for the project's headline result. The trend model is a promising candidate for a fresh evaluation.

## Files and reproduction

- Selected candidate bundle: `selected_trend_model.joblib`
- Validation candidate table: `trend_feature_validation.csv`
- Exploratory held-out comparison: `trend_feature_heldout_comparison.csv`
- Full experiment record: `trend_feature_results.json`
- Feature-generation change: `src/run_pipeline.py` (`make_features(..., include_trends=True)`)
- Experiment script: `src/trend_feature_experiment.py`

Run from the project directory:

```bash
python -m src.trend_feature_experiment --data-dir data/raw --output-dir outputs/trend_experiment --bundle outputs/trained_models.joblib
```

The saved candidate bundle uses the expanded feature schema and the validation-selected threshold. It is for research and demonstration only, not clinical decisions.
