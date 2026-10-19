# Last-observation feature experiment

## Question

Do causal features describing the most recent earlier measurement and its age improve the existing trend-feature model? This is a model-selection experiment using training and validation data only; the held-out test split was not loaded or used.

## Feature and model change

For each of 34 measured variables, the feature pipeline adds:

- the most recent non-missing value strictly before the current hour;
- the number of hours since that prior observation.

The current reading is not used as its own prior observation, and no future row is consulted. Together with the existing features, this gives the candidate 190 features. The selected model is class-weighted Histogram Gradient Boosting with learning rate 0.06, 200 iterations, 15 leaf nodes, minimum leaf size 100, and L2 regularization 4.0. The alert threshold is 0.65 with the existing 1-of-5 policy.

## Validation comparison

Threshold selection maximized validation challenge utility subject to the previous trend candidate's validation patient false-alert rate (26.24%).

| Validation metric | Previous trend candidate | Last-observation candidate |
| --- | ---: | ---: |
| Normalized challenge utility | 0.3920 | **0.4152** |
| PR-AUC | 0.0962 | **0.1002** |
| ROC-AUC | 0.8143 | **0.8342** |
| Non-sepsis patients with any alert | 26.24% | **22.34%** |
| Sepsis warning-window coverage | **76.36%** | 76.14% |

The validation result is encouraging but not sufficient alone to establish generalization, especially because the threshold was selected on this split.

## Three-fold patient-grouped comparison

Three folds were formed within the original training partition, stratified by whether a patient ever had a positive label. Both models used threshold 0.65 and the same 1-of-5 alert policy. The original validation and test partitions were excluded from fitting these fold models.

| Three-fold mean | Previous trend candidate | Last-observation candidate |
| --- | ---: | ---: |
| Normalized challenge utility | 0.3887 | **0.3944** |
| PR-AUC | 0.1065 | **0.1126** |
| ROC-AUC | 0.8161 | **0.8373** |
| Non-sepsis patients alerted | 25.39% | **21.25%** |
| Sepsis warning-window coverage | **75.00%** | 73.20% |

The candidate's utility was higher in two folds and lower in one. Its PR-AUC was slightly higher in all three folds. Mean warning coverage was about 1.8 percentage points lower, while mean patient false-alert rate was about 4.1 points lower. Fold-level outcomes are available in `last_observation_cv_folds.csv`; summary values are in `last_observation_cv_summary.csv`.

## Fit and limitations

The selected candidate's training PR-AUC was 0.1891 versus 0.1002 on validation, a gap of 0.0888. The former trend model's corresponding gap was about 0.048. The larger gap suggests overfitting remains possible even though the internal fold comparison is favorable on average. Three folds from one training partition are a limited internal check, not external validation. The dataset's target is the challenge's hourly label window, not a clinician-confirmed diagnosis. Scores are not calibrated probabilities; thresholds are not medical action cutoffs.

The new bundle is kept separately at `best_last_observation_candidate.joblib`. The old model and all earlier result artifacts remain unchanged, and the web demo can be rolled back with:

```bash
python -m src.app --bundle outputs/trend_experiment/selected_trend_model.joblib
```
