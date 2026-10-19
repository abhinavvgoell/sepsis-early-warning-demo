# Initial Held-Out Results

This is a research prototype on the PhysioNet/CinC Challenge 2019 training data. It is not validated for clinical use.

The patient-level split contains 28,235 training, 6,050 validation, and 6,051 test patients. Thresholds were selected on validation patients, not on test labels. The held-out test split has been examined across multiple exploratory comparisons, so these figures should not be treated as a final untouched test estimate.

## Held-out model comparison

| Model | Threshold | Accuracy | PR-AUC | ROC-AUC | Sensitivity | Precision | Normalized utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_prevalence_baseline | 0.50 | 0.982 | 0.018 | 0.500 | 0.000 | 0.000 | 0.000 |
| logistic_regression | 0.55 | 0.827 | 0.085 | 0.785 | 0.601 | 0.061 | 0.340 |
| hist_gradient_boosting | 0.55 | 0.831 | 0.123 | 0.831 | 0.658 | 0.068 | 0.402 |
| logistic_regression_accuracy | 0.70 | 0.982 | 0.088 | 0.781 | 0.000 | 0.000 | -0.000 |
| hist_gradient_boosting_accuracy | 0.95 | 0.982 | 0.118 | 0.825 | 0.000 | 0.000 | 0.000 |

Accuracy-named variants were trained without class weighting and selected a threshold to maximize validation accuracy. The other classifiers use class weighting and a threshold selected for normalized challenge utility.

The accuracy-focused variants reach roughly the majority-class baseline accuracy but have zero sensitivity on the test split: their selected cutoffs generate essentially no positive alerts. They do not improve sepsis detection. The baseline's high accuracy comes from the rarity of positive hourly labels.

## Patient-level alert review

| Model | Sepsis patients alerted in utility window | Window coverage | Non-sepsis patients with any alert | False-alert rate among non-sepsis patients | Median lead (hours, timely alerts) |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 383 | 0.870 | 2,554 | 0.455 | 12.0 |
| hist_gradient_boosting | 386 | 0.877 | 2,379 | 0.424 | 12.0 |
| logistic_regression_accuracy | 0 | 0.000 | 1 | 0.000 | n/a |
| hist_gradient_boosting_accuracy | 0 | 0.000 | 0 | 0.000 | n/a |

## Interpretation and limits

Compare the accuracy column with sensitivity, PR-AUC, normalized utility, and the patient-level false-alert rate. Since positive hourly labels are rare, an accuracy-oriented model can score highly simply by predicting few or no alerts. The patient-level table reports whether a non-sepsis patient received any alert and whether a sepsis patient received an alert during the challenge's beneficial time window.

The dataset contains historical ICU stays from the challenge's source hospitals; these results do not establish external validity, prospective benefit, calibration, or clinical safety. Thresholds were selected on validation data and must not be interpreted as medical action cutoffs. Further model selection should use a new untouched test set or external dataset for a final unbiased estimate.
