# Clean, Fair Evaluation: Frozen LightGBM vs. HGB Baseline

## Evaluation Setup and Untouched Data Status

This evaluation addresses the primary methodological limitation of previous iterations: repeated examination of the original test partition and mismatched training sets.

- **Untouched Data Status**: The public PhysioNet/CinC 2019 Challenge dataset contains 40,336 patients in sets A and B; challenge test set C remains private. There is **no genuinely untouched patient data** in the public release. All results are therefore marked as **exploratory research prototype findings**, not clinical validation.
- **Fair Split Protocol**: To ensure rigorous evaluation within the available data, the 6,051 patients from the previously examined held-out test split were **strictly excluded**. From the remaining 34,285 patients, a new patient-level split was established with fixed seed `EVAL_SEED=2026`:
  - **Training Split**: 23,999 patients
  - **Calibration & Threshold Tuning Split**: 5,143 patients (strictly separate)
  - **Final Held-Out Evaluation Split**: 5,143 patients (evaluated exactly once)
- **Matched Training Conditions**: Both LightGBM and the Histogram Gradient Boosting (HGB) baseline were trained on the **identical training cohort** with the **same 200 causal features**.
- **Strict Leakage Prevention**: Isotonic probability calibration and 1-of-5 alert threshold selection were fitted strictly on the separate calibration split and applied without adjustment to the held-out split.

## Disclosure on Hyperparameter Search (Nested vs. Non-Nested)

> [!IMPORTANT]
> **The hyperparameter search was NOT nested.** The LightGBM configuration (`num_leaves=15`, `min_child_samples=600`, `feature_fraction=0.5`, `bagging_fraction=0.7`, `lambda_l2=10.0`, `scale_pos_weight=10.0`, 233 rounds) was searched globally on the development cohort prior to this evaluation. Consequently, performance estimates reflect an upper-bound configuration selection and cannot be claimed as an unbiased generalization guarantee. True prospective validation requires testing on an entirely independent external clinical cohort.

## One-Time Held-Out Evaluation Results

| Evaluation Metric | Baseline HGB | Frozen LightGBM | Difference (LightGBM − HGB) | 95% Bootstrap CI (Diff) |
| :--- | ---: | ---: | ---: | ---: |
| **Normalized Utility** | 0.4021 | **0.4171** | +0.0146 | [-0.0063, +0.0370] |
| **PR-AUC (raw scores)** | 0.1039 | **0.1092** | +0.0052 | [-0.0043, +0.0133] |
| **ROC-AUC (raw scores)** | 0.8313 | **0.8353** | — | — |
| **Warning-Window Coverage** | 74.1% | **75.1%** | — | — |
| **Non-Sepsis Alert Rate (FAR)** | 22.2% | **22.0%** | — | — |
| **Hourly Sensitivity** | 60.4% | **61.3%** | — | — |
| **Hourly Specificity** | 86.8% | **87.1%** | — | — |

### Bootstrap Confidence Intervals (Patient-Level 95% CI)

| Model | Metric | Mean | 95% Confidence Interval |
| :--- | :--- | ---: | :---: |
| Baseline HGB | utility | 0.4026 | [0.3598, 0.4438] |
| Baseline HGB | pr_auc | 0.1055 | [0.0875, 0.1250] |
| Baseline HGB | roc_auc | 0.8323 | [0.8157, 0.8486] |
| Baseline HGB | coverage | 0.7413 | [0.6994, 0.7877] |
| Baseline HGB | far | 0.2222 | [0.2107, 0.2343] |
| Baseline HGB | sensitivity | 0.6048 | [0.5619, 0.6474] |
| Baseline HGB | specificity | 0.8677 | [0.8573, 0.8788] |
| Frozen LightGBM | utility | 0.4172 | [0.3749, 0.4616] |
| Frozen LightGBM | pr_auc | 0.1107 | [0.0923, 0.1280] |
| Frozen LightGBM | roc_auc | 0.8360 | [0.8187, 0.8510] |
| Frozen LightGBM | coverage | 0.7519 | [0.7075, 0.8000] |
| Frozen LightGBM | far | 0.2204 | [0.2088, 0.2326] |
| Frozen LightGBM | sensitivity | 0.6142 | [0.5704, 0.6607] |
| Frozen LightGBM | specificity | 0.8712 | [0.8611, 0.8826] |

## Statistical Findings

- **Paired Utility Difference**: Mean difference is `+0.0146` with 95% bootstrap interval `[-0.0063, +0.0370]`. Probability of superiority $P(\Delta > 0) = 91.6\%$.
- **Paired PR-AUC Difference**: Mean difference is `+0.0052` with 95% bootstrap interval `[-0.0043, +0.0133]`. Probability of superiority $P(\Delta > 0) = 89.5\%$.

## Research Prototype Statement

This project is a research prototype developed for educational and experimental analysis of time-series ICU data. The models, scores, and alert policies have not been clinically validated, do not produce medical diagnostic probabilities, and must not be used for patient management or care decisions.