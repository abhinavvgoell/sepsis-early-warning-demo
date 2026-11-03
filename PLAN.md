# Project Plan: Early Sepsis Warning

## Objective

Build a reproducible research prototype that estimates the PhysioNet 2019 hourly sepsis label from current and past ICU measurements. Compare a simple linear baseline with a stronger tree-based model, enforce patient-level separation, and report the official patient-level normalized utility alongside conventional classification metrics.

## Milestones

| Week | Work | Deliverable |
| --- | --- | --- |
| 1 | Acquire public data, audit cohort/labels/missingness, document the challenge label timing | Data notes and exploratory notebook |
| 2 | Implement causal history features; create stratified patient train/validation/test splits | Reproducible feature and split pipeline |
| 3 | Train baseline and boosted tree; choose alert threshold on validation utility | Model comparison and held-out metrics |
| 4 | Review patient-level alerts, inspect false positives/negatives, document limitations and reproducibility | Final notebook, report, and run instructions |

## Evaluation commitments

- Split by patient so no hourly record from a patient appears in more than one partition.
- Build features from current and prior rows only; never use future measurements.
- Tune alert thresholds on validation patients; use held-out patients for exploratory comparison and reserve a fresh test set for the final unbiased estimate if the project adds more model selection.
- Report precision-recall, AUROC, sensitivity, specificity, F1, and PhysioNet normalized utility. Do not use accuracy as the headline metric for the imbalanced task.
- Label scores as research findings on a historical dataset, not evidence of clinical readiness.

## Current scope

The project compares Logistic Regression, Histogram Gradient Boosting, a regularized Random Forest, probability blends, causal vital-trend and last-observation features, LightGBM with richer causal features, and k-of-n alert rules. It includes patient-level error/alert summaries, internal grouped folds, and bootstrap comparison. The current candidate remains a research model, not a clinically validated system.

## Progress snapshot

- **Complete:** project scaffold, public data downloader, cohort/label audit notebook, causal rolling and trend features, patient-level splits, baseline and alternative model comparisons, validation threshold selection, the 1–8 hour k-of-n alert grid, train-versus-validation fit review, final project report, and a local-only web form with an HTTP prediction backend.
- **Clean Fair Evaluation (Part 1):** Implemented in `src/final_evaluation.py`. The candidate LightGBM configuration (15 leaves, min child 600, feature fraction 0.5, bagging 0.7, L2 10.0, pos weight 10.0, 233 rounds) and 200 features are frozen. The previously examined 6,051 test patients are strictly excluded. LightGBM and the HGB baseline are trained on the identical training partition with the same features. Calibration and 1-of-5 threshold selection are performed on a strictly separate calibration partition. Reports patient-level bootstrap 95% CIs and paired bootstrap comparison.
- **Alert Noise Reduction (Part 2):** Implemented in `src/alert_policy_v2.py`. Compares baseline 1-of-5 against rising-score (3h mean), EWMA, and 6h refractory cooldown rules tuned on OOF predictions under 10%, 15%, and 20% non-sepsis false-alert rate caps. Reports utility, coverage, FAR, and alerts per patient-day.
- **Quality & Testing:** Unit test suite expanded in `tests/test_sepsis_pipeline.py` with 27 automated tests covering patient split separation, feature causality (no future row leakage), and hand-calculated alert rule correctness.
- **Exploratory Status & Limitations:** The public PhysioNet 2019 dataset does not contain untouched patient data (sets A & B were used across earlier development; set C is private). All evaluations remain exploratory research findings with non-nested hyperparameter selection disclosed. No clinical validity is claimed.
