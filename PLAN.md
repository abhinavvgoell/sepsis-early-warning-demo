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

- **Complete:** project scaffold, public data download, cohort/label audit notebook, causal rolling and trend features, patient-level splits, baseline and alternative model comparisons, validation threshold selection, the 1–8 hour k-of-n alert grid, train-versus-validation fit review, final project report, and a local-only web form with an HTTP prediction backend.
- **Current demo candidate:** LightGBM with 200 causal features, an isotonic score mapping, and a 1-of-5 alert policy selected from out-of-fold predictions. Five patient-grouped folds averaged PR-AUC 0.121 (SD 0.007). The exploratory test comparison showed higher utility and warning-window coverage than the retrained original HGB, but its paired utility interval includes zero; the baseline also used fewer training patients.
- **Preserved fallbacks:** the trend and last-observation HGB candidates remain available with `python -m src.app --bundle <bundle-path>`; previous artifacts were not overwritten.
- **App integration:** `src.app` supports both the LightGBM text bundle and the previous scikit-learn bundles. LightGBM is included in `requirements.txt`.
- **Next evaluation:** freeze the candidate and alert policy before evaluation on a genuinely untouched patient cohort or external dataset. The OOF calibration metric is optimistic, the test split has been examined before, and the cross-hospital run has no complete saved results. The current app is a demonstration only.
