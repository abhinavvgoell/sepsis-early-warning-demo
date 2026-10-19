# Model and Generalization Check

This comparison used the existing patient-level training and validation split only. The previously explored held-out test split was not used. Every candidate used the same **1-of-5** alert rule, and its threshold was selected on validation data to maximize challenge utility subject to a non-sepsis patient alert rate no higher than the original boosted-tree model's validation rate (26.3%). Training metrics are in-sample and will usually look better than validation metrics.

## Validation comparison and train–validation gaps

| Candidate | Train PR-AUC | Validation PR-AUC | Train–validation PR-AUC gap | Validation ROC-AUC | Validation utility | Validation non-sepsis patient alert rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Regularized trend HistGradientBoosting** | 0.1441 | 0.0962 | **0.0480** | 0.8143 | **0.3920** | 26.2% |
| 50/50 blend: original + trend HistGradientBoosting | 0.1575 | **0.0976** | 0.0599 | **0.8153** | 0.3891 | 25.2% |
| 75% trend / 25% original blend | 0.1527 | 0.0972 | 0.0555 | 0.8152 | 0.3898 | 22.0% |
| Original HistGradientBoosting | 0.1603 | 0.0964 | 0.0638 | 0.8129 | 0.3844 | 26.3% |
| 25% trend / 75% original blend | 0.1600 | 0.0975 | 0.0625 | 0.8146 | 0.3841 | 25.4% |
| Regularized Random Forest | 0.1515 | 0.0877 | 0.0638 | 0.8043 | 0.3459 | 24.7% |
| Logistic Regression | 0.0785 | 0.0751 | 0.0034 | 0.7626 | 0.3238 | 24.8% |

## What this says about fit

- The boosted trees show a noticeable train-to-validation gap (about 0.05–0.06 PR-AUC). This is a sign of some overfitting, though training metrics are expected to be optimistic because they are measured on data used to fit the model.
- The regularized trend model has the smallest PR-AUC gap among the stronger candidates and the best validation utility. This suggests that the added features plus stronger regularization improved generalization relative to the original tree model.
- The Random Forest has a large ROC-AUC gap (0.0775) and lower validation PR-AUC and utility. It overfits more in this setup and is not a good replacement.
- Logistic Regression has a small gap, but its validation PR-AUC and utility are much lower. That points to a model that is too simple for this dataset compared with the tree models; it is not the better fit just because its gap is small.
- The 50/50 blend has the best validation PR-AUC, but utility is slightly below the regularized trend model. Use the trend model if utility is the target; consider the blend only if ranking positives higher is the priority and its benefit is confirmed on fresh patients.

## Recommendation

Keep the **regularized trend HistGradientBoosting** as the leading validation candidate. The 50/50 blend is a reasonable challenger. Do not choose a winner from the previously explored test set; a final claim needs new untouched patients or external data. No new test metrics were calculated in this comparison.

## Reproduce

```bash
python -m src.model_generalization_experiment --data-dir data/raw --output-dir outputs/generalization_experiment --bundle outputs/trained_models.joblib --trend-bundle outputs/trend_experiment/selected_trend_model.joblib
```

Detailed values are in [generalization_comparison.csv](generalization_comparison.csv) and [generalization_results.json](generalization_results.json). The experiment code is [model_generalization_experiment.py](../../src/model_generalization_experiment.py).
