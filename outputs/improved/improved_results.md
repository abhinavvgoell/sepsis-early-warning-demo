# Improved LightGBM experiment

## What changed

The new candidate uses LightGBM with 200 causal features: current measurements, last-observation-carried-forward values, elapsed hours since the last measurement, deltas, rolling summaries, cumulative extremes, and a few clinical-rule-inspired composites. Feature construction uses the current row and earlier rows from the same patient. The selected configuration is `num_leaves=15`, `min_child_samples=600`, `feature_fraction=0.5`, `bagging_fraction=0.7`, `lambda_l2=10`, `scale_pos_weight=10`, and 233 boosting rounds.

Hyperparameters were searched using the training/validation split. The selected setting was then evaluated with five patient-grouped folds on the development patients (training plus validation): mean PR-AUC **0.1209** (standard deviation **0.0072**) and mean ROC-AUC **0.8453**. This is useful evidence of repeatability, but it is not nested cross-validation because the hyperparameters were selected before the fold comparison.

## Alert policy and reported test comparison

OOF predictions were used to fit an isotonic score mapping and choose alert thresholds. The demo uses the **1-of-5** policy selected under an out-of-fold false-alert cap. A more utility-oriented **1-of-2** policy was also evaluated; it raises more alerts and therefore produces more false alarms.

| Test-split metric | Retrained original HGB | LightGBM, 1-of-5 |
| --- | ---: | ---: |
| Normalized challenge utility | 0.4231 | **0.4437** |
| Raw PR-AUC | 0.1219 | **0.1478** |
| ROC-AUC | 0.8411 | **0.8544** |
| Sepsis warning-window coverage | 76.1% | **81.4%** |
| Non-sepsis patients with any alert | 23.8% | 24.9% |

The test false-alert rates are close, but not exactly matched. The paired patient bootstrap estimated a mean utility difference of **+0.0206**, with a 95% interval of **−0.0035 to +0.0446**. Since that interval includes zero, this result does not establish a reliable utility improvement. The PR-AUC comparison was more consistently favorable in the bootstrap, but the held-out split has already been examined in earlier project work.

The 1-of-2 utility-optimal rule had test utility **0.4478**, coverage **85.0%**, and patient false-alert rate **33.7%**. It catches more warning windows at the cost of substantially more non-sepsis patients receiving an alert, so it is not the default demo rule.

## Evaluation caveats

- The held-out test split was used in earlier experiments. All reported test comparisons are exploratory, not an untouched final estimate.
- The baseline was retrained using training patients only; the new model was fitted using training plus validation patients. The test comparison is therefore not perfectly matched for training-set size.
- The isotonic calibrator is fitted on pooled OOF predictions, and the reported OOF calibration score is measured on those same predictions. That calibration score is optimistic. Threshold and policy selection also use OOF outcomes.
- The ZIP's results JSON does not contain completed cross-hospital results. The accompanying progress note reported one direction and said the reverse direction was still running; no completed cross-hospital claim is made here.
- The label is the Challenge 2019 hourly label, which begins six hours before recorded onset and stays positive afterward. This is not an independent clinical diagnosis. No score or threshold is suitable for care decisions.

## Artifacts and reproducibility

### Shared-fold comparison with the previous HGB candidate

Both pipelines were evaluated on identical three-fold patient splits from a reproducible, stratified 10,000-patient sample of the original training partition; the validation and test partitions were excluded. Mean hourly PR-AUC was 0.0896 ± 0.0039 for HGB and 0.0923 ± 0.0053 for LightGBM; mean ROC-AUC was 0.8133 ± 0.0020 and 0.8235 ± 0.0079. This is a modest signal favoring the LightGBM pipeline in this sample, not proof of better alert utility or clinical performance. The feature sets differ, so this compares the full pipelines rather than the classifiers in isolation. Threshold-dependent metrics were omitted because thresholds came from different experiments. Fold details are in `../head_to_head/`.

- `improved_model.joblib`: LightGBM model text, a version-neutral isotonic interpolation map, feature schema, and selected 1-of-5 policy. The interpolation map was checked against the original calibrator with zero difference over 10,001 sampled scores.
- `improved_results.json`: metrics and bootstrap summary.
- `feature_importance.csv`: model gain summaries (descriptive, not causal explanations).
- `hyperparameter_search.csv`: search results.
- `src/improved_pipeline.py`: feature and training/evaluation pipeline.

Install `requirements.txt`, download the PhysioNet files into `data/raw`, then run:

```bash
python -m src.improved_pipeline --data-dir data/raw --output-dir outputs/improved --cache-dir outputs/improved/cache
```

The project demo loads the LightGBM bundle by default. Earlier HGB model bundles are preserved and can be selected with `python -m src.app --bundle <bundle-path>`.
