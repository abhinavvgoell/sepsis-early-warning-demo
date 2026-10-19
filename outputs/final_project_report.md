# Early Sepsis Warning from ICU Time Series

## Summary

This educational research prototype estimates whether an hourly ICU record falls in the PhysioNet/CinC Challenge 2019 sepsis-label window. It is not a clinical diagnosis, a care recommendation, or a medical probability.

## Dataset and target

The project uses 40,336 patient records from the public [PhysioNet/CinC Challenge 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/). Patient-level splits contain 28,235 training, 6,050 validation, and 6,051 held-out patients. The challenge label starts six hours before recorded onset and remains positive after onset. The model predicts this dataset label, not an independent clinical diagnosis.

## Model

The current demo candidate is class-weighted **LightGBM** with 200 causal features. Features include current measurements, last-observation-carried-forward values, hours since measurements, changes, rolling summaries, cumulative extremes, and a small set of rule-inspired composites. The selected settings are 15 leaves, minimum child size 600, feature fraction 0.5, bagging fraction 0.7, L2 regularization 10, positive-class weight 10, and 233 boosting rounds. An isotonic score mapping is applied. The default alert rule is 1-of-5: a rule is active if at least one of the last five hourly scores meets the selected cutoff.

## Results and interpretation

Five patient-grouped folds on the development patients gave mean PR-AUC **0.1209 ± 0.0072** and mean ROC-AUC **0.8453**. Hyperparameters were chosen using the training/validation split before this fold comparison, so this is not nested cross-validation.

The exploratory test comparison was:

| Metric | Retrained original HGB | LightGBM, 1-of-5 |
| --- | ---: | ---: |
| Normalized challenge utility | 0.423 | **0.444** |
| Raw PR-AUC | 0.122 | **0.148** |
| ROC-AUC | 0.841 | **0.854** |
| Warning-window coverage | 76.1% | **81.4%** |
| Non-sepsis patients alerted | **23.8%** | 24.9% |

The paired bootstrap's utility difference was +0.0206, with a 95% interval from −0.0035 to +0.0446. Since the interval includes zero, the results do not establish that utility improved reliably. The false-alert rates are similar, but not exactly matched. The utility-optimal 1-of-2 policy reached utility 0.448 and coverage 85.0%, while raising the patient false-alert rate to 33.7%; the demo therefore uses 1-of-5.

## Explicit next-six-hour target experiment

An isolated candidate was trained for a more direct question: among records before an observed onset, does onset occur within the next six hours? It uses only the six pre-onset hourly rows as positive examples and excludes post-onset rows and 15 septic records whose onset could not be observed before the record ended. On the same exploratory patient split and with each threshold selected on validation under a 25% non-sepsis-patient alert cap, the future-onset candidate reached row PR-AUC **0.083**, sensitivity **58.4%**, precision **5.2%**, warning-window coverage **69.2%**, and a **24.1%** non-sepsis-patient alert rate. The existing model scored **0.127**, **67.4%**, **5.5%**, **77.6%**, and **22.5%** on those same metrics. The candidate was not promoted to the demo because it performed worse on this comparison. The accuracy values (87.4% candidate, 86.3% current model) look high because negative hourly records dominate; they do not outweigh the low precision. Since the source data and patient split were examined earlier, these remain exploratory results, not independent validation. See [future_onset_results.md](future_onset_experiment/future_onset_results.md).

These results are exploratory. This held-out split was examined in earlier project iterations, and the baseline was trained on fewer patients than the improved model. No unbiased final test or clinical-performance claim is made. The calibration mapping and policy were selected using pooled out-of-fold predictions; the reported calibration score on those same predictions is optimistic.

### Shared-fold model check

To make a more comparable check, both the last-observation HGB pipeline and LightGBM v2 were evaluated on the same three patient-grouped folds drawn from a reproducible, stratified 10,000-patient sample of the original training partition. The validation and test partitions were not included. Mean hourly row-level PR-AUC was **0.0896 ± 0.0039** for HGB and **0.0923 ± 0.0053** for LightGBM; mean ROC-AUC was **0.8133 ± 0.0020** and **0.8235 ± 0.0079**, respectively. This is a modest discrimination advantage for the LightGBM feature/model pipeline in this sample, not a guarantee of better patient alert utility. The two pipelines use different causal feature sets, so this compares complete pipelines rather than isolating the classifier alone. Alert utility and false-alert rates are omitted because thresholds were selected under different prior experiments. See [head_to_head_cv.json](head_to_head/head_to_head_cv.json) and [head_to_head_cv_folds.csv](head_to_head/head_to_head_cv_folds.csv).

### Cross-hospital transfer check

The original class-weighted HGB baseline was trained on one PhysioNet hospital group and evaluated on the other, with the alert threshold chosen only on a held-out 15% of the source hospital. Using the same 1-of-5 rule, A→B gave hourly PR-AUC **0.0596**, ROC-AUC **0.7252**, sepsis warning-window coverage **54.5%**, and alerts for **18.7%** of non-sepsis patients. B→A gave hourly PR-AUC **0.0810**, ROC-AUC **0.7484**, coverage **76.6%**, and alerts for **56.4%** of non-sepsis patients.

The same protocol was then run for the **LightGBM v2 pipeline** using the project's selected hyperparameters. A→B had **52.9%** warning-window coverage and alerted **15.0%** of destination non-sepsis patients, compared with **30.3%** non-sepsis alerts on source validation. B→A had **84.7%** warning-window coverage and alerted **61.5%** of destination non-sepsis patients, compared with **31.2%** on source validation. This direction-dependent difference shows that a threshold chosen on one cohort may not control alert burden on another. Both experiments are retrospective cross-hospital checks, not clinical validation. The saved demo model was not overwritten. Details: [HGB cross-hospital report](cross_hospital_experiment/cross_hospital_results.md) and [LightGBM cross-hospital report](cross_hospital_experiment/lightgbm_cross_hospital_results.md).

## Limitations and next evaluation

- The historical challenge label and data may not generalize to other hospitals, patient populations, or current clinical workflows.
- Scores and thresholds are demonstration settings, not medical action cutoffs.
- The cross-hospital checks use only the two historical PhysioNet groups; they do not establish transfer to a current or prospective hospital. The LightGBM model was retrained within each source group for this experiment, so its result is a pipeline transfer check rather than an evaluation of the saved demo bundle as deployed.
- A future clinical evaluation would require a frozen model and alert policy followed by independent, prospective site validation and workflow review.

## Demo and reproduction

Run `python -m src.app` and open `http://127.0.0.1:8766`. The local server processes submitted records in memory without saving them. Use synthetic or de-identified records only. The project uses Python, pandas, NumPy, scikit-learn, LightGBM, and joblib. See [README.md](../README.md) for setup and reproduction instructions. See [improved_results.md](improved/improved_results.md) for experiment details and artifact descriptions.
