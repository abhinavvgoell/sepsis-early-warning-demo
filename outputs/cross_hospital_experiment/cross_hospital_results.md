# Cross-hospital evaluation

This first cross-hospital check evaluates the project's original class-weighted HistGradientBoosting baseline, not the current LightGBM demo bundle. The cutoff for each direction was selected using a held-out 15% of source-hospital patients. The destination hospital was not used to choose the cutoff. Both directions use the same 1-of-5 rule.

| Train → evaluate | Cutoff | PR-AUC (hourly) | ROC-AUC (hourly) | Accuracy (hourly) | Sensitivity (hourly) | Precision (hourly) | Sepsis warning-window coverage | Non-sepsis patients with alert | Challenge utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| training_setA_to_training_setB | 0.600 | 0.0596 | 0.7252 | 89.3% | 44.8% | 6.0% | 54.5% | 18.7% | 0.2558 |
| training_setB_to_training_setA | 0.675 | 0.0810 | 0.7484 | 74.8% | 60.2% | 5.1% | 76.6% | 56.4% | 0.2827 |

## Interpretation

A large drop in warning-window coverage, PR-AUC, or utility across a direction suggests the model does not transfer reliably between these hospital cohorts. A high hourly accuracy alone is not sufficient because negative hourly rows dominate.

This is retrospective research on the PhysioNet challenge cohorts. It does not show clinical readiness or justify use for patient care.

The difference between A→B and B→A is substantial, especially in the fraction of non-sepsis patients receiving an alert. This suggests the threshold and model behavior are sensitive to which cohort they are trained on. The current LightGBM demo model still needs this same source-only training and destination-only evaluation before we can make a cross-hospital statement about it.

Reproduce from the project directory with: `python -m src.cross_hospital_experiment`
