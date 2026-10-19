# Next-six-hour sepsis onset experiment

This isolated candidate predicts the dataset-inferred onset within the next six hours using only pre-onset records. It excludes post-onset rows and cases where onset cannot be observed before the record ends. It does not diagnose sepsis or establish clinical usefulness.

- Retained observed sepsis events: 2,917
- Excluded censored/unobserved septic records: 15
- Patient split: 28,224 train / 6,048 validation / 6,049 test
- Validation false-alert cap: 25%

| Candidate on same held-out patients | Row PR-AUC | Row ROC-AUC | Sensitivity | Precision | Sepsis warning-window coverage | Non-sepsis patient alert rate |
|---|---:|---:|---:|---:|---:|---:|
| Future-onset model | 0.083 | 0.841 | 58.4% | 5.2% | 69.2% | 24.1% |
| Current label model | 0.127 | 0.876 | 67.4% | 5.5% | 77.6% | 22.5% |

The test split and source data were examined in prior project iterations, so results are exploratory and not independent final estimates. A public-dataset comparison is not clinical validation. The current demo model was not overwritten.
