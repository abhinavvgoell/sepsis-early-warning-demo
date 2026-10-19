# LightGBM cross-hospital evaluation

This tests the current LightGBM pipeline configuration, retrained separately within each source hospital. For each direction, 85% of source patients were used for fitting; the remaining 15% selected the 1-of-5 cutoff by challenge utility. The destination cohort was held out from both fitting and threshold selection.

| Train → evaluate | Source-selected cutoff | Source validation false-alert rate | Destination warning-window coverage | Destination non-sepsis patients alerted | Destination utility | Destination hourly sensitivity | Destination hourly precision | Destination hourly accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| training_setA_to_training_setB | 0.244006 | 30.3% | 52.9% | 15.0% | 0.2638 | 42.8% | 6.6% | 90.6% |
| training_setB_to_training_setA | 0.185122 | 31.2% | 84.7% | 61.5% | 0.2979 | 69.0% | 4.8% | 69.4% |

## Interpretation

Compare destination warning-window coverage and non-sepsis patient alert rate with the source validation values. A sizeable difference suggests the model or its threshold does not transfer cleanly. Hourly accuracy is included for completeness but can be misleading because negative hourly rows dominate.

The saved demo model was not modified. This retrospective evaluation on the two PhysioNet hospital groups does not establish clinical readiness.

Reproduce with: `python -m src.cross_hospital_lightgbm`
