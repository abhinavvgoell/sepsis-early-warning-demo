# Full Alert Policy Grid (1–8 Hours)

I compared all **36 k-of-n rules** for windows of 1 through 8 hourly scores. A rule such as 2-of-3 alerts when at least 2 of the latest 3 scores exceed the cutoff. A rule such as 1-of-5 keeps the alert active if any of the latest 5 scores exceed the cutoff; it does not wait five hours before alerting. For each rule, its probability cutoff was selected on validation patients by normalized challenge utility, with ties favoring fewer non-sepsis patients receiving alerts.

The figures below are from the patient-held-out test split. That split was already inspected in earlier exploratory comparisons and the grid evaluates 36 choices, so the best observed rule may be optimistic. Treat these as project findings, not a final unbiased estimate or medical advice. The model is not validated for clinical use.

## Strongest observed candidates

| Rule | Cutoff | Accuracy | Hourly sensitivity | Sepsis warning-window coverage | Non-sepsis patients with any alert | False-positive hours | Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1-of-5** | 0.65 | 84.3% | 66.0% | 78.4% (345/440) | 26.0% (1,458/5,611) | 35,291 | **0.427** |
| 1-of-6 | 0.65 | 83.6% | 67.2% | 78.6% (346/440) | 26.0% (1,458/5,611) | 37,047 | 0.427 |
| 2-of-6 | 0.60 | 84.3% | 64.2% | 77.5% (341/440) | **25.6% (1,436/5,611)** | 35,243 | 0.411 |
| 2-of-3 | 0.55 | 84.4% | 63.6% | 79.1% (348/440) | 31.2% (1,752/5,611) | 34,952 | 0.403 |
| 4-of-7 | 0.55 | 85.7% | 60.1% | 73.2% (322/440) | 23.9% (1,341/5,611) | 31,881 | 0.393 |
| 8-of-8 | 0.45 | 88.1% | 51.8% | 67.3% (296/440) | **23.6% (1,323/5,611)** | 25,931 | 0.352 |

## How I’d interpret it

- **Best observed utility:** 1-of-5, at 0.427. It improves on instant alerts (1-of-1, utility 0.402; 42.4% non-sepsis patient alert rate) while retaining 78.4% warning-window coverage. It carries an alert forward for up to five hourly scores after a high score.
- **Similar false-alert rate with more utility:** 2-of-6 has a slightly lower non-sepsis patient alert rate than 1-of-5 (25.6% vs. 26.0%), with lower utility and slightly lower warning-window coverage.
- **Lower false-alert option:** 4-of-7 lowers the non-sepsis patient alert rate to 23.9%, with 73.2% warning-window coverage and 0.393 utility. The strict 8-of-8 rule lowers it a little further, but warning-window coverage falls to 67.3%.
- These results show a trade-off. There is no rule that maximizes detection, minimizes false alarms, and preserves alert timing at once. Choose based on which outcome the project prioritizes, then evaluate that choice on new untouched or external data.

The full 36-rule table, including every threshold and metric, is [alert_policy_grid.csv](policy_grid/alert_policy_grid.csv). The threshold-selection summary is [alert_policy_grid.json](policy_grid/alert_policy_grid.json).

## Reproduce

From the project directory:

```bash
python -m src.alert_policy_grid --data-dir data/raw --output-dir outputs/policy_grid --bundle outputs/trained_models.joblib --max-hours 8
```
