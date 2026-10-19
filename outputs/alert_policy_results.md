# Alert Policy Comparison

This experiment applies simple temporal alert rules to the trained Histogram Gradient Boosting model. Each rule's probability threshold was selected on validation patients to maximize the challenge-style normalized utility, then assessed on the patient-held-out test split. The test split was previously examined during exploratory work, so these numbers are useful for project comparison but are not a final untouched-test estimate. This is a research exercise, not a clinical system.

## Results

| Alert rule | Threshold | Accuracy | Hourly sensitivity | Timely sepsis-window coverage | Non-sepsis patients with any alert | False-positive hours | Normalized utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Immediate alert | 0.55 | 83.1% | 65.8% | 87.7% (386/440) | 42.4% (2,379/5,611) | 38,132 | 0.402 |
| At least 2 positive scores in 3 hours | 0.55 | 84.4% | 63.6% | 79.1% (348/440) | 31.2% (1,752/5,611) | 34,952 | 0.403 |
| At least 3 positive scores in 5 hours | 0.55 | 85.1% | 61.9% | 76.1% (335/440) | 27.2% (1,527/5,611) | 33,284 | 0.400 |
| 4 consecutive positive scores | 0.45 | 83.5% | 62.4% | 79.5% (350/440) | 35.6% (2,000/5,611) | 37,020 | 0.377 |
| Immediate alert + six-hour cooldown | 0.65 | 95.8% | 11.7% | 76.6% (337/440) | 26.0% (1,458/5,611) | 6,073 | 0.074 |
| 2 of 3 hours + six-hour cooldown | 0.55 | 95.2% | 12.1% | 78.9% (347/440) | 31.2% (1,752/5,611) | 7,459 | 0.070 |
| 3 of 5 hours + six-hour cooldown | 0.55 | 95.5% | 11.3% | 75.9% (334/440) | 27.2% (1,527/5,611) | 6,749 | 0.067 |

## What the comparison suggests

- The **2-of-3-hour rule** is the strongest overall balance in this sweep: normalized utility is slightly higher than immediate alerts (0.403 vs. 0.402), and the non-sepsis patient alert rate is lower by 11.2 percentage points. Timely sepsis-window coverage drops by 8.6 points.
- The **3-of-5-hour rule** reduces non-sepsis patient alerts by 15.2 percentage points versus immediate alerts (a 35.8% relative reduction). Its utility is nearly unchanged (0.400 vs. 0.402), while timely coverage falls by 11.1 points. This is the more conservative choice if fewer patients receive any false alert is the priority.
- The **4-consecutive-hour rule** lands between the other persistence rules for coverage (79.5%), but its non-sepsis alert rate (35.6%) and utility (0.377) are worse than the 2-of-3 and 3-of-5 rules in this comparison.
- A six-hour cooldown greatly lowers repeated false-positive alert hours, but also suppresses many useful positive hours. Utility falls from about 0.40 to about 0.07 and hourly sensitivity to about 11–12%. High accuracy here is misleading because most hours have a negative label.

The first comparison tested a small set of representative rules. A broader sweep across every k-of-n rule for windows up to eight hours is now available in [alert_policy_grid_results.md](alert_policy_grid_results.md). That sweep found stronger candidates, so use its results for the current comparison. Do not call any rule “best clinically”: the dataset is historical, the held-out test split has been explored, and no prospective or external validation has been done.

## Reproduce

From the project directory, run:

```bash
python -m src.alert_policy_experiment --data-dir data/raw --output-dir outputs --bundle outputs/trained_models.joblib
```

The detailed metrics are in `alert_policy_comparison.csv`; validation-selected thresholds and experiment notes are in `alert_policy_results.json`.
The four-consecutive-hour result is saved separately in `four_hour/alert_policy_comparison.csv`.
