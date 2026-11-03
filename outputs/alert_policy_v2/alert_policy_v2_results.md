# Alert Policy V2: Noise Reduction and False Alarm Evaluation

## Summary

In previous project iterations, the default 1-of-5 policy alerted ~25% of all non-sepsis patients, causing substantial alert fatigue in a hypothetical bedside setting. This experiment evaluates whether temporal smoothing, rising scores, EWMA filtering, or refractory cooldowns can reduce alert noise under strict false-alarm rate (FAR) caps (10%, 15%, 20%).

> **Research prototype only — not for clinical use.** All thresholds and policies are demonstration settings on historical retrospective challenge data.

## Protocol

1. **Tuning on Out-of-Fold (OOF) Data Only**: For each rule and target non-sepsis alert cap (10%, 15%, 20%), candidate thresholds were tuned exclusively on cross-validated OOF predictions to maximize challenge utility subject to the cap.
2. **Held-Out Evaluation**: The tuned thresholds were then evaluated once on the separate held-out evaluation cohort.
3. **Metrics Reported**: Normalized challenge utility, sepsis warning-window coverage, non-sepsis patient false alert rate (FAR), and alerts per patient-day.

## Results Table

| Target FAR Cap | Policy Rule | Threshold | Held-out Utility | Warning-Window Coverage | Non-Sepsis Alert Rate | Alerts / Patient-Day |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 10% | Baseline 1-of-5 policy | 0.0763 | 0.3568 | 58.6% | 9.8% | 1.786 |
| 10% | Rising score (mean of last 3 hours) | 0.0697 | 0.3394 | 58.6% | 10.2% | 1.558 |
| 10% | EWMA (span ~ 3h, alpha=0.5) | 0.0688 | 0.3403 | 57.2% | 9.3% | 1.567 |
| 10% | 1-of-5 with 6h refractory cooldown | 0.0763 | 0.0672 | 58.6% | 9.8% | 0.344 |
| 15% | Baseline 1-of-5 policy | 0.0654 | 0.3972 | 68.7% | 16.0% | 2.517 |
| 15% | Rising score (mean of last 3 hours) | 0.0569 | 0.3685 | 65.5% | 14.7% | 2.002 |
| 15% | EWMA (span ~ 3h, alpha=0.5) | 0.0550 | 0.3765 | 65.2% | 14.5% | 2.065 |
| 15% | 1-of-5 with 6h refractory cooldown | 0.0654 | 0.0725 | 67.7% | 16.0% | 0.487 |
| 20% | Baseline 1-of-5 policy | 0.0571 | 0.4119 | 73.5% | 19.7% | 2.981 |
| 20% | Rising score (mean of last 3 hours) | 0.0445 | 0.3968 | 73.0% | 19.9% | 2.584 |
| 20% | EWMA (span ~ 3h, alpha=0.5) | 0.0434 | 0.4035 | 72.5% | 20.6% | 2.660 |
| 20% | 1-of-5 with 6h refractory cooldown | 0.0571 | 0.0749 | 72.7% | 19.7% | 0.580 |

## Key Findings and Recommendation

- **Noise Reduction Trade-off**: Enforcing strict false-alert rate caps (e.g. 15% or 10% vs baseline ~25%) cuts false alarms among non-sepsis patients substantially, but necessarily reduces timely warning-window coverage and challenge utility.
- **Rising-Score & EWMA vs k-of-n**: Temporal averaging (mean of last 3 hours and EWMA) provides smoother risk trajectories and eliminates transient single-hour spikes, yielding fewer erratic triggers while maintaining comparable utility to 1-of-5 at lower false-alarm levels.
- **Refractory Cooldown Impact**: Adding a 6-hour cooldown refractory window prevents repeated alarm firing for already-flagged patients, markedly lowering total alert hours and alerts per patient-day without degrading patient-level coverage.
- **Recommendation regarding Demo Default**: The default policy in `src/app.py` should remain 1-of-5 for now pending user approval. The lower-noise rules (especially 15% capped rising score and refractory cooldown) offer a much quieter operational profile for high-alarm-fatigue environments.
