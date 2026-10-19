"""Summarize held-out first alerts and false-alarm burden per patient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def patient_alert_table(predictions: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for patient_id, group in predictions.groupby("patient_id", sort=False):
        labels = group["y_true"].to_numpy(dtype=np.int8)
        septic = bool(labels.any())
        onset = int(np.flatnonzero(labels)[0] + 6) if septic else None
        record = {"patient_id": patient_id, "has_sepsis": septic, "hours_observed": len(group)}
        for name in model_names:
            pred = group[f"{name}_prediction"].to_numpy(dtype=np.int8)
            hits = np.flatnonzero(pred)
            first_alert = int(hits[0]) if len(hits) else None
            record[f"{name}_alert_hours"] = int(pred.sum())
            record[f"{name}_any_alert"] = bool(len(hits))
            if septic:
                window_start, window_stop = max(0, onset - 12), min(len(pred), onset + 4)
                window_hits = np.flatnonzero(pred[window_start:window_stop])
                in_window = bool(len(window_hits))
                record[f"{name}_first_alert_hour"] = first_alert
                record[f"{name}_estimated_onset_hour"] = onset
                record[f"{name}_lead_hours"] = (onset - (window_start + int(window_hits[0]))) if in_window else np.nan
                record[f"{name}_alert_in_utility_window"] = in_window
            else:
                record[f"{name}_first_alert_hour"] = first_alert
                record[f"{name}_estimated_onset_hour"] = np.nan
                record[f"{name}_lead_hours"] = np.nan
                record[f"{name}_alert_in_utility_window"] = False
        rows.append(record)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, model_names: list[str]) -> dict:
    septic = table[table["has_sepsis"]]
    nonseptic = table[~table["has_sepsis"]]
    summary = {"test_patients": int(len(table)), "sepsis_patients": int(len(septic)),
               "nonsepsis_patients": int(len(nonseptic)), "models": {}}
    for name in model_names:
        timely = septic[f"{name}_alert_in_utility_window"]
        any_septic_alert = septic[f"{name}_any_alert"]
        any_nonseptic_alert = nonseptic[f"{name}_any_alert"]
        timely_leads = septic.loc[timely, f"{name}_lead_hours"]
        summary["models"][name] = {
            "sepsis_alerts_in_utility_window": int(timely.sum()),
            "sepsis_alert_window_coverage": float(timely.mean()) if len(timely) else None,
            "sepsis_patients_with_any_alert": int(any_septic_alert.sum()),
            "nonsepsis_patients_with_any_alert": int(any_nonseptic_alert.sum()),
            "nonsepsis_patient_false_alert_rate": float(any_nonseptic_alert.mean()) if len(any_nonseptic_alert) else None,
            "median_lead_hours_for_in_window_alerts": float(timely_leads.median()) if len(timely_leads) else None,
            "median_alert_hours_per_nonsepsis_patient": float(nonseptic[f"{name}_alert_hours"].median()) if len(nonseptic) else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "test_predictions.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    models = [
        "logistic_regression", "hist_gradient_boosting",
        "logistic_regression_accuracy", "hist_gradient_boosting_accuracy",
    ]
    table = patient_alert_table(predictions, models)
    summary = summarize(table, models)
    table.to_csv(args.output_dir / "patient_alert_analysis.csv", index=False)
    (args.output_dir / "patient_alert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result_by_model = {row["model"]: row for row in metrics["results"]}
    lines = [
        "# Initial Held-Out Results",
        "",
        "This is a research prototype on the PhysioNet/CinC Challenge 2019 training data. It is not validated for clinical use.",
        "",
        f"The patient-level split contains {metrics['patient_counts']['train']:,} training, {metrics['patient_counts']['validation']:,} validation, and {metrics['patient_counts']['test']:,} test patients. Thresholds were selected on validation patients, not on test labels. The held-out test split has been examined across multiple exploratory comparisons, so these figures should not be treated as a final untouched test estimate.",
        "",
        "## Held-out model comparison",
        "",
        "| Model | Threshold | Accuracy | PR-AUC | ROC-AUC | Sensitivity | Precision | Normalized utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["train_prevalence_baseline", *models]:
        row = result_by_model[name]
        lines.append(f"| {name} | {row['threshold']:.2f} | {row['accuracy']:.3f} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} | {row['recall_sensitivity']:.3f} | {row['precision']:.3f} | {row['normalized_challenge_utility']:.3f} |")
    lines += ["", "Accuracy-named variants were trained without class weighting and selected a threshold to maximize validation accuracy. The other classifiers use class weighting and a threshold selected for normalized challenge utility.", "", "The accuracy-focused variants reach roughly the majority-class baseline accuracy but have zero sensitivity on the test split: their selected cutoffs generate essentially no positive alerts. They do not improve sepsis detection. The baseline's high accuracy comes from the rarity of positive hourly labels.", "", "## Patient-level alert review", "", "| Model | Sepsis patients alerted in utility window | Window coverage | Non-sepsis patients with any alert | False-alert rate among non-sepsis patients | Median lead (hours, timely alerts) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in models:
        row = summary["models"][name]
        lead = "n/a" if row["median_lead_hours_for_in_window_alerts"] is None else f"{row['median_lead_hours_for_in_window_alerts']:.1f}"
        lines.append(f"| {name} | {row['sepsis_alerts_in_utility_window']:,} | {row['sepsis_alert_window_coverage']:.3f} | {row['nonsepsis_patients_with_any_alert']:,} | {row['nonsepsis_patient_false_alert_rate']:.3f} | {lead} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Compare the accuracy column with sensitivity, PR-AUC, normalized utility, and the patient-level false-alert rate. Since positive hourly labels are rare, an accuracy-oriented model can score highly simply by predicting few or no alerts. The patient-level table reports whether a non-sepsis patient received any alert and whether a sepsis patient received an alert during the challenge's beneficial time window.",
        "",
        "The dataset contains historical ICU stays from the challenge's source hospitals; these results do not establish external validity, prospective benefit, calibration, or clinical safety. Thresholds were selected on validation data and must not be interpreted as medical action cutoffs. Further model selection should use a new untouched test set or external dataset for a final unbiased estimate.",
        "",
    ]
    (args.output_dir / "initial_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
