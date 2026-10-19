"""Compare validation-tuned alert policies for the saved boosted-tree model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, normalized_utility, read_patients


def apply_policy(probabilities: np.ndarray, threshold: float, policy: str) -> np.ndarray:
    """Apply causal smoothing and/or cooldown to a single patient's risk scores."""
    raw = (probabilities >= threshold).astype(np.int8)
    match = re.match(r"^(\d+)_of_(\d+)", policy)
    if match:
        required, window = int(match.group(1)), int(match.group(2))
    elif policy == "4_consecutive":
        window, required = 4, 4
    else:
        window, required = 1, 1
    if window > 1:
        counts = np.convolve(raw, np.ones(window, dtype=np.int8), mode="full")[:len(raw)]
        raw = (counts >= required).astype(np.int8)
    if "cooldown_6h" not in policy:
        return raw
    output = np.zeros_like(raw)
    next_allowed = 0
    for hour, flag in enumerate(raw):
        if flag and hour >= next_allowed:
            output[hour] = 1
            next_allowed = hour + 6
    return output


def policy_predictions(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, policy: str) -> np.ndarray:
    output = np.zeros(len(frame), dtype=np.int8)
    patient_values = frame[ID_COLUMN].to_numpy()
    start = 0
    while start < len(frame):
        patient_id = patient_values[start]
        stop = start + 1
        while stop < len(frame) and patient_values[stop] == patient_id:
            stop += 1
        output[start:stop] = apply_policy(probabilities[start:stop], threshold, policy)
        start = stop
    return output


def patient_summary(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    septic_count = nonseptic_count = timely_count = septic_alert_count = nonseptic_alert_count = 0
    lead_times = []
    flagged_nonsepsis_hours = []
    for (_, group), pred_group in zip(frame.groupby(ID_COLUMN, sort=False), _group_predictions(frame, predictions)):
        labels = group[TARGET].to_numpy(dtype=np.int8)
        pred = pred_group
        if labels.any():
            septic_count += 1
            if pred.any():
                septic_alert_count += 1
            onset = int(np.argmax(labels)) + 6
            lower, upper = max(0, onset - 12), min(len(labels), onset + 4)
            hits = np.flatnonzero(pred[lower:upper])
            if len(hits):
                timely_count += 1
                lead_times.append(onset - (lower + int(hits[0])))
        else:
            nonseptic_count += 1
            if pred.any():
                nonseptic_alert_count += 1
            flagged_nonsepsis_hours.append(int(pred.sum()))
    return {
        "sepsis_patients": septic_count,
        "sepsis_patients_with_any_alert": septic_alert_count,
        "sepsis_alert_window_coverage": timely_count / septic_count if septic_count else None,
        "nonsepsis_patients": nonseptic_count,
        "nonsepsis_patients_with_any_alert": nonseptic_alert_count,
        "nonsepsis_patient_false_alert_rate": nonseptic_alert_count / nonseptic_count if nonseptic_count else None,
        "median_lead_hours_for_in_window_alerts": float(np.median(lead_times)) if lead_times else None,
        "mean_false_alert_hours_per_nonsepsis_patient": float(np.mean(flagged_nonsepsis_hours)) if flagged_nonsepsis_hours else None,
    }


def _group_predictions(frame: pd.DataFrame, predictions: np.ndarray):
    start = 0
    for _, group in frame.groupby(ID_COLUMN, sort=False):
        stop = start + len(group)
        yield predictions[start:stop]
        start = stop


def evaluate_policy(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, policy: str) -> dict:
    predictions = policy_predictions(frame, probabilities, threshold, policy)
    y = frame[TARGET].astype(np.int8)
    patient_ids = frame[ID_COLUMN]
    tp = int(((y.to_numpy() == 1) & (predictions == 1)).sum())
    fp = int(((y.to_numpy() == 0) & (predictions == 1)).sum())
    fn = int(((y.to_numpy() == 1) & (predictions == 0)).sum())
    tn = int(((y.to_numpy() == 0) & (predictions == 0)).sum())
    return {
        "policy": policy,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "normalized_challenge_utility": normalized_utility(y, predictions, patient_ids),
        "alert_hours": int(predictions.sum()),
        "true_positive_hours": tp,
        "false_positive_hours": fp,
        "false_negative_hours": fn,
        "true_negative_hours": tn,
        **patient_summary(frame, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    parser.add_argument("--policy", choices=["instant", "2_of_3", "2_of_3_cooldown_6h", "3_of_5", "3_of_5_cooldown_6h", "instant_cooldown_6h", "4_consecutive"])
    args = parser.parse_args()

    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw)
    raw_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = raw_target.index.to_numpy()
    _, remainder_ids = train_test_split(ids, test_size=0.30, random_state=SEED, stratify=raw_target.to_numpy())
    remainder_target = raw_target.loc[remainder_ids]
    valid_ids, test_ids = train_test_split(remainder_ids, test_size=0.50, random_state=SEED, stratify=remainder_target.to_numpy())
    bundle = joblib.load(args.bundle)
    model_name = "hist_gradient_boosting"
    model = bundle["models"][model_name]
    feature_columns = bundle["feature_names"]
    validation_mask = patient_ids.isin(valid_ids).to_numpy()
    test_mask = patient_ids.isin(test_ids).to_numpy()

    def ordered_frame(mask: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({ID_COLUMN: patient_ids.loc[mask].to_numpy(), TARGET: labels.loc[mask].to_numpy()}).reset_index(drop=True)

    validation_frame = ordered_frame(validation_mask)
    test_frame = ordered_frame(test_mask)
    validation_probabilities = model.predict_proba(features.loc[validation_mask].reindex(columns=feature_columns))[:, 1]
    test_probabilities = model.predict_proba(features.loc[test_mask].reindex(columns=feature_columns))[:, 1]

    policies = [args.policy] if args.policy else ["instant", "2_of_3", "2_of_3_cooldown_6h", "3_of_5", "3_of_5_cooldown_6h", "instant_cooldown_6h"]
    # Use the same 0.05 step as the main pipeline to keep the sweep practical.
    thresholds = np.linspace(0.05, 0.95, 19)
    validation_records = []
    test_records = []
    for policy in policies:
        candidates = [evaluate_policy(validation_frame, validation_probabilities, float(threshold), policy) for threshold in thresholds]
        # Maximize challenge utility; break ties using fewer non-sepsis patients alerted.
        best = max(candidates, key=lambda row: (row["normalized_challenge_utility"], -row["nonsepsis_patient_false_alert_rate"], row["threshold"]))
        validation_records.append(best)
        test_records.append(evaluate_policy(test_frame, test_probabilities, best["threshold"], policy))
        print(f"Finished {policy}: validation threshold={best['threshold']:.3f}, utility={best['normalized_challenge_utility']:.3f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_records).to_csv(args.output_dir / "alert_policy_comparison.csv", index=False)
    result = {
        "model": model_name,
        "thresholds_selected_on": "validation patients",
        "policies": [{"validation": validation, "test": test} for validation, test in zip(validation_records, test_records)],
        "caveat": "The held-out test patients have been inspected in prior exploratory comparisons; use a fresh or external test set for final unbiased evaluation.",
    }
    (args.output_dir / "alert_policy_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(pd.DataFrame([{
        "policy": test["policy"], "threshold": test["threshold"], "utility": test["normalized_challenge_utility"],
        "sepsis_window_coverage": test["sepsis_alert_window_coverage"],
        "nonsepsis_alert_rate": test["nonsepsis_patient_false_alert_rate"],
        "false_alert_hours": test["false_positive_hours"], "accuracy": test["accuracy"],
    } for test in test_records]).to_string(index=False))


if __name__ == "__main__":
    main()
