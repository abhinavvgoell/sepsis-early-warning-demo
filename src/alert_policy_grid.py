"""Efficient validation sweep of every k-of-n alert rule for windows up to 8 hours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, read_patients


def group_bounds(patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(patient_ids[1:] != patient_ids[:-1]) + 1]
    return starts, np.r_[starts[1:], len(patient_ids)]


def k_of_n_risk(probabilities: np.ndarray, starts: np.ndarray, ends: np.ndarray, n: int, k: int) -> np.ndarray:
    """Return a per-hour threshold score equivalent to k-of-n thresholding."""
    # A finite value below all searched cutoffs marks incomplete prefixes where
    # fewer than k scores are available; it also keeps ranking metrics defined.
    risk = np.full(len(probabilities), -1.0, dtype=np.float32)
    for start, end in zip(starts, ends):
        scores = probabilities[start:end]
        prefix_end = min(n - 1, len(scores))
        for local_end in range(k - 1, prefix_end):
            risk[start + local_end] = np.partition(scores[:local_end + 1], local_end + 1 - k)[local_end + 1 - k]
        if len(scores) >= n:
            windows = np.lib.stride_tricks.sliding_window_view(scores, n)
            kth_largest = np.partition(windows, n - k, axis=1)[:, n - k]
            risk[start + n - 1:end] = kth_largest
    return risk


def utility_components(frame: pd.DataFrame, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Precompute additive per-hour utility terms and patient-level masks."""
    labels = frame[TARGET].to_numpy(dtype=np.int8)
    reward = np.zeros(len(frame), dtype=np.float64)
    miss = np.zeros(len(frame), dtype=np.float64)
    septic_patients = np.zeros(len(starts), dtype=bool)
    timely_window = np.zeros(len(frame), dtype=bool)
    best = 0.0
    for patient_index, (start, end) in enumerate(zip(starts, ends)):
        y = labels[start:end]
        length = len(y)
        if not np.any(y):
            reward[start:end] = -0.05
            continue
        septic_patients[patient_index] = True
        onset = int(np.argmax(y)) + 6
        times = np.arange(length, dtype=float)
        rel = times - onset
        early = times <= onset - 6
        late = (times > onset - 6) & (times <= onset + 3)
        m1, b1 = 1.0 / 6.0, 2.0
        m2, b2 = -1.0 / 9.0, 1.0 / 3.0
        m3, b3 = -2.0 / 9.0, -(4.0 / 3.0)
        patient_reward = np.zeros(length, dtype=float)
        patient_reward[early] = np.maximum(m1 * rel[early] + b1, -0.05)
        patient_reward[late] = m2 * rel[late] + b2
        patient_miss = np.zeros(length, dtype=float)
        patient_miss[late] = m3 * rel[late] + b3
        reward[start:end] = patient_reward
        miss[start:end] = patient_miss
        lo, hi = max(0, onset - 12), min(length, onset + 4)
        timely_window[start + lo:start + hi] = True
        ideal = np.zeros(length, dtype=bool)
        ideal[lo:hi] = True
        best += patient_reward[ideal].sum()
    inactive = miss.sum()
    return labels, reward, miss, septic_patients, timely_window, best, inactive


def validation_score(risk: np.ndarray, threshold: float, starts: np.ndarray, septic: np.ndarray,
                     reward: np.ndarray, miss: np.ndarray, denominator: float) -> tuple[float, float]:
    pred = risk >= threshold
    patient_any = np.maximum.reduceat(pred.astype(np.int8), starts).astype(bool)
    utility = (np.dot(reward, pred) + np.dot(miss, ~pred) - miss.sum()) / denominator if denominator else 0.0
    nonseptic_rate = float(patient_any[~septic].mean()) if np.any(~septic) else 0.0
    return float(utility), nonseptic_rate


def evaluate_test(frame: pd.DataFrame, risk: np.ndarray, threshold: float, starts: np.ndarray,
                  ends: np.ndarray, septic: np.ndarray, timely_window: np.ndarray,
                  reward: np.ndarray, miss: np.ndarray, denominator: float) -> dict:
    pred = (risk >= threshold).astype(np.int8)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    patient_any = np.maximum.reduceat(pred, starts).astype(bool)
    timely_any = np.maximum.reduceat((pred.astype(bool) & timely_window).astype(np.int8), starts).astype(bool)
    false_patient_count = int(np.sum(patient_any & ~septic))
    timely_count = int(np.sum(timely_any & septic))
    false_hours = int(np.sum((y == 0) & (pred == 1)))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(np.sum((y == 0) & (pred == 0)) / max(1, np.sum(y == 0))),
        "pr_auc": float(average_precision_score(y, risk)),
        "roc_auc": float(roc_auc_score(y, risk)),
        "normalized_challenge_utility": float((np.dot(reward, pred) + np.dot(miss, pred == 0) - miss.sum()) / denominator) if denominator else 0.0,
        "timely_sepsis_patients": timely_count,
        "sepsis_patients": int(septic.sum()),
        "sepsis_alert_window_coverage": float(timely_count / max(1, septic.sum())),
        "nonsepsis_patients_with_alert": false_patient_count,
        "nonsepsis_patients": int((~septic).sum()),
        "nonsepsis_patient_false_alert_rate": float(false_patient_count / max(1, (~septic).sum())),
        "false_positive_hours": false_hours,
        "alert_hours": int(pred.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    parser.add_argument("--max-hours", type=int, default=8)
    args = parser.parse_args()
    if args.max_hours < 1:
        raise ValueError("--max-hours must be at least 1")

    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw)
    raw_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = raw_target.index.to_numpy()
    _, remainder = train_test_split(ids, test_size=0.30, random_state=SEED, stratify=raw_target.to_numpy())
    remainder_target = raw_target.loc[remainder]
    valid_ids, test_ids = train_test_split(remainder, test_size=0.50, random_state=SEED, stratify=remainder_target.to_numpy())
    bundle = joblib.load(args.bundle)
    model = bundle["models"]["hist_gradient_boosting"]
    columns = bundle["feature_names"]
    valid_mask = patient_ids.isin(valid_ids).to_numpy()
    test_mask = patient_ids.isin(test_ids).to_numpy()

    def make_split(mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        patient_array = patient_ids.loc[mask].to_numpy()
        frame = pd.DataFrame({ID_COLUMN: patient_array, TARGET: labels.loc[mask].to_numpy()}).reset_index(drop=True)
        starts, ends = group_bounds(patient_array)
        probs = model.predict_proba(features.loc[mask].reindex(columns=columns))[:, 1].astype(np.float32)
        return frame, probs, starts, ends

    valid, valid_prob, valid_starts, valid_ends = make_split(valid_mask)
    test, test_prob, test_starts, test_ends = make_split(test_mask)
    vy, vreward, vmiss, vseptic, vwindow, vbest, vinactive = utility_components(valid, valid_starts, valid_ends)
    ty, treward, tmiss, tseptic, twindow, tbest, tinactive = utility_components(test, test_starts, test_ends)
    vdenom, tdenom = vbest - vinactive, tbest - tinactive
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []

    for hours in range(1, args.max_hours + 1):
        valid_window_scores = {k: k_of_n_risk(valid_prob, valid_starts, valid_ends, hours, k) for k in range(1, hours + 1)}
        test_window_scores = {k: k_of_n_risk(test_prob, test_starts, test_ends, hours, k) for k in range(1, hours + 1)}
        for required in range(1, hours + 1):
            risk_v, risk_t = valid_window_scores[required], test_window_scores[required]
            candidates = []
            for cutoff in thresholds:
                utility, false_rate = validation_score(risk_v, float(cutoff), valid_starts, vseptic,
                                                       vreward, vmiss, vdenom)
                candidates.append((utility, -false_rate, float(cutoff), false_rate))
            best_utility, _, chosen_threshold, validation_false_rate = max(candidates)
            metrics = evaluate_test(test, risk_t, chosen_threshold, test_starts, test_ends, tseptic,
                                    twindow, treward, tmiss, tdenom)
            results.append({
                "rule": f"{required}_of_{hours}", "required_positive_scores": required,
                "window_hours": hours, "validation_threshold": chosen_threshold,
                "validation_utility": best_utility,
                "validation_nonsepsis_patient_false_alert_rate": validation_false_rate,
                **metrics,
            })
        print(f"Completed all {hours}-hour windows", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).sort_values(
        ["normalized_challenge_utility", "nonsepsis_patient_false_alert_rate"], ascending=[False, True]
    )
    table.to_csv(args.output_dir / "alert_policy_grid.csv", index=False)
    result = {
        "grid": f"all k-of-n rules for n=1..{args.max_hours} hours ({len(results)} rules)",
        "thresholds": thresholds.tolist(),
        "threshold_selection": "validation normalized utility; ties favor lower non-sepsis patient false-alert rate, then higher threshold",
        "test_caveat": "The held-out test patients were examined in earlier exploratory comparisons; these results are not an untouched final estimate.",
        "top_by_utility": table.head(10).to_dict(orient="records"),
        "lowest_false_alert_rates": table.sort_values("nonsepsis_patient_false_alert_rate").head(10).to_dict(orient="records"),
    }
    (args.output_dir / "alert_policy_grid.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nTop rules on held-out patients:")
    print(table[["rule", "validation_threshold", "accuracy", "sensitivity", "sepsis_alert_window_coverage",
                 "nonsepsis_patient_false_alert_rate", "false_positive_hours", "normalized_challenge_utility"]]
          .head(12).to_string(index=False))


if __name__ == "__main__":
    main()
