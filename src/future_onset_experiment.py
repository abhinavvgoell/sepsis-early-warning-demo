"""Compare the current model with a strictly pre-onset, next-six-hour target.

This is an isolated research experiment. It never overwrites the current demo model.
The target is positive only in the six hourly records before an observed onset;
post-onset records and records with an unobserved/censored onset are excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.alert_policy_grid import group_bounds, k_of_n_risk
from src.improved_pipeline import make_features_v2
from src.run_pipeline import ID_COLUMN, SEED, TARGET, read_patients


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def define_future_target(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep pre-onset rows; label the six hours immediately before onset as positive."""
    y = np.zeros(len(raw), dtype=np.int8)
    keep = np.ones(len(raw), dtype=bool)
    event_count = censored_count = 0
    for _, group in raw.groupby(ID_COLUMN, sort=False):
        positions = group.index.to_numpy()
        old_y = group[TARGET].to_numpy(dtype=np.int8)
        positive = np.flatnonzero(old_y)
        if not len(positive):
            continue
        onset = int(positive[0]) + 6
        if onset >= len(group):
            # The available record ends before an onset time can be verified.
            keep[positions] = False
            censored_count += 1
            continue
        event_count += 1
        keep[positions[onset:]] = False
        start = max(0, onset - 6)
        y[positions[start:onset]] = 1
    summary = {"observed_sepsis_events": event_count,
               "censored_or_unobserved_onsets_excluded": censored_count,
               "rows_kept": int(keep.sum()), "positive_rows": int(y[keep].sum())}
    return y, keep, summary


def apply_calibrator(bundle: dict, raw_scores: np.ndarray) -> np.ndarray:
    cal = bundle["calibrator"]
    if isinstance(cal, dict):
        return np.interp(raw_scores, np.asarray(cal["x_thresholds"]), np.asarray(cal["y_thresholds"]))
    return cal.predict(raw_scores)


def select_threshold(labels: np.ndarray, scores: np.ndarray, ids: np.ndarray,
                     false_alert_cap: float) -> dict:
    """On validation data, maximize hourly sensitivity under a patient alert cap."""
    starts, ends = group_bounds(ids)
    risk = k_of_n_risk(scores.astype(np.float32), starts, ends, n=5, k=1)
    septic = np.maximum.reduceat(labels.astype(np.int8), starts).astype(bool)
    thresholds = np.unique(np.quantile(risk, np.linspace(0.80, 0.9995, 140)))
    best = None
    for threshold in thresholds:
        pred = risk >= threshold
        patient_any = np.maximum.reduceat(pred.astype(np.int8), starts).astype(bool)
        far = float(patient_any[~septic].mean()) if (~septic).any() else 0.0
        if far > false_alert_cap + 1e-12:
            continue
        tp = int(np.sum(pred & (labels == 1)))
        fn = int(np.sum(~pred & (labels == 1)))
        sensitivity = tp / max(1, tp + fn)
        fp = int(np.sum(pred & (labels == 0)))
        precision = tp / max(1, tp + fp)
        item = {"threshold": float(threshold), "far": far,
                "sensitivity": float(sensitivity), "precision": float(precision)}
        if best is None or (item["sensitivity"], item["precision"], item["threshold"]) > (
                best["sensitivity"], best["precision"], best["threshold"]):
            best = item
    return best


def evaluate(labels: np.ndarray, scores: np.ndarray, ids: np.ndarray,
             threshold: float) -> dict:
    starts, ends = group_bounds(ids)
    risk = k_of_n_risk(scores.astype(np.float32), starts, ends, n=5, k=1)
    pred = risk >= threshold
    septic = np.maximum.reduceat(labels.astype(np.int8), starts).astype(bool)
    patient_any = np.maximum.reduceat(pred.astype(np.int8), starts).astype(bool)
    patient_in_window = np.maximum.reduceat((pred & (labels == 1)).astype(np.int8), starts).astype(bool)
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    return {
        "accuracy": float((tp + tn) / max(1, len(labels))),
        "sensitivity": float(tp / max(1, tp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "specificity": float(tn / max(1, tn + fp)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "warning_window_coverage": float(np.sum(patient_in_window & septic) / max(1, septic.sum())),
        "nonsepsis_patient_false_alert_rate": float(np.sum(patient_any & ~septic) / max(1, (~septic).sum())),
        "false_positive_hours": fp,
        "alert_hours": int(np.sum(pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/future_onset_experiment"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/improved/improved_model.joblib"))
    parser.add_argument("--false-alert-cap", type=float, default=0.25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("reading hourly patient records")
    raw = read_patients(args.data_dir).sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
    new_y, keep, target_summary = define_future_target(raw)
    patient_ids_all = raw[ID_COLUMN].to_numpy()
    positive_patient = pd.Series(new_y[keep]).groupby(patient_ids_all[keep]).max()
    patients = positive_patient.index.to_numpy()
    train_ids, remainder = train_test_split(patients, test_size=0.30, random_state=SEED,
                                             stratify=positive_patient.to_numpy())
    valid_ids, test_ids = train_test_split(remainder, test_size=0.50, random_state=SEED,
                                           stratify=positive_patient.loc[remainder].to_numpy())

    cache_path = args.output_dir / "feature_cache.joblib"
    if cache_path.exists():
        log(f"loading cached features: {cache_path}")
        cached = joblib.load(cache_path)
        features = cached["features"]
    else:
        log("building causal feature matrix (first run may take a few minutes)")
        features = make_features_v2(raw)
        joblib.dump({"features": features}, cache_path, compress=3)

    rows = np.flatnonzero(keep)
    ids = patient_ids_all[keep]
    labels = new_y[keep]
    x = features.iloc[rows].reset_index(drop=True)
    masks = [np.isin(ids, split_ids) for split_ids in (train_ids, valid_ids, test_ids)]
    (train_mask, valid_mask, test_mask) = masks

    bundle = joblib.load(args.bundle)
    params = bundle["params"]
    rounds = int(bundle["rounds"])
    feature_names = bundle["feature_names"]
    x = x.reindex(columns=feature_names)

    log(f"training on {len(train_ids):,} patients; target rows={int(train_mask.sum()):,}")
    dtrain = lgb.Dataset(x.loc[train_mask], labels[train_mask], free_raw_data=True)
    model = lgb.train({"objective": "binary", "metric": "average_precision", "learning_rate": 0.05,
                       "verbosity": -1, "num_threads": 12, "seed": SEED, "bagging_freq": 1,
                       "max_bin": 255, "deterministic": True, **params},
                      dtrain, num_boost_round=rounds)

    log("scoring the validation patients and selecting a 1-of-5 rule under the fixed alert cap")
    valid_new_raw = model.predict(x.loc[valid_mask])
    new_selection = select_threshold(labels[valid_mask], valid_new_raw, ids[valid_mask], args.false_alert_cap)
    if new_selection is None:
        raise RuntimeError("No future-onset threshold met the chosen validation false-alert cap.")

    current = joblib.load(args.bundle)
    current_booster = lgb.Booster(model_str=current["model_text"])
    valid_old = apply_calibrator(current, current_booster.predict(x.loc[valid_mask]))
    old_selection = select_threshold(labels[valid_mask], valid_old, ids[valid_mask], args.false_alert_cap)
    if old_selection is None:
        raise RuntimeError("No current-model threshold met the chosen validation false-alert cap.")

    log("evaluating both candidates on the patient-held-out exploratory split")
    test_new = model.predict(x.loc[test_mask])
    test_old = apply_calibrator(current, current_booster.predict(x.loc[test_mask]))
    evaluated = {}
    for name, scores, selection in (("future_onset_candidate", test_new, new_selection),
                                    ("current_challenge_label_model", test_old, old_selection)):
        metrics = evaluate(labels[test_mask], scores, ids[test_mask], float(selection["threshold"]))
        metrics.update({"row_pr_auc": metrics.pop("pr_auc"),
                        "row_roc_auc": metrics.pop("roc_auc"),
                        "policy": "1-of-5", "validation_threshold": float(selection["threshold"]),
                        "validation_false_alert_rate": float(selection["far"])})
        evaluated[name] = metrics
        log(f"{name}: PR-AUC={metrics['row_pr_auc']:.4f}, sensitivity={metrics['sensitivity']:.3f}, "
            f"precision={metrics['precision']:.3f}, alert FAR={metrics['nonsepsis_patient_false_alert_rate']:.3f}")

    result = {"experiment": "predict onset in the next six hours, restricted to pre-onset rows",
              "target_definition": "positive for observed septic patients in the six rows before inferred onset; all post-onset rows and septic patients whose onset cannot be observed in their record are excluded",
              "target_summary": target_summary,
              "patient_split_counts": {"train": len(train_ids), "validation": len(valid_ids), "test": len(test_ids)},
              "false_alert_cap_selected_before_test": args.false_alert_cap,
              "test_split_caveat": "The patient split and source dataset were examined in prior project iterations. This comparison is exploratory and not an independent clinical validation.",
              "validation_selection": {"future_onset_candidate": new_selection,
                                       "current_challenge_label_model": old_selection},
              "test_metrics": evaluated}
    (args.output_dir / "future_onset_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    joblib.dump({"candidate_only": True, "model_text": model.model_to_string(), "feature_names": feature_names,
                 "params": params, "rounds": rounds,
                 "policy": {**new_selection, "n": 5, "k": 1},
                 "target": result["target_definition"],
                 "feature_builder": "src.improved_pipeline.make_features_v2"},
                args.output_dir / "future_onset_model.joblib", compress=3)
    (args.output_dir / "future_onset_results.md").write_text(render_report(result), encoding="utf-8")
    log(f"wrote results and isolated candidate to {args.output_dir}")


def render_report(result: dict) -> str:
    future = result["test_metrics"]["future_onset_candidate"]
    current = result["test_metrics"]["current_challenge_label_model"]
    lines = ["# Next-six-hour sepsis onset experiment", "",
             "This isolated candidate predicts the dataset-inferred onset within the next six hours using only pre-onset records. It excludes post-onset rows and cases where onset cannot be observed before the record ends. It does not diagnose sepsis or establish clinical usefulness.", "",
             f"- Retained observed sepsis events: {result['target_summary']['observed_sepsis_events']:,}",
             f"- Excluded censored/unobserved septic records: {result['target_summary']['censored_or_unobserved_onsets_excluded']:,}",
             f"- Patient split: {result['patient_split_counts']['train']:,} train / {result['patient_split_counts']['validation']:,} validation / {result['patient_split_counts']['test']:,} test",
             f"- Validation false-alert cap: {result['false_alert_cap_selected_before_test']:.0%}", "",
             "| Candidate on same held-out patients | Row PR-AUC | Row ROC-AUC | Sensitivity | Precision | Sepsis warning-window coverage | Non-sepsis patient alert rate |",
             "|---|---:|---:|---:|---:|---:|---:|",
             f"| Future-onset model | {future['row_pr_auc']:.3f} | {future['row_roc_auc']:.3f} | {future['sensitivity']:.1%} | {future['precision']:.1%} | {future['warning_window_coverage']:.1%} | {future['nonsepsis_patient_false_alert_rate']:.1%} |",
             f"| Current label model | {current['row_pr_auc']:.3f} | {current['row_roc_auc']:.3f} | {current['sensitivity']:.1%} | {current['precision']:.1%} | {current['warning_window_coverage']:.1%} | {current['nonsepsis_patient_false_alert_rate']:.1%} |", "",
             "The test split and source data were examined in prior project iterations, so results are exploratory and not independent final estimates. A public-dataset comparison is not clinical validation. The current demo model was not overwritten.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
