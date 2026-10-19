"""Validation-only model, ensemble, and generalization comparison (no test set)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components, validation_score
from src.run_pipeline import ID_COLUMN, SEED, TARGET, read_patients, make_features


THRESHOLDS = np.arange(0.05, 0.951, 0.025)
POLICY_N = 5


def select_threshold(probabilities: np.ndarray, starts: np.ndarray, ends: np.ndarray, septic: np.ndarray,
                     reward: np.ndarray, miss: np.ndarray, denominator: float, false_cap: float | None = None) -> tuple[float, dict]:
    risk = k_of_n_risk(probabilities, starts, ends, POLICY_N, 1)
    choices = []
    for cutoff in THRESHOLDS:
        utility, false_rate = validation_score(risk, float(cutoff), starts, septic, reward, miss, denominator)
        choices.append({"threshold": float(cutoff), "utility": utility, "false_rate": false_rate})
    eligible = [row for row in choices if false_cap is None or row["false_rate"] <= false_cap + 1e-12]
    if not eligible:
        eligible = choices
    selected = max(eligible, key=lambda row: (row["utility"], -row["false_rate"], row["threshold"]))
    return selected["threshold"], selected


def split_metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float,
                  starts: np.ndarray, ends: np.ndarray, septic: np.ndarray, timely: np.ndarray,
                  reward: np.ndarray, miss: np.ndarray, denominator: float) -> dict:
    risk = k_of_n_risk(probabilities, starts, ends, POLICY_N, 1)
    result = evaluate_test(frame, risk, threshold, starts, ends, septic, timely, reward, miss, denominator)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    result["raw_probability_pr_auc"] = float(average_precision_score(y, probabilities))
    result["raw_probability_roc_auc"] = float(roc_auc_score(y, probabilities))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generalization_experiment"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    parser.add_argument("--trend-bundle", type=Path, default=Path("outputs/trend_experiment/selected_trend_model.joblib"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records and creating causal trend features...", flush=True)
    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw, include_trends=True)
    patient_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = patient_target.index.to_numpy()
    train_ids, remainder = train_test_split(ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy())
    valid_ids, _unused_test = train_test_split(
        remainder, test_size=0.50, random_state=SEED,
        stratify=patient_target.loc[remainder].to_numpy(),
    )
    train_mask, valid_mask = patient_ids.isin(train_ids).to_numpy(), patient_ids.isin(valid_ids).to_numpy()

    train_patients, valid_patients = patient_ids.loc[train_mask].to_numpy(), patient_ids.loc[valid_mask].to_numpy()
    train_frame = pd.DataFrame({ID_COLUMN: train_patients, TARGET: labels.loc[train_mask].to_numpy()}).reset_index(drop=True)
    valid_frame = pd.DataFrame({ID_COLUMN: valid_patients, TARGET: labels.loc[valid_mask].to_numpy()}).reset_index(drop=True)
    train_starts, train_ends = group_bounds(train_patients)
    valid_starts, valid_ends = group_bounds(valid_patients)
    train_y = labels.loc[train_mask].to_numpy(dtype=np.int8)
    valid_y = labels.loc[valid_mask].to_numpy(dtype=np.int8)
    ty, treward, tmiss, tseptic, twindow, tbest, tinactive = utility_components(train_frame, train_starts, train_ends)
    vy, vreward, vmiss, vseptic, vwindow, vbest, vinactive = utility_components(valid_frame, valid_starts, valid_ends)
    tden, vden = tbest - tinactive, vbest - vinactive

    baseline_bundle = joblib.load(args.bundle)
    trend_bundle = joblib.load(args.trend_bundle)
    baseline_hgb = baseline_bundle["models"]["hist_gradient_boosting"]
    baseline_lr = baseline_bundle["models"]["logistic_regression"]
    baseline_columns = baseline_bundle["feature_names"]
    trend_model = trend_bundle["model"]

    x_train, x_valid = features.loc[train_mask], features.loc[valid_mask]
    candidate_probability: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_probability["baseline_logistic_regression"] = (
        baseline_lr.predict_proba(x_train.reindex(columns=baseline_columns))[:, 1],
        baseline_lr.predict_proba(x_valid.reindex(columns=baseline_columns))[:, 1],
    )
    candidate_probability["baseline_hist_gradient_boosting"] = (
        baseline_hgb.predict_proba(x_train.reindex(columns=baseline_columns))[:, 1],
        baseline_hgb.predict_proba(x_valid.reindex(columns=baseline_columns))[:, 1],
    )
    candidate_probability["trend_regularized_hist_gradient_boosting"] = (
        trend_model.predict_proba(x_train)[:, 1], trend_model.predict_proba(x_valid)[:, 1],
    )
    base_train_prob, base_valid_prob = candidate_probability["baseline_hist_gradient_boosting"]
    trend_train_prob, trend_valid_prob = candidate_probability["trend_regularized_hist_gradient_boosting"]
    for weight in (0.25, 0.50, 0.75):
        name = f"blend_baseline_{1-weight:.2f}_trend_{weight:.2f}"
        candidate_probability[name] = (
            (1 - weight) * base_train_prob + weight * trend_train_prob,
            (1 - weight) * base_valid_prob + weight * trend_valid_prob,
        )

    print("Training a regularized Random Forest comparison...", flush=True)
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=32, max_depth=14, min_samples_leaf=200,
            max_features=0.4, max_samples=0.25, bootstrap=True,
            class_weight="balanced_subsample", n_jobs=2, random_state=SEED,
        ),
    )
    forest.fit(x_train, train_y)
    candidate_probability["random_forest_regularized"] = (
        forest.predict_proba(x_train)[:, 1], forest.predict_proba(x_valid)[:, 1],
    )

    # Match candidate false-alarm rates to the existing HGB's validation rate.
    _, base_validation_selection = select_threshold(base_valid_prob, valid_starts, valid_ends, vseptic,
                                                     vreward, vmiss, vden)
    false_alert_cap = base_validation_selection["false_rate"]
    rows = []
    for name, (train_prob, valid_prob) in candidate_probability.items():
        threshold, selection = select_threshold(valid_prob, valid_starts, valid_ends, vseptic,
                                                vreward, vmiss, vden, false_alert_cap)
        train_metrics = split_metrics(train_frame, train_prob, threshold, train_starts, train_ends,
                                      tseptic, twindow, treward, tmiss, tden)
        valid_metrics = split_metrics(valid_frame, valid_prob, threshold, valid_starts, valid_ends,
                                      vseptic, vwindow, vreward, vmiss, vden)
        rows.append({
            "model": name,
            "threshold_selected_on_validation": threshold,
            "validation_utility": selection["utility"],
            "validation_nonsepsis_patient_false_alert_rate": selection["false_rate"],
            "train_pr_auc": train_metrics["raw_probability_pr_auc"],
            "validation_pr_auc": valid_metrics["raw_probability_pr_auc"],
            "pr_auc_train_minus_validation": train_metrics["raw_probability_pr_auc"] - valid_metrics["raw_probability_pr_auc"],
            "train_roc_auc": train_metrics["raw_probability_roc_auc"],
            "validation_roc_auc": valid_metrics["raw_probability_roc_auc"],
            "roc_auc_train_minus_validation": train_metrics["raw_probability_roc_auc"] - valid_metrics["raw_probability_roc_auc"],
            "train_utility_at_validation_threshold": train_metrics["normalized_challenge_utility"],
            "validation_warning_window_coverage": valid_metrics["sepsis_alert_window_coverage"],
            "validation_precision": valid_metrics["precision"],
            "validation_hourly_sensitivity": valid_metrics["sensitivity"],
        })
        print(f"Scored {name}: val utility={selection['utility']:.4f}, val PR-AUC={valid_metrics['raw_probability_pr_auc']:.4f}", flush=True)

    table = pd.DataFrame(rows).sort_values(
        ["validation_utility", "validation_nonsepsis_patient_false_alert_rate"], ascending=[False, True]
    )
    table.to_csv(args.output_dir / "generalization_comparison.csv", index=False)
    selected = table.iloc[0].to_dict()
    (args.output_dir / "generalization_results.json").write_text(json.dumps({
        "selection": "Validation normalized utility at or below the baseline HGB validation non-sepsis patient alert rate.",
        "validation_false_alert_cap": false_alert_cap,
        "selected_model_on_validation_only": selected["model"],
        "caveat": "No test split was used. Training metrics are in-sample and expected to be optimistic; compare train-to-validation gaps, not train scores alone.",
        "results": rows,
    }, indent=2), encoding="utf-8")
    print("\nTrain-versus-validation comparison (no test data used):")
    print(table[["model", "train_pr_auc", "validation_pr_auc", "pr_auc_train_minus_validation",
                 "train_roc_auc", "validation_roc_auc", "roc_auc_train_minus_validation",
                 "validation_utility", "validation_nonsepsis_patient_false_alert_rate"]].to_string(index=False))
    print(f"\nValidation-selected candidate: {selected['model']}")


if __name__ == "__main__":
    main()
