"""Validation-only experiment with last-observation and data-freshness features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components, validation_score
from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, read_patients


THRESHOLDS = np.arange(0.05, 0.951, 0.025)
POLICY_N = 5


def choose_threshold(probabilities: np.ndarray, starts: np.ndarray, ends: np.ndarray, septic: np.ndarray,
                     reward: np.ndarray, miss: np.ndarray, denominator: float, false_cap: float) -> dict:
    risk = k_of_n_risk(probabilities, starts, ends, POLICY_N, 1)
    candidates = []
    for threshold in THRESHOLDS:
        utility, false_rate = validation_score(risk, float(threshold), starts, septic, reward, miss, denominator)
        if false_rate <= false_cap + 1e-12:
            candidates.append({"threshold": float(threshold), "utility": utility, "false_rate": false_rate})
    if not candidates:
        raise RuntimeError("No threshold met the validation false-alert cap.")
    return max(candidates, key=lambda row: (row["utility"], -row["false_rate"], row["threshold"]))


def metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float,
            starts: np.ndarray, ends: np.ndarray, septic: np.ndarray, timely: np.ndarray,
            reward: np.ndarray, miss: np.ndarray, denominator: float) -> dict:
    risk = k_of_n_risk(probabilities, starts, ends, POLICY_N, 1)
    result = evaluate_test(frame, risk, threshold, starts, ends, septic, timely, reward, miss, denominator)
    labels = frame[TARGET].to_numpy(dtype=np.int8)
    result["raw_probability_pr_auc"] = float(average_precision_score(labels, probabilities))
    result["raw_probability_roc_auc"] = float(roc_auc_score(labels, probabilities))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/last_observation_experiment"))
    parser.add_argument("--trend-bundle", type=Path, default=Path("outputs/trend_experiment/selected_trend_model.joblib"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and adding causal last-value/freshness features...", flush=True)
    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw, include_trends=True, include_last_observation=True)
    patient_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = patient_target.index.to_numpy()
    train_ids, remainder_ids = train_test_split(ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy())
    valid_ids, _unused_test = train_test_split(
        remainder_ids, test_size=0.50, random_state=SEED,
        stratify=patient_target.loc[remainder_ids].to_numpy(),
    )
    train_mask, valid_mask = patient_ids.isin(train_ids).to_numpy(), patient_ids.isin(valid_ids).to_numpy()
    train_patients, valid_patients = patient_ids.loc[train_mask].to_numpy(), patient_ids.loc[valid_mask].to_numpy()
    train_frame = pd.DataFrame({ID_COLUMN: train_patients, TARGET: labels.loc[train_mask].to_numpy()}).reset_index(drop=True)
    valid_frame = pd.DataFrame({ID_COLUMN: valid_patients, TARGET: labels.loc[valid_mask].to_numpy()}).reset_index(drop=True)
    train_starts, train_ends = group_bounds(train_patients)
    valid_starts, valid_ends = group_bounds(valid_patients)
    y_train, y_valid = labels.loc[train_mask].to_numpy(dtype=np.int8), labels.loc[valid_mask].to_numpy(dtype=np.int8)
    _, tr_reward, tr_miss, tr_septic, tr_window, tr_best, tr_inactive = utility_components(train_frame, train_starts, train_ends)
    _, va_reward, va_miss, va_septic, va_window, va_best, va_inactive = utility_components(valid_frame, valid_starts, valid_ends)
    tr_denominator, va_denominator = tr_best - tr_inactive, va_best - va_inactive

    prior = joblib.load(args.trend_bundle)
    prior_model = prior["model"]
    prior_columns = prior["feature_names"]
    prior_threshold = float(prior["threshold"])
    train_x, valid_x = features.loc[train_mask], features.loc[valid_mask]
    prior_train_prob = prior_model.predict_proba(train_x.reindex(columns=prior_columns))[:, 1]
    prior_valid_prob = prior_model.predict_proba(valid_x.reindex(columns=prior_columns))[:, 1]
    prior_valid = metrics(valid_frame, prior_valid_prob, prior_threshold, valid_starts, valid_ends,
                          va_septic, va_window, va_reward, va_miss, va_denominator)
    false_alert_cap = prior_valid["nonsepsis_patient_false_alert_rate"]
    results = [{
        "model": "current_trend_candidate",
        "threshold": prior_threshold,
        "validation_utility": prior_valid["normalized_challenge_utility"],
        "validation_false_alert_rate": false_alert_cap,
        "train_metrics": metrics(train_frame, prior_train_prob, prior_threshold, train_starts, train_ends,
                                  tr_septic, tr_window, tr_reward, tr_miss, tr_denominator),
        "validation_metrics": prior_valid,
        "config": "previous selected candidate; 122 features",
    }]

    configs = {
        "last_observation_strong_regularization": {
            "learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
            "min_samples_leaf": 100, "l2_regularization": 4.0,
        },
        "last_observation_moderate_regularization": {
            "learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
            "min_samples_leaf": 60, "l2_regularization": 3.0,
        },
    }
    fitted = {}
    for name, config in configs.items():
        print(f"Training {name} with {len(features.columns)} features...", flush=True)
        model = HistGradientBoostingClassifier(
            **config, class_weight="balanced", random_state=SEED, early_stopping=False,
        )
        model.fit(train_x, y_train)
        train_prob, valid_prob = model.predict_proba(train_x)[:, 1], model.predict_proba(valid_x)[:, 1]
        selected = choose_threshold(valid_prob, valid_starts, valid_ends, va_septic,
                                    va_reward, va_miss, va_denominator, false_alert_cap)
        row = {
            "model": name,
            "threshold": selected["threshold"],
            "validation_utility": selected["utility"],
            "validation_false_alert_rate": selected["false_rate"],
            "train_metrics": metrics(train_frame, train_prob, selected["threshold"], train_starts, train_ends,
                                      tr_septic, tr_window, tr_reward, tr_miss, tr_denominator),
            "validation_metrics": metrics(valid_frame, valid_prob, selected["threshold"], valid_starts, valid_ends,
                                           va_septic, va_window, va_reward, va_miss, va_denominator),
            "config": config,
        }
        results.append(row)
        fitted[name] = model
        print(f"Finished {name}: validation utility={selected['utility']:.4f}, false-alert rate={selected['false_rate']:.3f}", flush=True)

    candidates = results[1:]
    best_new = max(candidates, key=lambda row: (row["validation_utility"], -row["validation_false_alert_rate"]))
    chosen_model = fitted[best_new["model"]]
    joblib.dump({
        "model": chosen_model,
        "model_name": best_new["model"],
        "feature_names": list(features.columns),
        "threshold": best_new["threshold"],
        "alert_policy": "1_of_5",
        "config": best_new["config"],
        "seed": SEED,
        "target": TARGET,
    }, args.output_dir / "best_last_observation_candidate.joblib", compress=3)

    summary = []
    for row in results:
        tr, va = row["train_metrics"], row["validation_metrics"]
        summary.append({
            "model": row["model"], "threshold": row["threshold"],
            "validation_utility": row["validation_utility"],
            "validation_false_alert_rate": row["validation_false_alert_rate"],
            "validation_warning_window_coverage": va["sepsis_alert_window_coverage"],
            "train_pr_auc": tr["raw_probability_pr_auc"],
            "validation_pr_auc": va["raw_probability_pr_auc"],
            "pr_auc_train_minus_validation": tr["raw_probability_pr_auc"] - va["raw_probability_pr_auc"],
            "train_roc_auc": tr["raw_probability_roc_auc"],
            "validation_roc_auc": va["raw_probability_roc_auc"],
            "roc_auc_train_minus_validation": tr["raw_probability_roc_auc"] - va["raw_probability_roc_auc"],
        })
    table = pd.DataFrame(summary).sort_values(["validation_utility", "validation_false_alert_rate"], ascending=[False, True])
    table.to_csv(args.output_dir / "last_observation_validation.csv", index=False)
    payload = {
        "selection": "Validation utility at a false-alert rate no higher than the existing trend candidate.",
        "false_alert_cap": false_alert_cap,
        "selected_new_candidate": best_new["model"],
        "test_split_used": False,
        "results": results,
    }
    (args.output_dir / "last_observation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nValidation-only results (test split untouched):")
    print(table.to_string(index=False))
    print(f"Saved separate candidate bundle to {args.output_dir / 'best_last_observation_candidate.joblib'}")


if __name__ == "__main__":
    main()
