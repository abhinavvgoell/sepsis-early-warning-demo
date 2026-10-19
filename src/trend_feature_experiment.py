"""Train and compare causal vital-trend features under a fixed alert policy."""

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


THRESHOLDS = np.linspace(0.05, 0.95, 19)
ALERT_POLICY_N = 5
ALERT_POLICY_K = 1


def choose_validation_threshold(probabilities: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                                septic: np.ndarray, reward: np.ndarray, miss: np.ndarray,
                                denominator: float, false_alert_cap: float | None = None) -> tuple[float, dict]:
    risk = k_of_n_risk(probabilities, starts, ends, ALERT_POLICY_N, ALERT_POLICY_K)
    choices = []
    for threshold in THRESHOLDS:
        utility, false_rate = validation_score(risk, float(threshold), starts, septic, reward, miss, denominator)
        choices.append({"threshold": float(threshold), "utility": utility, "false_alert_rate": false_rate})
    feasible = [row for row in choices if false_alert_cap is None or row["false_alert_rate"] <= false_alert_cap + 1e-12]
    if not feasible:
        feasible = choices
    selected = max(feasible, key=lambda row: (row["utility"], -row["false_alert_rate"], row["threshold"]))
    return selected["threshold"], selected


def score_split(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float,
                starts: np.ndarray, ends: np.ndarray, septic: np.ndarray, timely_window: np.ndarray,
                reward: np.ndarray, miss: np.ndarray, denominator: float) -> dict:
    risk = k_of_n_risk(probabilities, starts, ends, ALERT_POLICY_N, ALERT_POLICY_K)
    metrics = evaluate_test(frame, risk, threshold, starts, ends, septic, timely_window,
                            reward, miss, denominator)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    metrics["model_pr_auc_raw_scores"] = float(average_precision_score(y, probabilities))
    metrics["model_roc_auc_raw_scores"] = float(roc_auc_score(y, probabilities))
    metrics["alert_policy"] = "1-of-5 (alert remains active if any of the last five hourly scores cross threshold)"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trend_experiment"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading patient records and building causal features...", flush=True)
    raw = read_patients(args.data_dir)
    expanded, labels, patient_ids = make_features(raw, include_trends=True)
    patient_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = patient_target.index.to_numpy()
    train_ids, remainder_ids = train_test_split(
        ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy()
    )
    remainder_target = patient_target.loc[remainder_ids]
    valid_ids, test_ids = train_test_split(
        remainder_ids, test_size=0.50, random_state=SEED, stratify=remainder_target.to_numpy()
    )
    train_mask = patient_ids.isin(train_ids).to_numpy()
    valid_mask = patient_ids.isin(valid_ids).to_numpy()
    test_mask = patient_ids.isin(test_ids).to_numpy()

    bundle = joblib.load(args.bundle)
    base_model = bundle["models"]["hist_gradient_boosting"]
    base_features = bundle["feature_names"]
    train_x = expanded.loc[train_mask]
    valid_x = expanded.loc[valid_mask]
    test_x = expanded.loc[test_mask]
    y_train = labels.loc[train_mask]
    y_valid = labels.loc[valid_mask]
    y_test = labels.loc[test_mask]
    valid_patients = patient_ids.loc[valid_mask].to_numpy()
    test_patients = patient_ids.loc[test_mask].to_numpy()
    valid_frame = pd.DataFrame({ID_COLUMN: valid_patients, TARGET: y_valid.to_numpy()}).reset_index(drop=True)
    test_frame = pd.DataFrame({ID_COLUMN: test_patients, TARGET: y_test.to_numpy()}).reset_index(drop=True)
    valid_starts, valid_ends = group_bounds(valid_patients)
    test_starts, test_ends = group_bounds(test_patients)
    vy, vreward, vmiss, vseptic, vwindow, vbest, vinactive = utility_components(valid_frame, valid_starts, valid_ends)
    ty, treward, tmiss, tseptic, twindow, tbest, tinactive = utility_components(test_frame, test_starts, test_ends)
    vdenominator, tdenominator = vbest - vinactive, tbest - tinactive

    # Original saved model establishes the validation false-alert budget.
    print("Scoring the existing model to set a fair false-alert cap...", flush=True)
    base_valid_prob = base_model.predict_proba(valid_x.reindex(columns=base_features))[:, 1]
    base_threshold, base_selection = choose_validation_threshold(
        base_valid_prob, valid_starts, valid_ends, vseptic, vreward, vmiss, vdenominator
    )
    false_alert_cap = base_selection["false_alert_rate"]
    base_valid_metrics = score_split(valid_frame, base_valid_prob, base_threshold, valid_starts, valid_ends,
                                     vseptic, vwindow, vreward, vmiss, vdenominator)

    configurations = {
        "trend_default": {
            "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 20,
            "min_samples_leaf": 20, "l2_regularization": 1.0,
        },
        "trend_regularized": {
            "learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
            "min_samples_leaf": 40, "l2_regularization": 2.0,
        },
        "trend_higher_capacity": {
            "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31,
            "min_samples_leaf": 40, "l2_regularization": 2.0,
        },
    }
    candidate_results = [{
        "model": "existing_hist_gradient_boosting",
        "features": "original causal features",
        "threshold": base_threshold,
        "validation_utility": base_selection["utility"],
        "validation_nonsepsis_patient_false_alert_rate": base_selection["false_alert_rate"],
        "validation_metrics": base_valid_metrics,
        "config": "existing saved model",
    }]
    fitted_candidates = {}
    validation_probabilities = {"existing_hist_gradient_boosting": base_valid_prob}
    test_probabilities = {}

    for name, config in configurations.items():
        print(f"Training {name} with {len(expanded.columns)} features...", flush=True)
        model = HistGradientBoostingClassifier(
            **config, class_weight="balanced", random_state=SEED, early_stopping=False,
        )
        model.fit(train_x, y_train)
        valid_prob = model.predict_proba(valid_x)[:, 1]
        threshold, selected = choose_validation_threshold(
            valid_prob, valid_starts, valid_ends, vseptic, vreward, vmiss,
            vdenominator, false_alert_cap=false_alert_cap,
        )
        metrics = score_split(valid_frame, valid_prob, threshold, valid_starts, valid_ends,
                              vseptic, vwindow, vreward, vmiss, vdenominator)
        test_prob = model.predict_proba(test_x)[:, 1]
        candidate_results.append({
            "model": name,
            "features": "original features + current-vs-prior deltas, 1-row deltas, and prior 6-hour standard deviations for 8 vital signs",
            "threshold": threshold,
            "validation_utility": selected["utility"],
            "validation_nonsepsis_patient_false_alert_rate": selected["false_alert_rate"],
            "validation_metrics": metrics,
            "config": config,
        })
        fitted_candidates[name] = model
        validation_probabilities[name] = valid_prob
        test_probabilities[name] = test_prob
        print(f"Finished {name}: validation utility={selected['utility']:.4f}, false-alert rate={selected['false_alert_rate']:.3f}", flush=True)

    improved_candidates = [row for row in candidate_results[1:]
                           if row["validation_nonsepsis_patient_false_alert_rate"] <= false_alert_cap + 1e-12]
    chosen = max(improved_candidates or candidate_results[1:],
                 key=lambda row: (row["validation_utility"], -row["validation_nonsepsis_patient_false_alert_rate"]))
    chosen_name = chosen["model"]
    chosen_threshold = chosen["threshold"]
    print(f"Selected on validation only: {chosen_name} at threshold {chosen_threshold:.2f}", flush=True)

    # Final exploratory comparison on the previously viewed held-out split.
    baseline_test_prob = base_model.predict_proba(test_x.reindex(columns=base_features))[:, 1]
    baseline_test = score_split(test_frame, baseline_test_prob, base_threshold, test_starts, test_ends,
                                tseptic, twindow, treward, tmiss, tdenominator)
    chosen_test_prob = test_probabilities[chosen_name]
    chosen_test = score_split(test_frame, chosen_test_prob, chosen_threshold, test_starts, test_ends,
                              tseptic, twindow, treward, tmiss, tdenominator)
    if chosen_name == "existing_hist_gradient_boosting":
        chosen_model = base_model
        chosen_feature_names = base_features
    else:
        chosen_model = fitted_candidates[chosen_name]
        chosen_feature_names = list(expanded.columns)
    joblib.dump({
        "model": chosen_model,
        "model_name": chosen_name,
        "feature_names": chosen_feature_names,
        "threshold": chosen_threshold,
        "alert_policy": "1_of_5",
        "config": chosen["config"],
        "target": TARGET,
        "seed": SEED,
    }, args.output_dir / "selected_trend_model.joblib", compress=3)

    result = {
        "objective": "Compare causal trend features at a non-sepsis patient false-alert rate no higher than the existing model's validation-selected 1-of-5 policy rate.",
        "alert_policy": "1-of-5 (carry a threshold crossing forward for up to five hourly scores)",
        "validation_false_alert_rate_cap": false_alert_cap,
        "selected_on_validation_only": chosen_name,
        "selected_threshold": chosen_threshold,
        "caveat": "The held-out test split has been explored in prior work. Test results are exploratory and were not used to select the model.",
        "candidates": candidate_results,
        "heldout_comparison": {
            "existing_hist_gradient_boosting": baseline_test,
            "selected_candidate": {"model": chosen_name, **chosen_test},
        },
    }
    (args.output_dir / "trend_feature_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "model": row["model"], "threshold": row["threshold"],
        "validation_utility": row["validation_utility"],
        "validation_nonsepsis_patient_false_alert_rate": row["validation_nonsepsis_patient_false_alert_rate"],
        "validation_model_pr_auc": row["validation_metrics"]["model_pr_auc_raw_scores"],
        "validation_warning_window_coverage": row["validation_metrics"]["sepsis_alert_window_coverage"],
    } for row in candidate_results]).to_csv(args.output_dir / "trend_feature_validation.csv", index=False)
    summary = pd.DataFrame([
        {"model": "existing_hist_gradient_boosting", **baseline_test},
        {"model": chosen_name, **chosen_test},
    ])
    summary.to_csv(args.output_dir / "trend_feature_heldout_comparison.csv", index=False)
    print("\nValidation model selection:")
    print(pd.read_csv(args.output_dir / "trend_feature_validation.csv").to_string(index=False))
    print("\nExploratory held-out comparison (not used for selection):")
    print(summary[["model", "threshold", "model_pr_auc_raw_scores", "accuracy", "sensitivity",
                   "sepsis_alert_window_coverage", "nonsepsis_patient_false_alert_rate",
                   "normalized_challenge_utility"]].to_string(index=False))
    print(f"Saved selected model to {args.output_dir / 'selected_trend_model.joblib'}")


if __name__ == "__main__":
    main()
