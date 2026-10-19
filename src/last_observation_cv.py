"""Patient-grouped cross-validation for current versus last-observation features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components
from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, read_patients


THRESHOLD = 0.65


def evaluate(frame: pd.DataFrame, probability: np.ndarray, starts: np.ndarray, ends: np.ndarray,
             septic: np.ndarray, timely: np.ndarray, reward: np.ndarray, miss: np.ndarray,
             denominator: float) -> dict:
    risk = k_of_n_risk(probability, starts, ends, n=5, k=1)
    result = evaluate_test(frame, risk, THRESHOLD, starts, ends, septic, timely, reward, miss, denominator)
    target = frame[TARGET].to_numpy(dtype=np.int8)
    result["raw_probability_pr_auc"] = float(average_precision_score(target, probability))
    result["raw_probability_roc_auc"] = float(roc_auc_score(target, probability))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/last_observation_experiment"))
    parser.add_argument("--trend-bundle", type=Path, default=Path("outputs/trend_experiment/selected_trend_model.joblib"))
    args = parser.parse_args()

    print("Preparing features for patient-grouped cross-validation...", flush=True)
    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw, include_trends=True, include_last_observation=True)
    patient_target = raw.groupby(ID_COLUMN)[TARGET].max()
    all_ids = patient_target.index.to_numpy()
    original_train_ids, _remainder_ids = train_test_split(
        all_ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy()
    )
    ids = original_train_ids
    outcome = patient_target.loc[ids].to_numpy()
    baseline_bundle = joblib.load(args.trend_bundle)
    baseline_columns = baseline_bundle["feature_names"]

    configs = {
        "current_trend_candidate": {
            "learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
            "min_samples_leaf": 40, "l2_regularization": 2.0,
        },
        "last_observation_candidate": {
            "learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
            "min_samples_leaf": 100, "l2_regularization": 4.0,
        },
    }
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    records = []
    for fold, (train_ix, valid_ix) in enumerate(folds.split(ids, outcome), start=1):
        train_ids, valid_ids = ids[train_ix], ids[valid_ix]
        train_mask = patient_ids.isin(train_ids).to_numpy()
        valid_mask = patient_ids.isin(valid_ids).to_numpy()
        train_patients = patient_ids.loc[train_mask].to_numpy()
        valid_patients = patient_ids.loc[valid_mask].to_numpy()
        train_frame = pd.DataFrame({ID_COLUMN: train_patients, TARGET: labels.loc[train_mask].to_numpy()}).reset_index(drop=True)
        valid_frame = pd.DataFrame({ID_COLUMN: valid_patients, TARGET: labels.loc[valid_mask].to_numpy()}).reset_index(drop=True)
        train_starts, train_ends = group_bounds(train_patients)
        valid_starts, valid_ends = group_bounds(valid_patients)
        _, reward, miss, valid_septic, valid_window, best, inactive = utility_components(valid_frame, valid_starts, valid_ends)
        denominator = best - inactive
        y_train = labels.loc[train_mask].to_numpy(dtype=np.int8)
        x_train, x_valid = features.loc[train_mask], features.loc[valid_mask]

        for name, config in configs.items():
            print(f"Fold {fold}/3: fitting {name}...", flush=True)
            model = HistGradientBoostingClassifier(
                **config, class_weight="balanced", random_state=SEED, early_stopping=False,
            )
            train_features = x_train if name == "last_observation_candidate" else x_train.reindex(columns=baseline_columns)
            valid_features = x_valid if name == "last_observation_candidate" else x_valid.reindex(columns=baseline_columns)
            model.fit(train_features, y_train)
            probabilities = model.predict_proba(valid_features)[:, 1]
            result = evaluate(valid_frame, probabilities, valid_starts, valid_ends, valid_septic,
                              valid_window, reward, miss, denominator)
            records.append({"fold": fold, "model": name, **result})
            print(f"  validation utility={result['normalized_challenge_utility']:.4f}, PR-AUC={result['raw_probability_pr_auc']:.4f}", flush=True)

    table = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "last_observation_cv_folds.csv", index=False)
    summary = table.groupby("model").agg(
        folds=("fold", "count"),
        mean_validation_utility=("normalized_challenge_utility", "mean"),
        sd_validation_utility=("normalized_challenge_utility", "std"),
        mean_validation_pr_auc=("raw_probability_pr_auc", "mean"),
        sd_validation_pr_auc=("raw_probability_pr_auc", "std"),
        mean_validation_roc_auc=("raw_probability_roc_auc", "mean"),
        mean_patient_false_alert_rate=("nonsepsis_patient_false_alert_rate", "mean"),
        mean_warning_window_coverage=("sepsis_alert_window_coverage", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "last_observation_cv_summary.csv", index=False)
    payload = {
        "split": "3-fold stratified by patient-level sepsis status within the original training patients",
        "threshold": THRESHOLD,
        "policy": "1-of-5",
        "test_split_used": False,
        "fold_results": records,
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "last_observation_cv.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nCross-validation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
