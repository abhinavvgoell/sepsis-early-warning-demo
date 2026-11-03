"""Clean, fair final evaluation of frozen LightGBM candidate against HGB baseline.

Enforces:
- Clean patient-level separation with fixed seed.
- Exclusion of the previously examined held-out test cohort.
- Same training patients and same features for both LightGBM and HGB baseline.
- Isotonic calibration and threshold tuning strictly on data separate from the evaluation split.
- Patient-level bootstrap 95% confidence intervals and paired bootstrap comparison.
- Clear documentation of non-nested hyperparameter search and exploratory status.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import warnings
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.alert_policy_grid import group_bounds, k_of_n_risk, utility_components
from src.improved_pipeline import LGB_BASE, Split, expected_calibration_error, make_features_v2
from src.run_pipeline import ID_COLUMN, SEED, TARGET, read_patients

EVAL_SEED = 2026
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_ROUNDS_PR = 200

# Frozen LightGBM configuration from improved_results.md
FROZEN_LGB_PARAMS = {
    "num_leaves": 15,
    "min_child_samples": 600,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "lambda_l2": 10.0,
    "scale_pos_weight": 10.0,
}
FROZEN_LGB_ROUNDS = 233


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_lightgbm(x: pd.DataFrame, y: np.ndarray, rounds: int = FROZEN_LGB_ROUNDS) -> lgb.Booster:
    """Fit LightGBM using the frozen candidate hyperparameters."""
    dtrain = lgb.Dataset(x, y)
    return lgb.train({**LGB_BASE, **FROZEN_LGB_PARAMS}, dtrain, num_boost_round=rounds)


def fit_hgb_baseline(x: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> HistGradientBoostingClassifier:
    """Fit HistGradientBoostingClassifier baseline on the same dataset."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x, y)
    return model


def tune_1_of_5_threshold(
    split: Split,
    probabilities: np.ndarray,
    n: int = 5,
    k: int = 1,
    far_cap: float | None = 0.25,
) -> float:
    """Tune 1-of-5 threshold on calibration/validation split only."""
    risk = split.risk(probabilities, n=n, k=k)
    valid_scores = risk[risk >= 0]
    if len(valid_scores) == 0:
        return 0.5
    grid = np.unique(np.quantile(valid_scores, np.linspace(0.60, 0.9995, 140)))
    grid = grid[np.isfinite(grid)]

    best_thresh = float(grid[0])
    best_util = -999.0
    best_far = 1.0

    for thr in grid:
        util, far = split.utility_and_far(risk, float(thr))
        if far_cap is not None and far > far_cap + 1e-12:
            continue
        if (util, -far) > (best_util, -best_far):
            best_util = util
            best_far = far
            best_thresh = float(thr)

    return best_thresh


def evaluate_split_full(
    split: Split,
    raw_prob: np.ndarray,
    cal_prob: np.ndarray,
    threshold: float,
    n: int = 5,
    k: int = 1,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate full set of metrics once on a split."""
    risk = split.risk(cal_prob, n=n, k=k)
    pred = (risk >= threshold).astype(np.int8)
    y = split.labels

    # Hourly metrics
    acc = float(accuracy_score(y, pred))
    prec = float(precision_score(y, pred, zero_division=0))
    sens = float(recall_score(y, pred, zero_division=0))
    spec = float(np.sum((y == 0) & (pred == 0)) / max(1, np.sum(y == 0)))
    pr_auc = float(average_precision_score(y, raw_prob))
    roc_auc = float(roc_auc_score(y, raw_prob))

    # Patient-level metrics
    patient_any = np.maximum.reduceat(pred, split.starts).astype(bool)
    timely_any = np.maximum.reduceat((pred.astype(bool) & split.window).astype(np.int8), split.starts).astype(bool)

    septic = split.septic
    n_septic = int(septic.sum())
    n_nonseptic = int((~septic).sum())

    timely_septic = int(np.sum(timely_any & septic))
    coverage = float(timely_septic / max(1, n_septic))

    false_nonseptic = int(np.sum(patient_any & ~septic))
    far = float(false_nonseptic / max(1, n_nonseptic))

    utility = float((np.dot(split.reward, pred) + np.dot(split.miss, pred == 0) - split.miss.sum()) / split.denominator)

    metrics = {
        "threshold": float(threshold),
        "normalized_challenge_utility": utility,
        "raw_pr_auc": pr_auc,
        "raw_roc_auc": roc_auc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "sepsis_alert_window_coverage": coverage,
        "nonsepsis_patient_false_alert_rate": far,
        "timely_sepsis_patients": timely_septic,
        "total_sepsis_patients": n_septic,
        "nonsepsis_patients_with_alert": false_nonseptic,
        "total_nonsepsis_patients": n_nonseptic,
        "alert_hours": int(pred.sum()),
        "false_positive_hours": int(np.sum((y == 0) & (pred == 1))),
    }
    return metrics, risk, pred


def patient_level_bootstrap_evaluation(
    split: Split,
    models: dict[str, dict[str, Any]],
    n_boot: int = BOOTSTRAP_ROUNDS,
    n_boot_pr: int = BOOTSTRAP_ROUNDS_PR,
    seed: int = EVAL_SEED,
) -> dict[str, Any]:
    """Calculate 95% bootstrap confidence intervals and paired differences for models."""
    rng = np.random.default_rng(seed)
    n_patients = split.n_patients
    septic = split.septic
    denom_p = split.best_minus_inactive_p

    names = list(models.keys())
    boot_records = {
        name: {
            "utility": [],
            "coverage": [],
            "far": [],
            "pr_auc": [],
            "roc_auc": [],
            "sensitivity": [],
            "specificity": [],
        }
        for name in names
    }

    log(f"Running {n_boot} patient-level bootstrap iterations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            patient_counts = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients).astype(float)
            den = float(np.dot(patient_counts, denom_p))
            ns = float(np.dot(patient_counts, septic))
            nn = float(np.dot(patient_counts, ~septic))

            row_weights = patient_counts[split.row_patient]

            for name in names:
                m = models[name]
                pred = m["pred"]
                y = split.labels

                # Patient-level metrics
                numer_p = m["patient_terms"]["numer"]
                timely_p = m["patient_terms"]["timely"]
                any_p = m["patient_terms"]["any"]

                u = float(np.dot(patient_counts, numer_p) / den) if den != 0 else 0.0
                cov = float(np.dot(patient_counts, timely_p & septic) / ns) if ns != 0 else 0.0
                f = float(np.dot(patient_counts, any_p & ~septic) / nn) if nn != 0 else 0.0

                boot_records[name]["utility"].append(u)
                boot_records[name]["coverage"].append(cov)
                boot_records[name]["far"].append(f)

                # Row-level metrics (weighted by patient cluster frequency)
                tp = float(np.sum(row_weights * (y == 1) * (pred == 1)))
                fn = float(np.sum(row_weights * (y == 1) * (pred == 0)))
                tn = float(np.sum(row_weights * (y == 0) * (pred == 0)))
                fp = float(np.sum(row_weights * (y == 0) * (pred == 1)))

                boot_records[name]["sensitivity"].append(tp / max(1.0, tp + fn))
                boot_records[name]["specificity"].append(tn / max(1.0, tn + fp))

                if b < n_boot_pr:
                    boot_records[name]["pr_auc"].append(
                        float(average_precision_score(y, m["raw_prob"], sample_weight=row_weights))
                    )
                    boot_records[name]["roc_auc"].append(
                        float(roc_auc_score(y, m["raw_prob"], sample_weight=row_weights))
                    )

    results: dict[str, Any] = {}
    for name in names:
        results[name] = {}
        for metric, values in boot_records[name].items():
            if len(values) > 0:
                results[name][metric] = {
                    "mean": float(np.mean(values)),
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                }

    # Paired bootstrap comparison: LightGBM vs Baseline
    if "lightgbm" in names and "baseline_hgb" in names:
        paired: dict[str, Any] = {}
        lgb_dict = boot_records["lightgbm"]
        hgb_dict = boot_records["baseline_hgb"]

        for metric in ["utility", "pr_auc", "roc_auc", "coverage", "far"]:
            lgb_vals = np.array(lgb_dict[metric])
            hgb_vals = np.array(hgb_dict[metric])
            min_len = min(len(lgb_vals), len(hgb_vals))
            diff = lgb_vals[:min_len] - hgb_vals[:min_len]

            paired[f"diff_{metric}_(lightgbm_minus_hgb)"] = {
                "mean": float(np.mean(diff)),
                "ci_lower": float(np.percentile(diff, 2.5)),
                "ci_upper": float(np.percentile(diff, 97.5)),
                "prob_lightgbm_superior": float(np.mean(diff > 0)),
            }
        results["paired_comparison"] = paired

    return results


def run_clean_evaluation(
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    max_patients: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "features.joblib"
    if cache_file.exists():
        log(f"Loading cached features from {cache_file}")
        cached = joblib.load(cache_file)
        features, labels, ids = cached["v2"], cached["labels"], cached["ids"]
    else:
        log(f"Reading raw patient data from {data_dir}...")
        raw = read_patients(data_dir)
        raw = raw.sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
        if max_patients is not None:
            sampled_ids = raw[ID_COLUMN].drop_duplicates().head(max_patients)
            raw = raw[raw[ID_COLUMN].isin(sampled_ids)].reset_index(drop=True)
        labels = raw[TARGET].to_numpy(np.int8)
        ids = raw[ID_COLUMN].to_numpy()
        log(f"Generating 200 causal features (make_features_v2)...")
        features = make_features_v2(raw)
        joblib.dump({"v2": features, "labels": labels, "ids": ids}, cache_file)

    log(f"Loaded {len(features):,} rows across {len(np.unique(ids)):,} patients with {features.shape[1]} features.")

    patient_targets = pd.Series(labels).groupby(ids).max()
    all_patient_ids = patient_targets.index.to_numpy()

    # Step 1: Reconstruct the old held-out test split from run_pipeline.py (SEED=42) and EXCLUDE it
    _, old_rest = train_test_split(
        all_patient_ids, test_size=0.30, random_state=SEED, stratify=patient_targets.to_numpy()
    )
    _, old_test_ids = train_test_split(
        old_rest, test_size=0.50, random_state=SEED, stratify=patient_targets.loc[old_rest].to_numpy()
    )
    old_test_set = set(old_test_ids)

    # Eligible pool: development patients only (strictly excluding old test split)
    eligible_ids = np.array([pid for pid in all_patient_ids if pid not in old_test_set])
    log(f"Excluded {len(old_test_ids):,} old test patients. Remaining clean pool: {len(eligible_ids):,} patients.")

    eligible_targets = patient_targets.loc[eligible_ids].to_numpy()

    # Step 2: Create a NEW patient-level split with fixed EVAL_SEED
    # 70% Train, 15% Calibration/Threshold-Tuning, 15% Final Held-Out Evaluation
    train_ids, eval_rest = train_test_split(
        eligible_ids, test_size=0.30, random_state=EVAL_SEED, stratify=eligible_targets
    )
    eval_rest_targets = patient_targets.loc[eval_rest].to_numpy()
    calib_ids, test_ids = train_test_split(
        eval_rest, test_size=0.50, random_state=EVAL_SEED, stratify=eval_rest_targets
    )

    log(f"New patient split: {len(train_ids):,} train | {len(calib_ids):,} calib/threshold | {len(test_ids):,} final held-out test.")

    mask_train = np.isin(ids, train_ids)
    mask_calib = np.isin(ids, calib_ids)
    mask_test = np.isin(ids, test_ids)

    x_train, y_train = features[mask_train], labels[mask_train]
    x_calib, y_calib = features[mask_calib], labels[mask_calib]
    x_test, y_test = features[mask_test], labels[mask_test]

    split_calib = Split(ids[mask_calib], y_calib)
    split_test = Split(ids[mask_test], y_test)

    # Step 3: Train LightGBM AND HGB Baseline on the SAME training patients with SAME features
    log("Fitting LightGBM on training partition...")
    lgb_model = fit_lightgbm(x_train, y_train, rounds=FROZEN_LGB_ROUNDS)

    log("Fitting Histogram Gradient Boosting baseline on SAME training partition...")
    hgb_model = fit_hgb_baseline(x_train, y_train, seed=EVAL_SEED)

    # Predict raw scores
    lgb_calib_raw = lgb_model.predict(x_calib)
    lgb_test_raw = lgb_model.predict(x_test)

    hgb_calib_raw = hgb_model.predict_proba(x_calib)[:, 1]
    hgb_test_raw = hgb_model.predict_proba(x_test)[:, 1]

    # Step 4: Fit Isotonic Calibration ONLY on separate calibration partition
    log("Fitting Isotonic Calibration on separate calibration partition...")
    iso_lgb = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(lgb_calib_raw, y_calib)
    iso_hgb = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(hgb_calib_raw, y_calib)

    lgb_calib_cal = iso_lgb.predict(lgb_calib_raw)
    lgb_test_cal = iso_lgb.predict(lgb_test_raw)

    hgb_calib_cal = iso_hgb.predict(hgb_calib_raw)
    hgb_test_cal = iso_hgb.predict(hgb_test_raw)

    # Step 5: Pick 1-of-5 Alert Threshold ONLY on separate calibration partition
    log("Tuning 1-of-5 alert threshold on separate calibration partition...")
    lgb_threshold = tune_1_of_5_threshold(split_calib, lgb_calib_cal, n=5, k=1, far_cap=0.25)
    hgb_threshold = tune_1_of_5_threshold(split_calib, hgb_calib_cal, n=5, k=1, far_cap=0.25)

    log(f"Selected 1-of-5 thresholds on calibration split: LightGBM = {lgb_threshold:.4f}, HGB = {hgb_threshold:.4f}")

    # Step 6: Evaluate ONCE on the Final Held-Out Evaluation Split
    log("Evaluating once on final held-out evaluation split...")
    lgb_metrics, lgb_risk, lgb_pred = evaluate_split_full(
        split_test, lgb_test_raw, lgb_test_cal, lgb_threshold, n=5, k=1
    )
    hgb_metrics, hgb_risk, hgb_pred = evaluate_split_full(
        split_test, hgb_test_raw, hgb_test_cal, hgb_threshold, n=5, k=1
    )

    # Step 7: Bootstrap 95% Confidence Intervals & Paired Comparison
    bootstrap_models = {
        "lightgbm": {
            "pred": lgb_pred,
            "raw_prob": lgb_test_raw,
            "patient_terms": split_test.patient_terms(lgb_risk, lgb_threshold),
        },
        "baseline_hgb": {
            "pred": hgb_pred,
            "raw_prob": hgb_test_raw,
            "patient_terms": split_test.patient_terms(hgb_risk, hgb_threshold),
        },
    }

    bootstrap_results = patient_level_bootstrap_evaluation(
        split_test, bootstrap_models, n_boot=BOOTSTRAP_ROUNDS, n_boot_pr=BOOTSTRAP_ROUNDS_PR, seed=EVAL_SEED
    )

    # Save predictions for Alert Policy V2
    joblib.dump(
        {
            "frame": pd.DataFrame({ID_COLUMN: ids[mask_calib], TARGET: y_calib}),
            "lightgbm_calibrated_prob": lgb_calib_cal,
            "baseline_calibrated_prob": hgb_calib_cal,
        },
        output_dir / "oof_predictions.joblib",
    )
    joblib.dump(
        {
            "frame": pd.DataFrame({ID_COLUMN: ids[mask_test], TARGET: y_test}),
            "lightgbm_calibrated_prob": lgb_test_cal,
            "baseline_calibrated_prob": hgb_test_cal,
        },
        output_dir / "heldout_predictions.joblib",
    )

    # Compile JSON results
    results_summary = {
        "evaluation_name": "Part 1 Clean Fair Evaluation",
        "cohort_notes": (
            "Untouched data status: The public PhysioNet 2019 release contains 40,336 patients across sets A and B (set C is private). "
            "Because all public patients were previously part of development or testing, there is no genuinely untouched external data. "
            "To maximize fairness, this evaluation strictly excluded the 6,051 previously examined test patients and drew a new "
            "patient-level split (70% train, 15% calibration/threshold tuning, 15% final held-out test) using EVAL_SEED=2026. "
            "Results are marked as exploratory."
        ),
        "hyperparameter_search_nested": False,
        "hyperparameter_search_disclosure": (
            "The hyperparameter search was NOT nested. LightGBM hyperparameters (15 leaves, min child 600, feature fraction 0.5, "
            "bagging 0.7, L2 10.0, pos weight 10.0) were searched on the overall development cohort in prior experiments. "
            "Therefore, reported discrimination scores carry potential selection bias and do not represent an unbiased estimate "
            "of prospective clinical performance."
        ),
        "split_counts": {
            "train_patients": len(train_ids),
            "calib_patients": len(calib_ids),
            "test_patients": len(test_ids),
            "excluded_old_test_patients": len(old_test_ids),
        },
        "lightgbm_metrics": lgb_metrics,
        "baseline_hgb_metrics": hgb_metrics,
        "bootstrap": bootstrap_results,
    }

    (output_dir / "final_evaluation_results.json").write_text(
        json.dumps(results_summary, indent=2, default=float), encoding="utf-8"
    )

    # Generate Markdown Report
    paired = bootstrap_results.get("paired_comparison", {})
    diff_u = paired.get("diff_utility_(lightgbm_minus_hgb)", {})
    diff_pr = paired.get("diff_pr_auc_(lightgbm_minus_hgb)", {})

    md_lines = [
        "# Clean, Fair Evaluation: Frozen LightGBM vs. HGB Baseline",
        "",
        "## Evaluation Setup and Untouched Data Status",
        "",
        "This evaluation addresses the primary methodological limitation of previous iterations: repeated examination of the original test partition and mismatched training sets.",
        "",
        "- **Untouched Data Status**: The public PhysioNet/CinC 2019 Challenge dataset contains 40,336 patients in sets A and B; challenge test set C remains private. There is **no genuinely untouched patient data** in the public release. All results are therefore marked as **exploratory research prototype findings**, not clinical validation.",
        "- **Fair Split Protocol**: To ensure rigorous evaluation within the available data, the 6,051 patients from the previously examined held-out test split were **strictly excluded**. From the remaining 34,285 patients, a new patient-level split was established with fixed seed `EVAL_SEED=2026`:",
        f"  - **Training Split**: {len(train_ids):,} patients",
        f"  - **Calibration & Threshold Tuning Split**: {len(calib_ids):,} patients (strictly separate)",
        f"  - **Final Held-Out Evaluation Split**: {len(test_ids):,} patients (evaluated exactly once)",
        "- **Matched Training Conditions**: Both LightGBM and the Histogram Gradient Boosting (HGB) baseline were trained on the **identical training cohort** with the **same 200 causal features**.",
        "- **Strict Leakage Prevention**: Isotonic probability calibration and 1-of-5 alert threshold selection were fitted strictly on the separate calibration split and applied without adjustment to the held-out split.",
        "",
        "## Disclosure on Hyperparameter Search (Nested vs. Non-Nested)",
        "",
        "> [!IMPORTANT]",
        "> **The hyperparameter search was NOT nested.** The LightGBM configuration (`num_leaves=15`, `min_child_samples=600`, `feature_fraction=0.5`, `bagging_fraction=0.7`, `lambda_l2=10.0`, `scale_pos_weight=10.0`, 233 rounds) was searched globally on the development cohort prior to this evaluation. Consequently, performance estimates reflect an upper-bound configuration selection and cannot be claimed as an unbiased generalization guarantee. True prospective validation requires testing on an entirely independent external clinical cohort.",
        "",
        "## One-Time Held-Out Evaluation Results",
        "",
        "| Evaluation Metric | Baseline HGB | Frozen LightGBM | Difference (LightGBM − HGB) | 95% Bootstrap CI (Diff) |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| **Normalized Utility** | {hgb_metrics['normalized_challenge_utility']:.4f} | **{lgb_metrics['normalized_challenge_utility']:.4f}** | {diff_u.get('mean', 0.0):+.4f} | [{diff_u.get('ci_lower', 0.0):+.4f}, {diff_u.get('ci_upper', 0.0):+.4f}] |",
        f"| **PR-AUC (raw scores)** | {hgb_metrics['raw_pr_auc']:.4f} | **{lgb_metrics['raw_pr_auc']:.4f}** | {diff_pr.get('mean', 0.0):+.4f} | [{diff_pr.get('ci_lower', 0.0):+.4f}, {diff_pr.get('ci_upper', 0.0):+.4f}] |",
        f"| **ROC-AUC (raw scores)** | {hgb_metrics['raw_roc_auc']:.4f} | **{lgb_metrics['raw_roc_auc']:.4f}** | — | — |",
        f"| **Warning-Window Coverage** | {hgb_metrics['sepsis_alert_window_coverage']*100:.1f}% | **{lgb_metrics['sepsis_alert_window_coverage']*100:.1f}%** | — | — |",
        f"| **Non-Sepsis Alert Rate (FAR)** | {hgb_metrics['nonsepsis_patient_false_alert_rate']*100:.1f}% | **{lgb_metrics['nonsepsis_patient_false_alert_rate']*100:.1f}%** | — | — |",
        f"| **Hourly Sensitivity** | {hgb_metrics['sensitivity']*100:.1f}% | **{lgb_metrics['sensitivity']*100:.1f}%** | — | — |",
        f"| **Hourly Specificity** | {hgb_metrics['specificity']*100:.1f}% | **{lgb_metrics['specificity']*100:.1f}%** | — | — |",
        "",
        "### Bootstrap Confidence Intervals (Patient-Level 95% CI)",
        "",
        "| Model | Metric | Mean | 95% Confidence Interval |",
        "| :--- | :--- | ---: | :---: |",
    ]

    for model_name, label in [("baseline_hgb", "Baseline HGB"), ("lightgbm", "Frozen LightGBM")]:
        b_dict = bootstrap_results.get(model_name, {})
        for metric in ["utility", "pr_auc", "roc_auc", "coverage", "far", "sensitivity", "specificity"]:
            if metric in b_dict:
                m_info = b_dict[metric]
                md_lines.append(f"| {label} | {metric} | {m_info['mean']:.4f} | [{m_info['ci_lower']:.4f}, {m_info['ci_upper']:.4f}] |")

    md_lines.extend([
        "",
        "## Statistical Findings",
        "",
        f"- **Paired Utility Difference**: Mean difference is `{diff_u.get('mean', 0.0):+.4f}` with 95% bootstrap interval `[{diff_u.get('ci_lower', 0.0):+.4f}, {diff_u.get('ci_upper', 0.0):+.4f}]`. "
        f"Probability of superiority $P(\\Delta > 0) = {diff_u.get('prob_lightgbm_superior', 0.0)*100:.1f}\\%$.",
        f"- **Paired PR-AUC Difference**: Mean difference is `{diff_pr.get('mean', 0.0):+.4f}` with 95% bootstrap interval `[{diff_pr.get('ci_lower', 0.0):+.4f}, {diff_pr.get('ci_upper', 0.0):+.4f}]`. "
        f"Probability of superiority $P(\\Delta > 0) = {diff_pr.get('prob_lightgbm_superior', 0.0)*100:.1f}\\%$.",
        "",
        "## Research Prototype Statement",
        "",
        "This project is a research prototype developed for educational and experimental analysis of time-series ICU data. The models, scores, and alert policies have not been clinically validated, do not produce medical diagnostic probabilities, and must not be used for patient management or care decisions.",
    ])

    (output_dir / "final_evaluation_results.md").write_text("\n".join(md_lines), encoding="utf-8")
    log("Final evaluation report and artifacts written successfully.")
    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_evaluation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/final_evaluation/cache"))
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    run_clean_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
