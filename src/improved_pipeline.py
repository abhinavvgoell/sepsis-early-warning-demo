"""Improved sepsis pipeline: richer causal features, tuned LightGBM, grouped CV,
calibration, utility-optimal alert policy, bootstrap CIs and a cross-hospital check.

Everything is causal: a feature at hour t uses rows <= t of the same patient only.
The patient-level train/validation/test split is identical to ``run_pipeline.py`` so
results are directly comparable with the existing model.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components
from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, read_patients

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
WORST_MAX = ["HR", "Resp", "Temp", "Lactate", "Creatinine", "Bilirubin_total", "WBC"]
WORST_MIN = ["O2Sat", "Temp", "SBP", "MAP", "Platelets"]
THREADS = 12


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- features
def make_features_v2(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal features: LOCF values, time-since-measured, windows, deltas, composites."""
    ids = raw[ID_COLUMN]
    hour = raw["hour_index"].to_numpy(np.float32)
    meas = [c for c in VITALS + LABS if c in raw]
    static = [c for c in STATIC if c in raw]
    cur = raw[meas].astype("float32")
    parts: dict[str, pd.DataFrame] = {"static": raw[static].astype("float32"), "cur": cur}

    locf = cur.groupby(ids, sort=False).ffill()
    parts["locf"] = locf.add_suffix("_locf")

    last_obs = pd.DataFrame(np.where(cur.notna().to_numpy(), hour[:, None], np.nan).astype("float32"),
                            columns=meas, index=cur.index).groupby(ids, sort=False).ffill()
    parts["since"] = (hour[:, None] - last_obs).add_suffix("_hours_since")

    lg = locf.groupby(ids, sort=False)
    parts["delta6"] = (locf - lg.shift(6)).add_suffix("_delta_6h")
    vit = [c for c in VITALS if c in locf]
    parts["delta1"] = (locf[vit] - lg[vit].shift(1)).add_suffix("_delta_1h")

    def rolled(window: int, how: str, min_periods: int = 1) -> pd.DataFrame:
        r = locf[vit].groupby(ids, sort=False).rolling(window, min_periods=min_periods)
        out = getattr(r, how)().reset_index(level=0, drop=True).sort_index()
        return out.astype("float32").add_suffix(f"_{how}_{window}h")

    parts["mean6"] = rolled(6, "mean")
    parts["std6"] = rolled(6, "std", 2)
    parts["max12"] = rolled(12, "max")
    parts["min12"] = rolled(12, "min")

    cum = {}
    for c in WORST_MAX:
        if c in locf:
            cum[f"{c}_max_so_far"] = lg[c].cummax()
    for c in WORST_MIN:
        if c in locf:
            cum[f"{c}_min_so_far"] = lg[c].cummin()
    parts["cum"] = pd.DataFrame(cum, index=raw.index).astype("float32")

    comp = pd.DataFrame(index=raw.index)
    l = locf
    if {"HR", "SBP"} <= set(l):
        comp["shock_index"] = l["HR"] / l["SBP"]
    sirs = ((l["Temp"] > 38) | (l["Temp"] < 36)).astype("float32") if "Temp" in l else 0
    sirs = sirs + (l["HR"] > 90) + (l["Resp"] > 20) + ((l["WBC"] > 12) | (l["WBC"] < 4))
    comp["sirs_count"] = sirs.astype("float32")
    comp["qsofa_like"] = ((l["Resp"] >= 22).astype("float32") + (l["SBP"] <= 100)).astype("float32")
    comp["map_lt_65"] = (l["MAP"] < 65).astype("float32").where(l["MAP"].notna())
    labs_present = [c for c in LABS if c in cur]
    comp["n_labs_so_far"] = cur[labs_present].notna().sum(axis=1).groupby(ids, sort=False).cumsum().astype("float32")
    comp["n_vitals_now"] = cur[vit].notna().sum(axis=1).astype("float32")
    parts["comp"] = comp.astype("float32")

    features = pd.concat(parts.values(), axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).astype("float32")
    return features


# --------------------------------------------------------------------------- evaluation
class Split:
    """Per-patient utility bookkeeping for a block of patients stored contiguously."""

    def __init__(self, patient_ids: np.ndarray, labels: np.ndarray):
        self.patient_ids = patient_ids
        self.labels = labels.astype(np.int8)
        self.frame = pd.DataFrame({ID_COLUMN: patient_ids, TARGET: self.labels})
        self.starts, self.ends = group_bounds(patient_ids)
        _, self.reward, self.miss, self.septic, self.window, best, inactive = utility_components(
            self.frame, self.starts, self.ends)
        self.denominator = best - inactive
        self.row_patient = np.repeat(np.arange(len(self.starts)), self.ends - self.starts)
        gain = self.reward - self.miss
        self.gain_p_fn = lambda pred: np.add.reduceat(gain * pred, self.starts)
        self.best_minus_inactive_p = (np.add.reduceat(self.reward * self.window, self.starts)
                                      - np.add.reduceat(self.miss, self.starts))
        self.n_patients = len(self.starts)

    def risk(self, prob: np.ndarray, n: int, k: int) -> np.ndarray:
        return k_of_n_risk(prob.astype(np.float32), self.starts, self.ends, n, k)

    def metrics(self, risk: np.ndarray, threshold: float) -> dict:
        return evaluate_test(self.frame, risk, threshold, self.starts, self.ends, self.septic,
                             self.window, self.reward, self.miss, self.denominator)

    def utility_and_far(self, risk: np.ndarray, threshold: float) -> tuple[float, float]:
        pred = risk >= threshold
        utility = float(np.dot(self.reward - self.miss, pred) / self.denominator)
        any_alert = np.maximum.reduceat(pred.astype(np.int8), self.starts).astype(bool)
        far = float(any_alert[~self.septic].mean()) if (~self.septic).any() else 0.0
        return utility, far

    def patient_terms(self, risk: np.ndarray, threshold: float) -> dict:
        pred = risk >= threshold
        return {
            "numer": self.gain_p_fn(pred.astype(np.float64)),
            "any": np.maximum.reduceat(pred.astype(np.int8), self.starts).astype(bool),
            "timely": np.maximum.reduceat((pred & self.window).astype(np.int8), self.starts).astype(bool),
        }


def threshold_grid(prob: np.ndarray) -> np.ndarray:
    grid = np.unique(np.quantile(prob, np.linspace(0.80, 0.9995, 140)))
    return grid[np.isfinite(grid)]


def search_policy(split: Split, prob: np.ndarray, policies: list[tuple[int, int]], far_cap: float | None):
    """Return the best (n, k, threshold) by utility, optionally under a false-alert-rate cap."""
    grid = threshold_grid(prob)
    best, table = None, []
    for n, k in policies:
        risk = split.risk(prob, n, k)
        for threshold in grid:
            utility, far = split.utility_and_far(risk, float(threshold))
            if far_cap is not None and far > far_cap + 1e-12:
                continue
            row = {"n": n, "k": k, "threshold": float(threshold), "utility": utility, "far": far}
            table.append(row)
            if best is None or (utility, -far) > (best["utility"], -best["far"]):
                best = row
    return best, pd.DataFrame(table)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def paired_bootstrap(split: Split, models: dict[str, dict], raw_probs: dict[str, np.ndarray],
                     n_boot: int, n_boot_pr: int, seed: int = SEED) -> dict:
    """Patient-level paired bootstrap of utility, coverage, false-alert rate and PR-AUC."""
    rng = np.random.default_rng(seed)
    septic = split.septic
    denom_p = split.best_minus_inactive_p
    names = list(models)
    out = {n: {"utility": [], "coverage": [], "far": [], "pr_auc": []} for n in names}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, split.n_patients, split.n_patients), minlength=split.n_patients).astype(float)
        den = float(np.dot(counts, denom_p))
        ns, nn = float(np.dot(counts, septic)), float(np.dot(counts, ~septic))
        for name in names:
            t = models[name]
            out[name]["utility"].append(float(np.dot(counts, t["numer"]) / den))
            out[name]["coverage"].append(float(np.dot(counts, t["timely"] & septic) / ns))
            out[name]["far"].append(float(np.dot(counts, t["any"] & ~septic) / nn))
        if b < n_boot_pr:
            row_w = counts[split.row_patient]
            for name in names:
                out[name]["pr_auc"].append(float(average_precision_score(split.labels, raw_probs[name], sample_weight=row_w)))
    summary = {}
    for name in names:
        summary[name] = {m: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
                         for m, v in out[name].items() if v}
    if len(names) == 2:
        a, b = names
        summary["difference_" + b + "_minus_" + a] = {
            m: {"mean": float(np.mean(np.array(out[b][m]) - np.array(out[a][m][:len(out[b][m])]))),
                "lo": float(np.percentile(np.array(out[b][m]) - np.array(out[a][m][:len(out[b][m])]), 2.5)),
                "hi": float(np.percentile(np.array(out[b][m]) - np.array(out[a][m][:len(out[b][m])]), 97.5)),
                "p_improvement": float(np.mean((np.array(out[b][m]) - np.array(out[a][m][:len(out[b][m])])) > 0))}
            for m in ("utility", "coverage", "far", "pr_auc") if out[a][m]}
    return summary


# --------------------------------------------------------------------------- modelling
LGB_BASE = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05,
            "verbosity": -1, "num_threads": THREADS, "seed": SEED, "bagging_freq": 1,
            "max_bin": 255, "deterministic": True}


def random_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = [{"num_leaves": 31, "min_child_samples": 100, "feature_fraction": 0.5,
                "bagging_fraction": 0.8, "lambda_l2": 10.0, "scale_pos_weight": 1.0}]
    while len(configs) < n:
        configs.append({
            "num_leaves": int(rng.choice([15, 31, 63, 127])),
            "min_child_samples": int(rng.choice([50, 100, 300, 600])),
            "feature_fraction": float(rng.choice([0.3, 0.5, 0.7])),
            "bagging_fraction": float(rng.choice([0.7, 0.9])),
            "lambda_l2": float(rng.choice([1.0, 10.0, 50.0])),
            "scale_pos_weight": float(rng.choice([1.0, 3.0, 10.0])),
        })
    return configs


def fit_lgb(params: dict, x: pd.DataFrame, y: np.ndarray, rounds: int) -> lgb.Booster:
    return lgb.train({**LGB_BASE, **params}, lgb.Dataset(x, y), num_boost_round=rounds)


def hgb_original() -> HistGradientBoostingClassifier:
    """The original project model configuration (v1 features, balanced class weights)."""
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150, max_leaf_nodes=20,
                                          l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/improved"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/improved/cache"))
    parser.add_argument("--baseline-bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    parser.add_argument("--configs", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=1500)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-pr", type=int, default=150)
    parser.add_argument("--skip-hospital", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"seed": SEED}

    # ---- data and features (cached) ------------------------------------------------
    cache = args.cache_dir / "features.joblib"
    if cache.exists():
        log(f"loading cached features from {cache}")
        cached = joblib.load(cache)
        v2, v1, labels, ids = cached["v2"], cached["v1"], cached["labels"], cached["ids"]
    else:
        log("reading raw patient files")
        raw = read_patients(args.data_dir)
        raw = raw.sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
        labels = raw[TARGET].to_numpy(np.int8)
        ids = raw[ID_COLUMN].to_numpy()
        log("building v2 features")
        v2 = make_features_v2(raw)
        log("building v1 (original) features")
        v1_feat, _, _ = make_features(raw)
        v1 = v1_feat.reset_index(drop=True)
        joblib.dump({"v2": v2, "v1": v1, "labels": labels, "ids": ids}, cache)
    log(f"rows={len(v2):,} v2_features={v2.shape[1]} v1_features={v1.shape[1]}")
    result["n_features_v2"], result["n_features_v1"] = int(v2.shape[1]), int(v1.shape[1])

    patient_target = pd.Series(labels).groupby(ids).max()
    all_ids = patient_target.index.to_numpy()
    train_ids, rest = train_test_split(all_ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy())
    valid_ids, test_ids = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=patient_target.loc[rest].to_numpy())
    dev_ids = np.concatenate([train_ids, valid_ids])
    m_train, m_valid, m_test = (np.isin(ids, s) for s in (train_ids, valid_ids, test_ids))
    m_dev = m_train | m_valid

    def block(mask: np.ndarray) -> Split:
        return Split(ids[mask], labels[mask])

    s_train, s_valid, s_test, s_dev = block(m_train), block(m_valid), block(m_test), block(m_dev)
    y_train, y_valid, y_test, y_dev = labels[m_train], labels[m_valid], labels[m_test], labels[m_dev]
    x_train, x_valid, x_test, x_dev = v2[m_train], v2[m_valid], v2[m_test], v2[m_dev]

    # ---- baseline (existing saved model, original features, 1-of-5 @ its threshold) ---
    log("scoring existing baseline model on the test patients")
    try:
        bundle = joblib.load(args.baseline_bundle)
        base_model = bundle["models"]["hist_gradient_boosting"]
        base_test_prob = base_model.predict_proba(v1[m_test].astype("float64").reindex(columns=bundle["feature_names"]))[:, 1]
        base_thr = 0.65
        result["baseline_source"] = "saved bundle, threshold 0.65"
    except Exception as exc:  # saved pickle written by a different scikit-learn build
        log(f"saved baseline unusable ({exc!r}); retraining the original configuration on the train split")
        base_model = hgb_original().fit(v1[m_train], y_train)
        valid_p = base_model.predict_proba(v1[m_valid])[:, 1]
        sel, _ = search_policy(s_valid, valid_p, [(5, 1)], None)
        base_thr = sel["threshold"]
        base_test_prob = base_model.predict_proba(v1[m_test])[:, 1]
        result["baseline_source"] = f"retrained original config, 1-of-5 validation-utility threshold {base_thr:.4f}"
    base_risk = s_test.risk(base_test_prob, 5, 1)
    base_metrics = s_test.metrics(base_risk, base_thr)
    base_metrics["raw_pr_auc"] = float(average_precision_score(y_test, base_test_prob))
    base_metrics["raw_roc_auc"] = float(roc_auc_score(y_test, base_test_prob))
    result["baseline_test"] = base_metrics
    log(f"baseline test: utility={base_metrics['normalized_challenge_utility']:.4f} "
        f"PR-AUC={base_metrics['raw_pr_auc']:.4f} FAR={base_metrics['nonsepsis_patient_false_alert_rate']:.3f}")

    # ---- hyper-parameter search on train -> valid -----------------------------------
    d_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    d_valid = lgb.Dataset(x_valid, y_valid, reference=d_train, free_raw_data=False)
    search = []
    search_cache = args.cache_dir / "search.csv"
    configs = random_configs(args.configs, SEED) if not search_cache.exists() else []
    if search_cache.exists():
        log("loading cached hyperparameter search")
    for i, cfg in enumerate(configs):
        t0 = time.time()
        booster = lgb.train({**LGB_BASE, **cfg}, d_train, num_boost_round=args.max_rounds, valid_sets=[d_valid],
                            callbacks=[lgb.early_stopping(75, verbose=False)])
        p = booster.predict(x_valid, num_iteration=booster.best_iteration)
        row = {"config": i, **cfg, "best_iteration": int(booster.best_iteration),
               "valid_pr_auc": float(average_precision_score(y_valid, p)),
               "valid_roc_auc": float(roc_auc_score(y_valid, p)), "seconds": round(time.time() - t0, 1)}
        search.append(row)
        log(f"config {i}: PR-AUC={row['valid_pr_auc']:.4f} ROC={row['valid_roc_auc']:.4f} "
            f"iters={row['best_iteration']} ({row['seconds']}s) {cfg}")
    if search_cache.exists():
        search_df = pd.read_csv(search_cache)
    else:
        search_df = pd.DataFrame(search)
        search_df.to_csv(search_cache, index=False)
    search_df = search_df.sort_values("valid_pr_auc", ascending=False)
    search_df.to_csv(args.output_dir / "hyperparameter_search.csv", index=False)
    best = search_df.iloc[0]
    best_params = {k: (int(best[k]) if k in ("num_leaves", "min_child_samples") else float(best[k]))
                   for k in ("num_leaves", "min_child_samples", "feature_fraction", "bagging_fraction",
                             "lambda_l2", "scale_pos_weight")}
    rounds = max(50, int(round(best["best_iteration"] * 1.1)))  # a little more data in CV/final fits
    result["best_params"], result["rounds"] = best_params, rounds
    result["search_best_valid_pr_auc"] = float(best["valid_pr_auc"])

    # ---- grouped (patient-level) CV on train+valid -> OOF probabilities ---------------
    log(f"{args.cv_folds}-fold patient-level CV on development patients")
    dev_patient_ids = np.unique(ids[m_dev])
    dev_patient_y = patient_target.loc[dev_patient_ids].to_numpy()
    oof = np.zeros(m_dev.sum())
    dev_row_ids = ids[m_dev]
    cv_rows = []
    skf = StratifiedKFold(args.cv_folds, shuffle=True, random_state=SEED)
    oof_cache = args.cache_dir / "oof.npy"
    cached_oof = np.load(oof_cache) if oof_cache.exists() else None
    for fold, (_, hold_idx) in enumerate(skf.split(dev_patient_ids, dev_patient_y)):
        hold = np.isin(dev_row_ids, dev_patient_ids[hold_idx])
        if cached_oof is not None:
            oof[hold] = cached_oof[hold]
        else:
            booster = fit_lgb(best_params, x_dev[~hold], y_dev[~hold], rounds)
            oof[hold] = booster.predict(x_dev[hold])
        cv_rows.append({"fold": fold, "pr_auc": float(average_precision_score(y_dev[hold], oof[hold])),
                        "roc_auc": float(roc_auc_score(y_dev[hold], oof[hold]))})
        log(f"fold {fold}: PR-AUC={cv_rows[-1]['pr_auc']:.4f} ROC={cv_rows[-1]['roc_auc']:.4f}")
    if cached_oof is None:
        np.save(oof_cache, oof)
    cv_df = pd.DataFrame(cv_rows)
    result["cv"] = {"folds": cv_rows, "pr_auc_mean": float(cv_df.pr_auc.mean()), "pr_auc_std": float(cv_df.pr_auc.std()),
                    "roc_auc_mean": float(cv_df.roc_auc.mean()), "oof_pr_auc": float(average_precision_score(y_dev, oof))}

    # ---- calibration + policy / threshold selection on OOF ---------------------------
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y_dev)
    oof_cal = iso.predict(oof)
    result["oof_calibration"] = {"brier_raw": float(brier_score_loss(y_dev, np.clip(oof, 0, 1))),
                                 "brier_calibrated_in_sample": float(brier_score_loss(y_dev, oof_cal))}
    policies = [(n, k) for n in range(1, 9) for k in range(1, min(n, 3) + 1)]
    log(f"searching {len(policies)} alert policies on OOF predictions")
    FAR_CAP = 0.263  # baseline's validation non-sepsis patient false-alert rate
    free_best, free_table = search_policy(s_dev, oof_cal, policies, None)
    matched_best, _ = search_policy(s_dev, oof_cal, [(5, 1)], FAR_CAP)
    matched_any, _ = search_policy(s_dev, oof_cal, policies, FAR_CAP)
    free_table.to_csv(args.output_dir / "oof_policy_search.csv", index=False)
    result["oof_selection"] = {"utility_optimal": free_best, "matched_far_1_of_5": matched_best,
                               "matched_far_any_policy": matched_any, "far_cap": FAR_CAP}
    log(f"OOF utility-optimal: {free_best}")
    log(f"OOF matched-FAR 1-of-5: {matched_best}")
    log(f"OOF matched-FAR any policy: {matched_any}")

    # ---- final model on all development patients -> test ------------------------------
    log("fitting final model on all development patients")
    final = fit_lgb(best_params, x_dev, y_dev, rounds)
    test_raw = final.predict(x_test)
    test_cal = iso.predict(test_raw)
    result["test_calibration"] = {
        "brier_raw": float(brier_score_loss(y_test, np.clip(test_raw, 0, 1))),
        "brier_calibrated": float(brier_score_loss(y_test, test_cal)),
        "ece_raw": expected_calibration_error(y_test, np.clip(test_raw, 0, 1)),
        "ece_calibrated": expected_calibration_error(y_test, test_cal),
        "baseline_ece_balanced_weights": expected_calibration_error(y_test, base_test_prob),
    }

    def eval_choice(label: str, sel: dict) -> dict:
        risk = s_test.risk(test_cal, sel["n"], sel["k"])
        m = s_test.metrics(risk, sel["threshold"])
        m.update({"policy": f"{sel['k']}-of-{sel['n']}", "raw_pr_auc": float(average_precision_score(y_test, test_cal)),
                  "raw_roc_auc": float(roc_auc_score(y_test, test_cal)), "oof_selected_threshold": sel["threshold"]})
        return m, risk

    new_metrics = {}
    risks = {}
    for label, sel in (("utility_optimal", free_best), ("matched_far_1_of_5", matched_best),
                       ("matched_far_any_policy", matched_any)):
        new_metrics[label], risks[label] = eval_choice(label, sel)
        m = new_metrics[label]
        log(f"TEST {label}: policy={m['policy']} utility={m['normalized_challenge_utility']:.4f} "
            f"coverage={m['sepsis_alert_window_coverage']:.3f} FAR={m['nonsepsis_patient_false_alert_rate']:.3f} "
            f"PR-AUC={m['raw_pr_auc']:.4f}")
    result["improved_test"] = new_metrics

    # ---- bootstrap CIs (baseline vs improved matched-FAR 1-of-5 and utility-optimal) --
    log("paired patient-level bootstrap on test patients")
    terms = {"baseline": s_test.patient_terms(base_risk, base_thr)}
    probs = {"baseline": base_test_prob}
    for label in ("matched_far_1_of_5", "utility_optimal"):
        sel = {"matched_far_1_of_5": matched_best, "utility_optimal": free_best}[label]
        terms[f"improved_{label}"] = s_test.patient_terms(risks[label], sel["threshold"])
        probs[f"improved_{label}"] = test_cal
    boot = {}
    for label in ("matched_far_1_of_5", "utility_optimal"):
        pair = {"baseline": terms["baseline"], f"improved_{label}": terms[f"improved_{label}"]}
        pair_p = {"baseline": probs["baseline"], f"improved_{label}": probs[f"improved_{label}"]}
        boot[label] = paired_bootstrap(s_test, pair, pair_p, args.bootstrap, args.bootstrap_pr)
    result["bootstrap"] = boot

    # ---- importances + artefacts -----------------------------------------------------
    imp = pd.DataFrame({"feature": v2.columns, "gain": final.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = imp.gain / imp.gain.sum()
    imp.to_csv(args.output_dir / "feature_importance.csv", index=False)
    calibration_map = {
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
    }
    joblib.dump({"model_text": final.model_to_string(), "calibrator": calibration_map,
                 "feature_names": list(v2.columns),
                 "policy": matched_best, "utility_optimal_policy": free_best, "params": best_params,
                 "rounds": rounds, "feature_builder": "src.improved_pipeline.make_features_v2"},
                args.output_dir / "improved_model.joblib", compress=3)
    pd.DataFrame({"patient_id": s_test.patient_ids, "y_true": y_test, "baseline_prob": base_test_prob,
                  "improved_prob_calibrated": test_cal}).to_csv(args.output_dir / "test_predictions.csv.gz", index=False)

    (args.output_dir / "improved_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log("main results written")

    # ---- cross-hospital generalisation (A->B, B->A) ----------------------------------
    if not args.skip_hospital:
        hosp_b = pd.Series(ids).str.slice(1, 2).to_numpy() == "1"  # p1xxxxx = training_setB
        hosp = {}
        for src_name, src_mask, dst_mask in (("A_to_B", ~hosp_b, hosp_b), ("B_to_A", hosp_b, ~hosp_b)):
            log(f"cross-hospital {src_name}")
            src_ids = np.unique(ids[src_mask])
            fit_ids, thr_ids = train_test_split(src_ids, test_size=0.15, random_state=SEED,
                                                stratify=patient_target.loc[src_ids].to_numpy())
            fit_m, thr_m = np.isin(ids, fit_ids), np.isin(ids, thr_ids)
            s_thr, s_dst = block(thr_m), block(dst_mask)
            row = {}
            for name, X, mdl in (("v1_hgb_balanced", v1, "hgb"), ("v2_lightgbm", v2, "lgb")):
                if mdl == "hgb":
                    # drop columns with no observed value in the fit hospital (breaks HGB binning)
                    cols = X.columns[X[fit_m].notna().any().to_numpy()]
                    model = hgb_original().fit(X.loc[fit_m, cols], labels[fit_m])
                    p_thr, p_dst = model.predict_proba(X.loc[thr_m, cols])[:, 1], model.predict_proba(X.loc[dst_mask, cols])[:, 1]
                else:
                    model = fit_lgb(best_params, X[fit_m], labels[fit_m], rounds)
                    p_thr, p_dst = model.predict(X[thr_m]), model.predict(X[dst_mask])
                sel, _ = search_policy(s_thr, p_thr, [(5, 1)], None)
                m = s_dst.metrics(s_dst.risk(p_dst, 5, 1), sel["threshold"])
                m["raw_pr_auc"] = float(average_precision_score(s_dst.labels, p_dst))
                m["raw_roc_auc"] = float(roc_auc_score(s_dst.labels, p_dst))
                m["threshold_from_source_hospital"] = sel["threshold"]
                row[name] = m
                log(f"  {src_name} {name}: PR-AUC={m['raw_pr_auc']:.4f} utility={m['normalized_challenge_utility']:.4f} "
                    f"FAR={m['nonsepsis_patient_false_alert_rate']:.3f}")
            hosp[src_name] = row
        result["cross_hospital"] = hosp

    (args.output_dir / "improved_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
