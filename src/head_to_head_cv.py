"""Shared patient-fold comparison of the saved LightGBM and HGB candidates.

Uses only the original training-patient partition; the validation/test partitions
are excluded. Threshold-dependent alert metrics are deliberately omitted because
the two models' alert thresholds were selected in different experiments.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.improved_pipeline import LGB_BASE, make_features_v2
from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    out_dir = root / "outputs" / "head_to_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(raw_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"No .psv files found under {raw_dir}")
    print(f"Reading labels for {len(files):,} patients...", flush=True)
    patient_labels = []
    for i, path in enumerate(files, 1):
        labels = pd.read_csv(path, sep="|", usecols=[TARGET])[TARGET]
        patient_labels.append((path, int(labels.max())))
        if i % 5000 == 0:
            print(f"  scanned {i:,}/{len(files):,} label files", flush=True)
    all_ids = np.asarray([path.stem for path, _ in patient_labels])
    all_y = np.asarray([label for _, label in patient_labels], dtype=np.int8)
    full_train_ids, _ = train_test_split(all_ids, test_size=0.30, random_state=SEED, stratify=all_y)
    train_label_map = dict(zip(all_ids, all_y))
    if len(full_train_ids) > 10_000:
        train_ids, _ = train_test_split(
            full_train_ids, train_size=10_000, random_state=SEED,
            stratify=np.asarray([train_label_map[x] for x in full_train_ids]),
        )
    else:
        train_ids = full_train_ids
    train_id_set = set(train_ids)
    print(f"Loading only {len(train_ids):,} original training patients...", flush=True)
    patients = []
    for i, (path, _) in enumerate(patient_labels, 1):
        if path.stem not in train_id_set:
            continue
        frame = pd.read_csv(path, sep="|")
        frame[ID_COLUMN] = path.stem
        frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
        patients.append(frame)
        if len(patients) % 5000 == 0:
            print(f"  loaded {len(patients):,}/{len(train_ids):,} training patients", flush=True)
    raw = pd.concat(patients, ignore_index=True).sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
    del patients, patient_labels
    y = raw[TARGET].to_numpy(dtype=np.int8)
    row_ids, _unique_train_ids = pd.factorize(raw[ID_COLUMN], sort=False)
    patient_target = raw.groupby(ID_COLUMN, sort=False)[TARGET].max()
    train_ids = np.arange(len(patient_target), dtype=np.int32)
    train_y = patient_target.to_numpy()
    print(f"Building HGB causal feature table for {len(raw):,} rows...", flush=True)
    hgb_x, _, _ = make_features(raw, include_trends=True, include_last_observation=True)
    del raw
    print(f"HGB features ready: {hgb_x.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    lgb_params = {"num_leaves": 15, "min_child_samples": 600, "feature_fraction": 0.5,
                  "bagging_fraction": 0.7, "lambda_l2": 10.0, "scale_pos_weight": 10.0}
    lgb_rounds = 233
    hgb_config = {"learning_rate": 0.06, "max_iter": 200, "max_leaf_nodes": 15,
                  "min_samples_leaf": 100, "l2_regularization": 4.0,
                  "class_weight": "balanced", "early_stopping": False, "random_state": SEED}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    records: list[dict] = []
    for fold, (fit_i, val_i) in enumerate(cv.split(train_ids, train_y), start=1):
        fit_mask, val_mask = np.isin(row_ids, fit_i), np.isin(row_ids, val_i)
        print(f"Fold {fold}/3: train rows={fit_mask.sum():,}, validation rows={val_mask.sum():,}", flush=True)
        y_fit, y_val = y[fit_mask], y[val_mask]

        print("  Fitting HGB last-observation candidate...", flush=True)
        hgb = HistGradientBoostingClassifier(**hgb_config)
        hgb.fit(hgb_x.loc[fit_mask], y_fit)
        p_hgb = hgb.predict_proba(hgb_x.loc[val_mask])[:, 1]
        records.append({"fold": fold, "model": "HGB_last_observation",
                        "pr_auc": float(average_precision_score(y_val, p_hgb)),
                        "roc_auc": float(roc_auc_score(y_val, p_hgb)),
                        "train_rows": int(fit_mask.sum()), "validation_rows": int(val_mask.sum())})
        del hgb, p_hgb

        print(f"  HGB fold {fold} done ({time.time()-t0:.0f}s)", flush=True)

    del hgb_x
    import gc
    gc.collect()
    print("Building LightGBM causal feature table...", flush=True)
    lgb_raw_parts = []
    for path in files:
        if path.stem not in train_id_set:
            continue
        frame = pd.read_csv(path, sep="|")
        frame[ID_COLUMN] = path.stem
        frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
        lgb_raw_parts.append(frame)
    lgb_raw = pd.concat(lgb_raw_parts, ignore_index=True).sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
    del lgb_raw_parts
    lgb_x = make_features_v2(lgb_raw)
    del lgb_raw
    gc.collect()
    print(f"LightGBM features ready: {lgb_x.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    for fold, (fit_i, val_i) in enumerate(cv.split(train_ids, train_y), start=1):
        fit_mask, val_mask = np.isin(row_ids, fit_i), np.isin(row_ids, val_i)
        print(f"LightGBM fold {fold}/3: train rows={fit_mask.sum():,}, validation rows={val_mask.sum():,}", flush=True)
        y_fit, y_val = y[fit_mask], y[val_mask]

        print("  Fitting LightGBM v2 candidate...", flush=True)
        dtrain = lgb.Dataset(lgb_x.loc[fit_mask], label=y_fit)
        booster = lgb.train({**LGB_BASE, **lgb_params}, dtrain, num_boost_round=lgb_rounds)
        p_lgb = booster.predict(lgb_x.loc[val_mask])
        records.append({"fold": fold, "model": "LightGBM_v2",
                        "pr_auc": float(average_precision_score(y_val, p_lgb)),
                        "roc_auc": float(roc_auc_score(y_val, p_lgb)),
                        "train_rows": int(fit_mask.sum()), "validation_rows": int(val_mask.sum())})
        del dtrain, booster, p_lgb
        print(f"  LightGBM fold {fold} done ({time.time()-t0:.0f}s)", flush=True)

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "head_to_head_cv_folds.csv", index=False)
    summary = table.groupby("model").agg(
        folds=("fold", "count"), mean_pr_auc=("pr_auc", "mean"), sd_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"),
    ).reset_index()
    summary.to_csv(out_dir / "head_to_head_cv_summary.csv", index=False)
    payload = {
        "split": "3-fold stratified by patient-level sepsis status, on a fixed 10,000-patient sample from original training patients only",
        "sampled_patients": int(len(train_ids)),
        "held_out_validation_and_test_partitions_used": False,
        "threshold_dependent_metrics": "not reported; thresholds were selected in different prior experiments",
        "models": {
            "HGB_last_observation": {"features": "run_pipeline.make_features(include_trends=True, include_last_observation=True)", **hgb_config},
            "LightGBM_v2": {"features": "improved_pipeline.make_features_v2", "params": lgb_params, "rounds": lgb_rounds},
        },
        "folds": records,
        "summary": summary.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "head_to_head_cv.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nShared-fold CV summary:")
    print(summary.to_string(index=False))
    print(f"Saved results in {out_dir}")


if __name__ == "__main__":
    main()
