"""Train and evaluate leakage-aware sepsis-label classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "SepsisLabel"
ID_COLUMN = "patient_id"
ROLLING_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
OBSERVATION_COLUMNS = ROLLING_COLUMNS + [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]


def read_patients(data_dir: Path) -> pd.DataFrame:
    """Read all hourly PSV files, preserving each patient's time order."""
    files = sorted(data_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under {data_dir}. Download/extract the PhysioNet training data first."
        )
    patients = []
    for path in files:
        frame = pd.read_csv(path, sep="|")
        if TARGET not in frame:
            raise ValueError(f"{path} is missing required target column {TARGET!r}")
        frame[ID_COLUMN] = path.stem
        frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
        patients.append(frame)
    data = pd.concat(patients, ignore_index=True)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype("int8")
    return data


def make_features(
    data: pd.DataFrame,
    include_trends: bool = False,
    include_last_observation: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create causal per-hour features; optional history features never use future data."""
    frame = data.sort_values([ID_COLUMN, "hour_index"]).copy()
    excluded = {TARGET, ID_COLUMN}
    measurements = [
        column for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]
    features = frame[measurements].copy()
    derived_features: dict[str, pd.Series] = {}
    groups = frame.groupby(ID_COLUMN, sort=False)
    for column in ROLLING_COLUMNS:
        if column not in frame:
            continue
        history = groups[column].shift(1)
        frame[f"{column}_mean_6h_past"] = history.groupby(frame[ID_COLUMN], sort=False).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
        frame[f"{column}_count_6h_past"] = history.notna().groupby(frame[ID_COLUMN], sort=False).rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
        derived_features[f"{column}_mean_6h_past"] = frame[f"{column}_mean_6h_past"]
        derived_features[f"{column}_count_6h_past"] = frame[f"{column}_count_6h_past"]
        if include_trends:
            # Differences compare the current observation with prior observations only.
            derived_features[f"{column}_delta_1h"] = frame[column] - history
            derived_features[f"{column}_deviation_from_6h_mean"] = frame[column] - frame[f"{column}_mean_6h_past"]
            derived_features[f"{column}_std_6h_past"] = history.groupby(frame[ID_COLUMN], sort=False).rolling(6, min_periods=2).std().reset_index(level=0, drop=True)
    if include_last_observation:
        for column in OBSERVATION_COLUMNS:
            if column not in frame:
                continue
            history = groups[column].shift(1)
            derived_features[f"{column}_last_value_past"] = history.groupby(frame[ID_COLUMN], sort=False).ffill()
            observed_hour = frame["hour_index"].where(frame[column].notna())
            previous_observed_hour = observed_hour.groupby(frame[ID_COLUMN], sort=False).shift(1)
            previous_observed_hour = previous_observed_hour.groupby(frame[ID_COLUMN], sort=False).ffill()
            derived_features[f"{column}_hours_since_last_past"] = frame["hour_index"] - previous_observed_hour
    # Per-measurement missingness is informative in sparsely sampled ICU data.
    for column in measurements:
        derived_features[f"{column}_missing"] = frame[column].isna().astype("int8")
    features = pd.concat([features, pd.DataFrame(derived_features, index=frame.index)], axis=1)
    labels = frame[TARGET].astype("int8")
    patients = frame[ID_COLUMN]
    return features.replace([np.inf, -np.inf], np.nan), labels, patients


def _patient_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """PhysioNet 2019 time-dependent utility for one patient's hourly sequence."""
    septic = bool(np.any(labels))
    if not septic:
        return float(-0.05 * np.count_nonzero(predictions))
    # Challenge labels start six hours before onset; recover the onset index.
    t_sepsis = int(np.argmax(labels)) + 6
    times = np.arange(len(labels), dtype=float)
    early, optimal, late = -12, -6, 3
    m1, b1 = 1.0 / (optimal - early), -1.0 / (optimal - early) * early
    m2, b2 = -1.0 / (late - optimal), -(-1.0 / (late - optimal)) * late
    m3, b3 = -2.0 / (late - optimal), -(-2.0 / (late - optimal)) * optimal
    rel = times - t_sepsis
    utility = np.zeros(len(labels), dtype=float)
    true_alert = predictions.astype(bool)
    early_region = times <= t_sepsis + optimal
    late_region = (times > t_sepsis + optimal) & (times <= t_sepsis + late)
    utility[true_alert & early_region] = np.maximum(m1 * rel[true_alert & early_region] + b1, -0.05)
    utility[true_alert & late_region] = m2 * rel[true_alert & late_region] + b2
    missed_late = (~true_alert) & late_region
    utility[missed_late] = m3 * rel[missed_late] + b3
    return float(utility.sum())


def normalized_utility(labels: pd.Series, predictions: np.ndarray, patient_ids: pd.Series) -> float:
    """Cohort utility normalized so no alerts score 0 and the ideal scores 1."""
    observed = best = inactive = 0.0
    frame = pd.DataFrame({"patient_id": patient_ids.to_numpy(), "label": labels.to_numpy(), "prediction": predictions})
    for _, group in frame.groupby("patient_id", sort=False):
        y = group["label"].to_numpy(dtype=np.int8)
        pred = group["prediction"].to_numpy(dtype=np.int8)
        observed += _patient_utility(y, pred)
        inactive += _patient_utility(y, np.zeros(len(y), dtype=np.int8))
        ideal = np.zeros(len(y), dtype=np.int8)
        if np.any(y):
            t_sepsis = int(np.argmax(y)) + 6
            ideal[max(0, t_sepsis - 12):min(t_sepsis + 4, len(y))] = 1
        best += _patient_utility(y, ideal)
    denominator = best - inactive
    return float((observed - inactive) / denominator) if denominator else 0.0


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, patient_ids: pd.Series) -> float:
    """Choose the validation threshold maximizing official normalized utility."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_score, best_threshold = -np.inf, 0.5
    for threshold in thresholds:
        score = normalized_utility(y_true, (probabilities >= threshold).astype("int8"), patient_ids)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def choose_accuracy_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a validation cutoff maximizing hourly accuracy; ties favor fewer alerts."""
    best_score, best_threshold = -np.inf, 0.5
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = probabilities >= threshold
        current = accuracy_score(y_true, predictions)
        if current > best_score or (current == best_score and threshold > best_threshold):
            best_score, best_threshold = current, float(threshold)
    return best_threshold


def score(y_true: pd.Series, probabilities: np.ndarray, threshold: float, patient_ids: pd.Series) -> dict:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "normalized_challenge_utility": normalized_utility(y_true, predictions.astype("int8"), patient_ids),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "positive_rate": float(y_true.mean()),
        "rows": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_patients(args.data_dir)
    features, labels, patient_ids = make_features(raw)
    patient_target = raw.groupby(ID_COLUMN)[TARGET].max()
    ids = patient_target.index.to_numpy()
    train_ids, remainder_ids = train_test_split(
        ids, test_size=0.30, random_state=SEED, stratify=patient_target.to_numpy()
    )
    remainder_target = patient_target.loc[remainder_ids]
    valid_ids, test_ids = train_test_split(
        remainder_ids, test_size=0.50, random_state=SEED,
        stratify=remainder_target.to_numpy()
    )
    train_mask = patient_ids.isin(train_ids).to_numpy()
    valid_mask = patient_ids.isin(valid_ids).to_numpy()
    test_mask = patient_ids.isin(test_ids).to_numpy()

    x_train, y_train = features.loc[train_mask], labels.loc[train_mask]
    x_valid, y_valid = features.loc[valid_mask], labels.loc[valid_mask]
    x_test, y_test = features.loc[test_mask], labels.loc[test_mask]

    prevalence = float(y_train.mean())
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=False), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=20,
            l2_regularization=1.0, class_weight="balanced", random_state=SEED,
        ),
        "logistic_regression_accuracy": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=False), StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED),
        ),
        "hist_gradient_boosting_accuracy": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=20,
            l2_regularization=1.0, random_state=SEED,
        ),
    }
    test_patient_ids = patient_ids.loc[test_mask]
    results = [{"model": "train_prevalence_baseline", **score(y_test, np.full(len(y_test), prevalence), 0.5, test_patient_ids)}]
    predictions = pd.DataFrame({"patient_id": patient_ids.loc[test_mask].to_numpy(), "y_true": y_test.to_numpy()})
    fitted_models = {}
    thresholds = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        valid_prob = model.predict_proba(x_valid)[:, 1]
        if name.endswith("_accuracy"):
            threshold = choose_accuracy_threshold(y_valid, valid_prob)
        else:
            threshold = choose_threshold(y_valid, valid_prob, patient_ids.loc[valid_mask])
        test_prob = model.predict_proba(x_test)[:, 1]
        results.append({"model": name, **score(y_test, test_prob, threshold, test_patient_ids)})
        fitted_models[name] = model
        thresholds[name] = threshold
        predictions[f"{name}_probability"] = test_prob
        predictions[f"{name}_prediction"] = (test_prob >= threshold).astype("int8")

    metrics = {"dataset": "PhysioNet/CinC Challenge 2019", "seed": SEED,
               "patient_counts": {"train": int(len(train_ids)), "validation": int(len(valid_ids)), "test": int(len(test_ids))},
               "row_counts": {"train": int(train_mask.sum()), "validation": int(valid_mask.sum()), "test": int(test_mask.sum())},
               "features": list(features.columns), "results": results}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(args.output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    model_bundle = {
        "models": fitted_models,
        "thresholds": thresholds,
        "feature_names": list(features.columns),
        "target": TARGET,
        "seed": SEED,
        "dataset": "PhysioNet/CinC Challenge 2019",
    }
    joblib.dump(model_bundle, args.output_dir / "trained_models.joblib", compress=3)
    print(json.dumps({"patient_counts": metrics["patient_counts"], "results": results}, indent=2))
    print(f"Saved fitted models and thresholds to {args.output_dir / 'trained_models.joblib'}")


if __name__ == "__main__":
    main()
