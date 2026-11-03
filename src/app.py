"""Local-only web demo for the saved sepsis research model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.alert_policy_grid import k_of_n_risk
from src.run_pipeline import ID_COLUMN, TARGET, make_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "outputs" / "improved" / "improved_model.joblib"
CLASSICAL_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "trained_models.joblib"
HTML_PATH = PROJECT_ROOT / "web" / "index.html"
HOST = "127.0.0.1"
PORT = 8766
MAX_HOURS = 168
MAX_REQUEST_BYTES = 2_000_000

RAW_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
STATIC_COLUMNS = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}
IGNORED_EXTRA_COLUMNS = {"SepsisLabel", "patient_id", "hour_index", "hour", "id"}
DEFAULT_MODEL = "lightgbm"
MODEL_OPTIONS = {
    "lightgbm": "LightGBM (200 causal features + calibration)",
    "ensemble": "Ensemble Blend (60% LightGBM + 30% HGB + 10% LR)",
    "logistic_regression": "Logistic Regression (class-weighted)",
    "hist_gradient_boosting": "Histogram Gradient Boosting (class-weighted)",
}
POLICY_OPTIONS = {
    "1_of_5": {"n": 5, "k": 1, "label": "1-of-5 Rule (Recommended - Matched FAR ~24.9%)"},
    "1_of_2": {"n": 2, "k": 1, "label": "1-of-2 Rule (Utility Optimal - High Sensitivity)"},
    "1_of_1": {"n": 1, "k": 1, "label": "1-of-1 Rule (Immediate Alert - No Persistence)"},
}


def parse_number(value: Any, field: str) -> float:
    """Parse numeric input safely, allowing missing values (NaN)."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned or cleaned in ("nan", "null", "none", "na", "n/a", "?", "-", "--", ""):
            return np.nan
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric or blank.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or blank.") from exc
    if math.isnan(number):
        return np.nan
    if not math.isfinite(number) or abs(number) > 1_000_000:
        raise ValueError(f"{field} must be a finite number.")
    return number


def sanitize_for_json(data: Any) -> Any:
    """Recursively convert numpy types to native Python and non-finite floats to None (JSON null)."""
    if isinstance(data, dict):
        return {str(k): sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    if isinstance(data, (float, np.floating)):
        return float(data) if math.isfinite(data) else None
    if isinstance(data, (int, np.integer)):
        return int(data)
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    if data is None:
        return None
    return data


def load_model_bundle(path: Path = BUNDLE_PATH, classical_path: Path = CLASSICAL_BUNDLE_PATH) -> dict:
    """Load both the improved LightGBM model bundle and the classical scikit-learn models."""
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if bundle.get("candidate_only"):
        raise RuntimeError("This experimental candidate is not configured for the demo app; use the default improved model bundle.")
    if bundle.get("feature_builder") == "src.improved_pipeline.make_features_v2":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("The selected model needs LightGBM. Install requirements.txt first.") from exc
        bundle["booster"] = lgb.Booster(model_str=bundle["model_text"])
    if not classical_path.exists():
        raise FileNotFoundError(f"Classical model bundle not found: {classical_path}")
    classical = joblib.load(classical_path)
    for model_key in ("logistic_regression", "hist_gradient_boosting"):
        if model_key not in classical.get("models", {}) or model_key not in classical.get("thresholds", {}):
            raise RuntimeError(f"The classical bundle is missing {model_key} or its saved threshold.")
    bundle["classical_bundle"] = classical
    return bundle


def _source_field(feature_name: str) -> str:
    """Group engineered features under their source field for readable explanations."""
    for column in sorted(RAW_COLUMNS, key=len, reverse=True):
        if feature_name == column or feature_name.startswith(f"{column}_"):
            return column
    return "Combined derived features"


def _top_contributors(values: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    """Group feature importances by raw clinical measurement and return top drivers."""
    grouped: dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, values):
        if math.isfinite(float(value)):
            grouped[_source_field(str(name))] += float(value)
    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:4]
    results = []
    for name, val in ranked:
        if abs(val) > 1e-5:
            results.append({
                "field": name,
                "direction": "raises" if val > 0 else "lowers",
                "impact": round(float(val), 4),
                "impact_formatted": f"{'+' if val > 0 else ''}{val:.2f}",
            })
    return results


def explain_predictions(
    model_key: str,
    model: Any,
    features: pd.DataFrame,
    bundle: dict,
) -> tuple[list[list[dict[str, Any]]] | None, str]:
    """Return per-row model contributions for LightGBM, Logistic Regression, or HGB."""
    feature_names = list(features.columns)
    if model_key == "lightgbm":
        # Native TreeSHAP contributions sum to raw margin
        contributions = np.asarray(bundle["booster"].predict(features, pred_contrib=True))[:, :-1]
        note = "LightGBM TreeSHAP contributions to raw margin before isotonic calibration. These reflect model attribution, not clinical causes."
        rows = [_top_contributors(row, feature_names) for row in contributions]
        return rows, note
    elif model_key == "logistic_regression":
        imputed = model.named_steps["simpleimputer"].transform(features)
        scaled = model.named_steps["standardscaler"].transform(imputed)
        coefficients = np.asarray(model.named_steps["logisticregression"].coef_[0])
        contributions = np.asarray(scaled) * coefficients[None, :]
        note = "Grouped additive contributions to logistic regression log-odds relative to intercept. These reflect model attribution, not clinical causes."
        rows = [_top_contributors(row, feature_names) for row in contributions]
        return rows, note
    elif model_key == "hist_gradient_boosting":
        # Fast vectorized marginal attribution across core clinical variables
        base_decision = model.decision_function(features)
        core_cols = ["HR", "Temp", "Resp", "SBP", "MAP", "O2Sat", "WBC", "Lactate", "Creatinine", "Platelets"]
        col_diffs = {}
        for col in core_cols:
            related = [c for c in feature_names if c == col or c.startswith(col + "_")]
            if related:
                f_mod = features.copy()
                for c in related:
                    f_mod[c] = np.nan
                col_diffs[col] = base_decision - model.decision_function(f_mod)
        rows = []
        for row_idx in range(len(features)):
            impacts = {col: float(diffs[row_idx]) for col, diffs in col_diffs.items() if abs(diffs[row_idx]) > 1e-4}
            ranked = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            rows.append([{
                "field": name,
                "direction": "raises" if val > 0 else "lowers",
                "impact": round(val, 4),
                "impact_formatted": f"{'+' if val > 0 else ''}{val:.2f}",
            } for name, val in ranked])
        note = "Marginal decision-margin impact of clinical measurements on Histogram Gradient Boosting. These reflect model attribution, not clinical causes."
        return rows, note
    else:
        return None, "Per-record explanation is not available for this model."


def safe_explain_predictions(
    model_key: str,
    model: Any,
    features: pd.DataFrame,
    bundle: dict,
) -> tuple[list[list[dict[str, Any]]] | None, str]:
    """Prevent explanation errors from failing the prediction."""
    try:
        return explain_predictions(model_key, model, features, bundle)
    except Exception as exc:
        print(f"Explanation error for {model_key}: {exc}", file=sys.stderr)
        return None, "A per-record explanation could not be produced for these inputs."


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def compute_clinical_flags(row: dict[str, Any]) -> dict[str, Any]:
    """Calculate standard ICU bedside clinical markers: SIRS, qSOFA, Shock Index, MAP < 65, NEWS2, MEWS, and SOFA."""
    hr = _safe_float(row.get("HR"))
    temp = _safe_float(row.get("Temp"))
    resp = _safe_float(row.get("Resp"))
    sbp = _safe_float(row.get("SBP"))
    map_val = _safe_float(row.get("MAP"))
    wbc = _safe_float(row.get("WBC"))
    lactate = _safe_float(row.get("Lactate"))
    o2sat = _safe_float(row.get("O2Sat"))
    fio2 = _safe_float(row.get("FiO2"))
    platelets = _safe_float(row.get("Platelets"))
    creatinine = _safe_float(row.get("Creatinine"))
    bilirubin = _safe_float(row.get("Bilirubin_total"))

    # ---- SIRS ----
    sirs_items: list[str] = []
    if temp is not None:
        if temp > 38.0 or temp < 36.0:
            sirs_items.append(f"Temp {temp:.1f} C")
    if hr is not None and hr > 90:
        sirs_items.append(f"HR {int(hr)} bpm")
    if resp is not None and resp > 20:
        sirs_items.append(f"Resp {int(resp)} /min")
    if wbc is not None and (wbc > 12.0 or wbc < 4.0):
        sirs_items.append(f"WBC {wbc:.1f} k/uL")

    # ---- qSOFA ----
    qsofa_items: list[str] = []
    if resp is not None and resp >= 22:
        qsofa_items.append("Resp >= 22 /min")
    if sbp is not None and sbp <= 100:
        qsofa_items.append("SBP <= 100 mmHg")

    # ---- Shock Index ----
    shock_index = None
    if hr is not None and sbp is not None and sbp > 0:
        shock_index = round(float(hr / sbp), 2)

    is_hypotensive = bool(map_val is not None and map_val < 65)
    is_lactate_high = bool(lactate is not None and lactate > 2.0)

    # ---- NEWS2 (Royal College of Physicians, UK) ----
    news2_scores: dict[str, int | None] = {}
    news2_triggers: list[str] = []

    # NEWS2 - Respiration rate
    if resp is not None:
        if resp <= 8:
            news2_scores["resp"] = 3
        elif resp <= 11:
            news2_scores["resp"] = 1
        elif resp <= 20:
            news2_scores["resp"] = 0
        elif resp <= 24:
            news2_scores["resp"] = 2
        else:
            news2_scores["resp"] = 3
        if news2_scores["resp"] > 0:
            news2_triggers.append(f"Resp {int(resp)}/min (score {news2_scores['resp']})")
    else:
        news2_scores["resp"] = None

    # NEWS2 - SpO2 Scale 1 (no known hypercapnic respiratory failure)
    if o2sat is not None:
        if o2sat <= 91:
            news2_scores["spo2"] = 3
        elif o2sat <= 93:
            news2_scores["spo2"] = 2
        elif o2sat <= 95:
            news2_scores["spo2"] = 1
        else:
            news2_scores["spo2"] = 0
        if news2_scores["spo2"] > 0:
            news2_triggers.append(f"SpO2 {o2sat:.0f}% (score {news2_scores['spo2']})")
    else:
        news2_scores["spo2"] = None

    # NEWS2 - Supplemental oxygen (inferred from FiO2)
    if fio2 is not None:
        if fio2 > 0.21:
            news2_scores["supp_o2"] = 2
            news2_triggers.append(f"Supplemental O2 FiO2={fio2:.2f} (score 2)")
        else:
            news2_scores["supp_o2"] = 0
    else:
        news2_scores["supp_o2"] = None

    # NEWS2 - Temperature
    if temp is not None:
        if temp <= 35.0:
            news2_scores["temp"] = 3
        elif temp <= 36.0:
            news2_scores["temp"] = 1
        elif temp <= 38.0:
            news2_scores["temp"] = 0
        elif temp <= 39.0:
            news2_scores["temp"] = 1
        else:
            news2_scores["temp"] = 2
        if news2_scores["temp"] > 0:
            news2_triggers.append(f"Temp {temp:.1f}C (score {news2_scores['temp']})")
    else:
        news2_scores["temp"] = None

    # NEWS2 - Systolic BP
    if sbp is not None:
        if sbp <= 90:
            news2_scores["sbp"] = 3
        elif sbp <= 100:
            news2_scores["sbp"] = 2
        elif sbp <= 110:
            news2_scores["sbp"] = 1
        elif sbp <= 219:
            news2_scores["sbp"] = 0
        else:
            news2_scores["sbp"] = 3
        if news2_scores["sbp"] > 0:
            news2_triggers.append(f"SBP {int(sbp)} mmHg (score {news2_scores['sbp']})")
    else:
        news2_scores["sbp"] = None

    # NEWS2 - Heart Rate
    if hr is not None:
        if hr <= 40:
            news2_scores["hr"] = 3
        elif hr <= 50:
            news2_scores["hr"] = 1
        elif hr <= 90:
            news2_scores["hr"] = 0
        elif hr <= 110:
            news2_scores["hr"] = 1
        elif hr <= 130:
            news2_scores["hr"] = 2
        else:
            news2_scores["hr"] = 3
        if news2_scores["hr"] > 0:
            news2_triggers.append(f"HR {int(hr)} bpm (score {news2_scores['hr']})")
    else:
        news2_scores["hr"] = None

    # NEWS2 totals
    has_any_vital = any(v is not None for v in (resp, o2sat, temp, sbp, hr))
    if has_any_vital:
        news2_scores["consciousness"] = 0
        news2_triggers.append("Consciousness not assessed (assumed Alert, score 0)")
        news2_available = [v for v in news2_scores.values() if v is not None]
        news2_components_assessed = len(news2_available)
        news2_any_extreme = any(v == 3 for v in news2_available)
        news2_score = sum(news2_available)
        if news2_score >= 7:
            news2_risk = "High"
        elif news2_score >= 5 or news2_any_extreme:
            news2_risk = "Medium"
        else:
            news2_risk = "Low"
    else:
        news2_components_assessed = 0
        news2_any_extreme = False
        news2_score = None
        news2_risk = "Insufficient data"

    # ---- MEWS (Modified Early Warning Score) ----
    mews_scores: dict[str, int | None] = {}
    mews_triggers: list[str] = []

    # MEWS - SBP
    if sbp is not None:
        if sbp <= 70:
            mews_scores["sbp"] = 3
        elif sbp <= 80:
            mews_scores["sbp"] = 2
        elif sbp <= 100:
            mews_scores["sbp"] = 1
        elif sbp <= 199:
            mews_scores["sbp"] = 0
        else:
            mews_scores["sbp"] = 2
        if mews_scores["sbp"] > 0:
            mews_triggers.append(f"SBP {int(sbp)} mmHg (score {mews_scores['sbp']})")
    else:
        mews_scores["sbp"] = None

    # MEWS - Heart Rate
    if hr is not None:
        if hr <= 40:
            mews_scores["hr"] = 2
        elif hr <= 50:
            mews_scores["hr"] = 1
        elif hr <= 100:
            mews_scores["hr"] = 0
        elif hr <= 110:
            mews_scores["hr"] = 1
        elif hr <= 129:
            mews_scores["hr"] = 2
        else:
            mews_scores["hr"] = 3
        if mews_scores["hr"] > 0:
            mews_triggers.append(f"HR {int(hr)} bpm (score {mews_scores['hr']})")
    else:
        mews_scores["hr"] = None

    # MEWS - Respiratory Rate
    if resp is not None:
        if resp < 9:
            mews_scores["resp"] = 2
        elif resp <= 14:
            mews_scores["resp"] = 0
        elif resp <= 20:
            mews_scores["resp"] = 1
        elif resp <= 29:
            mews_scores["resp"] = 2
        else:
            mews_scores["resp"] = 3
        if mews_scores["resp"] > 0:
            mews_triggers.append(f"Resp {int(resp)}/min (score {mews_scores['resp']})")
    else:
        mews_scores["resp"] = None

    # MEWS - Temperature
    if temp is not None:
        if temp < 35.0:
            mews_scores["temp"] = 2
        elif temp <= 38.4:
            mews_scores["temp"] = 0
        else:
            mews_scores["temp"] = 2
        if mews_scores["temp"] > 0:
            mews_triggers.append(f"Temp {temp:.1f}C (score {mews_scores['temp']})")
    else:
        mews_scores["temp"] = None

    # MEWS totals
    if has_any_vital:
        mews_scores["consciousness"] = 0
        mews_available = [v for v in mews_scores.values() if v is not None]
        mews_components_assessed = len(mews_available)
        mews_score = sum(mews_available)
    else:
        mews_components_assessed = 0
        mews_score = None

    # ---- SOFA Sub-Components (simplified from available data) ----

    # SOFA - Respiratory (SpO2/FiO2 proxy)
    if o2sat is not None or fio2 is not None:
        if fio2 is not None and fio2 >= 0.60:
            sofa_respiratory = 3
        elif o2sat is not None and o2sat < 92:
            sofa_respiratory = 2
        elif fio2 is not None and 0.40 <= fio2 < 0.60:
            sofa_respiratory = 2
        elif o2sat is not None and 92 <= o2sat <= 95:
            sofa_respiratory = 1
        elif fio2 is not None and 0.22 <= fio2 < 0.40:
            sofa_respiratory = 1
        else:
            sofa_respiratory = 0
    else:
        sofa_respiratory = None

    # SOFA - Coagulation (Platelets k/uL)
    if platelets is not None:
        if platelets < 20:
            sofa_coagulation = 4
        elif platelets < 50:
            sofa_coagulation = 3
        elif platelets < 100:
            sofa_coagulation = 2
        elif platelets < 150:
            sofa_coagulation = 1
        else:
            sofa_coagulation = 0
    else:
        sofa_coagulation = None

    # SOFA - Liver (Bilirubin_total mg/dL)
    if bilirubin is not None:
        if bilirubin >= 12.0:
            sofa_liver = 4
        elif bilirubin >= 6.0:
            sofa_liver = 3
        elif bilirubin >= 2.0:
            sofa_liver = 2
        elif bilirubin >= 1.2:
            sofa_liver = 1
        else:
            sofa_liver = 0
    else:
        sofa_liver = None

    # SOFA - Cardiovascular (MAP, Lactate as vasopressor proxy)
    if map_val is not None:
        if map_val < 55 or (lactate is not None and lactate > 4):
            sofa_cardiovascular = 3
        elif map_val < 65:
            sofa_cardiovascular = 2
        elif map_val < 70 or (lactate is not None and lactate > 2):
            sofa_cardiovascular = 1
        else:
            sofa_cardiovascular = 0
    else:
        sofa_cardiovascular = None

    # SOFA - Renal (Creatinine mg/dL)
    if creatinine is not None:
        if creatinine >= 5.0:
            sofa_renal = 4
        elif creatinine >= 3.5:
            sofa_renal = 3
        elif creatinine >= 2.0:
            sofa_renal = 2
        elif creatinine >= 1.2:
            sofa_renal = 1
        else:
            sofa_renal = 0
    else:
        sofa_renal = None

    # SOFA totals
    sofa_components = {
        "respiratory": sofa_respiratory,
        "coagulation": sofa_coagulation,
        "liver": sofa_liver,
        "cardiovascular": sofa_cardiovascular,
        "renal": sofa_renal,
    }
    sofa_available = [v for v in sofa_components.values() if v is not None]
    sofa_components_assessed = len(sofa_available)
    sofa_total = sum(sofa_available) if sofa_components_assessed > 0 else None

    return {
        "sirs_score": len(sirs_items),
        "sirs_positive": len(sirs_items) >= 2,
        "sirs_triggers": sirs_items,
        "qsofa_score": len(qsofa_items),
        "qsofa_positive": len(qsofa_items) >= 2,
        "qsofa_triggers": qsofa_items,
        "shock_index": shock_index,
        "shock_index_elevated": shock_index is not None and shock_index > 0.9,
        "hypotension": is_hypotensive,
        "hyperlactatemia": is_lactate_high,
        # NEWS2
        "news2_score": news2_score,
        "news2_components_assessed": news2_components_assessed,
        "news2_risk": news2_risk,
        "news2_any_extreme": news2_any_extreme,
        "news2_triggers": news2_triggers,
        # MEWS
        "mews_score": mews_score,
        "mews_components_assessed": mews_components_assessed,
        "mews_triggers": mews_triggers,
        # SOFA sub-components
        "sofa_respiratory": sofa_respiratory,
        "sofa_coagulation": sofa_coagulation,
        "sofa_liver": sofa_liver,
        "sofa_cardiovascular": sofa_cardiovascular,
        "sofa_renal": sofa_renal,
        "sofa_total": sofa_total,
        "sofa_components_assessed": sofa_components_assessed,
    }


def predict_records(bundle: dict, request_data: dict) -> dict:
    """Predict hourly sepsis probability, alert status, explanations, and clinical flags."""
    records = request_data.get("records")
    metadata = request_data.get("metadata", {})
    if not isinstance(records, list) or not records:
        raise ValueError("Enter at least one hourly record.")
    if len(records) > MAX_HOURS:
        raise ValueError(f"Enter no more than {MAX_HOURS} hourly records at a time.")
    if not isinstance(metadata, dict):
        raise ValueError("Patient context must be a JSON object.")
    model_key = request_data.get("model", DEFAULT_MODEL)
    if not isinstance(model_key, str) or model_key not in MODEL_OPTIONS:
        raise ValueError("Choose LightGBM, Logistic Regression, or Histogram Gradient Boosting.")

    policy_key = request_data.get("policy", "1_of_5")
    if policy_key not in POLICY_OPTIONS:
        policy_key = "1_of_5"

    unknown_metadata = set(metadata) - STATIC_COLUMNS - IGNORED_EXTRA_COLUMNS
    if unknown_metadata:
        raise ValueError(f"The request contains unsupported patient-context field(s): {', '.join(sorted(unknown_metadata))}")

    rows = []
    clinical_flags_list = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Hourly record {index + 1} must be an object.")
        unknown = set(record) - set(RAW_COLUMNS) - IGNORED_EXTRA_COLUMNS
        if unknown:
            raise ValueError(f"Hourly record {index + 1} contains unsupported field(s): {', '.join(sorted(unknown))}")
        row = {column: np.nan for column in RAW_COLUMNS}
        for column in RAW_COLUMNS:
            if column in STATIC_COLUMNS:
                value = metadata.get(column, record.get(column))
            else:
                value = record.get(column)
            row[column] = parse_number(value, column)
        if pd.isna(row["ICULOS"]):
            row["ICULOS"] = float(index + 1)
        rows.append(row)
        clinical_flags_list.append(compute_clinical_flags(row))

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame[ID_COLUMN] = "local_demo_patient"
    frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
    frame[TARGET] = np.zeros(len(frame), dtype=np.int8)

    # 1. LightGBM scoring
    from src.improved_pipeline import make_features_v2

    lgb_features = make_features_v2(frame).reindex(columns=bundle["feature_names"])
    lgb_raw = bundle["booster"].predict(lgb_features)
    calibrator = bundle["calibrator"]
    if isinstance(calibrator, dict):
        lgb_scores = np.interp(
            lgb_raw,
            np.asarray(calibrator["x_thresholds"], dtype=np.float64),
            np.asarray(calibrator["y_thresholds"], dtype=np.float64),
            left=float(calibrator["y_thresholds"][0]),
            right=float(calibrator["y_thresholds"][-1]),
        )
    else:
        lgb_scores = np.asarray(calibrator.predict(lgb_raw), dtype=np.float64)

    # 2. Classical models scoring
    classical = bundle["classical_bundle"]
    cls_features, _, _ = make_features(frame)
    cls_features = cls_features.reindex(columns=classical["feature_names"])
    lr_model = classical["models"]["logistic_regression"]
    lr_scores = lr_model.predict_proba(cls_features)[:, 1]
    hgb_model = classical["models"]["hist_gradient_boosting"]
    hgb_scores = hgb_model.predict_proba(cls_features)[:, 1]

    # Model scores registry
    # Normalized risk consensus ensemble (aligns probability scales across calibrated and uncalibrated models)
    thr_lgb = float(bundle["policy"]["threshold"])
    thr_hgb = float(classical["thresholds"]["hist_gradient_boosting"])
    thr_lr = float(classical["thresholds"]["logistic_regression"])
    ratio_ens = 0.60 * (lgb_scores / thr_lgb) + 0.25 * (hgb_scores / thr_hgb) + 0.15 * (lr_scores / thr_lr)
    ensemble_scores = ratio_ens * thr_lgb

    all_scores = {
        "lightgbm": lgb_scores,
        "ensemble": ensemble_scores,
        "logistic_regression": lr_scores,
        "hist_gradient_boosting": hgb_scores,
    }
    scores = all_scores[model_key]

    # Determine threshold and k-of-n policy
    if model_key in ("lightgbm", "ensemble"):
        if policy_key == "1_of_2" and "utility_optimal_policy" in bundle:
            pol = bundle["utility_optimal_policy"]
            policy_n, policy_k = int(pol["n"]), int(pol["k"])
            threshold = float(pol["threshold"])
            policy_desc = "1-of-2 Rule (Utility-Optimal, High Sensitivity)"
        elif policy_key == "1_of_1":
            policy_n, policy_k = 1, 1
            threshold = float(bundle["policy"]["threshold"])
            policy_desc = "1-of-1 Rule (Immediate Single-Hour Cutoff)"
        else:
            pol = bundle["policy"]
            policy_n, policy_k = int(pol["n"]), int(pol["k"])
            threshold = float(pol["threshold"])
            policy_desc = "1-of-5 Rule (Matched FAR ~24.9% · Recommended)"
    else:
        threshold = float(classical["thresholds"][model_key])
        if policy_key == "1_of_5":
            policy_n, policy_k = 5, 1
            policy_desc = "1-of-5 Rule (Persistence window)"
        elif policy_key == "1_of_2":
            policy_n, policy_k = 2, 1
            policy_desc = "1-of-2 Rule (Short persistence)"
        else:
            policy_n, policy_k = 1, 1
            policy_desc = "1-of-1 Rule (Immediate cutoff)"

    # Compute explanations for active model
    active_features = lgb_features if model_key in ("lightgbm", "ensemble") else cls_features
    active_estimator = bundle["booster"] if model_key in ("lightgbm", "ensemble") else classical["models"][model_key]
    explanation_rows, explanation_note = safe_explain_predictions(
        "lightgbm" if model_key == "ensemble" else model_key,
        active_estimator,
        active_features,
        bundle,
    )
    if model_key == "ensemble":
        explanation_note = "Ensemble feature attribution reflects the primary TreeSHAP contributions from the dominant LightGBM component (60% weight)."

    # Compute k-of-n alert policy activations
    patient_starts = np.array([0], dtype=int)
    patient_ends = np.array([len(scores)], dtype=int)
    policy_scores = k_of_n_risk(
        scores.astype(np.float32), patient_starts, patient_ends, n=policy_n, k=policy_k
    )
    active = policy_scores >= threshold

    hourly = []
    for index, score in enumerate(scores):
        hourly.append({
            "hour": index + 1,
            "model_score": round(float(score), 6),
            "all_model_scores": {
                "lightgbm": round(float(lgb_scores[index]), 6),
                "ensemble": round(float(ensemble_scores[index]), 6),
                "logistic_regression": round(float(lr_scores[index]), 6),
                "hist_gradient_boosting": round(float(hgb_scores[index]), 6),
            },
            "rule_active": bool(active[index]),
            "new_threshold_crossing": bool(score >= threshold),
            "explanation": explanation_rows[index] if explanation_rows is not None else None,
            "clinical_flags": clinical_flags_list[index],
        })

    # Patient journey trajectory summary
    peak_idx = int(np.argmax(scores))
    peak_score = float(scores[peak_idx])
    first_alert = next((item["hour"] for item in hourly if item["rule_active"]), None)

    if peak_score >= threshold * 1.5:
        risk_level = "Critical Sepsis Risk"
        risk_color = "red"
    elif peak_score >= threshold:
        risk_level = "Early Sepsis Warning"
        risk_color = "warning"
    elif peak_score >= threshold * 0.7:
        risk_level = "Borderline / Watch"
        risk_color = "amber"
    else:
        risk_level = "Low Risk / Normal"
        risk_color = "green"

    if len(scores) >= 2:
        trend_diff = scores[-1] - scores[0]
        if trend_diff > 0.05:
            trend_label = f"Rapidly Rising (+{trend_diff * 100:.1f}%)"
        elif trend_diff > 0.01:
            trend_label = f"Gradually Rising (+{trend_diff * 100:.1f}%)"
        elif trend_diff < -0.02:
            trend_label = f"Improving ({trend_diff * 100:.1f}%)"
        else:
            trend_label = "Stable"
    else:
        trend_label = "Single hour recorded"

    return {
        "model_name": MODEL_OPTIONS[model_key],
        "model_key": model_key,
        "threshold": threshold,
        "policy": policy_desc,
        "policy_key": policy_key,
        "available_policies": [
            {"key": k, "label": v["label"], "n": v["n"], "k": v["k"]}
            for k, v in POLICY_OPTIONS.items()
        ],
        "available_models": [
            {"key": k, "label": v} for k, v in MODEL_OPTIONS.items()
        ],
        "hours_scored": len(hourly),
        "latest": hourly[-1],
        "hourly": hourly,
        "summary": {
            "risk_level": risk_level,
            "risk_color": risk_color,
            "peak_hour": peak_idx + 1,
            "peak_score": round(peak_score, 4),
            "first_alert_hour": first_alert,
            "trend": trend_label,
            "latest_sirs": clinical_flags_list[-1]["sirs_score"],
            "latest_qsofa": clinical_flags_list[-1]["qsofa_score"],
            "latest_shock_index": clinical_flags_list[-1]["shock_index"],
            "latest_news2": clinical_flags_list[-1]["news2_score"],
            "latest_news2_risk": clinical_flags_list[-1]["news2_risk"],
            "latest_mews": clinical_flags_list[-1]["mews_score"],
            "latest_sofa_total": clinical_flags_list[-1]["sofa_total"],
        },
        "explanation_note": explanation_note,
        "warning": "Research demonstration only. Scores are not clinical probabilities and must not guide diagnosis or treatment.",
    }


class DemoHandler(BaseHTTPRequestHandler):
    bundle: dict = {}

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, "text/plain", b"")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())
            except OSError:
                self._send(500, "text/plain; charset=utf-8", b"The app interface file could not be loaded.")
        elif self.path.startswith("/assets/"):
            name = self.path[len("/assets/"):].split("?", 1)[0]
            asset = (HTML_PATH.parent / "assets" / name).resolve()
            types = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg"}
            if asset.parent != (HTML_PATH.parent / "assets").resolve() or asset.suffix not in types:
                self._send(404, "text/plain; charset=utf-8", b"Not found")
                return
            try:
                self._send(200, types[asset.suffix], asset.read_bytes())
            except OSError:
                self._send(404, "text/plain; charset=utf-8", b"Not found")
        elif self.path == "/health":
            self._send(200, "application/json", b'{"status":"ready"}')
        elif self.path == "/api/models":
            payload = {
                "default_model": DEFAULT_MODEL,
                "models": [{"key": key, "label": label} for key, label in MODEL_OPTIONS.items()],
                "policies": [{"key": k, "label": v["label"]} for k, v in POLICY_OPTIONS.items()],
            }
            self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self._send(404, "application/json", b'{"error":"Not found"}')
            return
        length = self.headers.get("Content-Length")
        try:
            size = int(length or "0")
        except ValueError:
            self._send(400, "application/json", b'{"error":"Invalid request length."}')
            return
        if size <= 0 or size > MAX_REQUEST_BYTES:
            self._send(413, "application/json", b'{"error":"Request is empty or too large."}')
            return
        if "application/json" not in self.headers.get("Content-Type", ""):
            self._send(415, "application/json", b'{"error":"Send hourly records as JSON."}')
            return
        try:
            raw_body = self.rfile.read(size)
            request_data = json.loads(raw_body)
            if not isinstance(request_data, dict):
                raise ValueError("The request must be a JSON object.")
            response = predict_records(self.bundle, request_data)
            sanitized = sanitize_for_json(response)
            body = json.dumps(sanitized, allow_nan=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        except (ValueError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, "application/json; charset=utf-8", body)
        except Exception as exc:
            print(f"Prediction exception: {exc}", file=sys.stderr)
            traceback.print_exc()
            body = json.dumps({"error": f"Prediction failed: {exc}"}).encode("utf-8")
            self._send(500, "application/json; charset=utf-8", body)

    def log_message(self, format: str, *args: Any) -> None:
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH,
                        help="Model bundle; defaults to the latest validation-selected candidate.")
    parser.add_argument("--classical-bundle", type=Path, default=CLASSICAL_BUNDLE_PATH,
                        help="Classical models bundle (Logistic Regression and HGB).")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default {PORT})")
    parser.add_argument("--host", type=str, default=HOST, help=f"Server host (default {HOST})")
    args = parser.parse_args()
    if not args.bundle.exists():
        raise FileNotFoundError(f"Model bundle not found: {args.bundle}")
    DemoHandler.bundle = load_model_bundle(args.bundle, args.classical_bundle)
    try:
        server = HTTPServer((args.host, args.port), DemoHandler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 10048 or "10048" in str(exc) or "Address already in use" in str(exc):
            print(f"\n[INFO] Port {args.port} is already in use by another process.", flush=True)
            print(f"The dashboard may already be running! Visit http://{args.host}:{args.port}", flush=True)
            print("To restart, please close the existing server process or specify a different port.\n", flush=True)
            return
        raise
    print(f"Sepsis research demo running at http://{args.host}:{args.port}", flush=True)
    print(f"Primary model bundle: {args.bundle}", flush=True)
    print(f"Classical model bundle: {args.classical_bundle}", flush=True)
    print("Local-only server. Use synthetic or de-identified data; requests are not saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
