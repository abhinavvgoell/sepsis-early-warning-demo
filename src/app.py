"""Local-only web demo for the saved sepsis research model."""

from __future__ import annotations

import json
import argparse
import math
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
DEFAULT_MODEL = "lightgbm"
MODEL_OPTIONS = {
    "lightgbm": "LightGBM (200 causal features + calibration)",
    "logistic_regression": "Logistic Regression (class-weighted)",
    "hist_gradient_boosting": "Histogram Gradient Boosting (class-weighted)",
}


def parse_number(value: Any, field: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.nan
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric or blank.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or blank.") from exc
    if not math.isfinite(number) or abs(number) > 1_000_000:
        raise ValueError(f"{field} must be a finite number.")
    return number


def load_model_bundle(path: Path, classical_path: Path = CLASSICAL_BUNDLE_PATH) -> dict:
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


def _top_contributors(values: np.ndarray, feature_names: list[str]) -> list[dict[str, str]]:
    grouped: dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, values):
        if math.isfinite(float(value)):
            grouped[_source_field(str(name))] += float(value)
    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    return [{
        "field": name,
        "direction": "raises" if value > 0 else "lowers" if value < 0 else "neutral",
    } for name, value in ranked if value != 0]


def explain_predictions(model_key: str, model: Any, features: pd.DataFrame, bundle: dict) -> tuple[list[list[dict[str, str]]] | None, str]:
    """Return per-row model contributions only for models with a direct additive explanation."""
    feature_names = list(features.columns)
    if model_key == "lightgbm":
        # LightGBM's native TreeSHAP contributions sum to its raw (pre-calibration) margin.
        contributions = np.asarray(bundle["booster"].predict(features, pred_contrib=True))[:, :-1]
        note = "Grouped LightGBM SHAP contributions to the raw model output; the final score is calibrated afterward. These are model associations, not clinical causes."
    elif model_key == "logistic_regression":
        # The saved baseline is median-imputation -> standardization -> logistic regression.
        imputed = model.named_steps["simpleimputer"].transform(features)
        scaled = model.named_steps["standardscaler"].transform(imputed)
        coefficients = np.asarray(model.named_steps["logisticregression"].coef_[0])
        contributions = np.asarray(scaled) * coefficients[None, :]
        note = "Grouped additive contributions to the logistic model's log-odds relative to its intercept. These are model associations, not clinical causes."
    else:
        return None, "A per-record explanation is not available for this model in the demo."
    rows = [_top_contributors(row, feature_names) for row in contributions]
    return rows, note


def safe_explain_predictions(model_key: str, model: Any, features: pd.DataFrame, bundle: dict) -> tuple[list[list[dict[str, str]]] | None, str]:
    """Keep optional explanation failures from preventing a valid model score."""
    try:
        return explain_predictions(model_key, model, features, bundle)
    except Exception:
        return None, "A per-record explanation could not be produced for these inputs."


def predict_records(bundle: dict, request_data: dict) -> dict:
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

    unknown_metadata = set(metadata) - STATIC_COLUMNS
    if unknown_metadata:
        raise ValueError("The request contains an unsupported patient-context field.")

    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Hourly record {index + 1} must be an object.")
        unknown = set(record) - set(RAW_COLUMNS)
        if unknown:
            raise ValueError(f"Hourly record {index + 1} contains an unsupported field.")
        row = {column: np.nan for column in RAW_COLUMNS}
        for column, value in record.items():
            if column not in STATIC_COLUMNS:
                row[column] = parse_number(value, column)
        for column in STATIC_COLUMNS:
            value = metadata.get(column, record.get(column))
            row[column] = parse_number(value, column)
        if pd.isna(row["ICULOS"]):
            row["ICULOS"] = float(index + 1)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame[ID_COLUMN] = "local_demo_patient"
    frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
    frame[TARGET] = np.zeros(len(frame), dtype=np.int8)  # placeholder; excluded from features
    if model_key == "lightgbm":
        if bundle.get("feature_builder") != "src.improved_pipeline.make_features_v2" or "booster" not in bundle:
            raise RuntimeError("The LightGBM demo bundle is unavailable or has an unsupported feature pipeline.")
        from src.improved_pipeline import make_features_v2

        features = make_features_v2(frame)
        features = features.reindex(columns=bundle["feature_names"])
        raw_scores = bundle["booster"].predict(features)
        explanation_rows, explanation_note = safe_explain_predictions(model_key, bundle["booster"], features, bundle)
        calibrator = bundle["calibrator"]
        if isinstance(calibrator, dict):
            scores = np.interp(
                raw_scores,
                np.asarray(calibrator["x_thresholds"], dtype=np.float64),
                np.asarray(calibrator["y_thresholds"], dtype=np.float64),
                left=float(calibrator["y_thresholds"][0]),
                right=float(calibrator["y_thresholds"][-1]),
            )
        else:  # support bundles produced before the portable calibration-map format
            scores = np.asarray(calibrator.predict(raw_scores), dtype=np.float64)
        policy = bundle["policy"]
        threshold = float(policy["threshold"])
        policy_n, policy_k = int(policy["n"]), int(policy["k"])
        policy_name = f"{policy_k}-of-{policy_n}"
        model_name = MODEL_OPTIONS[model_key]
    else:
        # These two baseline models share the original 98-feature pipeline.
        classical = bundle["classical_bundle"]
        features, _, _ = make_features(frame)
        features = features.reindex(columns=classical["feature_names"])
        model = classical["models"][model_key]
        scores = model.predict_proba(features)[:, 1]
        explanation_rows, explanation_note = safe_explain_predictions(model_key, model, features, bundle)
        threshold = float(classical["thresholds"][model_key])
        policy_n, policy_k = 5, 1
        policy_name = f"{policy_k}-of-{policy_n}"
        model_name = MODEL_OPTIONS[model_key]
    patient_starts = np.array([0], dtype=int)
    patient_ends = np.array([len(scores)], dtype=int)
    policy_scores = k_of_n_risk(scores.astype(np.float32), patient_starts, patient_ends,
                                 n=policy_n, k=policy_k)
    active = policy_scores >= threshold

    hourly = [{
        "hour": index + 1,
        "model_score": round(float(score), 6),
        "rule_active": bool(active[index]),
        "new_threshold_crossing": bool(score >= threshold),
        "explanation": explanation_rows[index] if explanation_rows is not None else None,
    } for index, score in enumerate(scores)]
    return {
        "model_name": model_name,
        "model_key": model_key,
        "threshold": threshold,
        "policy": f"{policy_name}: rule activates when at least {policy_k} of the latest {policy_n} hourly scores cross the threshold",
        "hours_scored": len(hourly),
        "latest": hourly[-1],
        "hourly": hourly,
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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())
            except OSError:
                self._send(500, "text/plain; charset=utf-8", b"The app interface file could not be loaded.")
        elif self.path == "/health":
            self._send(200, "application/json", b'{"status":"ready"}')
        elif self.path == "/api/models":
            payload = {"default_model": DEFAULT_MODEL,
                       "models": [{"key": key, "label": label} for key, label in MODEL_OPTIONS.items()]}
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
            request_data = json.loads(self.rfile.read(size))
            if not isinstance(request_data, dict):
                raise ValueError("The request must be a JSON object.")
            response = predict_records(self.bundle, request_data)
            body = json.dumps(response, allow_nan=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        except (ValueError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, "application/json; charset=utf-8", body)
        except Exception:
            # Do not return patient inputs or internal exception details to the browser.
            self._send(500, "application/json; charset=utf-8", b'{"error":"Prediction failed. Check the model bundle and input fields."}')

    def log_message(self, format: str, *args: Any) -> None:
        # Keep default logs limited to request metadata; never log request bodies.
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH,
                        help="Model bundle; defaults to the latest validation-selected candidate.")
    args = parser.parse_args()
    if not args.bundle.exists():
        raise FileNotFoundError(f"Model bundle not found: {args.bundle}")
    DemoHandler.bundle = load_model_bundle(args.bundle)
    server = HTTPServer((HOST, PORT), DemoHandler)
    print(f"Sepsis research demo running at http://{HOST}:{PORT}")
    print(f"Model bundle: {args.bundle}")
    print("Local-only server. Use synthetic or de-identified data; requests are not saved.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
