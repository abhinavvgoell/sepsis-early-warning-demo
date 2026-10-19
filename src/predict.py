"""Run a saved research model on one patient PSV time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.run_pipeline import ID_COLUMN, TARGET, make_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One pipe-delimited patient PSV file")
    parser.add_argument("--model", choices=[
        "logistic_regression", "hist_gradient_boosting",
        "logistic_regression_accuracy", "hist_gradient_boosting_accuracy",
    ], default="hist_gradient_boosting")
    parser.add_argument("--bundle", type=Path, default=Path("outputs/trained_models.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/patient_risk_predictions.csv"))
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    frame = pd.read_csv(args.input, sep="|").drop(columns=[TARGET], errors="ignore")
    frame[ID_COLUMN] = "inference_patient"
    frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
    frame[TARGET] = 0  # placeholder only; target is excluded from model features
    features, _, _ = make_features(frame)
    features = features.reindex(columns=bundle["feature_names"])
    model = bundle["models"][args.model]
    threshold = float(bundle["thresholds"][args.model])
    probabilities = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({
        "hour_index": np.arange(len(frame)),
        "risk_probability": probabilities,
        "alert_threshold": threshold,
        "predicted_alert": (probabilities >= threshold).astype("int8"),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} hourly predictions to {args.output}")
    print("Research demonstration only; not for clinical use.")


if __name__ == "__main__":
    main()
