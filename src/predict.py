"""Run a saved research model on one patient PSV time series with clinical metrics and alert policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.alert_policy_grid import k_of_n_risk
from src.run_pipeline import ID_COLUMN, TARGET, make_features


DEFAULT_IMPROVED_BUNDLE = Path("outputs/improved/improved_model.joblib")
DEFAULT_CLASSICAL_BUNDLE = Path("outputs/trained_models.joblib")


def compute_clinical_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bedside ICU clinical markers for each hour."""
    markers = []
    for _, row in df.iterrows():
        hr, temp, resp, sbp = row.get("HR"), row.get("Temp"), row.get("Resp"), row.get("SBP")
        map_val, wbc, lactate = row.get("MAP"), row.get("WBC"), row.get("Lactate")
        o2sat = row.get("O2Sat")
        fio2 = row.get("FiO2")
        platelets = row.get("Platelets")
        creatinine = row.get("Creatinine")
        bilirubin = row.get("Bilirubin_total")

        # ---- SIRS ----
        sirs = 0
        if pd.notna(temp) and (temp > 38.0 or temp < 36.0):
            sirs += 1
        if pd.notna(hr) and hr > 90:
            sirs += 1
        if pd.notna(resp) and resp > 20:
            sirs += 1
        if pd.notna(wbc) and (wbc > 12.0 or wbc < 4.0):
            sirs += 1

        # ---- qSOFA ----
        qsofa = 0
        if pd.notna(resp) and resp >= 22:
            qsofa += 1
        if pd.notna(sbp) and sbp <= 100:
            qsofa += 1

        shock_idx = float(hr / sbp) if (pd.notna(hr) and pd.notna(sbp) and sbp > 0) else np.nan

        # ---- NEWS2 ----
        news2_parts: list[int] = []
        has_any_news_vital = any(pd.notna(v) for v in (resp, o2sat, temp, sbp, hr))
        if pd.notna(resp):
            if resp <= 8:
                s = 3
            elif resp <= 11:
                s = 1
            elif resp <= 20:
                s = 0
            elif resp <= 24:
                s = 2
            else:
                s = 3
            news2_parts.append(s)
        if pd.notna(o2sat):
            if o2sat <= 91:
                s = 3
            elif o2sat <= 93:
                s = 2
            elif o2sat <= 95:
                s = 1
            else:
                s = 0
            news2_parts.append(s)
        # Supplemental oxygen
        if pd.notna(fio2):
            news2_parts.append(2 if fio2 > 0.21 else 0)
        if pd.notna(temp):
            if temp <= 35.0:
                s = 3
            elif temp <= 36.0:
                s = 1
            elif temp <= 38.0:
                s = 0
            elif temp <= 39.0:
                s = 1
            else:
                s = 2
            news2_parts.append(s)
        if pd.notna(sbp):
            if sbp <= 90:
                s = 3
            elif sbp <= 100:
                s = 2
            elif sbp <= 110:
                s = 1
            elif sbp <= 219:
                s = 0
            else:
                s = 3
            news2_parts.append(s)
        if pd.notna(hr):
            if hr <= 40:
                s = 3
            elif hr <= 50:
                s = 1
            elif hr <= 90:
                s = 0
            elif hr <= 110:
                s = 1
            elif hr <= 130:
                s = 2
            else:
                s = 3
            news2_parts.append(s)
        if has_any_news_vital:
            # Consciousness assumed Alert = 0
            news2_parts.append(0)

        if has_any_news_vital and len(news2_parts) > 0:
            news2_total = sum(news2_parts)
            news2_any3 = any(p == 3 for p in news2_parts)
            if news2_total >= 7:
                news2_risk = "High"
            elif news2_total >= 5 or news2_any3:
                news2_risk = "Medium"
            else:
                news2_risk = "Low"
        else:
            news2_total = np.nan
            news2_risk = "Insufficient data"

        # ---- MEWS ----
        mews_parts: list[int] = []
        if pd.notna(sbp):
            if sbp <= 70:
                s = 3
            elif sbp <= 80:
                s = 2
            elif sbp <= 100:
                s = 1
            elif sbp <= 199:
                s = 0
            else:
                s = 2
            mews_parts.append(s)
        if pd.notna(hr):
            if hr <= 40:
                s = 2
            elif hr <= 50:
                s = 1
            elif hr <= 100:
                s = 0
            elif hr <= 110:
                s = 1
            elif hr <= 129:
                s = 2
            else:
                s = 3
            mews_parts.append(s)
        if pd.notna(resp):
            if resp < 9:
                s = 2
            elif resp <= 14:
                s = 0
            elif resp <= 20:
                s = 1
            elif resp <= 29:
                s = 2
            else:
                s = 3
            mews_parts.append(s)
        if pd.notna(temp):
            if temp < 35.0:
                s = 2
            elif temp <= 38.4:
                s = 0
            else:
                s = 2
            mews_parts.append(s)
        if has_any_news_vital:
            mews_parts.append(0)
        mews_total = int(sum(mews_parts)) if (has_any_news_vital and len(mews_parts) > 0) else np.nan

        # ---- SOFA sub-components ----
        # Respiratory (SpO2/FiO2 proxy)
        if pd.notna(o2sat) or pd.notna(fio2):
            if pd.notna(fio2) and fio2 >= 0.60:
                sofa_resp = 3
            elif pd.notna(o2sat) and o2sat < 92:
                sofa_resp = 2
            elif pd.notna(fio2) and 0.40 <= fio2 < 0.60:
                sofa_resp = 2
            elif pd.notna(o2sat) and 92 <= o2sat <= 95:
                sofa_resp = 1
            elif pd.notna(fio2) and 0.22 <= fio2 < 0.40:
                sofa_resp = 1
            else:
                sofa_resp = 0
        else:
            sofa_resp = np.nan

        # Coagulation
        if pd.notna(platelets):
            if platelets < 20:
                sofa_coag = 4
            elif platelets < 50:
                sofa_coag = 3
            elif platelets < 100:
                sofa_coag = 2
            elif platelets < 150:
                sofa_coag = 1
            else:
                sofa_coag = 0
        else:
            sofa_coag = np.nan

        # Liver
        if pd.notna(bilirubin):
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
            sofa_liver = np.nan

        # Cardiovascular
        if pd.notna(map_val):
            if map_val < 55 or (pd.notna(lactate) and lactate > 4):
                sofa_cardio = 3
            elif map_val < 65:
                sofa_cardio = 2
            elif map_val < 70 or (pd.notna(lactate) and lactate > 2):
                sofa_cardio = 1
            else:
                sofa_cardio = 0
        else:
            sofa_cardio = np.nan

        # Renal
        if pd.notna(creatinine):
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
            sofa_renal = np.nan

        sofa_vals = [sofa_resp, sofa_coag, sofa_liver, sofa_cardio, sofa_renal]
        sofa_valid = [v for v in sofa_vals if pd.notna(v)]
        sofa_total = int(sum(sofa_valid)) if sofa_valid else np.nan

        markers.append({
            "sirs_score": sirs,
            "qsofa_score": qsofa,
            "shock_index": round(shock_idx, 2) if pd.notna(shock_idx) else np.nan,
            "hypotension_map_lt_65": int(pd.notna(map_val) and map_val < 65),
            "hyperlactatemia": int(pd.notna(lactate) and lactate > 2.0),
            "news2_score": news2_total,
            "news2_risk": news2_risk,
            "mews_score": mews_total,
            "sofa_respiratory": sofa_resp,
            "sofa_coagulation": sofa_coag,
            "sofa_liver": sofa_liver,
            "sofa_cardiovascular": sofa_cardio,
            "sofa_renal": sofa_renal,
            "sofa_total": sofa_total,
        })
    return pd.DataFrame(markers)


def run_prediction(
    input_path: Path,
    model_name: str = "lightgbm",
    bundle_path: Path | None = None,
    output_path: Path | None = None,
    policy: str = "1-of-5",
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine bundle
    if bundle_path is None or not bundle_path.exists():
        if model_name in ("lightgbm", "ensemble"):
            bundle_path = DEFAULT_IMPROVED_BUNDLE
            if not bundle_path.exists():
                bundle_path = DEFAULT_CLASSICAL_BUNDLE
                model_name = "hist_gradient_boosting"
        else:
            bundle_path = DEFAULT_CLASSICAL_BUNDLE

    bundle = joblib.load(bundle_path)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "|" in first_line:
        sep = "|"
    elif "\t" in first_line:
        sep = "\t"
    else:
        sep = ","
    try:
        raw_frame = pd.read_csv(input_path, sep=sep)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file {input_path} is empty.") from exc
    if len(raw_frame) == 0:
        raise ValueError(f"Input file {input_path} contains no observation rows.")
    has_target = TARGET in raw_frame.columns
    actual_labels = raw_frame[TARGET].to_numpy(dtype=np.int8) if has_target else None
    frame = raw_frame.drop(columns=[TARGET], errors="ignore").copy()
    frame[ID_COLUMN] = "inference_patient"
    frame["hour_index"] = np.arange(len(frame), dtype=np.int32)
    frame[TARGET] = 0

    if "feature_builder" in bundle and bundle.get("feature_builder") == "src.improved_pipeline.make_features_v2":
        # Improved LightGBM model
        from src.improved_pipeline import make_features_v2
        import lightgbm as lgb

        if "booster" not in bundle:
            bundle["booster"] = lgb.Booster(model_str=bundle["model_text"])
        features = make_features_v2(frame).reindex(columns=bundle["feature_names"])
        raw_scores = bundle["booster"].predict(features)
        calibrator = bundle["calibrator"]
        if isinstance(calibrator, dict):
            probabilities = np.interp(
                raw_scores,
                np.asarray(calibrator["x_thresholds"], dtype=np.float64),
                np.asarray(calibrator["y_thresholds"], dtype=np.float64),
                left=float(calibrator["y_thresholds"][0]),
                right=float(calibrator["y_thresholds"][-1]),
            )
        else:
            probabilities = np.asarray(calibrator.predict(raw_scores), dtype=np.float64)

        if policy == "1-of-2" and "utility_optimal_policy" in bundle:
            pol = bundle["utility_optimal_policy"]
            policy_n, policy_k, threshold = int(pol["n"]), int(pol["k"]), float(pol["threshold"])
        elif policy == "1-of-1":
            policy_n, policy_k = 1, 1
            threshold = float(bundle["policy"]["threshold"])
        else:
            pol = bundle["policy"]
            policy_n, policy_k, threshold = int(pol["n"]), int(pol["k"]), float(pol["threshold"])

        if model_name == "ensemble":
            # Load classical bundle for ensemble blend components
            if DEFAULT_CLASSICAL_BUNDLE.exists():
                cls_bundle = joblib.load(DEFAULT_CLASSICAL_BUNDLE)
                cls_features, _, _ = make_features(frame)
                cls_features = cls_features.reindex(columns=cls_bundle["feature_names"])
                prob_hgb = cls_bundle["models"]["hist_gradient_boosting"].predict_proba(cls_features)[:, 1]
                prob_lr = cls_bundle["models"]["logistic_regression"].predict_proba(cls_features)[:, 1]
                thr_hgb = float(cls_bundle["thresholds"]["hist_gradient_boosting"])
                thr_lr = float(cls_bundle["thresholds"]["logistic_regression"])
                ratio_ens = 0.60 * (probabilities / threshold) + 0.25 * (prob_hgb / thr_hgb) + 0.15 * (prob_lr / thr_lr)
                probabilities = ratio_ens * threshold
    else:
        # Classical model bundle
        features, _, _ = make_features(frame)
        features = features.reindex(columns=bundle["feature_names"])
        actual_model_name = model_name if model_name in bundle["models"] else "hist_gradient_boosting"
        model = bundle["models"][actual_model_name]
        threshold = float(bundle["thresholds"][actual_model_name])
        probabilities = model.predict_proba(features)[:, 1]
        if policy == "1-of-5":
            policy_n, policy_k = 5, 1
        elif policy == "1-of-2":
            policy_n, policy_k = 2, 1
        else:
            policy_n, policy_k = 1, 1

    # Temporal persistence risk
    starts = np.array([0], dtype=int)
    ends = np.array([len(probabilities)], dtype=int)
    policy_scores = k_of_n_risk(probabilities.astype(np.float32), starts, ends, n=policy_n, k=policy_k)
    predicted_alert = (policy_scores >= threshold).astype("int8")

    clinical_markers = compute_clinical_markers(raw_frame)

    result = pd.DataFrame({
        "hour_index": np.arange(1, len(frame) + 1),
        "risk_probability": np.round(probabilities, 5),
        "alert_threshold": round(threshold, 5),
        "policy": f"{policy_k}-of-{policy_n}",
        "predicted_alert": predicted_alert,
        "single_hour_crossing": (probabilities >= threshold).astype("int8"),
    })
    if has_target:
        result["actual_sepsis_label"] = actual_labels
    result = pd.concat([result, clinical_markers], axis=1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One pipe-delimited patient PSV file")
    parser.add_argument("--model", choices=[
        "lightgbm", "ensemble", "hist_gradient_boosting", "logistic_regression",
        "logistic_regression_accuracy", "hist_gradient_boosting_accuracy",
    ], default="lightgbm", help="Model to evaluate (default: lightgbm)")
    parser.add_argument("--bundle", type=Path, default=None, help="Path to model bundle joblib")
    parser.add_argument("--policy", choices=["1-of-5", "1-of-2", "1-of-1"], default="1-of-5",
                        help="Temporal alert persistence policy (default: 1-of-5)")
    parser.add_argument("--output", type=Path, default=Path("outputs/patient_risk_predictions.csv"),
                        help="Output path for predictions CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary table")
    args = parser.parse_args()

    result = run_prediction(
        input_path=args.input,
        model_name=args.model,
        bundle_path=args.bundle,
        output_path=args.output,
        policy=args.policy,
    )

    if not args.quiet:
        print("\n" + "=" * 70)
        print(f"SEPSIS EARLY WARNING - PATIENT RISK SUMMARY ({args.input.name})")
        print("=" * 70)
        print(f"Model: {args.model.upper()} | Alert Policy: {args.policy} | Hours: {len(result)}")
        peak_idx = int(result["risk_probability"].argmax())
        peak_score = result["risk_probability"].iloc[peak_idx]
        peak_hour = result["hour_index"].iloc[peak_idx]
        threshold = result["alert_threshold"].iloc[0]
        first_alert = result[result["predicted_alert"] == 1]["hour_index"].min()

        print(f"Alert Threshold: {threshold:.4f}")
        print(f"Peak Risk Score: {peak_score:.4f} at Hour {peak_hour}")
        if pd.notna(first_alert):
            print(f"ALERT ACTIVE: First triggered at Hour {int(first_alert)}")
        else:
            print("STATUS: No alert threshold triggered during observation window.")

        if "actual_sepsis_label" in result.columns and result["actual_sepsis_label"].sum() > 0:
            first_label = result[result["actual_sepsis_label"] == 1]["hour_index"].min()
            print(f"Dataset Positive Window: Starts at Hour {int(first_label)}")
            if pd.notna(first_alert):
                lead = first_label - first_alert
                if lead > 0:
                    print(f"EARLY WARNING LEAD TIME: {int(lead)} hours prior to dataset positive window!")
                elif lead == 0:
                    print("Detection: Exactly on time with dataset window.")
                else:
                    print(f"Detection: Alert fired {int(abs(lead))} hours after window start.")

        print("-" * 70)
        print("Hourly Highlights (Hours with elevated risk or alerts):")
        interesting = result[(result["predicted_alert"] == 1) | (result["risk_probability"] >= threshold * 0.7)]
        if len(interesting) == 0:
            interesting = result.tail(5)
        display_cols = ["hour_index", "risk_probability", "predicted_alert", "sirs_score", "qsofa_score", "news2_score", "mews_score", "sofa_total", "shock_index"]
        if "actual_sepsis_label" in result.columns:
            display_cols.append("actual_sepsis_label")
        print(interesting[display_cols].to_string(index=False))
        print("=" * 70)
        print(f"Full predictions saved to: {args.output}")
        print("Research demonstration only; not for clinical care.\n")


if __name__ == "__main__":
    main()
