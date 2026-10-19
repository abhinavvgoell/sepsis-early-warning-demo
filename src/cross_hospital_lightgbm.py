"""Cross-hospital evaluation for the project's LightGBM pipeline.

Each direction fits on 85% of source-hospital patients, selects the 1-of-5
threshold on the remaining source patients, and evaluates once on the other
hospital group. Does not modify the demo bundle.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.improved_pipeline import Split, fit_lgb, make_features_v2, search_policy
from src.run_pipeline import ID_COLUMN, SEED, TARGET, read_patients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_hospital_experiment"))
    parser.add_argument("--results", type=Path, default=Path("outputs/improved/improved_results.json"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = json.loads(args.results.read_text(encoding="utf-8"))
    params, rounds = training["best_params"], int(training["rounds"])

    site_data: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for site in ("training_setA", "training_setB"):
        print(f"Reading {site} and preparing causal LightGBM features...", flush=True)
        raw = read_patients(args.data_dir / site).sort_values([ID_COLUMN, "hour_index"], kind="stable").reset_index(drop=True)
        y = raw[TARGET].to_numpy(dtype=np.int8)
        ids = raw[ID_COLUMN].to_numpy()
        x = make_features_v2(raw)
        site_data[site] = (x, y, ids)
        print(f"Prepared {site}: {len(ids):,} hourly rows, {len(np.unique(ids)):,} patients, {x.shape[1]} features", flush=True)
        del raw
        gc.collect()

    results: dict[str, dict] = {}
    for source, destination in (("training_setA", "training_setB"), ("training_setB", "training_setA")):
        src_x, src_y, src_ids = site_data[source]
        dst_x, dst_y, dst_ids = site_data[destination]
        patient_y = pd.Series(src_y).groupby(src_ids).max()
        fit_ids, validation_ids = train_test_split(
            patient_y.index.to_numpy(), test_size=0.15, random_state=SEED,
            stratify=patient_y.to_numpy(),
        )
        fit_mask = np.isin(src_ids, fit_ids)
        validation_mask = np.isin(src_ids, validation_ids)
        validation_split = Split(src_ids[validation_mask], src_y[validation_mask])
        destination_split = Split(dst_ids, dst_y)

        print(f"Training LightGBM {source}->{destination} on {len(fit_ids):,} source patients...", flush=True)
        model = fit_lgb(params, src_x.loc[fit_mask], src_y[fit_mask], rounds)
        source_validation_prob = model.predict(src_x.loc[validation_mask])
        selection, _ = search_policy(validation_split, source_validation_prob, [(5, 1)], None)
        threshold = float(selection["threshold"])
        print(f"Evaluating destination at source-selected threshold {threshold:.6g}...", flush=True)
        destination_prob = model.predict(dst_x)
        dest_metrics = destination_split.metrics(destination_split.risk(destination_prob, 5, 1), threshold)
        results[f"{source}_to_{destination}"] = {
            "source_fit_patients": int(len(fit_ids)),
            "source_threshold_validation_patients": int(len(validation_ids)),
            "destination_patients": int(len(np.unique(dst_ids))),
            "threshold_selected_on_source_validation": threshold,
            "source_validation_utility": float(selection["utility"]),
            "source_validation_nonsepsis_patient_alert_rate": float(selection["far"]),
            "destination_metrics": dest_metrics,
        }
        print(f"{source}->{destination}: coverage={dest_metrics['sepsis_alert_window_coverage']:.1%}, "
              f"non-sepsis patient alerts={dest_metrics['nonsepsis_patient_false_alert_rate']:.1%}, "
              f"utility={dest_metrics['normalized_challenge_utility']:.4f}", flush=True)
        del model
        gc.collect()

    payload = {
        "model": "LightGBM v2 pipeline using the selected project hyperparameters and rounds",
        "threshold_protocol": "Each direction uses an 85/15 patient split within the source site; select 1-of-5 threshold by challenge utility on source validation; evaluate once on all destination-site patients.",
        "hyperparameters": params,
        "boosting_rounds": rounds,
        "caveats": [
            "The destination hospital group was not used for training or threshold selection.",
            "This retrospective cross-hospital experiment is not prospective or clinical validation.",
            "The project demo's saved model is not overwritten.",
            "Scores here are raw LightGBM scores; the saved demo model also applies its fitted isotonic mapping. Thresholds are retuned on source validation, so this evaluates ranking and alert-policy transfer rather than the demo's fixed numeric cutoff.",
        ],
        "directions": results,
    }
    out_json = args.output_dir / "lightgbm_cross_hospital_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for direction, value in results.items():
        m = value["destination_metrics"]
        rows.append({"direction": direction,
                     "threshold": value["threshold_selected_on_source_validation"],
                     "source_validation_false_alert_rate": value["source_validation_nonsepsis_patient_alert_rate"],
                     "destination_warning_window_coverage": m["sepsis_alert_window_coverage"],
                     "destination_nonsepsis_patient_alert_rate": m["nonsepsis_patient_false_alert_rate"],
                     "destination_utility": m["normalized_challenge_utility"],
                     "destination_precision": m["precision"],
                     "destination_sensitivity": m["sensitivity"],
                     "destination_accuracy": m["accuracy"],
                     "destination_patients": value["destination_patients"]})
    pd.DataFrame(rows).to_csv(args.output_dir / "lightgbm_cross_hospital_comparison.csv", index=False)
    lines = [
        "# LightGBM cross-hospital evaluation", "",
        "This tests the current LightGBM pipeline configuration, retrained separately within each source hospital. For each direction, 85% of source patients were used for fitting; the remaining 15% selected the 1-of-5 cutoff by challenge utility. The destination cohort was held out from both fitting and threshold selection.", "",
        "| Train → evaluate | Source-selected cutoff | Source validation false-alert rate | Destination warning-window coverage | Destination non-sepsis patients alerted | Destination utility | Destination hourly sensitivity | Destination hourly precision | Destination hourly accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['direction']} | {r['threshold']:.6g} | {r['source_validation_false_alert_rate']:.1%} | {r['destination_warning_window_coverage']:.1%} | {r['destination_nonsepsis_patient_alert_rate']:.1%} | {r['destination_utility']:.4f} | {r['destination_sensitivity']:.1%} | {r['destination_precision']:.1%} | {r['destination_accuracy']:.1%} |")
    lines += ["", "## Interpretation", "",
              "Compare destination warning-window coverage and non-sepsis patient alert rate with the source validation values. A sizeable difference suggests the model or its threshold does not transfer cleanly. Hourly accuracy is included for completeness but can be misleading because negative hourly rows dominate.", "",
              "The saved demo model was not modified. This retrospective evaluation on the two PhysioNet hospital groups does not establish clinical readiness.", "",
              "Reproduce with: `python -m src.cross_hospital_lightgbm`", ""]
    (args.output_dir / "lightgbm_cross_hospital_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LightGBM transfer results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
