"""Patient-split A-to-B/B-to-A evaluation of the project HGB model.

Thresholds are selected on a held-out source-hospital validation group. The
destination hospital is only used for the final evaluation. This experiment
does not modify or replace any model bundle used by the demo.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components
from src.run_pipeline import ID_COLUMN, SEED, TARGET, make_features, read_patients


def score_split(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    patient_ids = frame[ID_COLUMN].to_numpy()
    starts, ends = group_bounds(patient_ids)
    _, reward, miss, septic, window, best, inactive = utility_components(frame, starts, ends)
    risk = k_of_n_risk(probabilities.astype(np.float32), starts, ends, 5, 1)
    result = evaluate_test(frame, risk, threshold, starts, ends, septic, window,
                           reward, miss, best - inactive)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    result["row_pr_auc"] = float(average_precision_score(y, probabilities))
    result["row_roc_auc"] = float(roc_auc_score(y, probabilities))
    result["positive_row_rate"] = float(y.mean())
    result["threshold"] = float(threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_hospital_experiment"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for source, destination in (("training_setA", "training_setB"), ("training_setB", "training_setA")):
        print(f"Loading {source} and {destination}...", flush=True)
        src_raw = read_patients(args.data_dir / source)
        dst_raw = read_patients(args.data_dir / destination)
        src_x, src_y, src_ids = make_features(src_raw)
        dst_x, dst_y, dst_ids = make_features(dst_raw)
        src_x = src_x.astype(np.float32)
        dst_x = dst_x.astype(np.float32)
        src_labels = src_y.to_numpy(dtype=np.int8)
        dst_labels = dst_y.to_numpy(dtype=np.int8)
        src_ids = src_ids.to_numpy()
        dst_ids = dst_ids.to_numpy()
        src_patient_y = pd.Series(src_labels).groupby(src_ids).max()
        fit_ids, threshold_ids = train_test_split(
            src_patient_y.index.to_numpy(), test_size=0.15, random_state=SEED,
            stratify=src_patient_y.to_numpy(),
        )
        fit_mask, threshold_mask = np.isin(src_ids, fit_ids), np.isin(src_ids, threshold_ids)
        dest_mask = np.ones(len(dst_ids), dtype=bool)
        # HGB cannot handle an all-missing feature in its fit set; align both
        # source validation and destination to columns observable in source fit.
        columns = src_x.columns[src_x.loc[fit_mask].notna().any().to_numpy()]
        src_fit = src_x.loc[fit_mask, columns]
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=150, max_leaf_nodes=20,
            l2_regularization=1.0, class_weight="balanced", random_state=SEED,
        )
        print(f"Fitting HGB on {source} ({len(fit_ids):,} patients, {len(columns)} features)...", flush=True)
        model.fit(src_fit, src_labels[fit_mask])
        p_threshold = model.predict_proba(src_x.loc[threshold_mask, columns])[:, 1]
        threshold_frame = pd.DataFrame({ID_COLUMN: src_ids[threshold_mask], TARGET: src_labels[threshold_mask]})
        starts, ends = group_bounds(threshold_frame[ID_COLUMN].to_numpy())
        _, reward, miss, septic, _, best, inactive = utility_components(threshold_frame, starts, ends)
        best_choice = None
        for threshold in np.arange(0.05, 0.951, 0.025):
            risk = k_of_n_risk(p_threshold.astype(np.float32), starts, ends, 5, 1)
            metrics = evaluate_test(threshold_frame, risk, float(threshold), starts, ends, septic,
                                    np.zeros(len(threshold_frame), dtype=bool), reward, miss, best - inactive)
            choice = (metrics["normalized_challenge_utility"], -metrics["nonsepsis_patient_false_alert_rate"], float(threshold))
            if best_choice is None or choice > best_choice[0]:
                best_choice = (choice, float(threshold), metrics)
        selected_threshold = best_choice[1]
        threshold_metrics = best_choice[2]

        destination_frame = pd.DataFrame({ID_COLUMN: dst_ids, TARGET: dst_labels})
        print(f"Scoring {destination} at source-selected cutoff {selected_threshold:.3f}...", flush=True)
        p_destination = model.predict_proba(dst_x.loc[dest_mask, columns])[:, 1]
        destination_metrics = score_split(destination_frame, p_destination, selected_threshold)
        all_results[f"{source}_to_{destination}"] = {
            "source_patients_fit": int(len(fit_ids)),
            "source_patients_threshold_selection": int(len(threshold_ids)),
            "destination_patients": int(pd.Series(dst_ids).nunique()),
            "source_validation_selected_threshold": selected_threshold,
            "source_validation_utility": threshold_metrics["normalized_challenge_utility"],
            "source_validation_nonsepsis_patient_alert_rate": threshold_metrics["nonsepsis_patient_false_alert_rate"],
            "destination_metrics": destination_metrics,
            "features_used": int(len(columns)),
        }
        print(f"{source}->{destination}: PR-AUC={destination_metrics['row_pr_auc']:.4f}, "
              f"ROC-AUC={destination_metrics['row_roc_auc']:.4f}, "
              f"warning coverage={destination_metrics['sepsis_alert_window_coverage']:.1%}, "
              f"non-sepsis patient alerts={destination_metrics['nonsepsis_patient_false_alert_rate']:.1%}", flush=True)
        del src_raw, dst_raw, src_x, dst_x, model
        gc.collect()

    payload = {
        "experiment": "Cross-hospital generalization, patient-split source validation, HGB baseline, 1-of-5 rule",
        "selection_protocol": "For each direction, train on 85% of source-hospital patients; choose the 1-of-5 cutoff on the other 15% of source patients by normalized challenge utility; evaluate once on all destination-hospital patients.",
        "limitations": [
            "This is a retrospective public-dataset experiment, not clinical validation.",
            "The two PhysioNet cohorts are hospital groups, not a substitute for validation at a prospective deployment site.",
            "Hourly rows are correlated; row PR-AUC/ROC-AUC are supplementary to patient-level warning and false-alert metrics.",
        ],
        "directions": all_results,
    }
    (args.output_dir / "cross_hospital_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for direction, values in all_results.items():
        m = values["destination_metrics"]
        rows.append({"direction": direction, "threshold": values["source_validation_selected_threshold"],
                     "row_pr_auc": m["row_pr_auc"], "row_roc_auc": m["row_roc_auc"],
                     "accuracy": m["accuracy"], "sensitivity": m["sensitivity"],
                     "precision": m["precision"], "warning_window_coverage": m["sepsis_alert_window_coverage"],
                     "nonsepsis_patient_alert_rate": m["nonsepsis_patient_false_alert_rate"],
                     "normalized_challenge_utility": m["normalized_challenge_utility"]})
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "cross_hospital_comparison.csv", index=False)
    md = ["# Cross-hospital evaluation", "",
          "The cutoff for each direction was selected using a held-out 15% of source-hospital patients. The destination hospital was not used to choose the cutoff. Both directions use the same 1-of-5 rule and original HistGradientBoosting configuration.", "",
          "| Train → evaluate | Cutoff | PR-AUC (hourly) | ROC-AUC (hourly) | Accuracy (hourly) | Sensitivity (hourly) | Precision (hourly) | Sepsis warning-window coverage | Non-sepsis patients with alert | Challenge utility |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append(f"| {row['direction']} | {row['threshold']:.3f} | {row['row_pr_auc']:.4f} | {row['row_roc_auc']:.4f} | {row['accuracy']:.1%} | {row['sensitivity']:.1%} | {row['precision']:.1%} | {row['warning_window_coverage']:.1%} | {row['nonsepsis_patient_alert_rate']:.1%} | {row['normalized_challenge_utility']:.4f} |")
    md += ["", "## Interpretation", "",
           "A large drop in warning-window coverage, PR-AUC, or utility across a direction suggests the model does not transfer reliably between these hospital cohorts. A high hourly accuracy alone is not sufficient because negative hourly rows dominate.", "",
           "This is retrospective research on the PhysioNet challenge cohorts. It does not show clinical readiness or justify use for patient care.", "",
           "Reproduce from the project directory with: `python -m src.cross_hospital_experiment`", ""]
    (args.output_dir / "cross_hospital_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved report in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
