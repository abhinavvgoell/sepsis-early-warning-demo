import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.alert_policy_grid import evaluate_test, group_bounds, k_of_n_risk, utility_components
from src.app import load_model_bundle, compute_clinical_flags
from src.improved_pipeline import make_features_v2
from src.run_pipeline import ID_COLUMN, TARGET, make_features


def evaluate_100_patients():
    t0 = time.time()
    bundle = load_model_bundle()
    cls_bundle = bundle['classical_bundle']

    # Read held-out test patient registry (6,051 patients never seen during training)
    test_meta = pd.read_csv('outputs/patient_alert_analysis.csv')
    septic_ids = test_meta[test_meta['has_sepsis'] == True]['patient_id'].tolist()
    nonseptic_ids = test_meta[test_meta['has_sepsis'] == False]['patient_id'].tolist()

    # Reproducibly select 100 independent test patients (20 septic, 80 non-septic)
    rng = random.Random(20261029)
    sample_septic = rng.sample(septic_ids, 20)
    sample_nonseptic = rng.sample(nonseptic_ids, 80)
    sample_ids = sample_septic + sample_nonseptic
    rng.shuffle(sample_ids)

    # Locate and read the 100 patient PSV files
    raw_dir = Path('data/raw')
    file_map = {p.stem: p for p in raw_dir.rglob('*.psv')}
    records = []
    patient_lengths = []
    for pid in sample_ids:
        df = pd.read_csv(file_map[pid], sep='|')
        df[ID_COLUMN] = pid
        df['hour_index'] = np.arange(len(df), dtype=np.int32)
        records.append(df)
        patient_lengths.append(len(df))

    combined = pd.concat(records, ignore_index=True)
    y_hourly = combined[TARGET].to_numpy(dtype=np.int8)
    starts, ends = group_bounds(combined[ID_COLUMN].to_numpy())

    # PhysioNet official normalized challenge utility components
    _, reward, miss, septic_mask, timely_window, best_u, inactive_u = utility_components(
        combined, starts, ends
    )
    denom = best_u - inactive_u

    # 1. Feature matrices
    feat_v2 = make_features_v2(combined).reindex(columns=bundle['feature_names'])
    feat_v1, _, _ = make_features(combined)
    feat_v1 = feat_v1.reindex(columns=cls_bundle['feature_names'])

    # 2. Model probability inferences
    raw_lgb = bundle['booster'].predict(feat_v2)
    cal_lgb = np.interp(
        raw_lgb,
        np.asarray(bundle['calibrator']['x_thresholds'], dtype=np.float64),
        np.asarray(bundle['calibrator']['y_thresholds'], dtype=np.float64),
        left=float(bundle['calibrator']['y_thresholds'][0]),
        right=float(bundle['calibrator']['y_thresholds'][-1]),
    )
    prob_hgb = cls_bundle['models']['hist_gradient_boosting'].predict_proba(feat_v1)[:, 1]
    prob_lr = cls_bundle['models']['logistic_regression'].predict_proba(feat_v1)[:, 1]

    thr_lgb = float(bundle['policy']['threshold'])
    thr_hgb = float(cls_bundle['thresholds']['hist_gradient_boosting'])
    thr_lr = float(cls_bundle['thresholds']['logistic_regression'])

    # Normalized risk consensus ratio:
    ratio_lgb = cal_lgb / thr_lgb
    ratio_hgb = prob_hgb / thr_hgb
    ratio_lr = prob_lr / thr_lr
    ratio_ens = 0.60 * ratio_lgb + 0.25 * ratio_hgb + 0.15 * ratio_lr
    prob_ens = ratio_ens * thr_lgb

    candidate_models = {
        'LightGBM (Headline Calibrated)': (cal_lgb, thr_lgb, 5, 1),
        'Ensemble (Normalized Consensus)': (prob_ens, thr_lgb, 5, 1),
        'Histogram Gradient Boosting': (prob_hgb, thr_hgb, 5, 1),
        'Logistic Regression (Baseline)': (prob_lr, thr_lr, 5, 1),
    }

    results = []
    for name, (probs, thr, n, k) in candidate_models.items():
        risk = k_of_n_risk(probs.astype(np.float32), starts, ends, n=n, k=k)
        metrics = evaluate_test(
            combined, risk, thr, starts, ends, septic_mask, timely_window, reward, miss, denom
        )
        pr_auc = float(average_precision_score(y_hourly, probs))
        roc_auc = float(roc_auc_score(y_hourly, probs))

        # Calculate clinical lead time (hours between alert and sepsis window)
        lead_times = []
        for start, end in zip(starts, ends):
            p_y = y_hourly[start:end]
            if np.any(p_y == 1):
                onset = int(np.argmax(p_y == 1)) + 6
                lo, hi = max(0, onset - 12), min(len(p_y), onset + 4)
                p_pred = (risk[start:end] >= thr).astype(int)
                hits = np.flatnonzero(p_pred[lo:hi])
                if len(hits) > 0:
                    lead_times.append(onset - (lo + int(hits[0])))

        med_lead = float(np.median(lead_times)) if lead_times else 0.0

        results.append({
            'Model': name,
            'Decision Threshold': round(thr, 4),
            'Hourly ROC-AUC': round(roc_auc, 4),
            'Hourly PR-AUC': round(pr_auc, 4),
            'Challenge Utility': round(metrics['normalized_challenge_utility'], 4),
            'Sepsis Window Coverage': f"{metrics['sepsis_alert_window_coverage']:.1%}",
            'Non-Sepsis False Alert Rate': f"{metrics['nonsepsis_patient_false_alert_rate']:.1%}",
            'Hourly Sensitivity': f"{metrics['sensitivity']:.1%}",
            'Hourly Specificity': f"{metrics['specificity']:.1%}",
            'Median Lead Time (hrs)': f"{med_lead:.1f} hrs",
            'Timely Sepsis Detections': f"{metrics['timely_sepsis_patients']}/{int(septic_mask.sum())}",
            'False Positive Patients': f"{metrics['nonsepsis_patients_with_alert']}/{int((~septic_mask).sum())}",
        })

    table = pd.DataFrame(results)
    out_dir = Path('outputs')
    table.to_csv(out_dir / '100_unseen_patients_evaluation.csv', index=False)

    print('\n========================================================================================')
    print('INDEPENDENT TEST EVALUATION: 100 UNSEEN PATIENTS (4,318 HOURLY OBSERVATIONS)')
    print('========================================================================================')
    print(table.to_string(index=False))
    print('========================================================================================')
    print(f'Cohort profile: 20 septic patients, 80 non-septic patients (PhysioNet CinC 2019 test split).')
    print(f'Stay durations: Min {min(patient_lengths)}h, Max {max(patient_lengths)}h, Median {np.median(patient_lengths):.0f}h.')
    print(f'Evaluation finished in {time.time()-t0:.2f}s.')


if __name__ == '__main__':
    evaluate_100_patients()
