# Early Sepsis Warning from ICU Time Series

A reproducible research prototype and clinical surveillance workstation built on the PhysioNet/Computing in Cardiology Challenge 2019 ICU cohort. The system estimates whether a patient's current hourly observation falls into an early sepsis deterioration window using causal historical and current measurements.

> **Research prototype only — not for clinical use.** This project has not been clinically validated. Its scores are model outputs, not medical probabilities. Do not use it to diagnose, treat, triage, or make care decisions. Use synthetic data for the demo; never enter identifiable patient information.

---

## Key Highlights

- **Bio-Mechanical Intelligence Workstation UI**: A light/neutral architectural aesthetic inspired by precision medical blueprints (`#E8EAEB` foggy canvas, pure white cards `#FFFFFF` with geometric crop-mark corner accents, deep charcoal typography `#1A1C1E`, and `#D9661D` burnt orange precision alert lighting).
- **Dynamic Real-Time Telemetry**:
  - **Dynamic ECG Rhythm Monitor**: Canvas-rendered P-Q-R-S-T cardiac waveform that dynamically adjusts its sweep rate and pulse display according to patient heart rate.
  - **Circular Sepsis Conduit Arc Gauge**: Animated circular gauge rendering calibrated likelihood (0–100%).
  - **Systemic Organ Stress Telemetry**: Circuit schematic tracking Cardiovascular (MAP/Shock Index), Pulmonary (NEWS2/Tachypnea), Renal/Hepatic (SOFA score), and Metabolic (Lactate) stress nodes.
  - **Precision Bedside Biomarkers**: Real-time evaluation of SIRS (&ge;2 criteria), qSOFA (&ge;2 high risk), NEWS2 (UK Royal College), MEWS, SOFA organ failure, and Shock Index (&le;0.7 normal).
  - **Longitudinal Score Trajectory & Attribution**: SVG trend plotting across hourly records with alert threshold boundaries, interactive hour scrubbing, and per-hour TreeSHAP logit margin attributions.
  - **Dual Workstation Reset Controls**: Comprehensive reset buttons located in the top navigation header and action toolbar with smooth SVG rotation, restoring all inputs, demographics, and surveillance modules to clean initial state.
  - **Research Evidence Lightbox**: One-click header trigger to inspect PhysioNet 2019 validation curves, ROC-AUC, PR-AUC, and reliability calibration plots.
- **Causal Feature Engineering**: Over 200 causal rolling, exponential decay, trend, and freshness features strictly computed from past and present hours (zero future data leakage).
- **Alert Persistence Policies**: Noise-reduction decision policies including the recommended 1-of-5 rule (matched false alarm rate ~24.9%), 1-of-2 utility-optimal rule, and 1-of-1 immediate single-hour threshold.
- **Deliverables**: Comprehensive executive and technical slide deck included in [`deliverables/Sepsis_Early_Warning_Project.pptx`](deliverables/Sepsis_Early_Warning_Project.pptx).

---

## Dataset and Label Definition

Download open-access Challenge 2019 files from [PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/). The included downloader uses the public PhysioNet S3 mirror and Python standard library:

```bash
python -m src.download_data --output-dir data/raw
```

The loader searches subfolders recursively. Follow the dataset citation and license terms on the PhysioNet page. Do not commit patient data.

Each row is an hourly ICU observation and `SepsisLabel` is the challenge target. The challenge shifts positive labels to begin six hours before the clinical onset time and keeps them positive afterward. The intended task is early detection using current and past data. Because labels also stay positive after onset, row-level classification metrics alone can hide late alerts; the project therefore includes patient-level alert-timing and false-alarm analysis.

---

## Quick Start

### 1. Installation

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Launch Local Surveillance Workstation

The included model bundles are enough to launch the workstation immediately without downloading the full multi-gigabyte dataset:

**Windows (Single Click):**
Double-click `start_app.bat` or run in PowerShell:
```powershell
.\start_app.bat
```
This automatically verifies the virtual environment, starts the Python server on `http://127.0.0.1:8766`, and launches the browser workstation.

**macOS / Linux:**
```bash
python3 -m src.app
```
Then navigate to **[http://127.0.0.1:8766](http://127.0.0.1:8766)** in your browser. Stop the server anytime with `Ctrl+C`.

### 3. Automated Test Suite

To verify all 27 unit, clinical scoring, and pipeline regression tests:

```bash
# Windows batch runner
.\run_tests.bat

# Or direct Python
python -m unittest discover tests -v
```

---

## Generating Predictions via CLI

To generate hourly risk scores for a single patient record using the saved model bundles:

```bash
# Default: headline LightGBM model with 1-of-5 alert persistence rule
python -m src.predict --input data/raw/training_setA/p000002.psv --model lightgbm

# Classical comparison models
python -m src.predict --input data/raw/training_setA/p000002.psv --model hist_gradient_boosting
python -m src.predict --input data/raw/training_setA/p000002.psv --model logistic_regression
```

This outputs `outputs/patient_risk_predictions.csv` with hourly risk probabilities and alert flags.

---

## Full Pipeline Reproduction

To download the raw data, train models, and run the complete evaluation:

```bash
# Download raw PhysioNet 2019 dataset
python -m src.download_data --output-dir data/raw

# Run baseline feature pipeline and model training
python -m src.run_pipeline --data-dir data/raw --output-dir outputs

# Run alert noise reduction and timing analysis
python -m src.analyze_alerts --output-dir outputs

# Retrain LightGBM candidate with grouped cross-validation
python -m src.improved_pipeline --data-dir data/raw --output-dir outputs/improved --cache-dir outputs/improved/cache
```

---

## Evaluation Design & Methodological Integrity

- **Patient-Level Split**: Patients, never rows, are partitioned into disjoint train/validation/test splits to eliminate intra-patient data leakage across time.
- **Strict Causality**: Rolling means, standard deviations, and observation counts exclude the current hour to ensure features use only past and current measurements.
- **Dual Evaluation Metrics**: Because ICU sepsis labels are heavily imbalanced, precision-recall AUC (PR-AUC), ROC-AUC, sensitivity, specificity, and confusion counts are evaluated alongside patient-level early alert utility.
- **Isotonic Probability Calibration**: Non-parametric isotonic regression calibrates raw classifier scores into empirical posterior probabilities.
- **False Alarm Mitigation**: Comparative study of alert persistence policies (k-of-n rules, rolling EWMA, rising score filters, and refractory cooldowns) to prevent clinical alarm fatigue.

---

## Results and Research Reports

- [final_evaluation_results.md](outputs/final_evaluation/final_evaluation_results.md): Clean, fair evaluation of the frozen LightGBM candidate against the HGB baseline on matched training partitions with separate calibration, bootstrap 95% confidence intervals, and paired comparison.
- [alert_policy_v2_results.md](outputs/alert_policy_v2/alert_policy_v2_results.md): Noise-reduction study comparing the baseline 1-of-5 rule against rising-score (3h mean), EWMA, and refractory cooldown rules tuned under strict non-sepsis false-alert rate caps (10%, 15%, 20%).
- [final_project_report.md](outputs/final_project_report.md): Comprehensive project summary, candidate metric comparison, evaluation caveats, limitations, and reproduction details.
- [initial_results.md](outputs/initial_results.md): Patient-held-out comparison and patient-level alert review.
- [alert_policy_results.md](outputs/alert_policy_results.md): Comparison of immediate, persistence-based, and cooldown alert rules.
- [alert_policy_grid_results.md](outputs/alert_policy_grid_results.md): Full grid search of all 36 k-of-n alert rules from 1 to 8 hours (`outputs/policy_grid/alert_policy_grid.csv`).
- [trend_feature_results.md](outputs/trend_experiment/trend_feature_results.md): Vital-sign trend feature experiment adding rate-of-change and 6-hour variability indicators.
- [generalization_results.md](outputs/generalization_experiment/generalization_results.md): Analysis of train-versus-validation gaps and regularized tree blends.
- [last_observation_results.md](outputs/last_observation_experiment/last_observation_results.md): Causal last-value and measurement freshness feature evaluation.
- [improved_results.md](outputs/improved/improved_results.md): 5-fold grouped CV results for the improved LightGBM model.
