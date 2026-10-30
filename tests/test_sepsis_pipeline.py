"""Comprehensive automated test suite for the sepsis early warning pipeline.

Covers:
- Numeric input parsing and missing-value resilience.
- Bedside ICU clinical markers (SIRS, qSOFA, Shock Index, hypotension, lactate).
- Feature engineering robustness (v1 and v2, missing IDs/hours, sparse inputs).
- Alert persistence policy calculation (k-of-n).
- CLI inference across models (LightGBM, HGB, Logistic Regression) and delimiters (PSV, CSV, TSV).
- Backend prediction API logic and JSON serialization safety.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.alert_policy_grid import k_of_n_risk
from src.app import (
    DEFAULT_MODEL,
    compute_clinical_flags,
    load_model_bundle,
    parse_number,
    predict_records,
    sanitize_for_json,
)
from src.improved_pipeline import make_features_v2
from src.predict import compute_clinical_markers, run_prediction
from src.run_pipeline import ID_COLUMN, make_features


class TestInputParsing(unittest.TestCase):
    """Test parse_number on clean and dirty clinical inputs."""

    def test_valid_numbers(self):
        self.assertEqual(parse_number(80.5, "HR"), 80.5)
        self.assertEqual(parse_number("98.6", "Temp"), 98.6)
        self.assertEqual(parse_number(0, "Age"), 0.0)
        self.assertEqual(parse_number(-5.0, "BaseExcess"), -5.0)

    def test_missing_and_blank_representations(self):
        blanks = [None, "", "   ", "nan", "NaN", "null", "none", "na", "n/a", "?", "-", "--", np.nan]
        for val in blanks:
            result = parse_number(val, "HR")
            self.assertTrue(math.isnan(result), f"Expected NaN for input {val!r}")

    def test_invalid_types_raise(self):
        with self.assertRaises(ValueError):
            parse_number("invalid_string", "HR")
        with self.assertRaises(ValueError):
            parse_number(True, "HR")
        with self.assertRaises(ValueError):
            parse_number(False, "HR")
        with self.assertRaises(ValueError):
            parse_number(1e9, "HR")  # out of bounds


class TestClinicalMarkers(unittest.TestCase):
    """Test rule-based ICU sepsis scores."""

    def test_sirs_calculation(self):
        # All 4 SIRS criteria met
        flags = compute_clinical_flags({"HR": 110, "Temp": 38.5, "Resp": 24, "WBC": 15.0})
        self.assertEqual(flags["sirs_score"], 4)
        self.assertTrue(flags["sirs_positive"])

        # No SIRS criteria met
        flags_norm = compute_clinical_flags({"HR": 75, "Temp": 37.0, "Resp": 16, "WBC": 7.0})
        self.assertEqual(flags_norm["sirs_score"], 0)
        self.assertFalse(flags_norm["sirs_positive"])

    def test_qsofa_calculation(self):
        # High risk: Resp >= 22 and SBP <= 100
        flags = compute_clinical_flags({"Resp": 25, "SBP": 90})
        self.assertEqual(flags["qsofa_score"], 2)
        self.assertTrue(flags["qsofa_positive"])

        # Normal
        flags_norm = compute_clinical_flags({"Resp": 18, "SBP": 120})
        self.assertEqual(flags_norm["qsofa_score"], 0)
        self.assertFalse(flags_norm["qsofa_positive"])

    def test_shock_index(self):
        # HR 120 / SBP 80 = 1.5 (> 0.9 elevated shock index)
        flags = compute_clinical_flags({"HR": 120, "SBP": 80})
        self.assertEqual(flags["shock_index"], 1.5)
        self.assertTrue(flags["shock_index_elevated"])

        # SBP missing or zero
        flags_no_sbp = compute_clinical_flags({"HR": 120, "SBP": 0})
        self.assertIsNone(flags_no_sbp["shock_index"])

    def test_hypotension_and_lactate(self):
        flags = compute_clinical_flags({"MAP": 60, "Lactate": 3.2})
        self.assertTrue(flags["hypotension"])
        self.assertTrue(flags["hyperlactatemia"])

        flags_ok = compute_clinical_flags({"MAP": 80, "Lactate": 1.1})
        self.assertFalse(flags_ok["hypotension"])
        self.assertFalse(flags_ok["hyperlactatemia"])

    def test_news2_scoring(self):
        # Critical patient: High NEWS2
        flags = compute_clinical_flags({
            "Resp": 26,      # 3
            "O2Sat": 91,     # 3
            "Temp": 39.2,    # 2
            "SBP": 88,       # 3
            "HR": 135,       # 3
        })
        self.assertGreaterEqual(flags["news2_score"], 7)
        self.assertEqual(flags["news2_risk"], "High")
        self.assertTrue(flags["news2_any_extreme"])

        # Normal patient: Low NEWS2
        flags_norm = compute_clinical_flags({
            "Resp": 16,      # 0
            "O2Sat": 98,     # 0
            "Temp": 37.0,    # 0
            "SBP": 120,      # 0
            "HR": 75,        # 0
        })
        self.assertEqual(flags_norm["news2_score"], 0)
        self.assertEqual(flags_norm["news2_risk"], "Low")

    def test_mews_and_sofa_scoring(self):
        flags = compute_clinical_flags({
            "HR": 135, "SBP": 75, "Resp": 28, "Temp": 39.0,
            "Platelets": 45, "Bilirubin_total": 3.0, "Creatinine": 2.5, "MAP": 50, "Lactate": 4.5
        })
        self.assertIsNotNone(flags["mews_score"])
        self.assertGreater(flags["mews_score"], 4)
        self.assertIsNotNone(flags["sofa_total"])
        self.assertGreaterEqual(flags["sofa_total"], 6)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering resilience to incomplete input frames."""

    def test_make_features_v2_sparse_input(self):
        df = pd.DataFrame({"HR": [80.0, 95.0, 110.0], "SBP": [120.0, 110.0, 90.0]})
        features = make_features_v2(df)
        self.assertEqual(len(features), 3)
        self.assertEqual(features.shape[1], 200)
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_make_features_v1_sparse_input(self):
        df = pd.DataFrame({"HR": [80.0, 95.0], "Temp": [37.0, 38.2]})
        features, labels, patient_ids = make_features(df)
        self.assertEqual(len(features), 2)
        self.assertIn("HR", features.columns)
        self.assertEqual(len(patient_ids), 2)

    def test_single_row_make_features_v2(self):
        df = pd.DataFrame({"HR": [80.0]})
        features = make_features_v2(df)
        self.assertEqual(len(features), 1)
        self.assertEqual(features.shape[1], 200)


class TestAlertPolicy(unittest.TestCase):
    """Test k-of-n risk calculation."""

    def test_k_of_n_risk_1_of_5(self):
        probs = np.array([0.1, 0.7, 0.2, 0.3, 0.1, 0.1, 0.1], dtype=np.float32)
        starts = np.array([0])
        ends = np.array([len(probs)])
        risk = k_of_n_risk(probs, starts, ends, n=5, k=1)
        # Hour index 1 (0.7) should keep the alert score 0.7 active through hour index 5
        self.assertAlmostEqual(risk[1], 0.7, places=4)
        self.assertAlmostEqual(risk[2], 0.7, places=4)
        self.assertAlmostEqual(risk[3], 0.7, places=4)
        self.assertAlmostEqual(risk[4], 0.7, places=4)
        self.assertAlmostEqual(risk[5], 0.7, places=4)
        # By hour 6 (7th observation), the 0.7 rolls out of the 5-hour window
        self.assertAlmostEqual(risk[6], 0.3, places=4)

    def test_k_of_n_risk_1_of_1(self):
        probs = np.array([0.1, 0.7, 0.2], dtype=np.float32)
        starts = np.array([0])
        ends = np.array([len(probs)])
        risk = k_of_n_risk(probs, starts, ends, n=1, k=1)
        np.testing.assert_allclose(risk, probs, rtol=1e-5)


class TestPredictionCLI(unittest.TestCase):
    """Test run_prediction with different file formats and models."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()
        # Create temporary sample files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dir_path = Path(cls.temp_dir.name)

        rows = [
            "HR|O2Sat|Temp|SBP|MAP|Resp|WBC|Lactate|Age|Gender|ICULOS|SepsisLabel\n",
            "80|98|36.8|120|80|16|7.5|1.0|65|1|1|0\n",
            "95|96|37.5|110|75|20|9.2|1.4|65|1|2|0\n",
            "115|93|38.6|92|62|26|14.0|2.5|65|1|3|1\n",
        ]
        cls.psv_file = cls.dir_path / "patient.psv"
        cls.psv_file.write_text("".join(rows), encoding="utf-8")

        cls.csv_file = cls.dir_path / "patient.csv"
        cls.csv_file.write_text("".join(rows).replace("|", ","), encoding="utf-8")

        cls.tsv_file = cls.dir_path / "patient.tsv"
        cls.tsv_file.write_text("".join(rows).replace("|", "\t"), encoding="utf-8")

        cls.empty_file = cls.dir_path / "empty.csv"
        cls.empty_file.write_text("", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_predict_lightgbm_psv(self):
        res = run_prediction(self.psv_file, model_name="lightgbm", policy="1-of-5")
        self.assertEqual(len(res), 3)
        self.assertIn("risk_probability", res.columns)
        self.assertIn("predicted_alert", res.columns)
        self.assertIn("sirs_score", res.columns)
        # Hour 3 has elevated vitals
        self.assertGreater(res["risk_probability"].iloc[2], res["risk_probability"].iloc[0])

    def test_predict_hist_gradient_boosting_csv(self):
        res = run_prediction(self.csv_file, model_name="hist_gradient_boosting", policy="1-of-5")
        self.assertEqual(len(res), 3)
        self.assertIn("risk_probability", res.columns)

    def test_predict_logistic_regression_tsv(self):
        res = run_prediction(self.tsv_file, model_name="logistic_regression", policy="1-of-1")
        self.assertEqual(len(res), 3)
        self.assertIn("risk_probability", res.columns)

    def test_predict_empty_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            run_prediction(self.empty_file, model_name="lightgbm")
        self.assertIn("empty", str(ctx.exception).lower())


class TestAppBackend(unittest.TestCase):
    """Test web backend prediction handler and JSON serialization."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()

    def test_predict_records_lightgbm(self):
        request = {
            "metadata": {"Age": 68, "Gender": 1},
            "records": [
                {"HR": 80, "Temp": 36.8, "SBP": 120, "MAP": 80, "Resp": 16},
                {"HR": 115, "Temp": 38.6, "SBP": 92, "MAP": 62, "Resp": 26, "WBC": 14, "Lactate": 2.5},
            ],
            "model": "lightgbm",
            "policy": "1_of_5",
        }
        resp = predict_records(self.bundle, request)
        self.assertEqual(resp["hours_scored"], 2)
        self.assertIn("summary", resp)
        self.assertIn("hourly", resp)
        self.assertEqual(len(resp["hourly"]), 2)
        # Hour 2 should have attributions
        self.assertIsNotNone(resp["hourly"][1]["explanation"])

        # Verify JSON serialization with allow_nan=False
        sanitized = sanitize_for_json(resp)
        json_bytes = json.dumps(sanitized, allow_nan=False).encode("utf-8")
        parsed = json.loads(json_bytes)
        self.assertEqual(parsed["hours_scored"], 2)

    def test_predict_records_all_models(self):
        request = {
            "metadata": {"Age": 60, "Gender": 0},
            "records": [{"HR": 105, "Temp": 38.2, "SBP": 95, "MAP": 65, "Resp": 24}],
            "policy": "1_of_2",
        }
        for model_key in ["lightgbm", "ensemble", "logistic_regression", "hist_gradient_boosting"]:
            request["model"] = model_key
            resp = predict_records(self.bundle, request)
            self.assertEqual(resp["model_key"], model_key)
            self.assertEqual(len(resp["hourly"]), 1)
            sanitized = sanitize_for_json(resp)
            json_str = json.dumps(sanitized, allow_nan=False)
            self.assertTrue(len(json_str) > 0)

    def test_sanitize_for_json_converts_nan_to_none(self):
        data = {
            "normal_float": 1.23,
            "nan_float": float("nan"),
            "inf_float": float("inf"),
            "numpy_float": np.float32(4.56),
            "numpy_nan": np.float32(np.nan),
            "numpy_int": np.int64(42),
            "numpy_bool": np.bool_(True),
            "nested_list": [1.0, float("nan"), {"sub_nan": float("-inf")}],
        }
        clean = sanitize_for_json(data)
        self.assertIsNone(clean["nan_float"])
        self.assertIsNone(clean["inf_float"])
        self.assertIsNone(clean["numpy_nan"])
        self.assertEqual(clean["numpy_int"], 42)
        self.assertTrue(clean["numpy_bool"])
        self.assertIsNone(clean["nested_list"][1])
        self.assertIsNone(clean["nested_list"][2]["sub_nan"])

        # Ensure json.dumps passes without allow_nan error
        dumped = json.dumps(clean, allow_nan=False)
        self.assertIn('"nan_float": null', dumped)


if __name__ == "__main__":
    unittest.main()
