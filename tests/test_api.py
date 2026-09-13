"""
Integration tests for the secure inference REST API.
"""

import unittest
import json
import numpy as np
from health_pipeline.api.app import create_app
from health_pipeline.config import SECURITY_CONFIG


class TestSecureAPI(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.valid_token = SECURITY_CONFIG.API_AUTH_TOKEN
        self.auth_headers = {"Authorization": f"Bearer {self.valid_token}"}

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ONLINE")

    def test_authentication_enforcement(self):
        # 1. Missing auth
        res_no_auth = self.client.post("/v1/predict", json={})
        self.assertEqual(res_no_auth.status_code, 401)

        # 2. Invalid auth token
        bad_headers = {"Authorization": "Bearer invalid_secret_token_123"}
        res_bad_auth = self.client.post("/v1/predict", headers=bad_headers, json={})
        self.assertEqual(res_bad_auth.status_code, 401)

    def test_payload_sanitization_and_validation(self):
        # 1. Empty payload
        res = self.client.post("/v1/predict", headers=self.auth_headers, json={})
        self.assertEqual(res.status_code, 400)

        # 2. Extreme non-physiological voltage
        bad_voltage_sig = [100.0] * 500  # 100 mV is outside plausible physical range
        res_volt = self.client.post("/v1/predict", headers=self.auth_headers, json={"signal": bad_voltage_sig})
        self.assertEqual(res_volt.status_code, 400)
        self.assertIn("exceeds physical ECG range", res_volt.get_json()["message"])

    def test_sqi_rejection_of_flatline(self):
        # Flatline signal should be rejected by SQI with HTTP 422
        flatline = [0.0] * 1000
        res = self.client.post("/v1/predict", headers=self.auth_headers, json={"signal": flatline, "fs": 360.0})
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertEqual(data["status"], "REJECTED_LOW_QUALITY")
        self.assertIn("Flatline", data["rejection_reason"])

    def test_successful_inference(self):
        # Generate valid 5-sec ECG signal
        fs = 360.0
        t = np.linspace(0, 5.0, int(5.0 * fs), endpoint=False)
        valid_ecg = np.zeros_like(t)
        for beat in np.arange(0.5, 4.8, 0.8):
            valid_ecg += 1.2 * np.exp(-((t - beat) ** 2) / (2 * (0.03 ** 2)))

        payload = {
            "signal": valid_ecg.tolist(),
            "fs": fs,
            "ppg_signal": None,
            "spo2_signal": [98.5] * 100
        }

        res = self.client.post("/v1/predict", headers=self.auth_headers, json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("prediction", data)
        self.assertIn("clinical_condition", data["prediction"])
        self.assertIn("calibrated_confidence", data["prediction"])
        self.assertIn("triage_risk_tier", data["prediction"])
        self.assertIn(data["prediction"]["triage_risk_tier"], ["LOW_RISK", "MONITOR", "CRITICAL"])
        self.assertIn("clinical_vitals_summary", data)

        # Check security headers
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")


if __name__ == "__main__":
    unittest.main()
