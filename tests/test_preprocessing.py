"""
Unit tests for signal cleaning, SQI validation, and feature extraction.
"""

import unittest
import numpy as np
from health_pipeline.preprocessing.cleaner import SignalCleaner
from health_pipeline.preprocessing.sqi import SignalQualityAssessor
from health_pipeline.preprocessing.features import FeatureExtractor


class TestPreprocessingAndFeatures(unittest.TestCase):

    def setUp(self):
        self.fs = 360.0
        self.cleaner = SignalCleaner(fs=self.fs)
        self.assessor = SignalQualityAssessor(fs=self.fs)
        self.extractor = FeatureExtractor(fs=self.fs)

        # Generate a clean 10-second test ECG signal at 360 Hz (72 bpm -> ~12 beats)
        t = np.linspace(0, 10.0, int(10.0 * self.fs), endpoint=False)
        self.clean_signal = np.zeros_like(t)
        for beat_time in np.arange(0.5, 9.5, 0.833):
            # Add Gaussian QRS spike
            self.clean_signal += 1.2 * np.exp(-((t - beat_time) ** 2) / (2 * (0.03 ** 2)))
        
        # Add slight 50Hz hum and baseline drift
        self.noisy_signal = self.clean_signal + 0.05 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 0.2 * t)

    def test_cleaner_filtering(self):
        cleaned = self.cleaner.clean_ecg_signal(self.noisy_signal, notch_freq=50.0)
        self.assertEqual(len(cleaned), len(self.noisy_signal))
        # Ensure finite values and zero NaNs
        self.assertTrue(np.all(np.isfinite(cleaned)))
        # Check standard deviation is reasonable
        self.assertTrue(0.1 < np.std(cleaned) < 5.0)

    def test_sqi_assessment(self):
        # 1. Clean signal should pass SQI
        sqi_clean = self.assessor.evaluate_ecg_window(self.clean_signal)
        self.assertTrue(sqi_clean.is_acceptable)
        self.assertTrue(sqi_clean.quality_score > 0.4)

        # 2. Flatline signal should fail SQI
        flatline = np.zeros(int(10.0 * self.fs))
        sqi_flat = self.assessor.evaluate_ecg_window(flatline)
        self.assertFalse(sqi_flat.is_acceptable)
        self.assertIn("Flatline", sqi_flat.rejection_reason)

    def test_feature_extraction(self):
        cleaned = self.cleaner.clean_ecg_signal(self.clean_signal)
        features = self.extractor.extract_full_window_features(cleaned)

        # Check required HRV keys
        self.assertIn("mean_hr_bpm", features)
        self.assertIn("sdnn_ms", features)
        self.assertIn("rmssd_ms", features)
        self.assertIn("pnn50", features)
        self.assertIn("lf_hf_ratio", features)
        self.assertIn("stat_sample_entropy", features)
        self.assertIn("stat_kurtosis", features)

        # Heart rate should be close to 72 bpm
        self.assertTrue(60.0 <= features["mean_hr_bpm"] <= 85.0)
        # All feature values must be finite floats
        for k, v in features.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} is non-finite: {v}")


if __name__ == "__main__":
    unittest.main()
