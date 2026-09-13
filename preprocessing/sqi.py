"""
health_pipeline.preprocessing.sqi
Signal Quality Index (SQI) computation for physiological window validation.
Computes statistical, spectral, and artifact indices to filter out corrupted segments.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from health_pipeline.config import SIGNAL_CONFIG


@dataclass
class SQIResult:
    is_acceptable: bool
    quality_score: float  # 0.0 to 1.0
    metrics: Dict[str, float]
    rejection_reason: str = ""


class SignalQualityAssessor:
    """
    Assesses the clinical quality of ECG and PPG signal windows before feature extraction or inference.
    """

    def __init__(self, fs: float = SIGNAL_CONFIG.DEFAULT_ECG_FS):
        self.fs = fs

    def compute_kurtosis_sqi(self, window: np.ndarray) -> float:
        """
        kSQI: Kurtosis of physiological signal. Normal sinus rhythm has high peakedness (kurtosis ~ 4-15).
        Flatlines, heavy drift, or pure Gaussian noise have kurtosis ~ 0-3.
        """
        k = float(stats.kurtosis(window, fisher=True))
        return k

    def compute_skewness_sqi(self, window: np.ndarray) -> float:
        """
        sSQI: Skewness of physiological signal.
        """
        s = float(stats.skew(window))
        return s

    def compute_spectral_sqi(self, window: np.ndarray) -> Tuple[float, float]:
        """
        Computes baseline power ratio (< 1 Hz) and high-frequency EMG noise ratio (> 45 Hz).
        """
        freqs, psd = signal.welch(window, fs=self.fs, nperseg=min(len(window), int(self.fs * 2)))
        total_power = np.sum(psd) + 1e-9

        baseline_power = np.sum(psd[freqs < 1.0])
        emg_power = np.sum(psd[freqs > 45.0])

        bas_sqi = float(baseline_power / total_power)
        emg_sqi = float(emg_power / total_power)
        return bas_sqi, emg_sqi

    def evaluate_ecg_window(self, window: np.ndarray) -> SQIResult:
        """
        Evaluates an ECG segment and returns a quality score and pass/fail decision.
        """
        if len(window) < int(self.fs * 1.0):
            return SQIResult(
                is_acceptable=False, 
                quality_score=0.0, 
                metrics={}, 
                rejection_reason="Window duration is too short (< 1 sec)."
            )

        # Check flatline / zero variance
        var = float(np.var(window))
        if var < 1e-7:
            return SQIResult(
                is_acceptable=False,
                quality_score=0.0,
                metrics={"variance": var},
                rejection_reason="Flatline or sensor disconnection detected."
            )

        k_sqi = self.compute_kurtosis_sqi(window)
        s_sqi = self.compute_skewness_sqi(window)
        bas_sqi, emg_sqi = self.compute_spectral_sqi(window)

        rejection_reasons = []
        # Normal ECG exhibits distinct QRS spikes (kurtosis > 2)
        if k_sqi < SIGNAL_CONFIG.MIN_KURTOSIS_SQI:
            rejection_reasons.append("Low kurtosis: lacking distinct QRS complexes or excessive white noise.")
        elif k_sqi > SIGNAL_CONFIG.MAX_KURTOSIS_SQI * 2:
            rejection_reasons.append("Extreme kurtosis: high-amplitude impulse artifact / motion spike.")

        # Baseline wander check (> 60% power in < 1 Hz band)
        if bas_sqi > 0.60:
            rejection_reasons.append(f"Severe baseline wander ({bas_sqi*100:.1f}% low-frequency power).")

        # High frequency EMG noise (> 30% power in > 45 Hz band)
        if emg_sqi > 0.35:
            rejection_reasons.append(f"Severe muscle/EMG noise ({emg_sqi*100:.1f}% high-frequency power).")

        # Aggregate quality score (0.0 to 1.0)
        # Higher score for moderate positive kurtosis and low baseline/EMG contamination
        kurt_score = np.clip((k_sqi - 2.0) / 8.0, 0.0, 1.0) if k_sqi >= 2.0 else 0.0
        clean_score = max(0.0, 1.0 - (bas_sqi * 1.2 + emg_sqi * 1.5))
        overall_score = float(0.5 * kurt_score + 0.5 * clean_score)

        metrics = {
            "kurtosis_sqi": k_sqi,
            "skewness_sqi": s_sqi,
            "baseline_power_ratio": bas_sqi,
            "emg_power_ratio": emg_sqi,
            "variance": var
        }

        is_acceptable = len(rejection_reasons) == 0
        reason = "; ".join(rejection_reasons) if rejection_reasons else "Acceptable signal quality"

        return SQIResult(
            is_acceptable=is_acceptable,
            quality_score=overall_score,
            metrics=metrics,
            rejection_reason=reason
        )
