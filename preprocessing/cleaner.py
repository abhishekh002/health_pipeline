"""
health_pipeline.preprocessing.cleaner
Signal conditioning, Butterworth bandpass/notch filtering, artifact removal, and robust scaling.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.signal as signal
from health_pipeline.config import SIGNAL_CONFIG


class SignalCleaner:
    """
    Cleans physiological signals by filtering noise, suppressing powerline interference,
    removing baseline wander, and standardizing amplitudes.
    """

    def __init__(self, fs: float = SIGNAL_CONFIG.DEFAULT_ECG_FS):
        self.fs = fs

    def butter_bandpass_filter(
        self, 
        data: np.ndarray, 
        lowcut: float = SIGNAL_CONFIG.ECG_BANDPASS_LOW, 
        highcut: float = SIGNAL_CONFIG.ECG_BANDPASS_HIGH, 
        order: int = 4
    ) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter.
        Removes baseline wander (< lowcut) and high-frequency muscle noise (> highcut).
        """
        nyq = 0.5 * self.fs
        low = max(lowcut / nyq, 0.0001)
        high = min(highcut / nyq, 0.9999)
        b, a = signal.butter(order, [low, high], btype="bandpass")
        # Use filtfilt for forward-backward zero-phase distortion
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def notch_filter(
        self, 
        data: np.ndarray, 
        freq: float = 50.0, 
        q: float = SIGNAL_CONFIG.NOTCH_Q
    ) -> np.ndarray:
        """
        Applies an IIR notch filter to suppress powerline hum (50 Hz or 60 Hz).
        """
        nyq = 0.5 * self.fs
        if freq >= nyq:
            return data
        w0 = freq / nyq
        b, a = signal.iirnotch(w0, q)
        return signal.filtfilt(b, a, data)

    def remove_baseline_wander(self, data: np.ndarray, window_sec: float = 0.6) -> np.ndarray:
        """
        Removes respiratory baseline wander using median filtering (or high-pass subtraction).
        """
        kernel_size = int(window_sec * self.fs)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Scipy medfilt or high-pass subtraction
        baseline = signal.medfilt(data, kernel_size=min(kernel_size, 101))
        return data - baseline

    def handle_missing_values(
        self, 
        data: np.ndarray, 
        method: str = "interpolate"
    ) -> np.ndarray:
        """
        Handles NaN or missing data points via linear interpolation or median imputation.
        """
        cleaned = data.copy()
        nans = np.isnan(cleaned)
        if not np.any(nans):
            return cleaned

        if np.all(nans):
            return np.zeros_like(cleaned)

        x = np.arange(len(cleaned))
        valid_idx = x[~nans]
        valid_vals = cleaned[~nans]

        if method == "interpolate":
            cleaned[nans] = np.interp(x[nans], valid_idx, valid_vals)
        elif method == "median":
            cleaned[nans] = np.median(valid_vals)
        else:
            cleaned[nans] = 0.0

        return cleaned

    def robust_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Standardizes signal amplitudes using Median and Interquartile Range (IQR).
        Resistant to extreme peak outliers and cardiac ectopic spikes.
        """
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6:
            std = np.std(data)
            return (data - median) / (std + 1e-6)
        return (data - median) / iqr

    def clean_ecg_signal(
        self, 
        raw_ecg: np.ndarray,
        notch_freq: float = 50.0
    ) -> np.ndarray:
        """
        Executes the full clinical ECG conditioning pipeline:
        1. Impute missing/corrupted samples
        2. Zero-phase bandpass filter (0.5 - 45 Hz)
        3. Notch filter for mains hum
        4. Baseline wander removal
        5. Robust normalization
        """
        imputed = self.handle_missing_values(raw_ecg)
        bandpassed = self.butter_bandpass_filter(
            imputed, 
            lowcut=SIGNAL_CONFIG.ECG_BANDPASS_LOW, 
            highcut=SIGNAL_CONFIG.ECG_BANDPASS_HIGH
        )
        notched = self.notch_filter(bandpassed, freq=notch_freq)
        detrended = self.remove_baseline_wander(notched)
        standardized = self.robust_scale(detrended)
        return standardized

    def clean_ppg_signal(self, raw_ppg: np.ndarray) -> np.ndarray:
        """
        Executes PPG conditioning pipeline:
        1. Impute missing points
        2. Bandpass filter (0.5 - 5.0 Hz, covering 30 - 300 bpm pulse wave)
        3. Robust scaling
        """
        imputed = self.handle_missing_values(raw_ppg)
        bandpassed = self.butter_bandpass_filter(
            imputed,
            lowcut=SIGNAL_CONFIG.PPG_BANDPASS_LOW,
            highcut=SIGNAL_CONFIG.PPG_BANDPASS_HIGH
        )
        return self.robust_scale(bandpassed)
