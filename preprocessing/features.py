"""
health_pipeline.preprocessing.features
Comprehensive clinical feature extraction for physiological signals:
- R-peak detection & RR-interval extraction
- Time-domain Heart Rate Variability (HRV): SDNN, RMSSD, pNN50, Mean HR
- Frequency-domain HRV: VLF, LF, HF, LF/HF ratio via Welch PSD
- Morphological & Statistical Descriptors: Skewness, Kurtosis, Zero-Crossing Rate, Sample Entropy
- Multi-modal PPG & SpO2 features
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from health_pipeline.config import SIGNAL_CONFIG


class FeatureExtractor:
    """
    Extracts clinical and statistical features from physiological signals.
    """

    def __init__(self, fs: float = SIGNAL_CONFIG.DEFAULT_ECG_FS):
        self.fs = fs

    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detects R-peaks using an energy-envelope and derivative peak-finding algorithm
        (Pan-Tompkins inspired).
        """
        # Differentiate signal to highlight QRS slope
        diff = np.diff(ecg_signal)
        # Square signal to amplify high slopes and make positive
        squared = diff ** 2
        # Moving window integration (150ms window)
        window_len = max(int(0.15 * self.fs), 3)
        kernel = np.ones(window_len) / window_len
        integrated = np.convolve(squared, kernel, mode="same")

        # Minimum distance between beats (corresponding to max 220 bpm)
        min_dist = int(self.fs * 60.0 / 220.0)
        # Threshold: 30% of 90th percentile to avoid outlier suppression
        threshold = float(np.percentile(integrated, 90) * 0.30)
        threshold = max(threshold, 1e-4)

        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_dist)

        # Refine peaks to the actual local maximum in the raw/cleaned ECG
        refined_peaks = []
        search_radius = int(0.08 * self.fs)
        for p in peaks:
            start = max(0, p - search_radius)
            end = min(len(ecg_signal), p + search_radius + 1)
            local_max = start + np.argmax(ecg_signal[start:end])
            refined_peaks.append(local_max)

        # Remove potential duplicates
        refined_peaks = np.unique(refined_peaks)
        return refined_peaks

    def extract_hrv_time_domain(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Computes standard time-domain HRV metrics from R-peak indices:
        - Mean RR (ms)
        - SDNN (ms): Standard deviation of NN intervals
        - RMSSD (ms): Root mean square of successive differences (parasympathetic marker)
        - pNN50 (%): Percentage of successive intervals differing by > 50 ms
        - Mean HR (bpm), Min HR, Max HR, HR range
        """
        if len(r_peaks) < 3:
            return {
                "mean_rr_ms": 800.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "pnn50": 0.0,
                "mean_hr_bpm": 75.0,
                "min_hr_bpm": 75.0,
                "max_hr_bpm": 75.0,
                "hr_range_bpm": 0.0,
            }

        # RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks) / self.fs * 1000.0
        # Exclude physiologically implausible intervals (< 250ms or > 2000ms)
        valid_rr = rr_intervals[(rr_intervals >= 250.0) & (rr_intervals <= 2000.0)]
        if len(valid_rr) < 2:
            valid_rr = rr_intervals

        mean_rr = float(np.mean(valid_rr))
        sdnn = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else 0.0

        # Successive differences
        rr_diff = np.diff(valid_rr)
        rmssd = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else 0.0
        nn50 = np.sum(np.abs(rr_diff) > 50.0)
        pnn50 = float((nn50 / len(rr_diff)) * 100.0) if len(rr_diff) > 0 else 0.0

        # Instantaneous Heart Rate in bpm
        hr_series = 60000.0 / valid_rr
        mean_hr = float(np.mean(hr_series))
        min_hr = float(np.min(hr_series))
        max_hr = float(np.max(hr_series))
        hr_range = max_hr - min_hr

        return {
            "mean_rr_ms": mean_rr,
            "sdnn_ms": sdnn,
            "rmssd_ms": rmssd,
            "pnn50": pnn50,
            "mean_hr_bpm": mean_hr,
            "min_hr_bpm": min_hr,
            "max_hr_bpm": max_hr,
            "hr_range_bpm": hr_range,
        }

    def extract_hrv_frequency_domain(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Computes frequency-domain HRV metrics via Welch Power Spectral Density:
        - VLF power (0.0033 to 0.04 Hz)
        - LF power (0.04 to 0.15 Hz): Sympathetic & parasympathetic activity
        - HF power (0.15 to 0.40 Hz): Parasympathetic / respiratory sinus arrhythmia
        - LF/HF power ratio: Sympathovagal balance
        """
        default_res = {
            "vlf_power": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 1.0,
            "total_power": 0.0
        }

        if len(r_peaks) < 6:
            return default_res

        rr_intervals = np.diff(r_peaks) / self.fs  # In seconds
        rr_times = np.cumsum(rr_intervals)

        # Interpolate irregularly sampled RR intervals to uniform 4 Hz grid
        t_resampled = np.arange(rr_times[0], rr_times[-1], 0.25)
        if len(t_resampled) < 16:
            return default_res

        rr_resampled = np.interp(t_resampled, rr_times, rr_intervals)
        rr_resampled = rr_resampled - np.mean(rr_resampled)

        # Welch PSD at 4 Hz
        nperseg = min(len(rr_resampled), 256)
        freqs, psd = signal.welch(rr_resampled, fs=4.0, nperseg=nperseg)

        # Trapezoidal integration compatible across NumPy 1.x and 2.x
        import scipy.integrate as integrate
        trapz_func = getattr(integrate, "trapezoid", getattr(np, "trapezoid", getattr(np, "trapz", None)))

        vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)

        vlf_power = float(trapz_func(psd[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
        lf_power = float(trapz_func(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(trapz_func(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
        total_power = vlf_power + lf_power + hf_power

        lf_hf_ratio = float(lf_power / hf_power) if hf_power > 1e-7 else 1.0

        return {
            "vlf_power": max(0.0, vlf_power),
            "lf_power": max(0.0, lf_power),
            "hf_power": max(0.0, hf_power),
            "lf_hf_ratio": max(0.0, min(lf_hf_ratio, 25.0)),
            "total_power": max(0.0, total_power)
        }

    def compute_sample_entropy(self, data: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
        """
        Fast approximation of Sample Entropy (SampEn) to assess complexity/irregularity.
        Downsamples long arrays for efficiency.
        """
        if len(data) > 500:
            step = len(data) // 300
            data = data[::step]

        n = len(data)
        if n < 20:
            return 0.0

        r = r_scale * (np.std(data) + 1e-8)

        def _phi(dim):
            x = np.array([data[i:i + dim] for i in range(n - dim + 1)])
            # Chebyshev distance
            diff = np.abs(x[:, None, :] - x[None, :, :]).max(axis=2)
            matches = (diff <= r).sum() - len(x)  # Exclude self-matches
            return matches / (len(x) * (len(x) - 1) + 1e-8)

        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            if phi_m > 0 and phi_m1 > 0:
                return float(-np.log(phi_m1 / phi_m))
        except Exception:
            pass

        return 0.5

    def extract_statistical_morphology(self, signal_window: np.ndarray) -> Dict[str, float]:
        """
        Statistical and morphological descriptors for the physiological window:
        - Mean, Standard Deviation, Variance
        - Skewness & Kurtosis
        - Peak-to-Peak Amplitude
        - Zero-Crossing Rate (ZCR)
        - Sample Entropy
        """
        mean_val = float(np.mean(signal_window))
        std_val = float(np.std(signal_window))
        var_val = float(np.var(signal_window))
        skew_val = float(stats.skew(signal_window))
        kurt_val = float(stats.kurtosis(signal_window))
        p2p = float(np.ptp(signal_window))

        # Zero-Crossing Rate
        zero_crossings = np.nonzero(np.diff(signal_window > 0))[0]
        zcr = float(len(zero_crossings) / len(signal_window))

        # Sample Entropy
        samp_en = self.compute_sample_entropy(signal_window)

        return {
            "stat_mean": mean_val,
            "stat_std": std_val,
            "stat_var": var_val,
            "stat_skewness": skew_val,
            "stat_kurtosis": kurt_val,
            "stat_p2p_amp": p2p,
            "stat_zcr": zcr,
            "stat_sample_entropy": samp_en
        }

    def extract_spo2_ppg_features(
        self, 
        ppg_signal: Optional[np.ndarray] = None, 
        spo2_signal: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extracts SpO2 and PPG features:
        - Mean SpO2, Min SpO2, SpO2 variability
        - Hypoxic desaturation event count (< 90%)
        - PPG Perfusion Index (AC / DC ratio)
        """
        features = {
            "spo2_mean": 98.0,
            "spo2_min": 98.0,
            "spo2_std": 0.0,
            "spo2_desat_ratio": 0.0,
            "ppg_perfusion_index": 1.5
        }

        if spo2_signal is not None and len(spo2_signal) > 0:
            valid_spo2 = spo2_signal[~np.isnan(spo2_signal)]
            if len(valid_spo2) > 0:
                features["spo2_mean"] = float(np.mean(valid_spo2))
                features["spo2_min"] = float(np.min(valid_spo2))
                features["spo2_std"] = float(np.std(valid_spo2))
                features["spo2_desat_ratio"] = float(np.mean(valid_spo2 < 90.0))

        if ppg_signal is not None and len(ppg_signal) > 0:
            dc = float(np.mean(ppg_signal))
            ac = float(np.ptp(ppg_signal))
            if abs(dc) > 1e-4:
                features["ppg_perfusion_index"] = float((ac / abs(dc)) * 100.0)

        return features

    def extract_full_window_features(
        self,
        ecg_window: np.ndarray,
        ppg_window: Optional[np.ndarray] = None,
        spo2_window: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Combines all feature extraction steps into a single unified clinical descriptor vector.
        """
        # 1. Detect R peaks
        r_peaks = self.detect_r_peaks(ecg_window)
        # 2. Time-domain HRV
        time_hrv = self.extract_hrv_time_domain(r_peaks)
        # 3. Frequency-domain HRV
        freq_hrv = self.extract_hrv_frequency_domain(r_peaks)
        # 4. Statistical & Morphology
        stat_morph = self.extract_statistical_morphology(ecg_window)
        # 5. SpO2 / PPG
        multimodal = self.extract_spo2_ppg_features(ppg_window, spo2_window)

        combined = {}
        combined.update(time_hrv)
        combined.update(freq_hrv)
        combined.update(stat_morph)
        combined.update(multimodal)
        return combined
