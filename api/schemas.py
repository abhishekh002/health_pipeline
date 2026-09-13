"""
health_pipeline.api.schemas
Input sanitization, physiological bounds validation, and output formatting for inference APIs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from health_pipeline.config import SIGNAL_CONFIG


class APIValidationError(Exception):
    """Raised when an API payload fails schema, format, or physiological bounds checks."""
    pass


class PayloadValidator:
    """
    Sanitizes raw JSON requests, enforcing array sizes, sampling rates, and physical bounds.
    Protects against denial-of-service, memory exhaustion, and invalid inputs.
    """

    MAX_SAMPLES = 50000     # Maximum ~2.3 minutes at 360 Hz to prevent server memory saturation
    MIN_SAMPLES = 100       # Minimum samples required for digital filtering

    @classmethod
    def sanitize_prediction_request(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and sanitizes incoming prediction payload.
        """
        if not isinstance(payload, dict):
            raise APIValidationError("Request body must be a valid JSON object.")

        # Check for direct features vs raw signal
        has_features = "features" in payload and isinstance(payload["features"], (dict, list))
        has_signal = "signal" in payload and isinstance(payload["signal"], list)

        if not has_features and not has_signal:
            raise APIValidationError("Payload must contain either 'signal' (array of floats) or 'features' (dict/array).")

        sanitized: Dict[str, Any] = {}

        # 1. Validate raw signal array if provided
        if has_signal:
            raw_sig = payload["signal"]
            if len(raw_sig) < cls.MIN_SAMPLES:
                raise APIValidationError(f"Signal is too short ({len(raw_sig)} samples). Minimum is {cls.MIN_SAMPLES}.")
            if len(raw_sig) > cls.MAX_SAMPLES:
                raise APIValidationError(f"Signal exceeds maximum allowed length ({cls.MAX_SAMPLES} samples).")

            try:
                sig_arr = np.array(raw_sig, dtype=np.float64)
            except (ValueError, TypeError):
                raise APIValidationError("Signal elements must be numeric floating-point values.")

            if np.isinf(sig_arr).any():
                raise APIValidationError("Signal contains invalid infinite (Inf) values.")
            if np.isnan(sig_arr).all():
                raise APIValidationError("Signal consists entirely of NaN values.")

            # Physiological voltage check (-10 mV to +10 mV)
            finite_vals = sig_arr[~np.isnan(sig_arr)]
            min_v, max_v = float(np.min(finite_vals)), float(np.max(finite_vals))
            if min_v < -10.0 or max_v > 10.0:
                raise APIValidationError(f"Signal amplitude [{min_v:.2f}, {max_v:.2f}] mV exceeds physical ECG range.")

            sanitized["signal"] = sig_arr

        # 2. Validate sampling frequency
        fs = payload.get("fs", SIGNAL_CONFIG.DEFAULT_ECG_FS)
        try:
            fs = float(fs)
        except (ValueError, TypeError):
            raise APIValidationError("Sampling frequency 'fs' must be a numeric float.")
        if fs < 20.0 or fs > 2000.0:
            raise APIValidationError(f"Sampling frequency {fs} Hz is outside allowable medical device range [20, 2000] Hz.")
        sanitized["fs"] = fs

        # 3. Validate features dict/list if provided directly
        if has_features:
            features = payload["features"]
            if isinstance(features, dict):
                cleaned_feats = {}
                for k, v in features.items():
                    try:
                        f_val = float(v)
                        if not np.isfinite(f_val):
                            raise APIValidationError(f"Feature '{k}' contains non-finite value.")
                        cleaned_feats[str(k)] = f_val
                    except (ValueError, TypeError):
                        raise APIValidationError(f"Feature '{k}' must be numeric.")
                sanitized["features"] = cleaned_feats
            elif isinstance(features, list):
                try:
                    feat_arr = np.array(features, dtype=np.float64)
                    if not np.all(np.isfinite(feat_arr)):
                        raise APIValidationError("Features vector contains NaN or infinite values.")
                    sanitized["features"] = feat_arr
                except (ValueError, TypeError):
                    raise APIValidationError("Features list must contain only numeric floats.")

        # 4. Optional multimodal signals (PPG, SpO2)
        if "ppg_signal" in payload and payload["ppg_signal"] is not None:
            ppg_raw = payload["ppg_signal"]
            if not isinstance(ppg_raw, list) or len(ppg_raw) > cls.MAX_SAMPLES:
                raise APIValidationError("Invalid ppg_signal format or length.")
            sanitized["ppg_signal"] = np.array(ppg_raw, dtype=np.float64)
        else:
            sanitized["ppg_signal"] = None

        if "spo2_signal" in payload and payload["spo2_signal"] is not None:
            spo2_raw = payload["spo2_signal"]
            if not isinstance(spo2_raw, list) or len(spo2_raw) > cls.MAX_SAMPLES:
                raise APIValidationError("Invalid spo2_signal format or length.")
            spo2_arr = np.array(spo2_raw, dtype=np.float64)
            finite_spo2 = spo2_arr[~np.isnan(spo2_arr)]
            if len(finite_spo2) > 0 and (np.min(finite_spo2) < 40.0 or np.max(finite_spo2) > 100.0):
                raise APIValidationError("SpO2 values outside physiological range [40%, 100%].")
            sanitized["spo2_signal"] = spo2_arr
        else:
            sanitized["spo2_signal"] = None

        return sanitized
