"""
health_pipeline.data.validator
Validation of file integrity, schema definitions, sampling rates, and physiological ranges.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from health_pipeline.config import SIGNAL_CONFIG


class ValidationError(Exception):
    """Raised when data fails integrity, schema, or physiological validation."""
    pass


class DataValidator:
    """
    Validates file integrity, schemas, and signal properties before processing.
    """

    @staticmethod
    def verify_file_checksum(file_path: Union[str, Path], expected_sha256: Optional[str] = None) -> str:
        """
        Validates file existence and computes its SHA-256 checksum.
        If expected_sha256 is provided, verifies that hashes match.
        """
        p = Path(file_path)
        if not p.is_file():
            raise ValidationError(f"File not found: {file_path}")
        
        if p.stat().st_size == 0:
            raise ValidationError(f"File is empty (0 bytes): {file_path}")

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValidationError(
                f"Integrity check failed for {p.name}: expected {expected_sha256}, got {digest}"
            )

        return digest

    @staticmethod
    def validate_schema(columns: List[str], required_columns: List[str]) -> None:
        """
        Validates that required schema columns are present in the dataset.
        """
        available_set = set(col.strip().lower() for col in columns)
        missing = [req for req in required_columns if req.strip().lower() not in available_set]
        if missing:
            raise ValidationError(f"Schema violation: missing required columns: {missing}")

    @staticmethod
    def validate_sampling_rate(actual_fs: float, expected_fs: float, tolerance: float = 0.5) -> None:
        """
        Validates that the signal's sampling rate matches the required pipeline frequency.
        """
        if abs(actual_fs - expected_fs) > tolerance:
            raise ValidationError(
                f"Sampling rate mismatch: received {actual_fs} Hz, but pipeline expects {expected_fs} ± {tolerance} Hz."
            )

    @staticmethod
    def validate_signal_array(
        signal: np.ndarray, 
        signal_type: str = "ECG",
        allow_nan: bool = False
    ) -> Tuple[bool, str]:
        """
        Validates signal length, checks for NaN/Inf values, and confirms that
        values lie within clinically plausible physical ranges.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float64)

        if signal.size == 0:
            return False, "Signal array is empty."

        nan_count = np.isnan(signal).sum()
        inf_count = np.isinf(signal).sum()

        if inf_count > 0:
            return False, f"Signal contains {inf_count} infinite values."

        if not allow_nan and nan_count > 0:
            return False, f"Signal contains {nan_count} NaN values (NaN not permitted)."

        valid_vals = signal[~np.isnan(signal)]
        if len(valid_vals) == 0:
            return False, "Signal consists entirely of NaNs."

        # Physiological bounds check
        if signal_type.upper() == "ECG":
            min_val, max_val = float(np.min(valid_vals)), float(np.max(valid_vals))
            # Beyond -10mV or +10mV indicates extreme lead disconnection or electrical fault
            if min_val < SIGNAL_CONFIG.ECG_VOLTAGE_MIN_MV * 2 or max_val > SIGNAL_CONFIG.ECG_VOLTAGE_MAX_MV * 2:
                return False, f"ECG voltage [{min_val:.2f}, {max_val:.2f}] mV exceeds plausible limits."

        elif signal_type.upper() == "SPO2":
            min_val, max_val = float(np.min(valid_vals)), float(np.max(valid_vals))
            if min_val < SIGNAL_CONFIG.SPO2_MIN_PERCENT or max_val > SIGNAL_CONFIG.SPO2_MAX_PERCENT:
                return False, f"SpO2 value [{min_val:.1f}%, {max_val:.1f}%] outside physiological range [50%, 100%]."

        elif signal_type.upper() in ("HR", "HEART_RATE"):
            min_val, max_val = float(np.min(valid_vals)), float(np.max(valid_vals))
            if min_val < SIGNAL_CONFIG.HEART_RATE_MIN_BPM or max_val > SIGNAL_CONFIG.HEART_RATE_MAX_BPM:
                return False, f"Heart rate [{min_val:.1f}, {max_val:.1f}] bpm outside physiological range [30, 240] bpm."

        return True, "Valid signal"
