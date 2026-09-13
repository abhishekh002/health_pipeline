"""
health_pipeline.config
Configuration settings for the secure health monitoring machine learning pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import os


@dataclass(frozen=True)
class SecurityConfig:
    # Secret salt for patient de-identification / HMAC hashing (can be overridden via ENV)
    HMAC_SALT: str = os.getenv("HEALTH_PIPELINE_SALT", "SecureHealthML_Salt_Key_2026_@clinical#")
    # API authentication key for inference endpoint
    API_AUTH_TOKEN: str = os.getenv("HEALTH_API_KEY", "health_sec_token_9f83a2c0918bd47e")
    # HIPAA jitter bounds (in hours) to preserve relative intervals while masking exact date
    TIMESTAMP_JITTER_HOURS_MIN: int = -720  # -30 days
    TIMESTAMP_JITTER_HOURS_MAX: int = 720   # +30 days
    # Masking token for removed identifiers
    REDACTED_TEXT: str = "[DE-IDENTIFIED_PHI]"


@dataclass(frozen=True)
class SignalConfig:
    # Supported sampling rates (Hz)
    DEFAULT_ECG_FS: float = 360.0       # MIT-BIH Arrhythmia standard sampling frequency
    DEFAULT_PPG_FS: float = 125.0       # Standard PPG pulse oximetry frequency
    
    # ECG Filter cutoffs (Hz)
    ECG_BANDPASS_LOW: float = 0.5       # Baseline wander removal
    ECG_BANDPASS_HIGH: float = 45.0     # High-frequency EMG/muscle artifact removal
    NOTCH_FREQ_50HZ: float = 50.0       # Mains frequency (EU/Asia)
    NOTCH_FREQ_60HZ: float = 60.0       # Mains frequency (US)
    NOTCH_Q: float = 30.0               # Quality factor for notch filter
    
    # PPG Filter cutoffs (Hz)
    PPG_BANDPASS_LOW: float = 0.5
    PPG_BANDPASS_HIGH: float = 5.0
    
    # Signal Quality Index (SQI) Thresholds
    MIN_KURTOSIS_SQI: float = 2.0
    MAX_KURTOSIS_SQI: float = 30.0
    MIN_SKEWNESS_SQI: float = -3.0
    MAX_SKEWNESS_SQI: float = 3.0
    MIN_VALID_RATIO: float = 0.85       # Minimum fraction of non-artifact samples required in a window

    # Physiological Bounds for Validation & Input Sanitization
    ECG_VOLTAGE_MIN_MV: float = -5.0    # Millivolts
    ECG_VOLTAGE_MAX_MV: float = 5.0
    HEART_RATE_MIN_BPM: float = 30.0    # Extreme bradycardia floor
    HEART_RATE_MAX_BPM: float = 240.0   # Extreme tachycardia ceiling
    SPO2_MIN_PERCENT: float = 50.0      # Below 50% is non-physiological or sensor disconnection
    SPO2_MAX_PERCENT: float = 100.0


@dataclass(frozen=True)
class ModelConfig:
    RANDOM_SEED: int = 42
    N_SPLITS: int = 5
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.15
    CLASS_WEIGHT: str = "balanced"
    
    # Triage risk probability cutoffs
    LOW_RISK_THRESHOLD: float = 0.35
    HIGH_RISK_THRESHOLD: float = 0.70


@dataclass
class PipelinePaths:
    PROJECT_ROOT: Path = Path("D:/project")
    DATASET_DIR: Path = Path("D:/project/mit-bih-arrhythmia-database-1.0.0")
    ARTIFACTS_DIR: Path = Path("D:/project/health_pipeline/artifacts")
    MODELS_DIR: Path = Path("D:/project/health_pipeline/artifacts/models")
    LOGS_DIR: Path = Path("D:/project/health_pipeline/artifacts/logs")
    REPORTS_DIR: Path = Path("D:/project/health_pipeline/artifacts/reports")

    def __post_init__(self):
        for directory in [self.ARTIFACTS_DIR, self.MODELS_DIR, self.LOGS_DIR, self.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Global instances
SECURITY_CONFIG = SecurityConfig()
SIGNAL_CONFIG = SignalConfig()
MODEL_CONFIG = ModelConfig()
PATHS = PipelinePaths()
