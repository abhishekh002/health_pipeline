"""
health_pipeline.security.secure_logger
Custom privacy-preserving logger that redacts PHI and prevents raw physiological signal dumps in logs.
"""

import logging
import re
from pathlib import Path
from typing import Optional
from health_pipeline.config import PATHS, SECURITY_CONFIG
from health_pipeline.security.privacy import PrivacyProtector


class PHIFilter(logging.Filter):
    """
    Log filter that intercepts records, checks for patient identifiers,
    and masks any potentially identifying data or massive raw numeric dumps.
    """

    # Matches raw array prints like [0.123, -0.456, 1.234, ...]
    RAW_ARRAY_PATTERN = re.compile(r"\[\s*-?\d+\.\d+(?:,\s*-?\d+\.\d+){4,}\s*\]")
    
    def __init__(self, privacy_protector: Optional[PrivacyProtector] = None):
        super().__init__()
        self.protector = privacy_protector or PrivacyProtector()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        # 1. Scrub clinical text / PII
        msg = self.protector.scrub_text(msg)
        # 2. Prevent dumping raw physiological arrays into logs
        msg = self.RAW_ARRAY_PATTERN.sub("[RAW_SIGNAL_DATA_REDACTED]", msg)
        record.msg = msg
        return True


def get_secure_logger(name: str = "health_pipeline") -> logging.Logger:
    """
    Creates and configures a secure logger with file and console handlers,
    enforcing PHI redaction at all output sinks.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    phi_filter = PHIFilter()

    # Formatter with anonymized format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(phi_filter)
    logger.addHandler(console_handler)

    # Secure Log File Handler
    log_file = PATHS.LOGS_DIR / "secure_pipeline.log"
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(phi_filter)
    logger.addHandler(file_handler)

    return logger
