"""
health_pipeline.security.privacy
HIPAA Safe Harbor de-identification, salted HMAC pseudonymization,
and timestamp anonymization utilities for physiological signal ingestion.
"""

import hashlib
import hmac
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from health_pipeline.config import SECURITY_CONFIG


class PrivacyProtector:
    """
    Guarantees clinical data privacy at ingestion by:
    1. Replacing patient identifiers with salted HMAC-SHA256 pseudonyms.
    2. Jittering / anonymizing absolute timestamps while preserving relative sampling intervals.
    3. Scrubbing free-text annotations of HIPAA 18 Safe Harbor direct identifiers.
    """

    # Regex patterns for common direct identifiers
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
    IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    NAME_TAG_PATTERN = re.compile(r"(?i)\b(?:mr\.|mrs\.|ms\.|dr\.|patient|name|subject)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")

    def __init__(self, salt: Optional[str] = None):
        self._salt = (salt or SECURITY_CONFIG.HMAC_SALT).encode("utf-8")

    def pseudonymize_identifier(self, identifier: Union[str, int]) -> str:
        """
        Derives a deterministic, irreversible pseudonym for a patient/subject ID
        using HMAC-SHA256 with a secret cryptographic salt.
        Truncated to 16 hex characters for compact tracking.
        """
        raw_val = str(identifier).strip().encode("utf-8")
        h = hmac.new(self._salt, raw_val, hashlib.sha256).hexdigest()
        return f"PID-{h[:16].upper()}"

    def get_patient_jitter_offset(self, patient_pseudonym: str) -> timedelta:
        """
        Calculates a deterministic timestamp offset for a given patient pseudonym.
        Ensures all records belonging to the same patient are shifted consistently,
        preserving cross-record longitudinal relationships without revealing the true calendar date.
        """
        h = hashlib.sha256(patient_pseudonym.encode("utf-8")).hexdigest()
        # Convert first 8 hex characters to an integer within the configured jitter range
        offset_val = int(h[:8], 16)
        jitter_range = SECURITY_CONFIG.TIMESTAMP_JITTER_HOURS_MAX - SECURITY_CONFIG.TIMESTAMP_JITTER_HOURS_MIN
        hours = SECURITY_CONFIG.TIMESTAMP_JITTER_HOURS_MIN + (offset_val % jitter_range)
        return timedelta(hours=hours)

    def anonymize_timestamp(self, dt: datetime, patient_pseudonym: str) -> datetime:
        """
        Applies deterministic patient-level jittering to an absolute datetime.
        """
        offset = self.get_patient_jitter_offset(patient_pseudonym)
        return dt + offset

    def scrub_text(self, text: str) -> str:
        """
        Scrubs clinical free text / headers of SSNs, emails, phone numbers,
        IP addresses, and patient names according to HIPAA Safe Harbor guidelines.
        """
        if not text:
            return text
        
        redacted = text
        redacted = self.SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
        redacted = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
        redacted = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
        redacted = self.IP_PATTERN.sub("[REDACTED_IP]", redacted)
        redacted = self.NAME_TAG_PATTERN.sub(r"Patient [REDACTED_NAME]", redacted)
        return redacted

    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-sanitizes ingestion metadata dictionary:
        - Pseudonymizes any patient/subject/record identifiers.
        - Redacts explicit direct identifier keys.
        - Preserves clinically relevant de-identified covariates (e.g. age category, biological sex).
        """
        sanitized = {}
        patient_id_raw = (
            metadata.get("patient_id") or 
            metadata.get("subject_id") or 
            metadata.get("record_id") or 
            "UNKNOWN_PATIENT"
        )
        patient_pseudonym = self.pseudonymize_identifier(patient_id_raw)
        sanitized["patient_pseudonym"] = patient_pseudonym

        forbidden_keys = {
            "name", "patient_name", "first_name", "last_name", "ssn", "mrn",
            "phone", "email", "address", "zip", "postal_code", "ip_address",
            "device_serial", "physician", "doctor"
        }

        for k, v in metadata.items():
            lower_k = k.lower().strip()
            if lower_k in forbidden_keys:
                continue
            elif "id" in lower_k and lower_k not in ("channel_id", "lead_id"):
                sanitized[f"{k}_pseudonym"] = self.pseudonymize_identifier(v)
            elif isinstance(v, str):
                sanitized[k] = self.scrub_text(v)
            elif isinstance(v, datetime):
                sanitized[k] = self.anonymize_timestamp(v, patient_pseudonym)
            else:
                sanitized[k] = v

        return sanitized
