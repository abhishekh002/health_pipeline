"""
Unit tests for data privacy, HIPAA Safe Harbor de-identification, and HMAC pseudonymization.
"""

import unittest
from datetime import datetime
from health_pipeline.security.privacy import PrivacyProtector


class TestPrivacyProtector(unittest.TestCase):

    def setUp(self):
        self.privacy = PrivacyProtector(salt="ClinicalTestSalt_2026")

    def test_deterministic_pseudonymization(self):
        id1 = "PATIENT_98765"
        id2 = "PATIENT_98765"
        id3 = "PATIENT_12345"

        pseudo1 = self.privacy.pseudonymize_identifier(id1)
        pseudo2 = self.privacy.pseudonymize_identifier(id2)
        pseudo3 = self.privacy.pseudonymize_identifier(id3)

        self.assertTrue(pseudo1.startswith("PID-"))
        self.assertEqual(pseudo1, pseudo2)
        self.assertNotEqual(pseudo1, pseudo3)

    def test_hipaa_text_scrubbing(self):
        clinical_note = (
            "Patient John Doe (SSN: 123-45-6789) visited Dr. Smith. "
            "Contact: patient.test@hospital.org or phone 555-123-4567. IP: 192.168.1.50."
        )
        scrubbed = self.privacy.scrub_text(clinical_note)

        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("patient.test@hospital.org", scrubbed)
        self.assertNotIn("555-123-4567", scrubbed)
        self.assertNotIn("192.168.1.50", scrubbed)
        self.assertIn("[REDACTED_SSN]", scrubbed)
        self.assertIn("[REDACTED_EMAIL]", scrubbed)
        self.assertIn("[REDACTED_PHONE]", scrubbed)
        self.assertIn("[REDACTED_IP]", scrubbed)

    def test_timestamp_jittering(self):
        dt = datetime(2026, 5, 15, 10, 30, 0)
        pseudo_a = "PID-AAAA1111"
        pseudo_b = "PID-BBBB2222"

        jittered_a1 = self.privacy.anonymize_timestamp(dt, pseudo_a)
        jittered_a2 = self.privacy.anonymize_timestamp(dt, pseudo_a)
        jittered_b = self.privacy.anonymize_timestamp(dt, pseudo_b)

        # Consistency for the same patient
        self.assertEqual(jittered_a1, jittered_a2)
        # Obfuscated from original
        self.assertNotEqual(dt, jittered_a1)
        # Different patient has different jitter
        self.assertNotEqual(jittered_a1, jittered_b)

    def test_metadata_sanitization(self):
        raw_meta = {
            "patient_id": "SUBJ-999",
            "patient_name": "Jane Roe",
            "ssn": "000-11-2222",
            "age": 62,
            "sex": "F",
            "diagnosis": "Sinus rhythm evaluated by Dr. Adams"
        }
        sanitized = self.privacy.sanitize_metadata(raw_meta)

        self.assertIn("patient_pseudonym", sanitized)
        self.assertNotIn("patient_name", sanitized)
        self.assertNotIn("ssn", sanitized)
        self.assertEqual(sanitized["age"], 62)
        self.assertEqual(sanitized["sex"], "F")
        self.assertNotIn("Jane Roe", str(sanitized))


if __name__ == "__main__":
    unittest.main()
