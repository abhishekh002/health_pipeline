"""
health_pipeline.data.loader
Multi-format physiological data loader with built-in schema validation and HIPAA de-identification at ingestion.
Supports:
- WFDB (MIT-BIH Arrhythmia Database format 212 binary + .hea headers + .atr annotations)
- Sensor CSV files (e.g. multi-channel wearable recordings)
- Synthetic / benchmark physiological generators
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from health_pipeline.config import PATHS, SIGNAL_CONFIG
from health_pipeline.data.validator import DataValidator, ValidationError
from health_pipeline.security.privacy import PrivacyProtector
from health_pipeline.security.secure_logger import get_secure_logger

logger = get_secure_logger("health_pipeline.data.loader")


@dataclass
class PhysiologicalRecord:
    """
    Standardized in-memory container for loaded and de-identified physiological records.
    """
    patient_pseudonym: str
    record_id: str
    sampling_frequency: float
    signals: Dict[str, np.ndarray]  # channel_name -> 1D numpy array of signal values (mV or units)
    metadata: Dict[str, Any]        # Sanitized metadata (demographics, device info, dates)
    annotations: Optional[Dict[str, Any]] = None  # Beat labels, timestamps, R-peaks


class PhysiologicalDataLoader:
    """
    Unified data loader enforcing validation and privacy safeguards at ingestion.
    """

    def __init__(self, privacy_protector: Optional[PrivacyProtector] = None):
        self.privacy = privacy_protector or PrivacyProtector()
        self.validator = DataValidator()

    def load_wfdb_record(
        self, 
        record_path_or_dir: Union[str, Path], 
        record_name: str,
        max_samples: Optional[int] = None
    ) -> PhysiologicalRecord:
        """
        Loads a WFDB record (e.g. MIT-BIH Arrhythmia database 212 format).
        Parses the .hea header file, decodes the .dat binary signal file,
        extracts demographic covariates (Age, Sex) and de-identifies the patient.
        """
        base_dir = Path(record_path_or_dir)
        hea_file = base_dir / f"{record_name}.hea"
        dat_file = base_dir / f"{record_name}.dat"

        if not hea_file.exists():
            raise FileNotFoundError(f"WFDB header file not found: {hea_file}")
        if not dat_file.exists():
            raise FileNotFoundError(f"WFDB binary data file not found: {dat_file}")

        # 1. Parse header (.hea)
        header_meta = self._parse_wfdb_header(hea_file)
        fs = header_meta["fs"]
        num_channels = header_meta["num_channels"]
        channels_info = header_meta["channels"]

        # 2. Decode format 212 binary signal (.dat)
        raw_signals = self._decode_format_212(
            dat_file, 
            num_channels=num_channels, 
            channels_info=channels_info,
            max_samples=max_samples
        )

        # 3. Read annotations (.atr) if available
        atr_file = base_dir / f"{record_name}.atr"
        annotations = None
        if atr_file.exists():
            annotations = self._parse_wfdb_atr(atr_file, max_samples=max_samples)

        # 4. Ingestion Privacy Checks & De-identification
        raw_metadata = {
            "record_id": record_name,
            "patient_id": record_name,  # Record names in MIT-BIH correspond to patient subjects
            "source_format": "WFDB_212",
            "age": header_meta.get("age"),
            "sex": header_meta.get("sex"),
            "medications": header_meta.get("medications"),
            "original_fs": fs
        }
        sanitized_metadata = self.privacy.sanitize_metadata(raw_metadata)
        patient_pseudonym = sanitized_metadata["patient_pseudonym"]

        logger.info(
            f"Ingested WFDB record: {record_name} -> Pseudonym: {patient_pseudonym} | "
            f"Channels: {list(raw_signals.keys())} | fs: {fs} Hz | "
            f"Samples: {len(next(iter(raw_signals.values())))}"
        )

        return PhysiologicalRecord(
            patient_pseudonym=patient_pseudonym,
            record_id=record_name,
            sampling_frequency=fs,
            signals=raw_signals,
            metadata=sanitized_metadata,
            annotations=annotations
        )

    def _parse_wfdb_header(self, hea_path: Path) -> Dict[str, Any]:
        """Parses WFDB .hea ASCII header format."""
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        first_line = lines[0].split()
        record_name = first_line[0]
        num_channels = int(first_line[1])
        fs = float(first_line[2])
        total_samples = int(first_line[3]) if len(first_line) > 3 else 0

        channels = []
        age = None
        sex = None
        medications = []

        for line in lines[1:]:
            if line.startswith("#"):
                # Demographic line: e.g. '# 69 M 1085 1629 x1' or medication info
                comment = line.lstrip("#").strip()
                tokens = comment.split()
                if len(tokens) >= 2 and tokens[0].isdigit() and tokens[1].upper() in ("M", "F"):
                    age = int(tokens[0])
                    sex = tokens[1].upper()
                else:
                    medications.append(comment)
            else:
                parts = line.split()
                if len(parts) >= 9:
                    ch_info = {
                        "filename": parts[0],
                        "format": int(parts[1]),
                        "gain": float(parts[2].split("(")[0]) if "(" in parts[2] else float(parts[2]),
                        "baseline": int(parts[2].split("(")[1].split(")")[0]) if "(" in parts[2] else 1024,
                        "units": parts[2].split("/")[1] if "/" in parts[2] else "mV",
                        "adc_res": int(parts[3]),
                        "desc": parts[8] if len(parts) > 8 else f"ch_{len(channels)}"
                    }
                    channels.append(ch_info)

        return {
            "record_name": record_name,
            "num_channels": num_channels,
            "fs": fs,
            "total_samples": total_samples,
            "channels": channels,
            "age": age,
            "sex": sex,
            "medications": ", ".join(medications) if medications else None
        }

    def _decode_format_212(
        self, 
        dat_path: Path, 
        num_channels: int, 
        channels_info: List[Dict[str, Any]],
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decodes PhysioNet WFDB format 212.
        Each sample pair for channel 0 and 1 is packed across 3 bytes:
        b0, b1, b2 ->
          val0 = b0 | ((b1 & 0x0F) << 8)
          val1 = b2 | ((b1 & 0xF0) << 4)
        """
        with open(dat_path, "rb") as f:
            raw_bytes = f.read()

        num_triplets = len(raw_bytes) // 3
        if max_samples:
            num_triplets = min(num_triplets, max_samples)

        # Convert bytes to numpy uint8
        data = np.frombuffer(raw_bytes[: num_triplets * 3], dtype=np.uint8).reshape(-1, 3)

        b0 = data[:, 0].astype(np.int32)
        b1 = data[:, 1].astype(np.int32)
        b2 = data[:, 2].astype(np.int32)

        raw_ch0 = b0 | ((b1 & 0x0F) << 8)
        raw_ch1 = b2 | ((b1 & 0xF0) << 4)

        # 12-bit two's complement sign extension
        raw_ch0[raw_ch0 >= 2048] -= 4096
        raw_ch1[raw_ch1 >= 2048] -= 4096

        # Convert to physical units (mV) using header gain and baseline
        gain0 = channels_info[0]["gain"] if channels_info and len(channels_info) > 0 else 200.0
        base0 = channels_info[0]["baseline"] if channels_info and len(channels_info) > 0 else 1024
        ch0_name = channels_info[0]["desc"] if channels_info and len(channels_info) > 0 else "MLII"

        gain1 = channels_info[1]["gain"] if channels_info and len(channels_info) > 1 else 200.0
        base1 = channels_info[1]["baseline"] if channels_info and len(channels_info) > 1 else 1024
        ch1_name = channels_info[1]["desc"] if channels_info and len(channels_info) > 1 else "V5"

        phys_ch0 = (raw_ch0 - base0) / gain0
        phys_ch1 = (raw_ch1 - base1) / gain1

        return {
            ch0_name: phys_ch0,
            ch1_name: phys_ch1
        }

    def _parse_wfdb_atr(self, atr_path: Path, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Simple binary parser for WFDB .atr annotation files.
        Extracts sample indices and MIT-BIH annotation codes (Normal 'N', Ventricular 'V', etc.).
        """
        with open(atr_path, "rb") as f:
            data = f.read()

        sample_indices = []
        annotation_codes = []
        curr_sample = 0
        i = 0

        while i < len(data) - 1:
            b0 = data[i]
            b1 = data[i + 1]
            i += 2

            code = b1 >> 2
            delta = b0 | ((b1 & 0x03) << 8)

            if code == 59:  # SKIP
                if i + 4 <= len(data):
                    # 32-bit skip
                    skip = data[i] | (data[i+1] << 8) | (data[i+2] << 16) | (data[i+3] << 24)
                    curr_sample += skip
                    i += 4
            elif code == 63:  # AUX (text note)
                length = delta
                i += length
                if length % 2 != 0:
                    i += 1  # 16-bit word alignment
            elif code == 0:
                # End of file / null
                break
            else:
                curr_sample += delta
                if max_samples and curr_sample >= max_samples:
                    break
                sample_indices.append(curr_sample)
                # Map WFDB code (1=N, 5=V, 8=A, etc.)
                annotation_codes.append(code)

        return {
            "indices": np.array(sample_indices, dtype=np.int64),
            "codes": np.array(annotation_codes, dtype=np.int32)
        }

    def load_csv_record(
        self,
        file_path: Union[str, Path],
        patient_id: str,
        timestamp_col: Optional[str] = "timestamp",
        signal_cols: Optional[List[str]] = None,
        fs: float = 1.0
    ) -> PhysiologicalRecord:
        """
        Loads a CSV record, validates headers and rows, applies HIPAA de-identification,
        and extracts signal columns as numpy arrays.
        """
        p = Path(file_path)
        self.validator.verify_file_checksum(p)

        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not rows:
            raise ValidationError(f"CSV file {p.name} contains no data rows.")

        target_signals = signal_cols or [col for col in fieldnames if col != timestamp_col]
        self.validator.validate_schema(fieldnames, target_signals)

        signals_dict: Dict[str, List[float]] = {col: [] for col in target_signals}
        for row_idx, row in enumerate(rows):
            for col in target_signals:
                try:
                    val = float(row[col]) if row[col] != "" else np.nan
                except ValueError:
                    val = np.nan
                signals_dict[col].append(val)

        final_signals = {k: np.array(v, dtype=np.float64) for k, v in signals_dict.items()}

        raw_meta = {
            "patient_id": patient_id,
            "filename": p.name,
            "row_count": len(rows),
            "source_format": "CSV"
        }
        sanitized_meta = self.privacy.sanitize_metadata(raw_meta)

        return PhysiologicalRecord(
            patient_pseudonym=sanitized_meta["patient_pseudonym"],
            record_id=p.stem,
            sampling_frequency=fs,
            signals=final_signals,
            metadata=sanitized_meta
        )

    def generate_synthetic_record(
        self,
        patient_id: str,
        duration_sec: float = 60.0,
        fs: float = 360.0,
        condition: str = "NORMAL",
        demographics: Optional[Dict[str, Any]] = None
    ) -> PhysiologicalRecord:
        """
        Generates a validated synthetic multi-lead ECG & PPG benchmark record
        with known clinical ground truth (e.g. Normal Sinus Rhythm vs. Ventricular Arrhythmia / Hypoxia).
        Useful for end-to-end integration tests and API simulation.
        """
        n_samples = int(duration_sec * fs)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Baseline heart rate in Hz
        hr_bpm = 72.0 if condition == "NORMAL" else (140.0 if condition == "ARRHYTHMIA" else 80.0)
        f_hr = hr_bpm / 60.0

        # Synthetic ECG wave generation (P-Q-R-S-T synthesis via Gaussian wave superposition)
        ecg = np.zeros(n_samples)
        beat_period = 1.0 / f_hr
        r_peaks = []

        curr_time = 0.2
        while curr_time < duration_sec - 0.2:
            idx = int(curr_time * fs)
            r_peaks.append(idx)
            # Normal beat vs abnormal wide ectopic beat
            qrs_width = 0.04 if condition == "NORMAL" else 0.09
            amp_r = 1.2 if condition == "NORMAL" else 1.8

            for offset, amp, width in [
                (-0.16, 0.15, 0.05),   # P wave
                (-0.04, -0.20, 0.02),  # Q wave
                (0.00, amp_r, qrs_width), # R wave
                (0.04, -0.30, 0.03),   # S wave
                (0.22, 0.35, 0.08)     # T wave
            ]:
                ecg += amp * np.exp(-((t - (curr_time + offset)) ** 2) / (2 * width ** 2))

            # Add mild baseline wander (0.2 Hz) and 50 Hz powerline hum
            curr_time += beat_period * (1.0 + np.random.normal(0, 0.02))

        # Add physical noise
        ecg += 0.05 * np.sin(2 * np.pi * 0.2 * t)  # Baseline wander
        ecg += 0.02 * np.sin(2 * np.pi * 50.0 * t) # Mains hum
        ecg += np.random.normal(0, 0.015, n_samples) # Thermal noise

        # Synthetic PPG & SpO2
        ppg_fs = SIGNAL_CONFIG.DEFAULT_PPG_FS
        n_ppg = int(duration_sec * ppg_fs)
        t_ppg = np.linspace(0, duration_sec, n_ppg, endpoint=False)
        ppg = 0.5 + 0.3 * np.sin(2 * np.pi * f_hr * t_ppg) + 0.1 * np.sin(4 * np.pi * f_hr * t_ppg)
        
        spo2_val = 98.5 if condition != "HYPOXIA" else 84.0
        spo2_signal = np.full(n_ppg, spo2_val) + np.random.normal(0, 0.2, n_ppg)

        raw_meta = {
            "patient_id": patient_id,
            "condition": condition,
            "age": demographics.get("age", 55) if demographics else 55,
            "sex": demographics.get("sex", "M") if demographics else "M",
            "device": "SyntheticClinicalSensor-v1",
            "source_format": "SYNTHETIC"
        }
        sanitized_meta = self.privacy.sanitize_metadata(raw_meta)

        return PhysiologicalRecord(
            patient_pseudonym=sanitized_meta["patient_pseudonym"],
            record_id=f"SYN-{patient_id}",
            sampling_frequency=fs,
            signals={"ECG": ecg, "PPG": ppg, "SpO2": spo2_signal},
            metadata=sanitized_meta,
            annotations={"r_peaks": np.array(r_peaks), "condition": condition}
        )
