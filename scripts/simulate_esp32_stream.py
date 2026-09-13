"""
health_pipeline.scripts.simulate_esp32_stream
Simulates an ESP32 hardware device (AD8232 ECG + MAX30102 SpO2/BPM + DS18B20 Temp)
streaming live telemetry packets to the secure ML Health Pipeline API.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from health_pipeline.config import SECURITY_CONFIG, PATHS
from health_pipeline.data.loader import PhysiologicalDataLoader


def simulate_esp32_stream(
    target_url: str = "http://127.0.0.1:8000/api/esp32/telemetry",
    duration_sec: int = 30,
    interval_sec: float = 1.0
):
    print("=" * 65)
    print("ESP32 PATIENT HEALTH MONITOR HARDWARE SIMULATOR")
    print(f"Target Gateway: {target_url}")
    print(f"Sensors: AD8232 (ECG), MAX30102 (SpO2, BPM), DS18B20 (Temperature)")
    print("=" * 65)

    loader = PhysiologicalDataLoader()
    rec100 = loader.load_wfdb_record(PATHS.DATASET_DIR, "100", max_samples=10000)
    rec106 = loader.load_wfdb_record(PATHS.DATASET_DIR, "106", max_samples=10000)

    ecg_normal = rec100.signals["MLII"]
    ecg_arrhythmia = rec106.signals["MLII"]

    fs = 125  # ESP32 sampling rate
    window_samples = 250  # 2-sec window

    start_time = time.time()
    packet_id = 0

    while time.time() - start_time < duration_sec:
        packet_id += 1
        elapsed = time.time() - start_time

        # Simulate transitions: Normal (0-10s) -> Arrhythmia (10-20s) -> Leads-Off (20-25s) -> Normal
        if elapsed < 10.0:
            cond_desc = "Normal Sinus Rhythm"
            idx_start = (packet_id * 100) % (len(ecg_normal) - window_samples)
            signal_window = ecg_normal[idx_start : idx_start + window_samples].tolist()
            bpm = float(72.0 + np.random.normal(0, 1.5))
            spo2 = float(98.5 + np.random.normal(0, 0.4))
            temp_c = float(36.6 + np.random.normal(0, 0.1))
            leads_off = False
        elif elapsed < 20.0:
            cond_desc = "Ventricular Arrhythmia (PVCs)"
            idx_start = (packet_id * 100) % (len(ecg_arrhythmia) - window_samples)
            signal_window = ecg_arrhythmia[idx_start : idx_start + window_samples].tolist()
            bpm = float(135.0 + np.random.normal(0, 4.0))
            spo2 = float(92.0 + np.random.normal(0, 1.0))
            temp_c = float(37.4 + np.random.normal(0, 0.1))
            leads_off = False
        elif elapsed < 25.0:
            cond_desc = "AD8232 Electrode Detached (Leads-Off LO+/LO-)"
            signal_window = [0.0] * window_samples
            bpm = 0.0
            spo2 = 0.0
            temp_c = 36.6
            leads_off = True
        else:
            cond_desc = "Normal Rhythm Restored"
            idx_start = (packet_id * 100) % (len(ecg_normal) - window_samples)
            signal_window = ecg_normal[idx_start : idx_start + window_samples].tolist()
            bpm = 74.0
            spo2 = 98.0
            temp_c = 36.6
            leads_off = False

        payload = {
            "device_id": "ESP32-HOW2ELEC-NODE-01",
            "fs": fs,
            "leads_off": leads_off,
            "bpm": round(bpm, 1),
            "spo2": round(spo2, 1),
            "temperature_c": round(temp_c, 2),
            "signal": signal_window
        }

        try:
            req = urllib.request.Request(
                target_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {SECURITY_CONFIG.API_AUTH_TOKEN}"
                }
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                pred = res_body.get("prediction", {})
                triage = pred.get("triage_risk_tier", res_body.get("triage_risk_tier", "NORMAL"))
                print(
                    f"[{elapsed:4.1f}s | Pkt #{packet_id:02d}] Sent: {cond_desc} -> "
                    f"BPM: {bpm:5.1f} | SpO2: {spo2:4.1f}% | Temp: {temp_c:4.1f}C | "
                    f"ML Triage: >>> {triage} <<< ({pred.get('clinical_condition', '')})"
                )
        except Exception as e:
            print(f"[{elapsed:4.1f}s | Pkt #{packet_id:02d}] Gateway Connection Error: {e}")

        time.sleep(interval_sec)

    print("=" * 65)
    print("ESP32 Simulation Complete.")
    print("=" * 65)


if __name__ == "__main__":
    simulate_esp32_stream()
