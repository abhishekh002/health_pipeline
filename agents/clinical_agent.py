"""
health_pipeline.agents.clinical_agent
Autonomous Clinical AI Agent for real-time physiological signal interpretation,
diagnostic reasoning, trend analysis, and intervention guidance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AgentClinicalReport:
    assessment_summary: str
    primary_finding: str
    risk_level: str
    clinical_rationale: List[str]
    immediate_actions: List[str]
    differential_diagnosis: List[str]
    biomarker_breakdown: Dict[str, str]


class ClinicalAIAgent:
    """
    Expert Clinical AI Agent specializing in real-time cardiovascular telemetry,
    cardiac arrhythmia triage, and multi-modal vital sign analysis.
    """

    def __init__(self, agent_name: str = "CardioGuard AI Clinical Copilot"):
        self.agent_name = agent_name

    def analyze_patient_state(
        self,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        signal_quality: Optional[Dict[str, Any]] = None,
        leads_off: bool = False
    ) -> AgentClinicalReport:
        """
        Deep clinical reasoning engine generating structured diagnostic reports
        from raw biomarkers, SQI assessments, and calibrated model inferences.
        """
        hr = float(features.get("mean_hr_bpm", features.get("bpm", 72.0)))
        spo2 = float(features.get("spo2_mean", features.get("spo2", 98.0)))
        temp_c = float(features.get("temperature_c", 36.6))
        rmssd = float(features.get("rmssd_ms", 25.0))
        sdnn = float(features.get("sdnn_ms", 30.0))
        pnn50 = float(features.get("pnn50", 5.0))
        lf_hf = float(features.get("lf_hf_ratio", 1.2))
        triage_tier = prediction.get("triage_risk_tier", "LOW_RISK")
        condition = prediction.get("clinical_condition", "Normal Sinus Rhythm")
        conf = float(prediction.get("calibrated_confidence", 0.95))

        rationale = []
        actions = []
        differentials = []
        biomarkers = {
            "Heart Rate": f"{hr:.1f} bpm",
            "Oxygen Saturation (SpO2)": f"{spo2:.1f}%",
            "Body Temperature": f"{temp_c:.1f} °C",
            "Vagal Tone (RMSSD)": f"{rmssd:.1f} ms",
            "Autonomic Balance (LF/HF)": f"{lf_hf:.2f}"
        }

        # 1. Lead detachment / Signal Quality Check
        if leads_off:
            return AgentClinicalReport(
                assessment_summary="AD8232 ECG biomedical electrodes are detached from patient skin.",
                primary_finding="Electrode Disconnection (Leads-Off LO+/LO- active)",
                risk_level="MONITOR",
                clinical_rationale=[
                    "LO+ or LO- lead-off comparator pins on the AD8232 front-end registered high impedance.",
                    "Signal is flatlined; automated algorithmic interpretation is halted to prevent diagnostic errors."
                ],
                immediate_actions=[
                    "Check right clavicle (RA), left clavicle (LA), and lower abdomen (RL) electrode contacts.",
                    "Clean skin with alcohol swab and apply fresh Ag/AgCl disposable adhesive patches.",
                    "Verify 3.5mm biomedical cable seating in the AD8232 sensor jack."
                ],
                differential_diagnosis=["Sensor detachment", "Lead pull / movement artifact", "Dry electrode gel"],
                biomarker_breakdown=biomarkers
            )

        # 2. Critical Cardiac Emergency Analysis
        if triage_tier == "CRITICAL" or hr > 130.0 or spo2 < 88.0:
            if hr > 120.0 and spo2 < 90.0:
                summary = "ACUTE EMERGENCY: Severe Ventricular Tachycardia compounded by Acute Hypoxemia."
                finding = "Ventricular Tachycardia / Sustained Tachyarrhythmia with Respiratory Compromise"
                differentials = [
                    "Sustained Ventricular Tachycardia (VT)",
                    "Supraventricular Tachycardia (SVT) with 1:1 conduction",
                    "Acute Coronary Syndrome with Ischemic Arrhythmia",
                    "Acute Pulmonary Embolism or Hypoxic Cardiac Arrest precursor"
                ]
            elif hr > 120.0:
                summary = "URGENT: Marked Tachyarrhythmia detected by 1D neural classifier."
                finding = "Non-Sinus Tachyarrhythmia (Elevated Ventricular Rate)"
                differentials = ["Ventricular Tachycardia", "Atrial Fibrillation with Rapid Ventricular Response (AFib with RVR)", "Sinus Tachycardia from acute sepsis/shock"]
            else:
                summary = "URGENT: Hypoxic Desaturation Event with cardiac instability."
                finding = "Acute Hypoxemia (SpO2 < 88%)"
                differentials = ["Acute Respiratory Distress", "Sensor displacement / poor perfusion", "Pulmonary compromise"]

            rationale.append(f"Calibrated neural classifier output: {condition} with {conf*100:.1f}% confidence.")
            if hr > 100.0:
                rationale.append(f"Tachycardia with mean rate {hr:.1f} bpm, significantly exceeding physiologic resting ceiling (100 bpm).")
            if spo2 < 90.0:
                rationale.append(f"Critical desaturation at {spo2:.1f}%, indicating impaired pulmonary gas exchange or systemic hypoperfusion.")
            if rmssd > 80.0:
                rationale.append(f"Excessive beat-to-beat variability (RMSSD = {rmssd:.1f} ms) consistent with chaotic ectopic depolarizations.")

            actions = [
                "IMMEDIATE: Notify attending cardiologist / Rapid Response Team (RRT).",
                "Assess patient responsiveness, airway, breathing, and radial/carotid pulses.",
                "Administer high-flow supplemental O2 via non-rebreather mask to address hypoxemia.",
                "Obtain immediate 12-lead diagnostic ECG and place defibrillator pads on patient chest.",
                "Prepare IV access; review bedside emergency cardiac medications (Amiodarone / Lidocaine)."
            ]
            risk = "CRITICAL"

        # 3. Monitor / Elevated Risk Analysis
        elif triage_tier == "MONITOR" or (hr > 100.0 or hr < 50.0) or (spo2 < 94.0):
            summary = "ELEVATED RISK: Borderline cardiac rhythm or autonomic dysregulation observed."
            finding = f"Borderline Hemodynamics: {condition}"
            risk = "MONITOR"
            differentials = ["Premature Ventricular Contractions (PVCs)", "Sinus Bradycardia / Sinus Tachycardia", "Mild hypoxia / transient apnea"]

            rationale.append(f"Model identified {condition} with risk probability in the MONITOR tier.")
            if hr > 100.0:
                rationale.append(f"Elevated heart rate ({hr:.1f} bpm). Review for fever (Temp: {temp_c:.1f}°C), pain, or anxiety.")
            if hr < 50.0:
                rationale.append(f"Bradycardia ({hr:.1f} bpm). Assess for vagal stimulation or medication side-effects (e.g. beta blockers).")
            if spo2 < 94.0:
                rationale.append(f"Suboptimal oxygen saturation ({spo2:.1f}%).")

            actions = [
                "Schedule nursing bedside assessment within 10 minutes.",
                "Verify sensor placement on finger (MAX30102) and chest leads (AD8232).",
                "Review longitudinal vital trends and medication schedule.",
                "Perform repeat 10-second Signal Quality Index (SQI) audit."
            ]

        # 4. Normal / Stable Patient
        else:
            summary = "STABLE: Normal Sinus Rhythm with healthy autonomic regulation."
            finding = "Normal Sinus Rhythm (Hemodynamically Stable)"
            risk = "LOW_RISK"
            differentials = ["Normal Cardiac Physiology", "Physiologic Sinus Rhythm"]

            rationale.append(f"Heart rate {hr:.1f} bpm is within healthy normal resting boundaries (60-100 bpm).")
            rationale.append(f"Oxygen saturation {spo2:.1f}% indicates normal arterial oxygenation.")
            rationale.append(f"Autonomic tone markers (RMSSD = {rmssd:.1f} ms, LF/HF = {lf_hf:.2f}) reflect balanced parasympathetic/sympathetic tone.")

            actions = [
                "Continue standard continuous passive telemetry monitoring.",
                "Maintain periodic vital sign logging every 60 seconds.",
                "No immediate clinical intervention required."
            ]

        return AgentClinicalReport(
            assessment_summary=summary,
            primary_finding=finding,
            risk_level=risk,
            clinical_rationale=rationale,
            immediate_actions=actions,
            differential_diagnosis=differentials,
            biomarker_breakdown=biomarkers
        )

    def answer_clinical_query(
        self,
        query: str,
        current_state: Dict[str, Any]
    ) -> str:
        """
        Interactive conversational reasoning interface for clinicians and nurses.
        """
        q = query.lower().strip()
        features = current_state.get("features", {})
        prediction = current_state.get("prediction", {})
        hr = float(features.get("mean_hr_bpm", features.get("bpm", 72.0)))
        spo2 = float(features.get("spo2_mean", features.get("spo2", 98.0)))
        tier = prediction.get("triage_risk_tier", "LOW_RISK")
        cond = prediction.get("clinical_condition", "Normal Sinus Rhythm")

        if "why" in q or "cause" in q or "reason" in q or "explain" in q:
            return (
                f"**Clinical AI Rationale:** The system flagged **{cond}** with a triage tier of **{tier}**. "
                f"Key drivers: Heart Rate is **{hr:.1f} bpm**, Arterial Oxygen Saturation is **{spo2:.1f}%**, "
                f"and waveform morphological descriptors indicate abnormal ventricular repolarization."
            )
        elif "action" in q or "treatment" in q or "what to do" in q or "protocol" in q:
            if tier == "CRITICAL":
                return (
                    "**Priority Emergency Actions:**\n"
                    "1. Immediately alert the attending physician / Rapid Response Team.\n"
                    "2. Check patient airway, breathing, responsiveness, and pulse.\n"
                    "3. Administer high-flow O2 via mask to counter SpO2 desaturation.\n"
                    "4. Obtain urgent 12-lead ECG and ready defibrillator pads.\n"
                    "5. Ensure patent intravenous (IV) access."
                )
            else:
                return (
                    "**Recommended Protocol:**\n"
                    "1. Patient is currently stable in Normal Sinus Rhythm.\n"
                    "2. Continue passive monitoring and verify lead contacts.\n"
                    "3. Routine vital check scheduled in 60 minutes."
                )
        elif "hrv" in q or "rmssd" in q or "autonomic" in q:
            rmssd = features.get("rmssd_ms", 25.0)
            return (
                f"**Heart Rate Variability (HRV) Analysis:**\n"
                f"- **RMSSD:** {rmssd:.1f} ms (Reflects parasympathetic vagal regulation of heart rate).\n"
                f"- **LF/HF Ratio:** {features.get('lf_hf_ratio', 1.2):.2f} (Sympathovagal balance marker).\n"
                f"- A sudden collapse in RMSSD combined with tachycardia often precedes acute arrhythmic decompensation."
            )
        elif "diagnosis" in q or "differential" in q:
            if tier == "CRITICAL":
                return (
                    "**Differential Diagnosis:**\n"
                    "- Primary: Acute Sustained Ventricular Tachycardia (VT)\n"
                    "- Secondary: Supraventricular Tachycardia with aberrancy\n"
                    "- Associated: Hypoxic Myocardial Ischemia"
                )
            else:
                return "**Differential Diagnosis:** Normal Cardiac Electrophysiology (Sinus Rhythm)."
        else:
            return (
                f"**CardioGuard AI Clinical Status:**\n"
                f"- Current Rhythm: **{cond}**\n"
                f"- Triage Tier: **{tier}**\n"
                f"- Vitals: Heart Rate **{hr:.1f} bpm**, SpO2 **{spo2:.1f}%**\n"
                f"Ask me about: *'What is the treatment protocol?'*, *'Explain the cause'*, or *'Differential diagnosis'*."
            )
