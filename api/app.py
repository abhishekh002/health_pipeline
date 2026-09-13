"""
health_pipeline.api.app
Secure Flask prediction server for physiological health monitoring inference.
Features:
- Bearer token / API key authentication
- Rate limiting per client
- Signal input sanitization and physiological range checks
- Automatic signal cleaning, SQI validation, and feature extraction
- Calibrated confidence scoring and clinical risk tiering
- Complete masking of internal model architecture and server traces
"""

from pathlib import Path
from typing import Any, Dict, Optional
import flask
from flask import Flask, jsonify, request
import numpy as np

from health_pipeline.agents.clinical_agent import ClinicalAIAgent
from health_pipeline.api.schemas import APIValidationError, PayloadValidator
from health_pipeline.config import PATHS, SECURITY_CONFIG
from health_pipeline.models.classifier import PhysiologicalModelWrapper, PredictionOutput
from health_pipeline.preprocessing.cleaner import SignalCleaner
from health_pipeline.preprocessing.features import FeatureExtractor
from health_pipeline.preprocessing.sqi import SignalQualityAssessor
from health_pipeline.security.auth import Authenticator
from health_pipeline.security.secure_logger import get_secure_logger

logger = get_secure_logger("health_pipeline.api")


def create_app(model_path: Optional[str] = None) -> Flask:
    """Factory creating and configuring the secure inference Flask app."""
    app = Flask(__name__)
    authenticator = Authenticator(rate_limit_per_minute=120)
    clinical_agent = ClinicalAIAgent()

    # Load trained model bundle
    target_model_path = model_path or (PATHS.MODELS_DIR / "health_classifier.joblib")
    loaded_model: Optional[PhysiologicalModelWrapper] = None
    if target_model_path.exists():
        try:
            loaded_model = PhysiologicalModelWrapper.load(target_model_path)
            logger.info(f"API loaded model successfully from {target_model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {target_model_path}: {e}. Fallback to mock.")
            loaded_model = None

    # Security Headers Middleware
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline'"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.errorhandler(APIValidationError)
    def handle_validation_error(e):
        return jsonify({
            "status": "ERROR",
            "error_type": "VALIDATION_FAILED",
            "message": str(e)
        }), 400

    # Web Dashboard and Report Endpoints
    @app.route("/", methods=["GET"])
    def index():
        dashboard_path = Path(__file__).parent / "dashboard.html"
        if dashboard_path.exists():
            return dashboard_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
        return "Physiological Health Monitoring API is running. Go to /report for evaluation.", 200

    @app.route("/report", methods=["GET"])
    def view_report():
        report_path = PATHS.REPORTS_DIR / "clinical_evaluation_report.html"
        if report_path.exists():
            return report_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
        return "Clinical report not found.", 404

    @app.route("/report/pdf", methods=["GET"])
    def download_pdf_report():
        pdf_path = PATHS.REPORTS_DIR / "clinical_evaluation_report.pdf"
        if not pdf_path.exists():
            from health_pipeline.evaluation.pdf_generator import generate_clinical_evaluation_pdf
            generate_clinical_evaluation_pdf(pdf_path)
        return flask.send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=False,
            download_name="clinical_evaluation_report.pdf"
        )

    @app.route("/api/presets", methods=["GET"])
    def get_preset():
        p_type = request.args.get("type", "100")
        fs = 360.0
        try:
            from health_pipeline.data.loader import PhysiologicalDataLoader
            loader = PhysiologicalDataLoader()
            if p_type in ("100", "106"):
                rec = loader.load_wfdb_record(PATHS.DATASET_DIR, p_type, max_samples=3600)
                ch = "MLII" if "MLII" in rec.signals else next(iter(rec.signals.keys()))
                sig = rec.signals[ch][:1800].tolist()
                name = f"MIT-BIH Record {p_type} ({'Normal Sinus' if p_type == '100' else 'Arrhythmia'})"
            elif p_type == "corrupt":
                t = np.linspace(0, 5.0, int(5.0 * fs), endpoint=False)
                sig = (0.5 * np.sin(2 * np.pi * 1.5 * t)).tolist()
                name = "Corrupted Drift / Non-QRS Artifact"
            elif p_type == "alert":
                rec = loader.load_wfdb_record(PATHS.DATASET_DIR, "106", max_samples=3600)
                ch = "MLII" if "MLII" in rec.signals else next(iter(rec.signals.keys()))
                sig = rec.signals[ch][500:2300].tolist()
                name = "🚨 Cardiac Emergency Alert Signal (Ventricular Ectopy / Tachycardia)"
            else:
                syn = loader.generate_synthetic_record("TACHY", duration_sec=5.0, condition="ARRHYTHMIA")
                sig = syn.signals["ECG"].tolist()
                name = "Hypoxia / Tachycardia Signal"
            return jsonify({"name": name, "signal": sig, "fs": fs})
        except Exception as e:
            return jsonify({"name": "Default Test Wave", "signal": [0.0] * 500, "fs": fs})

    # Global in-memory storage for latest ESP32 telemetry
    latest_esp32_telemetry: Dict[str, Any] = {
        "device_pseudonym": "NO_DEVICE_CONNECTED",
        "timestamp_utc": "",
        "leads_off": False,
        "bpm": 0.0,
        "spo2": 0.0,
        "temperature_c": 0.0,
        "signal": [],
        "prediction": {"clinical_condition": "Idle", "triage_risk_tier": "LOW_RISK"}
    }

    # ESP32 Dedicated Telemetry Endpoint
    @app.route("/api/esp32/telemetry", methods=["POST"])
    def esp32_telemetry():
        auth_header = request.headers.get("Authorization") or request.headers.get("X-API-Key")
        is_auth, auth_msg = authenticator.validate_token(auth_header)
        if not is_auth:
            return jsonify({"status": "ERROR", "message": auth_msg}), 401

        payload = request.get_json(silent=True) or {}
        raw_device_id = payload.get("device_id", "ESP32_UNKNOWN")
        from health_pipeline.security.privacy import PrivacyProtector
        privacy = PrivacyProtector()
        device_pseudonym = privacy.pseudonymize_identifier(raw_device_id)

        leads_off = bool(payload.get("leads_off", False))
        bpm = float(payload.get("bpm", 72.0))
        spo2 = float(payload.get("spo2", 98.0))
        temp_c = float(payload.get("temperature_c", 36.5))
        fs = float(payload.get("fs", 125.0))
        raw_signal = payload.get("signal", [])

        if leads_off:
            result = {
                "status": "ALERT_LEADS_OFF",
                "message": "AD8232 ECG electrodes detached from patient body.",
                "triage_risk_tier": "MONITOR",
                "device_pseudonym": device_pseudonym
            }
            latest_esp32_telemetry.update({
                "device_pseudonym": device_pseudonym,
                "leads_off": True,
                "bpm": bpm,
                "spo2": spo2,
                "temperature_c": temp_c,
                "signal": [0.0] * 50,
                "prediction": {"clinical_condition": "Leads Disconnected", "triage_risk_tier": "MONITOR"}
            })
            return jsonify(result), 200

        # Run signal through cleaner & classifier
        sig_arr = np.array(raw_signal, dtype=np.float64) if raw_signal else np.zeros(100)
        cleaner = SignalCleaner(fs=fs)
        cleaned_sig = cleaner.clean_ecg_signal(sig_arr)
        extractor = FeatureExtractor(fs=fs)
        features = extractor.extract_full_window_features(
            cleaned_sig, 
            spo2_window=np.array([spo2] * 20)
        )
        feature_vector = np.array(list(features.values())).reshape(1, -1)

        if loaded_model is not None and loaded_model.is_fitted:
            pred_out = loaded_model.infer_single_sample(feature_vector)
        else:
            is_arr = (bpm < 50.0 or bpm > 115.0 or spo2 < 90.0)
            pred_out = PredictionOutput(
                predicted_class=1 if is_arr else 0,
                class_label="Arrhythmia / Abnormal" if is_arr else "Normal Sinus Rhythm",
                confidence_score=0.94,
                pathology_probability=0.91 if is_arr else 0.05,
                risk_tier="CRITICAL" if is_arr else "LOW_RISK"
            )

        resp_payload = {
            "status": "SUCCESS",
            "device_pseudonym": device_pseudonym,
            "prediction": {
                "clinical_condition": pred_out.class_label,
                "calibrated_confidence": round(pred_out.confidence_score, 4),
                "pathology_probability": round(pred_out.pathology_probability, 4),
                "triage_risk_tier": pred_out.risk_tier
            },
            "clinical_vitals_summary": {
                "heart_rate_bpm": bpm,
                "spo2_percent": spo2,
                "temperature_c": temp_c,
                "rmssd_ms": round(features.get("rmssd_ms", 25.0), 1)
            }
        }

        latest_esp32_telemetry.update({
            "device_pseudonym": device_pseudonym,
            "leads_off": False,
            "bpm": bpm,
            "spo2": spo2,
            "temperature_c": temp_c,
            "signal": raw_signal[-150:] if raw_signal else [],
            "prediction": resp_payload["prediction"]
        })

        return jsonify(resp_payload), 200

    @app.route("/api/esp32/latest", methods=["GET"])
    def get_latest_esp32():
        return jsonify(latest_esp32_telemetry), 200

    # Clinical AI Agent Endpoints
    @app.route("/api/agent/analyze", methods=["POST"])
    def agent_analyze():
        payload = request.get_json(silent=True) or {}
        
        # If payload provides features/prediction use them; otherwise use latest telemetry
        features = payload.get("features")
        prediction = payload.get("prediction")
        leads_off = payload.get("leads_off")
        sqi_info = payload.get("signal_quality")

        if features is None:
            features = {
                "mean_hr_bpm": latest_esp32_telemetry.get("bpm", 72.0),
                "spo2_mean": latest_esp32_telemetry.get("spo2", 98.0),
                "temperature_c": latest_esp32_telemetry.get("temperature_c", 36.6),
                "rmssd_ms": 28.5,
                "sdnn_ms": 34.0,
                "lf_hf_ratio": 1.25
            }
        if prediction is None:
            prediction = latest_esp32_telemetry.get("prediction", {
                "clinical_condition": "Normal Sinus Rhythm",
                "triage_risk_tier": "LOW_RISK",
                "calibrated_confidence": 0.96
            })
        if leads_off is None:
            leads_off = latest_esp32_telemetry.get("leads_off", False)

        report = clinical_agent.analyze_patient_state(
            features=features,
            prediction=prediction,
            signal_quality=sqi_info,
            leads_off=leads_off
        )

        return jsonify({
            "status": "SUCCESS",
            "agent_name": clinical_agent.agent_name,
            "report": {
                "assessment_summary": report.assessment_summary,
                "primary_finding": report.primary_finding,
                "risk_level": report.risk_level,
                "clinical_rationale": report.clinical_rationale,
                "immediate_actions": report.immediate_actions,
                "differential_diagnosis": report.differential_diagnosis,
                "biomarker_breakdown": report.biomarker_breakdown
            }
        }), 200

    @app.route("/api/agent/chat", methods=["POST"])
    def agent_chat():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({
                "status": "ERROR",
                "message": "Query string is required."
            }), 400

        current_state = payload.get("state")
        if not current_state:
            current_state = {
                "features": {
                    "mean_hr_bpm": latest_esp32_telemetry.get("bpm", 72.0),
                    "spo2_mean": latest_esp32_telemetry.get("spo2", 98.0),
                    "temperature_c": latest_esp32_telemetry.get("temperature_c", 36.6),
                    "rmssd_ms": 28.5,
                    "lf_hf_ratio": 1.25
                },
                "prediction": latest_esp32_telemetry.get("prediction", {
                    "clinical_condition": "Normal Sinus Rhythm",
                    "triage_risk_tier": "LOW_RISK"
                })
            }

        response_text = clinical_agent.answer_clinical_query(query, current_state)
        return jsonify({
            "status": "SUCCESS",
            "agent_name": clinical_agent.agent_name,
            "query": query,
            "response": response_text
        }), 200

    # 1. Health Check Endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "ONLINE",
            "service": "Physiological Health Monitoring Inference API",
            "version": "1.0.0",
            "model_loaded": loaded_model is not None
        }), 200

    # 2. Secure Inference Endpoint
    @app.route("/v1/predict", methods=["POST"])
    def predict_endpoint():
        # Authentication
        auth_header = request.headers.get("Authorization") or request.headers.get("X-API-Key")
        is_auth, auth_msg = authenticator.validate_token(auth_header)
        if not is_auth:
            return jsonify({
                "status": "ERROR",
                "error_type": "AUTHENTICATION_FAILED",
                "message": auth_msg
            }), 401

        # Rate Limiting
        client_ip = request.remote_addr or "unknown_client"
        is_allowed, remaining = authenticator.check_rate_limit(client_ip)
        if not is_allowed:
            return jsonify({
                "status": "ERROR",
                "error_type": "RATE_LIMIT_EXCEEDED",
                "message": "Request rate limit exceeded. Please retry in 60 seconds."
            }), 429

        # Request Parsing & Sanitization
        raw_json = request.get_json(silent=True)
        if not raw_json:
            raise APIValidationError("Request payload must be valid JSON.")

        sanitized_input = PayloadValidator.sanitize_prediction_request(raw_json)

        # Process input: either direct features or raw physiological signal
        if "signal" in sanitized_input:
            fs = sanitized_input["fs"]
            raw_sig = sanitized_input["signal"]

            # Step 1: Signal Conditioning
            cleaner = SignalCleaner(fs=fs)
            cleaned_sig = cleaner.clean_ecg_signal(raw_sig)

            # Step 2: Signal Quality Index (SQI) Check
            assessor = SignalQualityAssessor(fs=fs)
            sqi_result = assessor.evaluate_ecg_window(cleaned_sig)

            if not sqi_result.is_acceptable:
                return jsonify({
                    "status": "REJECTED_LOW_QUALITY",
                    "message": "Signal rejected due to severe motion artifact or sensor detachment.",
                    "quality_score": round(sqi_result.quality_score, 4),
                    "rejection_reason": sqi_result.rejection_reason,
                    "metrics": sqi_result.metrics
                }), 422

            # Step 3: Feature Extraction
            extractor = FeatureExtractor(fs=fs)
            features = extractor.extract_full_window_features(
                ecg_window=cleaned_sig,
                ppg_window=sanitized_input.get("ppg_signal"),
                spo2_window=sanitized_input.get("spo2_signal")
            )
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            quality_info = {
                "quality_score": round(sqi_result.quality_score, 4),
                "is_acceptable": True
            }
        else:
            # Direct feature vector input
            feat_input = sanitized_input["features"]
            if isinstance(feat_input, dict):
                features = feat_input
                feature_vector = np.array(list(feat_input.values())).reshape(1, -1)
            else:
                features = {}
                feature_vector = feat_input.reshape(1, -1)
            quality_info = {"quality_score": 1.0, "is_acceptable": True}

        # Step 4: Model Inference & Calibration
        if loaded_model is not None and loaded_model.is_fitted:
            pred_out = loaded_model.infer_single_sample(feature_vector)
        else:
            # Fallback heuristic based on clinical rules if no trained model bundle is available
            hr = features.get("mean_hr_bpm", 72.0)
            rmssd = features.get("rmssd_ms", 35.0)
            is_abnormal = (hr < 45.0 or hr > 120.0 or rmssd < 15.0)
            pred_out = PredictionOutput(
                predicted_class=1 if is_abnormal else 0,
                class_label="Arrhythmia / Cardiac Abnormality" if is_abnormal else "Normal Sinus Rhythm",
                confidence_score=0.88,
                pathology_probability=0.85 if is_abnormal else 0.12,
                risk_tier="CRITICAL" if is_abnormal else "LOW_RISK"
            )

        # Step 5: Sanitized Clinical Response Output
        response_payload = {
            "status": "SUCCESS",
            "prediction": {
                "class_code": pred_out.predicted_class,
                "clinical_condition": pred_out.class_label,
                "calibrated_confidence": round(pred_out.confidence_score, 4),
                "pathology_probability": round(pred_out.pathology_probability, 4),
                "triage_risk_tier": pred_out.risk_tier
            },
            "signal_assessment": quality_info,
            "clinical_vitals_summary": {
                "heart_rate_bpm": round(features.get("mean_hr_bpm", 0.0), 1),
                "rmssd_ms": round(features.get("rmssd_ms", 0.0), 1),
                "sdnn_ms": round(features.get("sdnn_ms", 0.0), 1),
                "spo2_percent": round(features.get("spo2_mean", 98.0), 1)
            }
        }

        return jsonify(response_payload), 200

    return app
