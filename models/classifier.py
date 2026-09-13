"""
health_pipeline.models.classifier
Calibrated clinical classifier wrapper with risk triage stratification and secure serialization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from health_pipeline.config import MODEL_CONFIG, PATHS
from health_pipeline.security.secure_logger import get_secure_logger

logger = get_secure_logger("health_pipeline.models.classifier")


@dataclass
class PredictionOutput:
    predicted_class: int
    class_label: str
    confidence_score: float      # Calibrated probability of the predicted class
    pathology_probability: float # Calibrated probability of abnormal/pathology class
    risk_tier: str               # LOW_RISK, MONITOR, CRITICAL
    status: str = "SUCCESS"


class PhysiologicalModelWrapper:
    """
    Wraps an ensemble classifier with probability calibration and clinical triage stratification.
    """

    CLASS_MAPPING = {
        0: "Normal Sinus Rhythm",
        1: "Arrhythmia / Cardiac Abnormality"
    }

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        calibrate: bool = True
    ):
        if base_estimator is None:
            # High-performance gradient booster supporting class weights and non-linear interactions
            self.model = HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=150,
                learning_rate=0.08,
                max_depth=6,
                random_state=MODEL_CONFIG.RANDOM_SEED
            )
        else:
            self.model = base_estimator

        self.feature_names = feature_names or []
        self.calibrate = calibrate
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None
    ) -> "PhysiologicalModelWrapper":
        """
        Fits the base model and calibrates posterior probabilities using Platt scaling.
        """
        logger.info(f"Fitting model on {len(X)} samples with {X.shape[1]} features...")
        if self.calibrate and len(X) >= 50:
            # Calibrate probabilities via sigmoid (Platt scaling)
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.model, 
                method="sigmoid", 
                cv=3
            )
            self.calibrated_model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)

        self.is_fitted = True
        logger.info("Model fitting and calibration complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Cannot predict with an unfitted model.")
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary class predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def infer_single_sample(self, feature_vector: np.ndarray) -> PredictionOutput:
        """
        Runs clinical inference on a single 1D feature array, returning calibrated probabilities
        and a risk triage score (LOW_RISK, MONITOR, CRITICAL).
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        probas = self.predict_proba(feature_vector)[0]
        pred_class = int(np.argmax(probas))
        conf_score = float(probas[pred_class])
        pathology_prob = float(probas[1]) if len(probas) > 1 else conf_score

        # Clinical Triage Mapping
        if pathology_prob >= MODEL_CONFIG.HIGH_RISK_THRESHOLD:
            risk_tier = "CRITICAL"
        elif pathology_prob >= MODEL_CONFIG.LOW_RISK_THRESHOLD:
            risk_tier = "MONITOR"
        else:
            risk_tier = "LOW_RISK"

        class_label = self.CLASS_MAPPING.get(pred_class, f"Class {pred_class}")

        return PredictionOutput(
            predicted_class=pred_class,
            class_label=class_label,
            confidence_score=conf_score,
            pathology_probability=pathology_prob,
            risk_tier=risk_tier
        )

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves the serialized model bundle to disk."""
        target_path = Path(filepath or (PATHS.MODELS_DIR / "health_classifier.joblib"))
        bundle = {
            "model": self.calibrated_model if self.calibrated_model else self.model,
            "feature_names": self.feature_names,
            "calibrate": self.calibrate,
            "class_mapping": self.CLASS_MAPPING
        }
        joblib.dump(bundle, target_path)
        logger.info(f"Model saved to {target_path}")
        return target_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PhysiologicalModelWrapper":
        """Loads a model bundle from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model file does not exist: {p}")
        bundle = joblib.load(p)
        wrapper = cls(
            base_estimator=bundle["model"],
            feature_names=bundle.get("feature_names", []),
            calibrate=bundle.get("calibrate", True)
        )
        wrapper.calibrated_model = bundle["model"] if isinstance(bundle["model"], CalibratedClassifierCV) else None
        wrapper.is_fitted = True
        return wrapper
