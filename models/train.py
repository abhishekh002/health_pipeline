"""
health_pipeline.models.train
Model training harness with class imbalance mitigation, patient-isolated cross-validation,
hyperparameter optimization, and privacy-preserving training telemetry.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from health_pipeline.config import MODEL_CONFIG, PATHS
from health_pipeline.models.classifier import PhysiologicalModelWrapper
from health_pipeline.preprocessing.splitter import DatasetSplits, PatientLevelDataSplitter
from health_pipeline.security.secure_logger import get_secure_logger

logger = get_secure_logger("health_pipeline.models.train")


@dataclass
class CrossValidationSummary:
    n_folds: int
    mean_auc: float
    std_auc: float
    mean_sensitivity: float
    mean_specificity: float
    mean_f1: float
    fold_metrics: List[Dict[str, float]]


class ModelTrainer:
    """
    Orchestrates the training lifecycle, handling class imbalance,
    patient-grouped cross validation, and privacy-conscious logging.
    """

    def __init__(
        self,
        random_seed: int = MODEL_CONFIG.RANDOM_SEED,
        class_weight: str = MODEL_CONFIG.CLASS_WEIGHT
    ):
        self.random_seed = random_seed
        self.class_weight = class_weight

    def handle_class_imbalance(self, y: np.ndarray) -> np.ndarray:
        """
        Computes cost-sensitive sample weights inversely proportional to class frequencies.
        In clinical monitoring, pathological events (arrhythmias) are often a minority class (< 10%).
        Assigning higher loss weights to minority cases prevents the classifier from favoring
        the dominant normal class at the cost of clinical safety.
        """
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        return sample_weights

    def run_patient_grouped_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_pseudonyms: List[str],
        n_splits: int = 5
    ) -> CrossValidationSummary:
        """
        Performs patient-grouped cross-validation, guaranteeing no patient overlaps across folds.
        Logs fold metrics securely without patient identifiers.
        """
        splitter = PatientLevelDataSplitter(n_splits=n_splits, random_seed=self.random_seed)
        fold_gen = splitter.get_group_kfold_generator(X, y, patient_pseudonyms)

        fold_metrics = []
        logger.info(f"Starting {n_splits}-fold Patient-Grouped Cross-Validation...")

        for fold_idx, (train_idx, val_idx) in enumerate(fold_gen):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            # Compute sample weights for the training fold
            sample_w = self.handle_class_imbalance(y_tr)

            clf = HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=120,
                learning_rate=0.08,
                max_depth=5,
                random_state=self.random_seed + fold_idx
            )
            clf.fit(X_tr, y_tr, sample_weight=sample_w)

            y_pred = clf.predict(X_vl)
            y_prob = clf.predict_proba(X_vl)[:, 1] if len(clf.classes_) > 1 else y_pred

            # Calculate clinical metrics
            auc = float(roc_auc_score(y_vl, y_prob)) if len(np.unique(y_vl)) > 1 else 0.5
            sens = float(recall_score(y_vl, y_pred, zero_division=0))
            f1 = float(f1_score(y_vl, y_pred, zero_division=0))
            
            # Specificity = TN / (TN + FP)
            tn = np.sum((y_vl == 0) & (y_pred == 0))
            fp = np.sum((y_vl == 0) & (y_pred == 1))
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

            m = {
                "fold": fold_idx + 1,
                "val_samples": len(val_idx),
                "auc": auc,
                "sensitivity": sens,
                "specificity": spec,
                "f1_score": f1
            }
            fold_metrics.append(m)
            logger.info(
                f"[CV Fold {fold_idx + 1}/{n_splits}] Validation AUC: {auc:.4f} | "
                f"Sensitivity: {sens:.4f} | Specificity: {spec:.4f} | F1: {f1:.4f}"
            )

        mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        std_auc = float(np.std([m["auc"] for m in fold_metrics]))
        mean_sens = float(np.mean([m["sensitivity"] for m in fold_metrics]))
        mean_spec = float(np.mean([m["specificity"] for m in fold_metrics]))
        mean_f1 = float(np.mean([m["f1_score"] for m in fold_metrics]))

        logger.info(
            f"[CV Summary] Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} | "
            f"Mean Sensitivity: {mean_sens:.4f} | Mean Specificity: {mean_spec:.4f} | Mean F1: {mean_f1:.4f}"
        )

        return CrossValidationSummary(
            n_folds=n_splits,
            mean_auc=mean_auc,
            std_auc=std_auc,
            mean_sensitivity=mean_sens,
            mean_specificity=mean_spec,
            mean_f1=mean_f1,
            fold_metrics=fold_metrics
        )

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_pseudonyms: List[str]
    ) -> Dict[str, Any]:
        """
        Explores candidate hyperparameter configurations using patient-grouped cross-validation
        to optimize clinical AUC-ROC while guarding against overfitting.
        """
        param_grid = [
            {"learning_rate": 0.05, "max_iter": 100, "max_depth": 4},
            {"learning_rate": 0.08, "max_iter": 120, "max_depth": 5},
            {"learning_rate": 0.10, "max_iter": 150, "max_depth": 6}
        ]

        best_score = -1.0
        best_params = param_grid[0]
        logger.info("Executing patient-isolated hyperparameter search...")

        for params in param_grid:
            splitter = PatientLevelDataSplitter(n_splits=3, random_seed=self.random_seed)
            gen = splitter.get_group_kfold_generator(X, y, patient_pseudonyms)
            scores = []
            for tr_idx, vl_idx in gen:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_vl, y_vl = X[vl_idx], y[vl_idx]
                clf = HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=self.random_seed,
                    **params
                )
                clf.fit(X_tr, y_tr, sample_weight=self.handle_class_imbalance(y_tr))
                if len(np.unique(y_vl)) > 1:
                    score = roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1])
                else:
                    score = 0.5
                scores.append(score)
            
            mean_score = float(np.mean(scores))
            logger.info(f"Params {params} -> Mean Validation AUC: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        logger.info(f"Optimal Hyperparameters selected: {best_params} (Score: {best_score:.4f})")
        return best_params

    def train_final_model(
        self,
        splits: DatasetSplits,
        best_params: Optional[Dict[str, Any]] = None
    ) -> PhysiologicalModelWrapper:
        """
        Trains and calibrates the final clinical model on the combined training partition,
        logging progress securely.
        """
        params = best_params or {"learning_rate": 0.08, "max_iter": 150, "max_depth": 5}
        base_estimator = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=self.random_seed,
            **params
        )

        sample_weights = self.handle_class_imbalance(splits.y_train)

        wrapper = PhysiologicalModelWrapper(
            base_estimator=base_estimator,
            feature_names=splits.feature_names,
            calibrate=True
        )
        wrapper.fit(splits.X_train, splits.y_train, sample_weight=sample_weights)

        # Evaluate on validation split
        val_proba = wrapper.predict_proba(splits.X_val)[:, 1]
        val_preds = wrapper.predict(splits.X_val)
        val_auc = float(roc_auc_score(splits.y_val, val_proba)) if len(np.unique(splits.y_val)) > 1 else 0.5
        val_sens = float(recall_score(splits.y_val, val_preds, zero_division=0))

        logger.info(f"Final Model Validation: AUC = {val_auc:.4f} | Sensitivity = {val_sens:.4f}")
        return wrapper
