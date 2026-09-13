"""
health_pipeline.evaluation.metrics
Clinical performance evaluation metrics accounting for clinical risk asymmetry:
- Sensitivity (Recall) & Specificity
- Positive Predictive Value (PPV) & Negative Predictive Value (NPV)
- AUC-ROC and PR-AUC
- Brier Score (Calibration error)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    brier_score_loss
)


@dataclass
class ClinicalEvaluationMetrics:
    sensitivity: float           # Recall / True Positive Rate
    specificity: float           # True Negative Rate
    ppv: float                   # Positive Predictive Value / Precision
    npv: float                   # Negative Predictive Value
    f1_score: float              # Harmonic mean of Precision and Sensitivity
    auc_roc: float               # Area Under ROC Curve
    pr_auc: float                # Area Under Precision-Recall Curve
    brier_score: float           # Calibration mean squared error
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    prevalence: float            # Proportion of positive pathological cases in test set

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "positive_predictive_value": round(self.ppv, 4),
            "negative_predictive_value": round(self.npv, 4),
            "f1_score": round(self.f1_score, 4),
            "auc_roc": round(self.auc_roc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier_score": round(self.brier_score, 4),
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "TN": self.true_negatives,
                "FN": self.false_negatives
            },
            "total_samples": self.total_samples,
            "disease_prevalence": round(self.prevalence, 4)
        }


class ClinicalEvaluator:
    """
    Computes rigorous clinical diagnostic metrics.
    Emphasizes sensitivity and PR-AUC because failing to detect an acute cardiac event
    involves higher clinical risk than a false alarm.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> ClinicalEvaluationMetrics:
        """
        Calculates all clinical diagnostic metrics from ground truth and predictions.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            auc_roc = float(roc_auc_score(y_true, y_prob))
            precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall_pts, precision_pts))
        else:
            auc_roc = 0.5
            pr_auc = 0.0

        brier = float(brier_score_loss(y_true, y_prob))
        prevalence = float(np.mean(y_true))

        return ClinicalEvaluationMetrics(
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1_score=f1,
            auc_roc=auc_roc,
            pr_auc=pr_auc,
            brier_score=brier,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            total_samples=len(y_true),
            prevalence=prevalence
        )
