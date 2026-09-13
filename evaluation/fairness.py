"""
health_pipeline.evaluation.fairness
Algorithmic fairness and demographic bias auditing across patient subgroups (Sex, Age cohorts).
Evaluates Demographic Parity, Disparate Impact, and Equalized Odds (TPR and FPR parity).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SubgroupFairnessResult:
    attribute_name: str
    subgroup_metrics: Dict[str, Dict[str, float]]
    demographic_parity_ratio: float
    disparate_impact: float
    tpr_disparity: float  # Equalized Odds: |TPR_group1 - TPR_group2|
    fpr_disparity: float  # |FPR_group1 - FPR_group2|
    is_fair_80_percent_rule: bool


class FairnessAuditor:
    """
    Audits the ML pipeline for fairness disparities across demographic covariates
    to ensure the monitoring system performs equitably across all patient groups.
    """

    @staticmethod
    def audit_subgroup(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subgroups: List[Any],
        attribute_name: str = "sex"
    ) -> SubgroupFairnessResult:
        """
        Audits performance disparities across demographic subgroups.
        Computes selection rate, True Positive Rate (Sensitivity), and False Positive Rate for each group.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        subgroups = np.asarray(subgroups)

        unique_groups = np.unique(subgroups)
        group_stats: Dict[str, Dict[str, float]] = {}

        selection_rates = []
        tprs = []
        fprs = []

        for grp in unique_groups:
            mask = (subgroups == grp)
            n_grp = int(np.sum(mask))
            if n_grp == 0:
                continue

            grp_y_true = y_true[mask]
            grp_y_pred = y_pred[mask]

            selection_rate = float(np.mean(grp_y_pred))
            selection_rates.append(selection_rate)

            # TPR (Sensitivity) = TP / (TP + FN)
            pos_mask = (grp_y_true == 1)
            tpr = float(np.mean(grp_y_pred[pos_mask])) if np.sum(pos_mask) > 0 else 0.0
            tprs.append(tpr)

            # FPR = FP / (FP + TN)
            neg_mask = (grp_y_true == 0)
            fpr = float(np.mean(grp_y_pred[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
            fprs.append(fpr)

            # Specificity
            spec = 1.0 - fpr

            group_stats[str(grp)] = {
                "sample_count": n_grp,
                "selection_rate": round(selection_rate, 4),
                "true_positive_rate": round(tpr, 4),
                "false_positive_rate": round(fpr, 4),
                "specificity": round(spec, 4)
            }

        # Demographic Parity / Disparate Impact (min selection rate / max selection rate)
        min_sr = min(selection_rates) if selection_rates else 0.0
        max_sr = max(selection_rates) if selection_rates else 1.0
        disparate_impact = float(min_sr / max_sr) if max_sr > 1e-6 else 1.0

        # Equalized Odds disparities
        tpr_disparity = float(max(tprs) - min(tprs)) if len(tprs) > 1 else 0.0
        fpr_disparity = float(max(fprs) - min(fprs)) if len(fprs) > 1 else 0.0

        # 4/5ths (80%) rule: Disparate Impact >= 0.80 and TPR disparity < 0.15
        is_fair = (disparate_impact >= 0.80) and (tpr_disparity <= 0.15)

        return SubgroupFairnessResult(
            attribute_name=attribute_name,
            subgroup_metrics=group_stats,
            demographic_parity_ratio=round(disparate_impact, 4),
            disparate_impact=round(disparate_impact, 4),
            tpr_disparity=round(tpr_disparity, 4),
            fpr_disparity=round(fpr_disparity, 4),
            is_fair_80_percent_rule=is_fair
        )
