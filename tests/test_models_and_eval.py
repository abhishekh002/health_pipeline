"""
Unit tests for model training, probability calibration, clinical evaluation metrics, and demographic fairness audits.
"""

import unittest
import numpy as np
from health_pipeline.models.classifier import PhysiologicalModelWrapper
from health_pipeline.models.train import ModelTrainer
from health_pipeline.evaluation.metrics import ClinicalEvaluator
from health_pipeline.evaluation.fairness import FairnessAuditor
from health_pipeline.evaluation.reporter import ClinicalEvaluationReporter


class TestModelsAndEvaluation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic dataset with imbalanced classes (80% Normal, 20% Pathology)
        self.n_samples = 150
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Construct synthetic pathology correlated with feature 0 and 1
        scores = self.X[:, 0] * 1.5 + self.X[:, 1] * 1.2 + np.random.randn(self.n_samples) * 0.5
        self.y = (scores > 1.0).astype(int)

    def test_model_wrapper_calibration_and_triage(self):
        wrapper = PhysiologicalModelWrapper(calibrate=True)
        wrapper.fit(self.X, self.y)

        # Predict single sample
        sample_vec = self.X[0]
        pred_out = wrapper.infer_single_sample(sample_vec)

        self.assertIn(pred_out.predicted_class, [0, 1])
        self.assertIn(pred_out.risk_tier, ["LOW_RISK", "MONITOR", "CRITICAL"])
        self.assertTrue(0.0 <= pred_out.confidence_score <= 1.0)
        self.assertTrue(0.0 <= pred_out.pathology_probability <= 1.0)

    def test_clinical_evaluator_metrics(self):
        # Known ground truth and predictions
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])  # 1 FP, 1 FN
        y_prob = np.array([0.1, 0.2, 0.1, 0.8, 0.3, 0.9, 0.85, 0.95])

        metrics = ClinicalEvaluator.evaluate(y_true, y_pred, y_prob)

        # TP = 3, FN = 1 -> Sensitivity = 3/4 = 0.75
        self.assertAlmostEqual(metrics.sensitivity, 0.75, places=2)
        # TN = 3, FP = 1 -> Specificity = 3/4 = 0.75
        self.assertAlmostEqual(metrics.specificity, 0.75, places=2)
        self.assertTrue(metrics.auc_roc > 0.8)
        self.assertTrue(metrics.brier_score < 0.2)

    def test_demographic_fairness_audit(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        subgroups = ["Group_A", "Group_A", "Group_A", "Group_A", "Group_B", "Group_B", "Group_B", "Group_B"]

        fairness = FairnessAuditor.audit_subgroup(y_true, y_pred, subgroups, attribute_name="test_group")

        self.assertEqual(fairness.attribute_name, "test_group")
        self.assertIn("Group_A", fairness.subgroup_metrics)
        self.assertIn("Group_B", fairness.subgroup_metrics)
        self.assertTrue(0.0 <= fairness.disparate_impact <= 1.0)
        self.assertTrue(0.0 <= fairness.tpr_disparity <= 1.0)


if __name__ == "__main__":
    unittest.main()
