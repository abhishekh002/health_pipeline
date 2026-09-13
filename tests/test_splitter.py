"""
Unit tests for zero-leakage patient-isolated data splitting.
"""

import unittest
import numpy as np
from health_pipeline.preprocessing.splitter import PatientLevelDataSplitter


class TestPatientSplitter(unittest.TestCase):

    def test_zero_leakage_guarantee(self):
        # 10 distinct patients, each having 20 signal segments (total 200 samples)
        n_patients = 10
        samples_per_patient = 20
        total_samples = n_patients * samples_per_patient

        feature_matrix = np.random.randn(total_samples, 8)
        labels = np.random.choice([0, 1], size=total_samples, p=[0.7, 0.3])
        
        patient_pseudonyms = []
        meta_list = []
        for i in range(n_patients):
            pid = f"PID-TEST-PATIENT-{i:03d}"
            for _ in range(samples_per_patient):
                patient_pseudonyms.append(pid)
                meta_list.append({"patient_pseudonym": pid, "sex": "M" if i % 2 == 0 else "F", "age": 50 + i})

        splitter = PatientLevelDataSplitter(test_size=0.20, val_size=0.20, random_seed=42)
        splits = splitter.split_patient_records(
            feature_matrix=feature_matrix,
            labels=labels,
            patient_pseudonyms=patient_pseudonyms,
            metadata_list=meta_list,
            feature_names=[f"feat_{j}" for j in range(8)]
        )

        train_pts = set(splits.patients_train)
        val_pts = set(splits.patients_val)
        test_pts = set(splits.patients_test)

        # 1. Total patients accounted for
        self.assertEqual(len(train_pts | val_pts | test_pts), n_patients)

        # 2. Complete disjointness: Zero patient leakage across splits
        self.assertEqual(len(train_pts.intersection(val_pts)), 0)
        self.assertEqual(len(train_pts.intersection(test_pts)), 0)
        self.assertEqual(len(val_pts.intersection(test_pts)), 0)

        # 3. verify_zero_leakage method should pass without assertion error
        splitter.verify_zero_leakage(splits.patients_train, splits.patients_val, splits.patients_test)


if __name__ == "__main__":
    unittest.main()
