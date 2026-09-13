"""
health_pipeline.preprocessing.splitter
Patient-isolated, stratified train/validation/test splitting and GroupKFold cross-validation.
Guarantees ZERO patient-level data leakage across splits.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from health_pipeline.config import MODEL_CONFIG
from health_pipeline.security.secure_logger import get_secure_logger

logger = get_secure_logger("health_pipeline.preprocessing.splitter")


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    patients_train: List[str]
    meta_train: List[Dict[str, Any]]

    X_val: np.ndarray
    y_val: np.ndarray
    patients_val: List[str]
    meta_val: List[Dict[str, Any]]

    X_test: np.ndarray
    y_test: np.ndarray
    patients_test: List[str]
    meta_test: List[Dict[str, Any]]

    feature_names: List[str]


class PatientLevelDataSplitter:
    """
    Partitions windowed physiological datasets into train, validation, and test subsets
    while strictly grouping by patient pseudonym to prevent data leakage.
    """

    def __init__(
        self,
        test_size: float = MODEL_CONFIG.TEST_SIZE,
        val_size: float = MODEL_CONFIG.VAL_SIZE,
        n_splits: int = MODEL_CONFIG.N_SPLITS,
        random_seed: int = MODEL_CONFIG.RANDOM_SEED
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.random_seed = random_seed

    def verify_zero_leakage(
        self, 
        patients_train: List[str], 
        patients_val: List[str], 
        patients_test: List[str]
    ) -> None:
        """
        Formally verifies that sets of patient pseudonyms across splits have empty intersections.
        Raises AssertionError if patient leakage is detected.
        """
        set_train = set(patients_train)
        set_val = set(patients_val)
        set_test = set(patients_test)

        train_val_overlap = set_train.intersection(set_val)
        train_test_overlap = set_train.intersection(set_test)
        val_test_overlap = set_val.intersection(set_test)

        if train_val_overlap:
            raise AssertionError(f"CRITICAL PRIVACY & LEAKAGE BREACH: Train and Val share patients: {train_val_overlap}")
        if train_test_overlap:
            raise AssertionError(f"CRITICAL PRIVACY & LEAKAGE BREACH: Train and Test share patients: {train_test_overlap}")
        if val_test_overlap:
            raise AssertionError(f"CRITICAL PRIVACY & LEAKAGE BREACH: Val and Test share patients: {val_test_overlap}")

        logger.info(
            f"Zero-leakage verified: Train ({len(set_train)} patients), "
            f"Val ({len(set_val)} patients), Test ({len(set_test)} patients) - No overlap."
        )

    def split_patient_records(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        patient_pseudonyms: List[str],
        metadata_list: List[Dict[str, Any]],
        feature_names: List[str]
    ) -> DatasetSplits:
        """
        Splits dataset into stratified, patient-grouped Train, Validation, and Test sets.
        Uses StratifiedGroupKFold on patient IDs so condition proportions remain balanced.
        """
        patient_arr = np.array(patient_pseudonyms)
        labels_arr = np.array(labels)
        unique_patients = np.unique(patient_arr)

        if len(unique_patients) < 3:
            # Fallback for minimal testing sets: ensure disjoint assignment
            n_pts = len(unique_patients)
            train_pts = unique_patients[: max(1, n_pts - 2)]
            val_pts = unique_patients[max(1, n_pts - 2) : max(2, n_pts - 1)]
            test_pts = unique_patients[max(2, n_pts - 1) :]
        else:
            # First split out Test set (approx 20% of patients)
            sgkf_test = StratifiedGroupKFold(n_splits=int(1.0 / self.test_size), shuffle=True, random_state=self.random_seed)
            train_val_idx, test_idx = next(sgkf_test.split(feature_matrix, labels_arr, groups=patient_arr))

            X_tv, y_tv = feature_matrix[train_val_idx], labels_arr[train_val_idx]
            pts_tv = patient_arr[train_val_idx]
            meta_tv = [metadata_list[i] for i in train_val_idx]

            X_test, y_test = feature_matrix[test_idx], labels_arr[test_idx]
            pts_test = patient_arr[test_idx].tolist()
            meta_test = [metadata_list[i] for i in test_idx]

            # Next split Train and Val from Train+Val pool (approx 15% of total for Val)
            val_ratio_in_tv = self.val_size / (1.0 - self.test_size)
            n_splits_tv = max(2, int(round(1.0 / val_ratio_in_tv)))
            sgkf_val = StratifiedGroupKFold(n_splits=n_splits_tv, shuffle=True, random_state=self.random_seed)
            train_sub_idx, val_sub_idx = next(sgkf_val.split(X_tv, y_tv, groups=pts_tv))

            X_train, y_train = X_tv[train_sub_idx], y_tv[train_sub_idx]
            pts_train = pts_tv[train_sub_idx].tolist()
            meta_train = [meta_tv[i] for i in train_sub_idx]

            X_val, y_val = X_tv[val_sub_idx], y_tv[val_sub_idx]
            pts_val = pts_tv[val_sub_idx].tolist()
            meta_val = [meta_tv[i] for i in val_sub_idx]

        # Verify zero leakage
        self.verify_zero_leakage(pts_train, pts_val, pts_test)

        return DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            patients_train=pts_train,
            meta_train=meta_train,
            X_val=X_val,
            y_val=y_val,
            patients_val=pts_val,
            meta_val=meta_val,
            X_test=X_test,
            y_test=y_test,
            patients_test=pts_test,
            meta_test=meta_test,
            feature_names=feature_names
        )

    def get_group_kfold_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_pseudonyms: List[str]
    ):
        """
        Returns a StratifiedGroupKFold generator for patient-isolated cross-validation.
        """
        patient_arr = np.array(patient_pseudonyms)
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.random_seed
        )
        return sgkf.split(X, y, groups=patient_arr)
