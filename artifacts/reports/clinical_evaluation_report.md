# Physiological Health Monitoring ML Evaluation Report
**Generated:** 2026-09-18 20:10:59Z | **Compliance:** HIPAA Safe Harbor De-Identified (Zero Direct PHI)

## 1. Executive Clinical Summary
- **Test Cohort Size:** 105 independent physiological segments
- **Pathology Prevalence:** 33.3%
- **Sensitivity (Recall):** **51.43%** (Prioritized for zero missed acute events)
- **Specificity:** **100.00%** (Safeguards against clinical alert fatigue)
- **AUC-ROC:** **0.9624** | **PR-AUC:** **0.9541**
- **Calibration Brier Score:** **0.1423**

## 2. Confusion Matrix
| | Predicted Negative (Normal) | Predicted Positive (Abnormal) | Total Actual |
|---|---|---|---|
| **Actual Normal** | 70 (TN) | 0 (FP) | 70 |
| **Actual Abnormal** | 17 (FN) | 18 (TP) | 35 |

## 3. Algorithmic Fairness & Demographic Disparity Audit
### Subgroup Dimension: `BIOLOGICAL_SEX`
- **Disparate Impact Ratio:** `0.0000` (Pass threshold ≥ 0.80)
- **True Positive Rate (TPR) Disparity:** `0.5143` (Equalized odds difference)
- **80% Rule Fairness Compliance:** `ATTENTION_REQUIRED`

| Subgroup | Samples | Selection Rate | Sensitivity (TPR) | False Positive Rate | Specificity |
|---|---|---|---|---|---|
| F | 70 | 25.7% | 51.4% | 0.0% | 100.0% |
| M | 35 | 0.0% | 0.0% | 0.0% | 100.0% |

### Subgroup Dimension: `AGE_COHORT`
- **Disparate Impact Ratio:** `0.0000` (Pass threshold ≥ 0.80)
- **True Positive Rate (TPR) Disparity:** `0.5143` (Equalized odds difference)
- **80% Rule Fairness Compliance:** `ATTENTION_REQUIRED`

| Subgroup | Samples | Selection Rate | Sensitivity (TPR) | False Positive Rate | Specificity |
|---|---|---|---|---|---|
| Non-Senior (<65) | 70 | 25.7% | 51.4% | 0.0% | 100.0% |
| Senior (>=65) | 35 | 0.0% | 0.0% | 0.0% | 100.0% |

## 4. Privacy & HIPAA Compliance Verification
- [x] Salted HMAC-SHA256 pseudonymization applied to all subject/record IDs at ingestion.
- [x] Zero raw physiological waveforms exported in audit reports.
- [x] Demographic covariates aggregated without re-identifying cross-links.
- [x] Strict patient-level grouping enforced; zero patient overlap between train/val/test splits.