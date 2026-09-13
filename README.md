# Secure Physiological Health Pipeline

A Python machine-learning pipeline for physiological health monitoring. It supports ECG and pulse-oximeter processing, signal-quality assessment, patient-isolated model training, clinical evaluation, fairness auditing, privacy protection, and a Flask inference API with an ESP32 telemetry gateway.

## Features

- ECG cleaning, feature extraction, and signal-quality scoring
- Patient-level train, validation, and test splitting to reduce data leakage
- Class-imbalance handling and calibrated classification
- Clinical metrics including sensitivity, specificity, AUC-ROC, PR-AUC, and Brier score
- Demographic fairness analysis across age and sex groups
- Privacy-preserving pseudonymization and secure logging
- Web dashboard, REST inference API, and ESP32 telemetry endpoints
- HTML, Markdown, JSON, and PDF evaluation reports

## Requirements

- Python 3.10 or newer
- Optional MIT-BIH Arrhythmia Database at `D:/project/mit-bih-arrhythmia-database-1.0.0`

Install the runtime dependencies:

```powershell
python -m pip install Flask numpy scipy scikit-learn joblib
```

## Run the API Dashboard

From the repository root:

```powershell
python scripts/run_api.py
```

Open these URLs in a browser:

- Dashboard: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Clinical report: `http://127.0.0.1:8000/report`
- PDF report: `http://127.0.0.1:8000/report/pdf`

The API uses the default token from `config.py`. For deployments, set a unique token before starting the server:

```powershell
$env:HEALTH_API_KEY = "replace-with-a-secure-token"
python scripts/run_api.py
```

You can change the host and port with `API_HOST` and `API_PORT`:

```powershell
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
python scripts/run_api.py
```

## REST Inference

Authenticated prediction requests use the `Authorization` header:

```powershell
$headers = @{ Authorization = "Bearer $env:HEALTH_API_KEY" }
$body = @{
    signal = @(0.0, 0.1, 0.0)
    fs = 360.0
    spo2_signal = @(98.0, 98.0, 98.0)
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/predict `
    -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

The dashboard also provides signal presets and ESP32 live telemetry controls.

## Run the End-to-End Pipeline

```powershell
python scripts/run_pipeline.py
```

The pipeline writes models, logs, and reports under `artifacts/`.

## Simulate ESP32 Telemetry

Start the API first, then run:

```powershell
python scripts/simulate_esp32_stream.py
```

The telemetry endpoint is `POST /api/esp32/telemetry`, and the latest gateway state is available at `GET /api/esp32/latest`.

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Project Layout

```text
api/            Flask API, schemas, and dashboard
artifacts/      Generated models, logs, and evaluation reports
data/           Physiological data loading and validation
evaluation/     Clinical metrics, fairness, and report generation
models/         Classifier wrapper and training logic
preprocessing/  Signal cleaning, feature extraction, splitting, and SQI
scripts/        API, pipeline, and ESP32 simulator entry points
security/       Authentication, privacy, and secure logging
tests/          Unit and integration tests
```

## Configuration

Edit `config.py` or use environment variables for deployment-specific settings. Important values include:

- `HEALTH_API_KEY`: API authentication token
- `HEALTH_PIPELINE_SALT`: pseudonymization salt
- `API_HOST`: API bind address
- `API_PORT`: API port
- `PATHS.DATASET_DIR`: MIT-BIH dataset location

Do not use the development defaults for a production deployment. This project is for research and monitoring support; it is not a substitute for clinical diagnosis or emergency medical care.
