"""
health_pipeline.scripts.run_api
Production launcher and test client for the secure physiological inference API service.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
from health_pipeline.api.app import create_app
from health_pipeline.config import PATHS, SECURITY_CONFIG

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    app = create_app()
    print(f"Starting Secure Health Pipeline API on port {port}...")
    print(f"  -> Dashboard: http://localhost:{port}/ or http://127.0.0.1:{port}/")
    print(f"  -> Health:    http://localhost:{port}/health")
    print(f"  -> Report:    http://localhost:{port}/report")
    print(f"Auth Token: {SECURITY_CONFIG.API_AUTH_TOKEN}")
    app.run(host=host, port=port, debug=False)
