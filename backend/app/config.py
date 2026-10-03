"""Paths and limits. MODEL_DIR / FRONTEND_DIR can be overridden with environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", PROJECT_ROOT / "models"))
FRONTEND_DIR = Path(os.environ.get("CHURN_FRONTEND_DIR", PROJECT_ROOT / "frontend"))

MAX_CSV_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_CSV_ROWS = 5000
