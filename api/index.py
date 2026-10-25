"""Vercel Function entry point: the stateless API (backend/app/stateless.py)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DATA_DIR", "/tmp/jewellery-data")  # the deployment filesystem is read-only except /tmp
sys.path.insert(0, str(ROOT / "backend"))

from app.stateless import app  # noqa: E402,F401
