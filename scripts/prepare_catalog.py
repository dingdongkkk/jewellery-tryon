"""Prepare product cutouts once (BiRefNet on Hugging Face). Usage: python scripts/prepare_catalog.py [--force]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.catalog import prepare_all  # noqa: E402

if __name__ == "__main__":
    prepare_all(force="--force" in sys.argv)
