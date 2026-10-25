"""Paths and settings. Secrets (HF_TOKEN) come from backend/.env and never leave the backend."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ORIGINALS_DIR = DATA_DIR / "originals"
UPLOAD_ORIGINALS_DIR = ORIGINALS_DIR / "uploads"
PRODUCT_ORIGINALS_DIR = ORIGINALS_DIR / "products"
CUTOUTS_DIR = DATA_DIR / "cutouts"
MASKS_DIR = DATA_DIR / "masks"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "app.db"
CATALOG_DIR = ROOT / "catalog"
SAMPLES_DIR = ROOT / "samples"

for d in (UPLOAD_ORIGINALS_DIR, PRODUCT_ORIGINALS_DIR, CUTOUTS_DIR, MASKS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN") or None
FAL_KEY = os.getenv("FAL_KEY") or None

# Which try-on provider is used by default:
#   "fal-nano-banana" | "fal-qwen-edit" (paid, needs FAL_KEY) | "qwen-fast" | "qwen-official" (free HF Spaces) | "none"
PROVIDERS = ("fal-nano-banana", "fal-nano-banana-2", "fal-qwen-edit", "qwen-fast", "qwen-official", "none")
DEFAULT_PROVIDER = os.getenv("TRYON_PROVIDER") or ("fal-nano-banana" if FAL_KEY else "qwen-fast")
QWEN_FAST_SPACE = os.getenv("QWEN_FAST_SPACE", "prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast")
QWEN_OFFICIAL_SPACE = os.getenv("QWEN_OFFICIAL_SPACE", "Qwen/Qwen-Image-Edit-2511")
BIREFNET_SPACE = os.getenv("BIREFNET_SPACE", "ZhengPeng7/BiRefNet_demo")

# Upload validation
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_SIDE = 400
MAX_SIDE = 6000  # larger photos are downscaled once at upload

# Worker
MAX_TRANSIENT_RETRIES = 3
POLL_INTERVAL_S = 1.0
