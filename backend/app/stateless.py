"""Stateless API for serverless hosting (Vercel Functions).

No disk, no database, no background worker: the browser sends the (resized) photo with each request, body
landmarks come from MediaPipe running in the browser, the try-on runs inside the request, and results come back
in the response as data URLs. Nothing about the user is stored on the server.

Routes (under /api): GET products, GET health, POST autoplace, POST tryon.
If APP_PASSWORD is set, POST routes need the header X-App-Password.
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from . import config
from .autoplace import NotFound, autoplace
from .pipeline import normalise_upload
from .tryon import run_tryon

PREPARED = config.CATALOG_DIR / "prepared"
MAX_SIDE = 2000          # photos are resized in the browser too; this is the server-side cap
OUTPUT_LONG_SIDE = 2560  # result size: fits the 4.5 MB response limit as JPEG, and the 2 GB memory limit

app = FastAPI(title="Jewellery try-on (serverless)")


def top_attach_point(alpha: np.ndarray):
    """Topmost solid pixel (hook / bail), normalised. Same as catalog.top_attach_point, without its heavy imports."""
    ys, xs = np.nonzero(alpha > 128)
    y = ys.min()
    row = xs[ys <= y + max(2, alpha.shape[0] // 200)]
    return float(row.mean() / alpha.shape[1]), float(y / alpha.shape[0])


def _products() -> dict[str, dict]:
    manifest = json.loads((config.CATALOG_DIR / "catalog.json").read_text())
    out = {}
    for p in manifest["products"]:
        cut_path = PREPARED / "cutouts" / f"{p['id']}.png"
        if not cut_path.exists():
            continue
        cut = Image.open(cut_path)
        ax, ay = top_attach_point(np.asarray(cut.getchannel("A")))
        anchors = {"attach": [ax, ay], "center": [0.5, 0.5], **({"ring": p["ring"]} if p.get("ring") else {})}
        out[p["id"]] = {
            "id": p["id"], "name": p["name"], "category": p["category"],
            "cutout_path": str(cut_path),
            "original_path": str(PREPARED / "products" / f"{p['id']}.jpg"),
            "dimensions": {"cutout_w": cut.width, "cutout_h": cut.height, "approx_width_mm": p.get("approx_width_mm")},
            "anchors": anchors, "credit": p.get("credit"), "url": p.get("url"),
        }
    return out


PRODUCTS = _products()


def _public(p: dict) -> dict:
    return {
        "id": p["id"], "name": p["name"], "category": p["category"], "supported": True,
        "cutout_url": f"/catalog/cutouts/{p['id']}.png", "thumb_url": f"/catalog/cutouts/{p['id']}_thumb.png",
        "original_url": f"/catalog/products/{p['id']}.jpg", "dimensions": p["dimensions"], "anchors": p["anchors"],
        "reference_views": [], "credit": p["credit"], "source_url": p["url"],
    }


def _check_password(given: str | None):
    pw = os.getenv("APP_PASSWORD", "")
    if pw and not hmac.compare_digest(given or "", pw):
        raise HTTPException(401, "Password required")


def _decode_image(data_url: str) -> Image.Image:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        im = normalise_upload(io.BytesIO(raw))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "Not a readable image.")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


def _jpeg_data_url(path: Path, max_side: int, quality: int = 90) -> str | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/products")
def products():
    return {
        "products": [_public(p) for p in PRODUCTS.values()],
        "categories": [
            {"id": "earring", "label": "Earrings", "supported": True},
            {"id": "pendant", "label": "Pendants", "supported": True},
            {"id": "ring", "label": "Rings", "supported": True},
            {"id": "bracelet", "label": "Bracelets", "supported": False, "note": "Not supported yet"},
        ],
    }


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "stateless", "provider": config.DEFAULT_PROVIDER, "fal_key": bool(config.FAL_KEY),
            "hf_token": bool(config.HF_TOKEN), "password": bool(os.getenv("APP_PASSWORD"))}


class AutoPlaceBody(BaseModel):
    image: str
    product_id: str
    landmarks: dict | None = None


@app.post("/api/autoplace")
def auto_place(body: AutoPlaceBody, x_app_password: str | None = Header(default=None)):
    _check_password(x_app_password)
    prod = PRODUCTS.get(body.product_id)
    if not prod:
        raise HTTPException(404, "product not found")
    im = _decode_image(body.image)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "photo.png"
        im.save(path)
        try:
            return autoplace(path, prod, landmarks=body.landmarks or {})
        except NotFound as e:
            raise HTTPException(422, str(e))


class TryOnBody(BaseModel):
    image: str
    product_id: str
    placement: dict
    provider: str | None = None
    seed: int | None = None
    settings: dict = Field(default_factory=dict)


@app.post("/api/tryon")
def try_on(body: TryOnBody, x_app_password: str | None = Header(default=None)):
    _check_password(x_app_password)
    prod = PRODUCTS.get(body.product_id)
    if not prod:
        raise HTTPException(404, "product not found")
    provider = body.provider or config.DEFAULT_PROVIDER
    if provider not in config.PROVIDERS:
        raise HTTPException(400, "unknown provider")
    items = body.placement.get("items") or []
    if not 1 <= len(items) <= 2:
        raise HTTPException(400, "placement needs 1 or 2 items")
    im = _decode_image(body.image)
    job_id = "job_" + uuid.uuid4().hex[:12]
    seed = body.seed if body.seed is not None else random.randint(0, 2**31 - 1)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        photo = tmp / "photo.png"
        im.save(photo)
        job = {"id": job_id, "provider": provider, "seed": seed, "placement": body.placement,
               "settings": {"output_long_side": OUTPUT_LONG_SIDE,
                            "detail_pass": body.settings.get("detail_pass") is True}}
        try:
            r = run_tryon(job, prod, {"original_path": str(photo)}, tmp / "out", lambda s: None)
        except Exception as e:  # the same messages the worker would store on the job
            status = "quota_exceeded" if type(e).__name__ == "QuotaExceededError" else "failed"
            return {"id": job_id, "status": status, "error": str(e)[:600], "provider": provider, "seed": seed,
                    "placement": body.placement, "settings": {}, "attempts": 0, "timings": None,
                    "assets": {}, "tiles": [], "debug": {}}
        out = tmp / "out"
        tryon_url = _jpeg_data_url(out / "tryon.png", OUTPUT_LONG_SIDE, 90)
        timings = {**r["timings"], "total": round(time.time() - t0, 2), "info": r["info"]}
        return {
            "id": job_id, "status": "complete", "error": None, "provider": provider, "seed": r["seed"],
            "placement": body.placement, "settings": {}, "attempts": 1, "timings": timings,
            "assets": {"tryon": tryon_url, "detail": tryon_url, "preview": tryon_url,
                       "ai": _jpeg_data_url(out / "tryon_ai.png", 1600, 85),
                       "reference": f"/catalog/products/{prod['id']}.jpg"},
            "tiles": [], "debug": {},
        }
