"""FastAPI app. Long work is done by the separate worker process (app/worker.py)."""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .db import db, dumps, init_db, new_id, now, row_to_dict
from .pipeline import normalise_upload
from .tryon import DEFAULT_SETTINGS

app = FastAPI(title="Jewellery try-on API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5188", "http://127.0.0.1:5188"], allow_methods=["*"], allow_headers=["*"])
init_db()

# local file serving (prototype; put behind auth / signed URLs before deploying)
app.mount("/media", StaticFiles(directory=config.DATA_DIR), name="media")
app.mount("/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")


def media_url(path: str | None) -> str | None:
    if not path:
        return None
    rel = Path(path).resolve().relative_to(config.DATA_DIR.resolve())
    return f"/media/{rel.as_posix()}"


# ----------------------------------------------------------------------------- products


def product_out(p: dict) -> dict:
    meta = p.get("meta") or {}
    return {
        "id": p["id"],
        "name": p["name"],
        "category": p["category"],
        "supported": meta.get("supported", False),
        "cutout_url": media_url(p["cutout_path"]),
        "thumb_url": f"/media/{meta['thumb']}" if meta.get("thumb") else media_url(p["cutout_path"]),
        "original_url": media_url(p["original_path"]),
        "mask_url": media_url(p["mask_path"]),
        "dimensions": p["dimensions"],
        "anchors": p["anchors"],
        "reference_views": p["reference_views"],
        "credit": meta.get("credit"),
        "source_url": meta.get("url"),
    }


@app.get("/products")
def list_products():
    with db() as c:
        rows = [row_to_dict(r) for r in c.execute("SELECT * FROM products ORDER BY category, id")]
    return {
        "products": [product_out(p) for p in rows],
        "categories": [
            {"id": "earring", "label": "Earrings", "supported": True},
            {"id": "pendant", "label": "Pendants", "supported": True},
            {"id": "ring", "label": "Rings", "supported": True, "note": "Back-of-hand photos; the ring is placed on the ring finger"},
            {"id": "bracelet", "label": "Bracelets", "supported": False, "note": "Needs extra views or 3D: not yet supported"},
        ],
    }


@app.get("/samples-list")
def list_samples():
    files = sorted(p.name for p in config.SAMPLES_DIR.glob("*.jpg"))
    return {"samples": [f"/samples/{f}" for f in files]}


# ----------------------------------------------------------------------------- uploads


@app.post("/uploads")
async def create_upload(file: UploadFile = File(...)):
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is larger than 25 MB.")
    import io

    try:
        im = normalise_upload(io.BytesIO(data))
    except ValueError as e:
        raise HTTPException(400, str(e))
    uid = new_id("up")
    path = config.UPLOAD_ORIGINALS_DIR / f"{uid}.png"
    im.save(path, compress_level=3)
    with db() as c:
        c.execute(
            "INSERT INTO uploads (id, original_path, width, height, created_at) VALUES (?,?,?,?,?)",
            (uid, str(path), im.width, im.height, now()),
        )
    return {"id": uid, "width": im.width, "height": im.height, "url": media_url(str(path))}


@app.delete("/uploads/{upload_id}")
def delete_upload(upload_id: str):
    with db() as c:
        up = row_to_dict(c.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone())
        if not up:
            raise HTTPException(404, "upload not found")
        jobs = [r["id"] for r in c.execute("SELECT id FROM tryon_jobs WHERE upload_id=?", (upload_id,))]
        for jid in jobs:
            shutil.rmtree(config.OUTPUTS_DIR / jid, ignore_errors=True)
        c.execute("DELETE FROM tryon_jobs WHERE upload_id=?", (upload_id,))
        c.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    Path(up["original_path"]).unlink(missing_ok=True)
    return {"deleted": True, "jobs_deleted": len(jobs)}


# ----------------------------------------------------------------------------- try-ons


class PlacementItem(BaseModel):
    cx: float
    cy: float
    width: float = Field(gt=0)
    rotation: float = 0.0
    flip: bool = False
    occlude: dict | None = None


class Chain(BaseModel):
    points: list[list[float]] = Field(min_length=2, max_length=400)
    thickness: float = Field(default=2.0, gt=0, le=40)


class Placement(BaseModel):
    items: list[PlacementItem] = Field(min_length=1, max_length=2)
    chain: Chain | None = None
    whole_photo: bool = False  # no body landmarks found: the AI gets the whole photo and finds the spot itself


class AutoPlaceRequest(BaseModel):
    upload_id: str
    product_id: str


@app.post("/autoplace")
def auto_place(req: AutoPlaceRequest):
    """Detect ears / neck / ring finger and return a ready placement (runs locally, ~1 s)."""
    from .autoplace import NotFound, autoplace

    with db() as c:
        up = row_to_dict(c.execute("SELECT * FROM uploads WHERE id=?", (req.upload_id,)).fetchone())
        prod = row_to_dict(c.execute("SELECT * FROM products WHERE id=?", (req.product_id,)).fetchone())
    if not up or not prod:
        raise HTTPException(404, "upload or product not found")
    try:
        return autoplace(up["original_path"], prod)
    except NotFound as e:
        raise HTTPException(422, str(e))


class TryOnRequest(BaseModel):
    upload_id: str
    product_id: str
    placement: Placement
    provider: str | None = None  # qwen-fast | qwen-official | none
    seed: int | None = None
    settings: dict = Field(default_factory=dict)


def job_out(j: dict) -> dict:
    out_dir = config.OUTPUTS_DIR / j["id"]
    assets = {}
    if j["status"] == "complete":
        ref = next(out_dir.glob("reference.*"), None)
        assets = {
            "tryon": media_url(str(out_dir / "tryon.png")),
            "preview": media_url(str(out_dir / "preview.jpg")) if (out_dir / "preview.jpg").exists() else None,
            "detail": media_url(str(out_dir / "detail.png")),
            "reference": media_url(str(ref)) if ref else None,
            # The canonical source-based master and an optional AI draft are separate assets.
            "ai": media_url(str(out_dir / "tryon_ai.png")) if (out_dir / "tryon_ai.png").exists() else None,
        }
    tiles = []
    info = (j.get("timings") or {}).get("info") or {}
    for t in info.get("tiles", []):
        tiles.append({**t, "url": media_url(str(out_dir / t["file"]))})
    debug = {}
    for name in ("preplaced.png", "ai_context.png"):
        if (out_dir / name).exists():
            debug[name.split(".")[0]] = media_url(str(out_dir / name))
    return {
        "id": j["id"],
        "upload_id": j["upload_id"],
        "product_id": j["product_id"],
        "status": j["status"],
        "placement": j["placement"],
        "provider": j["provider"],
        "seed": j["seed"],
        "settings": j["settings"],
        "error": j["error"],
        "attempts": j["attempts"],
        "timings": j["timings"],
        "created_at": j["created_at"],
        "updated_at": j["updated_at"],
        "assets": assets,
        "tiles": tiles if j["status"] == "complete" else [],
        "debug": debug,
    }


@app.post("/tryons", status_code=201)
def create_tryon(req: TryOnRequest):
    provider = req.provider or config.DEFAULT_PROVIDER
    if provider not in config.PROVIDERS:
        raise HTTPException(400, "unknown provider")
    with db() as c:
        up = c.execute("SELECT id FROM uploads WHERE id=?", (req.upload_id,)).fetchone()
        prod = row_to_dict(c.execute("SELECT * FROM products WHERE id=?", (req.product_id,)).fetchone())
        if not up:
            raise HTTPException(404, "upload not found")
        if not prod:
            raise HTTPException(404, "product not found")
        if not (prod["meta"] or {}).get("supported"):
            raise HTTPException(400, f"{prod['category']} try-on is not supported yet")
        jid = new_id("job")
        seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        settings = {k: v for k, v in req.settings.items() if k in DEFAULT_SETTINGS or k == "context_mode"}
        target = settings.get("output_long_side", 6144)
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not 1024 <= target <= 8192:
            raise HTTPException(422, "Output size must be between 1024 and 8192 pixels.")
        c.execute(
            "INSERT INTO tryon_jobs (id, upload_id, product_id, status, placement, provider, seed, settings, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (jid, req.upload_id, req.product_id, "queued", dumps(req.placement.model_dump(exclude_none=True)), provider, seed, dumps(settings), now(), now()),
        )
        j = row_to_dict(c.execute("SELECT * FROM tryon_jobs WHERE id=?", (jid,)).fetchone())
    return job_out(j)


@app.get("/tryons/{job_id}")
def get_tryon(job_id: str):
    with db() as c:
        j = row_to_dict(c.execute("SELECT * FROM tryon_jobs WHERE id=?", (job_id,)).fetchone())
        if not j:
            raise HTTPException(404, "job not found")
        out = job_out(j)
        if j["status"] == "queued":
            out["queue_position"] = c.execute(
                "SELECT COUNT(*) FROM tryon_jobs WHERE status='queued' AND created_at < ?", (j["created_at"],)
            ).fetchone()[0]
    return out


@app.delete("/tryons/{job_id}")
def delete_tryon(job_id: str, keep_upload: bool = False):
    """Deletes the job, its output files and (unless other jobs still use it) the uploaded photo."""
    with db() as c:
        j = row_to_dict(c.execute("SELECT * FROM tryon_jobs WHERE id=?", (job_id,)).fetchone())
        if not j:
            raise HTTPException(404, "job not found")
        c.execute("DELETE FROM tryon_jobs WHERE id=?", (job_id,))
        others = c.execute("SELECT COUNT(*) FROM tryon_jobs WHERE upload_id=?", (j["upload_id"],)).fetchone()[0]
        upload_deleted = False
        if not keep_upload and others == 0:
            up = c.execute("SELECT original_path FROM uploads WHERE id=?", (j["upload_id"],)).fetchone()
            if up:
                Path(up["original_path"]).unlink(missing_ok=True)
                c.execute("DELETE FROM uploads WHERE id=?", (j["upload_id"],))
                upload_deleted = True
    shutil.rmtree(config.OUTPUTS_DIR / job_id, ignore_errors=True)
    return {"deleted": True, "upload_deleted": upload_deleted}


@app.get("/health")
def health():
    with db() as c:
        counts = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM tryon_jobs GROUP BY status")}
    return {
        "ok": True,
        "provider": config.DEFAULT_PROVIDER,
        "hf_token": bool(config.HF_TOKEN),
        "fal_key": bool(config.FAL_KEY),
        "jobs": counts,
        "pid": os.getpid(),
    }
