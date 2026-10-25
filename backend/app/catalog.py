"""Catalogue preparation: BiRefNet background removal, run once per product and cached on disk.

For each product we keep three files separately:
  originals/products/<id>.jpg  the (cropped) original product photo, full resolution
  masks/<id>.png               the BiRefNet alpha mask at the same resolution
  cutouts/<id>.png             RGBA cutout = original pixels + mask, tight-cropped (max 2048 px)
The cutout's RGB comes from the original photo, never from the model output.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image, ImageOps

from .config import BIREFNET_SPACE, CATALOG_DIR, CUTOUTS_DIR, HF_TOKEN, MASKS_DIR, PRODUCT_ORIGINALS_DIR
from .db import db, dumps, init_db

CUTOUT_MAX_SIDE = 2048
BIREFNET_INPUT_SIDE = 2048
# "local" runs BiRefNet-HR on this machine (default, no quota); "space" uses the HF Space
BIREFNET_BACKEND = os.getenv("BIREFNET_BACKEND", "local")
LOCAL_BIREFNET_REPO = "ZhengPeng7/BiRefNet"  # 1024px general model; HR at 2048 swaps on a 16 GB Mac
LOCAL_INPUT_SIDE = 1024


def birefnet_mask(img: Image.Image, client: Client) -> Image.Image:
    """Returns an L-mode mask the size of img."""
    small = img.copy()
    small.thumbnail((BIREFNET_INPUT_SIDE, BIREFNET_INPUT_SIDE), Image.LANCZOS)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        small.save(f.name)
        res = client.predict(handle_file(f.name), "2048x2048", "General-HR", api_name="/image")
    out = res[1]
    path = out["path"] if isinstance(out, dict) else out
    rgba = Image.open(path)
    mask = rgba.getchannel("A") if rgba.mode == "RGBA" else rgba.convert("L")
    return mask.resize(img.size, Image.BICUBIC)


_local_model = None


def birefnet_mask_local(img: Image.Image) -> Image.Image:
    """BiRefNet weights (MIT) run on this Mac (MPS or CPU): no Hugging Face GPU quota used."""
    global _local_model
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if _local_model is None:
        _local_model = AutoModelForImageSegmentation.from_pretrained(LOCAL_BIREFNET_REPO, trust_remote_code=True)
        _local_model.to(device).float().eval()
    tf = transforms.Compose([
        transforms.Resize((LOCAL_INPUT_SIDE, LOCAL_INPUT_SIDE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = _local_model(x)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(pred)
    return mask.resize(img.size, Image.BICUBIC)


def clean_mask(mask: Image.Image) -> Image.Image:
    a = np.asarray(mask, dtype=np.float32) / 255.0
    a[a < 0.04] = 0  # drop faint background haze
    a[a > 0.97] = 1
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def top_attach_point(alpha: np.ndarray):
    """Topmost solid pixel: where an earring hook / pendant bail attaches. Normalised (x, y)."""
    ys, xs = np.nonzero(alpha > 128)
    y = ys.min()
    row = xs[ys <= y + max(2, alpha.shape[0] // 200)]
    return float(row.mean() / alpha.shape[1]), float(y / alpha.shape[0])


def prepare_product(p: dict, client: Client | None, force=False) -> dict:
    pid = p["id"]
    orig_path = PRODUCT_ORIGINALS_DIR / f"{pid}.jpg"
    mask_path = MASKS_DIR / f"{pid}.png"
    cut_path = CUTOUTS_DIR / f"{pid}.png"
    thumb_path = CUTOUTS_DIR / f"{pid}_thumb.png"
    timing = None
    if force or not (orig_path.exists() and mask_path.exists() and cut_path.exists()):
        src = ImageOps.exif_transpose(Image.open(CATALOG_DIR / "sources" / p["source"])).convert("RGB")
        if p.get("crop"):
            x0, y0, x1, y1 = p["crop"]
            W, H = src.size
            src = src.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        src.save(orig_path, quality=95, subsampling=0)
        t = time.time()
        raw = birefnet_mask(src, client) if client is not None else birefnet_mask_local(src)
        mask = clean_mask(raw)
        timing = round(time.time() - t, 1)
        mask.save(mask_path)
        rgba = src.copy()
        rgba.putalpha(mask)
        bbox = mask.point(lambda v: 255 if v > 10 else 0).getbbox()
        pad = int(0.02 * max(src.size))
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(src.width, bbox[2] + pad), min(src.height, bbox[3] + pad))
        cut = rgba.crop(bbox)
        if max(cut.size) > CUTOUT_MAX_SIDE:
            cut.thumbnail((CUTOUT_MAX_SIDE, CUTOUT_MAX_SIDE), Image.LANCZOS)
        cut.save(cut_path)
        th = cut.copy()
        th.thumbnail((360, 360), Image.LANCZOS)
        th.save(thumb_path)
    cut = Image.open(cut_path)
    alpha = np.asarray(cut.getchannel("A"))
    ax, ay = top_attach_point(alpha)
    supported = p["category"] in ("earring", "pendant", "ring")
    row = {
        "id": pid,
        "name": p["name"],
        "category": p["category"],
        "original_path": str(orig_path),
        "cutout_path": str(cut_path),
        "mask_path": str(mask_path),
        "dimensions": dumps({
            "cutout_w": cut.width,
            "cutout_h": cut.height,
            "approx_width_mm": p.get("approx_width_mm"),
        }),
        "reference_views": dumps(p.get("reference_views", [])),
        "anchors": dumps({"attach": [ax, ay], "center": [0.5, 0.5], **({"ring": p["ring"]} if p.get("ring") else {})}),
        "meta": dumps({
            "credit": p.get("credit"),
            "url": p.get("url"),
            "supported": supported,
            "thumb": f"cutouts/{pid}_thumb.png",
            "birefnet_seconds": timing,
            "birefnet_backend": BIREFNET_BACKEND if timing is not None else None,
        }),
    }
    with db() as c:
        c.execute(
            "INSERT OR REPLACE INTO products (id,name,category,original_path,cutout_path,mask_path,dimensions,"
            "reference_views,anchors,meta) VALUES (:id,:name,:category,:original_path,:cutout_path,:mask_path,"
            ":dimensions,:reference_views,:anchors,:meta)",
            row,
        )
    return {"id": pid, "birefnet_seconds": timing, "cutout": [cut.width, cut.height]}


def prepare_all(force=False):
    init_db()
    manifest = json.loads((CATALOG_DIR / "catalog.json").read_text())
    client = None
    needs = force or any(
        not (CUTOUTS_DIR / f"{p['id']}.png").exists() for p in manifest["products"]
    )
    if needs and BIREFNET_BACKEND == "space":
        client = Client(BIREFNET_SPACE, token=HF_TOKEN, verbose=False)
    out = []
    for p in manifest["products"]:
        out.append(prepare_product(p, client, force))
        print(out[-1], flush=True)
    return out
