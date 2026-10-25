"""One try-on job, stage by stage: preparing -> generating -> compositing -> complete."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import pipeline as P
from .providers import get_provider
from .providers.base import NoImageGeneratedError
from .worn import generate_worn

HARD_CASE = {
    "earring": ("If an ear is hidden by hair or shadow, show the earring hanging below the hair, partly covered by "
                "it, exactly where that earlobe would be; never draw or reveal an ear that is not visible."),
    "ring": ("Look at which way the hand faces. If the palm faces the camera, put the ring on the fourth finger "
             "(ring finger) so its band wraps around that finger at the base; on the palm side only the band shows. "
             "The ring must encircle a finger; never lay it on the palm, never let it float, never put it on an object."),
    "pendant": ("The pendant hangs from a fine chain that goes around the back of the neck; it rests on the chest or "
                "clothing below the collarbones and never floats."),
}

DEFAULT_SETTINGS = {
    "detail_pass": False,        # opt-in extra AI request per detected piece
    "context_pad": 1.0,          # crop padding as a multiple of the jewellery size
    "relight_strength": 0.55,     # how much of the model's colour cast / shading to apply to the cutout
    "shading_strength": 1.0,     # how much of the model's contact shadows to apply to the skin
    "shadow_opacity": 0.14,      # our own soft contact shadow
    "output_long_side": 6144,    # master export; bounded to 64 megapixels
    "send_reference": False,      # a 2nd (product) image makes the model return a product close-up instead of editing the photo
    "min_alignment": 0.6,         # reject model outputs whose framing no longer matches the photo
    "min_local_alignment": 0.75,  # earrings/rings: reject outputs where the model moved the piece
    "ai_blend_min": 0.9,          # blend the AI image into the photo only if it kept the pose almost exactly
    "ai_attempts": 1,             # free GPU quota is small; raise to retry when the model moves the piece
    "auto_light": True,          # no-AI light match (colour cast, exposure, direction) from the photo
    "auto_light_strength": 0.85,
    "steps": 4,
    "guidance_scale": 1.0,
    "lora_adapter": "Anything2Real",
}


def default_context_mode(category: str) -> str:
    # Generated jewellery must never sit underneath the source cutout.
    return "shading"


def _alpha_layer(cut_pm, items, k, origin, shape):
    a = np.zeros(shape, np.float32)
    for it in items:
        patch, x0, y0 = P.place_patch(cut_pm, it, k, origin)
        if it.occlude:
            patch *= P.occlusion_keep(cut_pm.shape, it, k, origin, patch.shape, x0, y0)[..., None]
        ov = P._overlap(shape, patch.shape, x0, y0)
        if ov:
            cs, ps = ov
            pa = patch[..., 3][ps]
            a[cs] = a[cs] + pa * (1 - a[cs])
    return a


def _reference_image(cut: np.ndarray, max_side=768) -> Image.Image:
    rgb = cut[..., :3] * cut[..., 3:4] + (1 - cut[..., 3:4])  # on white
    im = P.to_pil(rgb)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def run_tryon(job: dict, product: dict, upload: dict, out_dir: Path, set_status) -> dict:
    timings = {}
    settings = {**DEFAULT_SETTINGS, **(job.get("settings") or {})}
    settings.setdefault("context_mode", default_context_mode(product["category"]))
    if job["provider"] == "fal-nano-banana":
        # Supply the product reference while keeping the same pre-placed geometry.
        user = job.get("settings") or {}
        settings["send_reference"] = user.get("send_reference", True)
    # Replacing the context with generated pixels can leave duplicate jewellery
    # under the real cutout. The faithful result only borrows light and shadows.
    settings["context_mode"] = "shading"
    target = settings["output_long_side"]
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not np.isfinite(target) or not 1024 <= target <= 8192:
        raise ValueError("output_long_side must be between 1024 and 8192 pixels")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- preparing
    t = time.time()
    set_status("preparing")
    img = P.load_rgb(upload["original_path"])
    H, W = img.shape[:2]
    cut = P.load_rgba(product["cutout_path"])
    cut_pm = P.premultiply(cut)
    ch_, cw_ = cut.shape[:2]
    aspect = ch_ / cw_
    items = [P.Item.from_dict(d) for d in job["placement"]["items"]]
    if not 1 <= len(items) <= 2:
        raise ValueError("placement needs 1 or 2 items")
    for it in items:
        if not (0 <= it.cx <= W and 0 <= it.cy <= H) or not (4 <= it.width <= W):
            raise ValueError("placement is outside the photo")
    boxes = [P.item_bbox(it, aspect) for it in items]
    crop_boxes = list(boxes)
    if job["placement"].get("chain"):
        cp = np.asarray(job["placement"]["chain"]["points"])
        crop_boxes.append((cp[:, 0].min(), cp[:, 1].min(), cp[:, 0].max(), cp[:, 1].max()))
    cx, cy, cw, chh = P.context_crop(crop_boxes, W, H, settings["context_pad"])
    crop = (cx, cy, cw, chh)
    q = min(1.0, 1024 / max(cw, chh))
    sw, sh = max(int(round(cw * q)), 8), max(int(round(chh * q)), 8)
    q_x, q_y = sw / cw, sh / chh
    qk = (q_x + q_y) / 2
    base_small = cv2.resize(img[cy:cy + chh, cx:cx + cw], (sw, sh), interpolation=cv2.INTER_AREA)
    pre = base_small.copy()
    chain = job["placement"].get("chain")
    chain_color = P.dominant_metal_color(cut)
    if chain:
        P.draw_chain(pre, chain, chain_color, crop, qk)
    P.render_layer(cut_pm, items, aspect, crop, qk, gains=None, shadow_opacity=0.0)(pre)
    jmask = _alpha_layer(cut_pm, items, qk, (cx, cy), (sh, sw))
    P.to_pil(pre).save(out_dir / "preplaced.png")
    timings["preparing"] = round(time.time() - t, 2)

    # Start with local lighting, then replace only the neighbourhood of each
    # accepted AI pass. Separate crops give small earrings meaningful model pixels.
    base = img.copy()
    gain_crop = np.ones((chh, cw, 3), np.float32)
    g = np.ones(3, np.float32)
    info = {"attempts": [], "lighting_passes": []}
    if settings.get("auto_light", True):
        alpha_full = _alpha_layer(cut_pm, items, 1.0, (cx, cy), (chh, cw))
        g, spatial = P.scene_light(img[cy:cy + chh, cx:cx + cw], alpha_full,
                                  max(it.width for it in items), settings["auto_light_strength"])
        info["auto_light_gain_rgb"] = [round(float(v), 3) for v in g]
        gain_crop = spatial[..., None] * g[None, None, :]
    provider = get_provider(job["provider"])
    seed = int(job["seed"])
    ai_draft = img.copy()
    has_draft = False
    # Worn mode preserves the model's natural placement, geometry and occlusion.
    render_items = items
    overlays = []
    worn_mode = provider is not None and settings.get("mode", "worn" if job["provider"] in ("fal-nano-banana", "fal-nano-banana-2") else "lighting") == "worn"
    if worn_mode:
        t = time.time()
        set_status("generating")
        w = None
        try:
            settings["whole_photo"] = bool(job["placement"].get("whole_photo"))
            settings["face_oval"] = job["placement"].get("face_oval")
            w = generate_worn(img, cut, product, items, aspect, crop, provider, settings, seed, out_dir)
            # Escalation, only when the first result is unusable: one retry with the stronger model (Nano Banana 2,
            # ~$0.08) and an instruction aimed at what went wrong.
            if not w["ok"] and settings.get("escalate", True) and job["provider"] == "fal-nano-banana":
                reason = str(w["info"].get("rejected", ""))
                hard = HARD_CASE.get(product["category"], "")
                if reason.startswith("the AI didn't put it on properly"):
                    hard = f"Last time this went wrong: {reason}. {hard}"
                elif "pose or framing" in reason:
                    # same framing as the first attempt (a wider crop made the model zoom in even more)
                    hard = ("Keep the exact framing and zoom of image 1. Do not move, turn, rescale or redraw the "
                            "person, hair or background in any way. " + hard)
                first = w["info"]
                retry = dict(settings, hard_case=hard, worn_attempts=1)
                w = generate_worn(img, cut, product, items, aspect, crop, get_provider("fal-nano-banana-2"), retry,
                                  seed + 7919, out_dir)
                w["info"]["escalated"] = {"model": "fal-nano-banana-2", "first_attempt": first}
        except NoImageGeneratedError as e:
            raise NoImageGeneratedError("The AI returned no image. Press Generate again.") from e
        if w is not None:
            info["worn"] = w["info"]
            if w.get("ai_full") is not None:
                P.to_pil(w["ai_full"]).save(out_dir / "tryon_ai.png", compress_level=4)
                info["ai_output"] = "blended into photo"
            if w["ok"]:
                base = w["base"]
                render_items = w["items"]
                overlays = w["overlays"]
                piece_boxes = w.get("piece_boxes") or [P.item_bbox(it, aspect) for it in render_items] + [
                    (x, y, x + s_, y + s_) for (x, y, s_, _) in (o["region"] for o in overlays)]
                if piece_boxes:
                    boxes = piece_boxes
                gain_crop = w["gain_crop"]
                g = np.ones(3, np.float32)
                chain = None  # the model drew the chain around the neck
            else:
                # never fall back to pasting the product photo: it doesn't look worn
                from .providers.base import ProviderError
                raise ProviderError(f"The AI couldn't put the piece on this photo ({w['info'].get('rejected')}). "
                                    "Press Generate again, or try a clearer photo.")
        timings["generating"] = round(time.time() - t, 2)
        provider = None  # don't run the lighting-only pass as well (no second paid call)
    if provider is not None:
        t = time.time()
        set_status("generating")
        ref = _reference_image(cut, 1536) if settings["send_reference"] else None
        piece_name = {"earring": "earring", "pendant": "pendant and chain", "ring": "ring"}.get(product["category"], "jewellery")
        prompt = (f"Edit image 1, the close-up photo with an existing {piece_name}. "
                  "Match the metal's illumination and soft contact shadows to the surrounding photograph. "
                  "Keep the existing jewellery at exactly the same location, scale and shape. "
                  "Preserve the skin, hair, background and crop framing. Return the edited photograph." 
                  + (" Image 2 is a product reference for the material only." if ref is not None else ""))
        info["prompt"] = prompt
        accepted_count = 0
        for index, (it, box) in enumerate(zip(items, boxes)):
            local_boxes = [box]
            if chain:
                local_boxes = crop_boxes
            lx, ly, lw, lh = P.context_crop(local_boxes, W, H, pad_factor=0.8, min_side=256)
            # Crop first, then render the original asset directly at model size.
            lk = 1024 / max(lw, lh)
            mw, mh = round(lw * lk), round(lh * lk)
            local_pre = cv2.resize(img[ly:ly+lh, lx:lx+lw], (mw, mh), interpolation=cv2.INTER_CUBIC)
            local_pre = np.clip(local_pre, 0, 1)
            local_region = (lx, ly, lw, lh)
            if chain:
                P.draw_chain(local_pre, chain, chain_color, local_region, lk)
            P.render_layer(cut_pm, [it], aspect, local_region, lk, shadow_opacity=0)(local_pre)
            mask = _alpha_layer(cut_pm, [it], lk, (lx, ly), (mh, mw))
            P.to_pil(local_pre).save(out_dir / f"lighting_input_{index}.png")
            accepted = None
            for attempt in range(max(1, min(3, int(settings["ai_attempts"])))):
                request_seed = seed + index * 100 + attempt
                try:
                    res = provider.generate_tryon(P.to_pil(local_pre), ref, prompt, request_seed, settings)
                except NoImageGeneratedError:
                    info.setdefault("warnings", []).append(f"Piece {index + 1}: the AI returned no image; local lighting used.")
                    break
                cand = np.asarray(res.image.convert("RGB").resize((mw, mh), Image.Resampling.LANCZOS), dtype=np.float32) / 255
                align = P.alignment_score(local_pre, cand, mask)
                local = P.local_alignment(local_pre, cand, mask, it.width * lk)
                info["attempts"].append({"piece": index, "seed": res.seed, "alignment": round(align, 3),
                                         "local_alignment": round(local, 3), "input_size": [mw, mh]})
                res.image.save(out_dir / f"lighting_output_{index}_{attempt}.png")
                if index == 0 and attempt == 0:
                    res.image.save(out_dir / "ai_context.png")
                if align >= settings["ai_blend_min"]:
                    blend = cv2.resize(P.pixel_blend_map((mh, mw)), (lw, lh))[..., None]
                    generated = cv2.resize(cand, (lw, lh), interpolation=cv2.INTER_CUBIC)
                    original = ai_draft[ly:ly+lh, lx:lx+lw]
                    ai_draft[ly:ly+lh, lx:lx+lw] = np.clip(blend * generated + (1-blend) * original, 0, 1)
                    has_draft = True
                if align >= settings["min_alignment"] and local >= settings["min_local_alignment"]:
                    accepted = cand
                    break
            info["lighting_passes"].append({"piece": index, "accepted": accepted is not None,
                                             "crop": list(local_region), "jewellery_model_width": round(it.width * lk)})
            if accepted is None:
                continue
            accepted_count += 1
            shadows = P.shading_map(local_pre, accepted, mask, it.width * lk, settings["shading_strength"])
            base[ly:ly+lh, lx:lx+lw] *= cv2.resize(shadows, (lw, lh))[..., None]
            gain, spatial = P.estimate_relight(local_pre, accepted, mask, it.width * lk, settings["relight_strength"])
            local_gain = cv2.resize(spatial, (lw, lh))[..., None] * gain
            weight = cv2.GaussianBlur(mask, (0, 0), max(2, 0.12 * it.width * lk))
            weight = cv2.resize(np.clip(weight * 3, 0, 1), (lw, lh))[..., None]
            # Both context crops are in upload coordinates; clip for boundary cases.
            x0, y0 = max(lx, cx), max(ly, cy)
            x1, y1 = min(lx+lw, cx+cw), min(ly+lh, cy+chh)
            a = weight[y0-ly:y1-ly, x0-lx:x1-lx]
            target_gain = gain_crop[y0-cy:y1-cy, x0-cx:x1-cx]
            target_gain[:] = a * local_gain[y0-ly:y1-ly, x0-lx:x1-lx] + (1-a) * target_gain
        info["accepted_lighting_passes"] = accepted_count
        if accepted_count < len(items):
            info["ai_rejected"] = "Some AI passes moved the piece or changed framing; local lighting used for those pieces."
        if has_draft:
            P.to_pil(ai_draft).save(out_dir / "tryon_ai.png", compress_level=4)
            info["ai_output"] = "blended into photo"
        timings["generating"] = round(time.time() - t, 2)

    # ---------------------------------------------------------------- compositing
    t = time.time()
    set_status("compositing")
    def gains(region, k, canvas_shape):
        if gain_crop is None:
            return None
        # Sample the smooth lighting map at global output pixel centres.
        hh, ww = canvas_shape[:2]
        xx = (np.arange(ww, dtype=np.float32) + 0.5) / k + region[0] - cx - 0.5
        yy = (np.arange(hh, dtype=np.float32) + 0.5) / k + region[1] - cy - 0.5
        mx, my = np.meshgrid(xx, yy)
        return cv2.remap(gain_crop, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

    if gain_crop is not None:
        chain_color = np.clip(chain_color * g, 0, 1)

    # A single master contains the final jewellery pixels. The viewer, download
    # and close-ups all use this exact image, without substituting zoom tiles.
    native_scale = max(cw_ / it.width for it in items)
    export_scale = max(1.0, min(target / max(W, H), (64_000_000 / (W * H)) ** 0.5,
                                native_scale, 8.0))
    output_w, output_h = round(W * export_scale), round(H * export_scale)
    # Render strips with an overlap for chain/shadow filtering. This bounds float
    # canvas memory on a 16 GB machine, even for an 8K PNG.
    master = P.to_pil(base).resize((output_w, output_h), Image.Resampling.LANCZOS)
    strip_rows = max(1, int(384 / export_scale))
    halo = max(16, int(max(it.width for it in items) * 0.15),
               int((chain or {}).get("thickness", 0) * 8))
    bands = []
    for y in range(0, H, strip_rows):
        end = min(H, y + strip_rows)
        lo, hi = max(0, y - halo), min(H, end + halo)
        # Exact global pixel phase for base and gain sampling across strip joins.
        oy0, oy1 = round(lo * export_scale), round(hi * export_scale)
        canvas = np.asarray(master.crop((0, oy0, output_w, oy1)), dtype=np.float32) / 255.0
        region = (0, oy0 / export_scale, W, (oy1 - oy0) / export_scale)
        if chain:
            P.draw_chain(canvas, chain, chain_color, region, export_scale)
        P.render_layer(cut_pm, render_items, aspect, region, export_scale, gains, settings["shadow_opacity"])(canvas)
        start_out, end_out = round(y * export_scale), round(end * export_scale)
        band = P.to_pil(canvas[start_out - oy0:end_out - oy0])
        # Don't feed already composited pixels into the next overlapping strip.
        # Keep strips separate until rendering has finished.
        bands.append((start_out, band))
    for start_out, band in bands:
        master.paste(band, (0, start_out))
    del bands
    # high-res AI detail passes (pieces the real photo couldn't be fitted to): blend them in at export scale
    for o in overlays:
        x, y, s_, _ = o["region"]
        X0, Y0 = round(x * export_scale), round(y * export_scale)
        S_ = min(round(s_ * export_scale), output_w - X0, output_h - Y0)
        if S_ < 4:
            continue
        hi = cv2.resize(o["image"], (S_, S_), interpolation=cv2.INTER_AREA if S_ < o["image"].shape[1] else cv2.INTER_CUBIC)
        m = cv2.resize(o["mask"], (S_, S_), interpolation=cv2.INTER_LINEAR)[..., None]
        under = np.asarray(master.crop((X0, Y0, X0 + S_, Y0 + S_)), dtype=np.float32) / 255.0
        master.paste(P.to_pil(np.clip(m * hi + (1 - m) * under, 0, 1)), (X0, Y0))
    full_path = out_dir / "tryon.png"
    master.save(full_path, compress_level=4)
    preview = master.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    preview.save(out_dir / "preview.jpg", quality=95, subsampling=0)
    info["quality"] = {
        "output_size": [output_w, output_h], "upload_size": [W, H],
        "export_scale": round(export_scale, 3), "source_size": [cw_, ch_],
        "jewellery_width_px": [round(it.width * export_scale) for it in items],
        "source_limited": any(it.width * export_scale > cw_ for it in items),
        "background_upscaled": export_scale > 1,
        "detail_is_exact_crop": True,
    }

    # 2) Literal pixel crops of the downloadable master (no independent redraw).
    tiles = []
    for i, b in enumerate(boxes):
        pad = 0.35 * max(b[2] - b[0], b[3] - b[1])
        x0, y0 = max(0, round((b[0] - pad) * export_scale)), max(0, round((b[1] - pad) * export_scale))
        x1, y1 = min(output_w, round((b[2] + pad) * export_scale)), min(output_h, round((b[3] + pad) * export_scale))
        name = f"detail_{i}.png"
        master.crop((x0, y0, x1, y1)).save(out_dir / name, compress_level=4)
        tiles.append({"file": name, "region": [x0, y0, x1 - x0, y1 - y0],
                      "scale": 1.0, "size": [x1 - x0, y1 - y0]})
    shutil.copyfile(out_dir / tiles[0]["file"], out_dir / "detail.png")

    # 3) original product reference
    shutil.copyfile(product["original_path"], out_dir / ("reference" + Path(product["original_path"]).suffix))
    timings["compositing"] = round(time.time() - t, 2)

    return {
        "output_path": str(full_path),
        "seed": seed,
        "timings": timings,
        "info": {
            **info,
            "crop": list(crop),
            "preplaced_preview_size": [sw, sh],
            "tiles": tiles,
            "context_mode": ("worn" if worn_mode and w is not None and w.get("ok")
                             else "shading" if info.get("accepted_lighting_passes") else "none"),
        },
    }
