"""AI-worn try-on with texture lock.

1. The model (Nano Banana) gets the untouched photo crop plus the product photo and puts the piece on the person
   itself: on the visible ear, chain around the neck, ring around the finger, with its own light and shadows.
2. We find where it drew the jewellery (difference from the original crop), fit the real product cutout to each
   drawn piece (position, size, angle, mirror) and replace the model's drawing with the real pixels, relit with the
   model's lighting. Leftover model jewellery outside the real cutout is cleaned back to the original photo.

The result looks worn (the model decided the pose-correct placement and occlusion) and keeps the product's real
texture when zoomed (the jewellery pixels come from the high-res product photo).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from . import pipeline as P

PROMPTS = {
    "earring": (
        "Put a matching pair of the earrings shown in image 2 on the person in image 1, as if she is really wearing "
        "them: one hanging from each pierced earlobe, attached only by its own wire or hinge exactly as shown in image "
        "2. Never change the hairstyle: do not move hair aside, shorten it or reveal an ear. If an ear is covered by "
        "hair, that earring hangs just below the hair at the earlobe, partly covered by it. Use exactly the design, "
        "metal, colours and stones of "
        "the earring in image 2, at a realistic earring size. Do not add any hook, clip, stud, ear cuff, chain or "
        "decoration that is not in image 2. Do not turn, tilt or move the head. Keep the person's face, expression, "
        "hair, skin, clothing, pose, lighting, framing and background exactly the same, pixel for pixel; only add the "
        "earrings, lit by the same light, with natural soft shadows."
    ),
    "pendant": (
        "Put the pendant shown in image 2 on the person in image 1, as if they are really wearing it: on a fine chain "
        "of the same metal that goes around the back of the neck, with the pendant hanging straight and resting "
        "naturally on the upper chest (over the clothing if the neckline is high). Use exactly the design, metal and "
        "details of the pendant in image 2, at a realistic size. Keep the face, hair, skin, clothing, pose, lighting, "
        "framing and background exactly the same; only add the necklace, lit by the same light, with soft shadows."
    ),
    "ring": (
        "Put the ring shown in image 2 on the ring finger of the hand in image 1, as if it is really worn: between the "
        "knuckle and the middle joint, the band wrapping around the finger and the top of the ring facing the camera. "
        "The ring must be on a finger, never on an object the hand is holding (a cup, compact, brush or phone); if "
        "the ring finger is hidden, put it on the most visible finger. If the palm faces the camera, the band wraps "
        "around the finger and shows on the palm side (the top of the ring is on the back of the finger, mostly "
        "hidden); never lay the ring on the palm or let it float. "
        "Use exactly the design, metal and stones of the ring in image 2, at a realistic size for that finger. Keep "
        "the hand, nails, skin, pose, lighting, framing and background exactly the same; only add the ring, lit by "
        "the same light, with natural soft shadows."
    ),
}


PROMPT_ONE_EAR = (
    "The person in image 1 is seen from the side, so only one ear is visible. Put one earring shown in image 2 on that "
    "visible ear, as if she is really wearing it: hanging from the pierced earlobe, attached only by its own wire or "
    "hinge exactly as shown in image 2. Do not put an earring or any jewellery anywhere else (not on the face, lips, "
    "nose or cheek). Use exactly the design, metal, colours and stones of the earring in image 2, at a realistic "
    "earring size. Do not add any hook, clip, stud, ear cuff or decoration that is not in image 2. Do not turn, tilt "
    "or move the head. Keep the face, hair, skin, clothing, pose, lighting, framing and background exactly the same; "
    "only add the earring, lit by the same light, with natural soft shadows."
)


def reference_image(cut: np.ndarray, max_side=1024) -> Image.Image:
    rgb = cut[..., :3] * cut[..., 3:4] + (1 - cut[..., 3:4])
    im = P.to_pil(rgb)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def jewellery_mask(orig: np.ndarray, gen: np.ndarray, threshold: float = 16.0) -> np.ndarray:
    """Where the model added jewellery: strong colour change vs the original (soft shadows stay below threshold)."""
    a = cv2.cvtColor(cv2.GaussianBlur(orig, (0, 0), 1.0), cv2.COLOR_RGB2LAB)
    b = cv2.cvtColor(cv2.GaussianBlur(gen, (0, 0), 1.0), cv2.COLOR_RGB2LAB)
    d = np.sqrt(((a - b) ** 2).sum(-1))  # Lab delta E (float input: L in 0..100)
    m = (d > threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def edge_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of edge maps: compares structure (face, ears, hair outlines), so flat backgrounds don't
    hide a moved head the way plain intensity correlation does."""
    def edges(x):
        L = cv2.GaussianBlur(P.luminance(x).astype(np.float32), (0, 0), 1.5)
        return np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
    ea, eb = edges(a), edges(b)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    return float((ea * eb).sum() / (np.sqrt((ea * ea).sum() * (eb * eb).sum()) + 1e-6))


def _edges(x: np.ndarray) -> np.ndarray:
    L = cv2.GaussianBlur(P.luminance(x).astype(np.float32), (0, 0), 1.5)
    return np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))


def face_alignment(ref: np.ndarray, gen: np.ndarray) -> float | None:
    """Edge correlation inside the face only (eyes, nose, mouth). Hair the model redraws or earrings it swaps
    don't count; a turned or moved face does. None when no face is found."""
    from .autoplace import FACE_OVAL, _detect, _pts

    res = _detect("face", P.to_pil(ref))
    if not res.face_landmarks:
        return None
    f = _pts(res.face_landmarks[0], ref.shape[1], ref.shape[0])
    m = np.zeros(ref.shape[:2], np.uint8)
    cv2.fillPoly(m, [f[FACE_OVAL].astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    if m.sum() < 500:
        return None
    a, b = _edges(ref)[m > 0], _edges(gen)[m > 0]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def warp_stats(ref_shape, warp_full) -> dict:
    A = warp_full[:, :2]
    return {"scale": float(np.sqrt(abs(np.linalg.det(A)))),
            "rotation": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
            "shift": float(np.hypot(*warp_full[:, 2]) / max(ref_shape[:2]))}


def align_to(ref: np.ndarray, mov: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, bool]:
    """Sub-pixel affine registration of the model output onto the original (ECC on luminance)."""
    h, w = ref.shape[:2]
    s = min(1.0, 512 / max(h, w))
    a = cv2.resize(P.luminance(ref), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(P.luminance(mov), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        return mov, False
    warp[:, 2] /= s
    if stats is not None:
        stats.update(warp_stats(ref.shape, warp))
    out = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1), True


def _components(mask, min_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append({"mask": (lab == i).astype(np.uint8), "bbox": (x, y, w, h), "area": int(area), "centroid": cent[i]})
    return out


DETAIL_PROMPT = (
    "Image 1 is a close-up crop of a photo of a person wearing the {piece} shown in image 2. Redraw only the {piece} "
    "in sharp, high-resolution detail that exactly matches image 2: the same design, metal, colours, stones, "
    "granulation and engraving, and nothing more: do not add any hook, clip, stud, cuff or part that is not in image "
    "2. Keep its position, size, angle, the way it hangs, and its lighting and shadows. Keep every other pixel of the "
    "crop (skin, hair, ear, clothing, background) exactly the same."
)


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.5, sigma)), 0, 1)


def detail_pass(provider, img_like: np.ndarray, region, ref, piece: str, seed, settings):
    """Second model call on a tight crop around one piece, at ~1024 px: sharp, product-faithful texture.
    Returns (image at model resolution aligned to the crop, alignment) or (None, score)."""
    x, y, w, h = region
    k = 1024 / max(w, h)
    mw, mh = max(8, round(w * k)), max(8, round(h * k))
    crop = np.clip(cv2.resize(img_like[y:y + h, x:x + w], (mw, mh), interpolation=cv2.INTER_CUBIC), 0, 1)
    res = provider.generate_tryon(P.to_pil(crop), ref, DETAIL_PROMPT.format(piece=piece), seed, settings)
    out = np.asarray(res.image.convert("RGB").resize((mw, mh), Image.LANCZOS), dtype=np.float32) / 255.0
    st = {}
    out, registered = align_to(crop, out, st)
    score = edge_alignment(crop, out)
    moved = (not registered or abs(st.get("scale", 1) - 1) > 0.05 or abs(st.get("rotation", 0)) > 3
             or st.get("shift", 0) > 0.05)
    ok = not moved and score >= settings.get("detail_min_edge_alignment", 0.7)
    return (out if ok else None), score, crop


def generate_worn(img, cut, product, hint_items, aspect, crop, provider, settings, seed, out_dir):
    """The model puts the piece on the person (reference mode); its image is accepted only if the person didn't
    move, then blended back into the full-res photo, with a sharper second pass per piece."""
    H, W = img.shape[:2]
    # give the model enough of the photo to understand the pose (at least ~half the image), at its working size
    cx, cy, cw, chh = crop
    if settings.get("whole_photo"):  # no landmarks: the model sees the whole photo and finds the spot itself
        cx, cy, cw, chh = 0, 0, W, H
    side = int(max(cw, chh, 0.5 * min(W, H)))
    ccx, ccy = cx + cw / 2, cy + chh / 2
    cw, chh = min(W, max(cw, side)), min(H, max(chh, side))
    cx = int(min(max(ccx - cw / 2, 0), W - cw))
    cy = int(min(max(ccy - chh / 2, 0), H - chh))
    out_crop = crop
    q = 1024 / max(cw, chh)  # small crops are upscaled for the model, large ones downscaled
    sw, sh = max(8, round(cw * q)), max(8, round(chh * q))
    small = np.clip(cv2.resize(img[cy:cy + chh, cx:cx + cw], (sw, sh),
                               interpolation=cv2.INTER_AREA if q < 1 else cv2.INTER_CUBIC), 0, 1)
    ref = reference_image(cut, 1024)
    cat = product["category"]
    prompt = PROMPTS.get(product["category"], PROMPTS["earring"])
    if product["category"] == "earring" and len(hint_items) == 1 and not settings.get("whole_photo"):
        prompt = PROMPT_ONE_EAR
    if settings.get("hard_case"):  # escalation attempt: say exactly what went wrong last time
        prompt = prompt + " " + settings["hard_case"]
    info = {"prompt": prompt, "attempts": []}
    gen = None
    for attempt in range(int(settings.get("worn_attempts", 1))):  # each attempt is a paid call
        res = provider.generate_tryon(P.to_pil(small), ref, prompt, seed + 1000 * attempt, settings)
        tag = "esc_" if settings.get("hard_case") else ""  # escalation keeps the first attempt's image too
        res.image.save(out_dir / (f"{tag}ai_context.png" if attempt == 0 else f"{tag}ai_context_{attempt}.png"))
        cand = np.asarray(res.image.convert("RGB").resize((sw, sh), Image.LANCZOS), dtype=np.float32) / 255.0
        st = {}
        cand, registered = align_to(small, cand, st)
        edge = edge_alignment(small, cand)
        face = face_alignment(small, cand)
        a = {"seed": res.seed, "edge_alignment": round(edge, 3), "face_alignment": None if face is None else round(face, 3),
             **{k: round(v, 3) for k, v in st.items()}}
        info["attempts"].append(a)
        # The person must not move: strict limits on the shift/scale/rotation needed to line the images up, and
        # the face itself must match (hair the model redraws, or earrings it swaps, don't count against it).
        moved = (not registered or abs(st.get("scale", 1) - 1) > settings.get("worn_max_scale", 0.035)
                 or abs(st.get("rotation", 0)) > 1.5 or st.get("shift", 0) > 0.03)
        if face is not None:
            changed = face < settings.get("worn_min_face_alignment", 0.25)
        else:
            changed = edge < settings.get("worn_min_edge_alignment", 0.55)
        if moved or changed:
            a["rejected"] = "the AI moved or reshaped the person"
            continue
        gen = cand
        info.update({"alignment": round(edge, 3), "registered": registered, "seed": res.seed, **res.info})
        if attempt and not tag:
            res.image.save(out_dir / "ai_context.png")
        break
    if gen is None:
        info["rejected"] = "the AI changed the person's pose or framing in every attempt"
        return {"ok": False, "info": info}

    hint_px = max(it.width for it in hint_items) * q
    hints = [((it.cx - cx) * q, (it.cy - cy) * q) for it in hint_items]
    cat = product["category"]

    # 1) The model's worn image, back in the full-res photo. Every pixel it changed (the jewellery, its shadows,
    #    hair falling in front) comes from the model, so pieces stay whole; pixels it left alone come from the
    #    original photo, so the face stays sharp.
    a_lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.0), cv2.COLOR_RGB2LAB)
    g_lab = cv2.cvtColor(cv2.GaussianBlur(gen, (0, 0), 1.0), cv2.COLOR_RGB2LAB)
    d = np.sqrt(((a_lab - g_lab) ** 2).sum(-1))
    w = np.clip((d - 3.0) / 7.0, 0, 1).astype(np.float32)
    w = cv2.dilate(w, np.ones((5, 5), np.uint8))
    w = cv2.GaussianBlur(w, (0, 0), 2.0)
    use_ai = cv2.resize(w, (cw, chh), interpolation=cv2.INTER_LINEAR)[..., None]
    gen_full = np.clip(cv2.resize(gen, (cw, chh), interpolation=cv2.INTER_LANCZOS4), 0, 1)
    ai_full = img.copy()
    ai_full[cy:cy + chh, cx:cx + cw] = use_ai * gen_full + (1 - use_ai) * img[cy:cy + chh, cx:cx + cw]

    # a black-and-white photo gets black-and-white jewellery (a gold piece in a B&W photo looks pasted)
    lab0 = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2LAB)
    if float(np.hypot(lab0[..., 1], lab0[..., 2]).mean()) < 4.0:
        gl = P.luminance(ai_full)
        ai_full = np.repeat(gl[..., None], 3, axis=2).astype(np.float32)
        info["monochrome_photo"] = True

    # 2) where the pieces are (for the high-res pass): strong changes near the expected spots, one piece per spot
    mask = jewellery_mask(small, gen)
    if cat == "pendant":  # leave the thin chain out of the piece (gently: the drawn pendant may be small)
        k = max(3, int(0.05 * hint_px)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    groups = {}
    for c in _components(mask, max(12, int(0.03 * hint_px * hint_px))):
        dists = [math.hypot(c["centroid"][0] - hx, c["centroid"][1] - hy) for hx, hy in hints]
        j = int(np.argmin(dists))
        if settings.get("whole_photo") or dists[j] < {"pendant": 4.0, "ring": 6.0}.get(cat, 2.5) * hint_px:
            groups.setdefault(j, np.zeros((sh, sw), np.uint8))
            groups[j] |= c["mask"]
    comps = []
    for m_ in groups.values():
        ys_, xs_ = np.nonzero(m_)
        comps.append({"mask": m_, "bbox": (xs_.min(), ys_.min(), xs_.max() - xs_.min() + 1, ys_.max() - ys_.min() + 1)})
    info["pieces_found"] = len(comps)
    if not comps:
        # couldn't pinpoint the piece; if the model clearly added something, keep its image (no sharpening pass)
        if mask.sum() < 0.0005 * mask.size:
            info["rejected"] = "the AI didn't add the jewellery"
            return {"ok": False, "info": info}
        info["note"] = "piece not located precisely; AI image used as is"

    # 2b) is it actually worn? A cheap vision model checks a close-up of each piece (the pixel checks can't tell a
    #     ring around a finger from one lying on the palm).
    if settings.get("verify_worn", True) and hasattr(provider, "verify"):
        checks = []
        regions = [c["bbox"] for c in comps] or [(0, 0, sw, sh)]
        for (x, y, w_, h_) in regions:
            s3 = int(max(w_, h_) * 3.2) + 16
            x0, y0 = int(max(0, (x + w_ / 2 - s3 / 2) / q + cx)), int(max(0, (y + h_ / 2 - s3 / 2) / q + cy))
            x1, y1 = int(min(W, x0 + s3 / q)), int(min(H, y0 + s3 / q))
            crop_img = P.to_pil(ai_full[y0:y1, x0:x1]) if comps else P.to_pil(ai_full)
            checks.append(provider.verify(crop_img, cat))
        info["worn_checks"] = checks
        bad = [c for c in checks if c.get("worn") is False]
        if bad:
            info["rejected"] = f"the AI didn't put it on properly ({bad[0].get('reason', 'not worn')})"
            return {"ok": False, "info": info}

    # 3) sharper jewellery: a second model pass on a tight crop around each piece (used only if it lines up)
    piece = {"earring": "earring", "pendant": "pendant", "ring": "ring"}.get(cat, "jewellery")
    overlays, details = [], []
    for i, c in enumerate(comps):
        x, y, w_, h_ = c["bbox"]
        if not settings.get("detail_pass", False):  # extra paid call per piece: opt-in
            details.append({"piece": i, "mode": "AI image"})
            continue
        side = int(max(w_, h_) / q * 1.4)
        ux, uy = int(cx + (x + w_ / 2) / q - side / 2), int(cy + (y + h_ / 2) / q - side / 2)
        ux, uy = max(0, min(W - side, ux)), max(0, min(H - side, uy))
        side = min(side, W - ux, H - uy)
        try:
            out, score, crop_in = detail_pass(provider, ai_full, (ux, uy, side, side), ref, piece, seed + 17 + i, settings)
        except Exception as e:  # keep the main result if the extra pass fails
            details.append({"piece": i, "mode": "AI image", "detail_error": str(e)[:200]})
            continue
        P.to_pil(crop_in).save(out_dir / f"detail_input_{i}.png")
        if out is None:
            details.append({"piece": i, "mode": "AI image", "detail_alignment": round(score, 3)})
            continue
        P.to_pil(out).save(out_dir / f"detail_ai_{i}.png")
        mh, mw = out.shape[:2]
        piece_full = np.zeros((H, W), np.float32)
        piece_full[cy:cy + chh, cx:cx + cw] = cv2.resize(c["mask"].astype(np.float32), (cw, chh), interpolation=cv2.INTER_LINEAR)
        pm = cv2.resize(piece_full[uy:uy + side, ux:ux + side], (mw, mh), interpolation=cv2.INTER_LINEAR)
        g2 = max(5, int(0.06 * mw)) | 1
        m = _feather(cv2.dilate((pm > 0.2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g2, g2))), g2 / 3)
        overlays.append({"region": (ux, uy, side, side), "image": out, "mask": m})
        details.append({"piece": i, "mode": "AI image + high-res AI pass", "detail_alignment": round(score, 3),
                        "detail_scale": round(mw / side, 2)})
    info["pieces"] = details
    info["model_crop"] = [cx, cy, cw, chh]
    return {"ok": True, "info": info, "ai_full": ai_full, "base": ai_full, "items": [],
            "gain_crop": np.ones((out_crop[3], out_crop[2], 3), np.float32), "overlays": overlays}
