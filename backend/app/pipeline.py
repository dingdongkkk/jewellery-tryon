"""Image pipeline: crop, pre-place, AI context, then detail-locked compositing.

Coordinates: a placement item is {cx, cy, width, rotation, flip} in pixels of the uploaded
(EXIF-corrected) photo. (cx, cy) is the centre of the product cutout, width is the rendered width
of the cutout, rotation is degrees clockwise (same as CSS), flip mirrors horizontally.

Detail lock: the generative model only ever supplies lighting / shading / context. The jewellery
pixels in every output are resampled straight from the original high-res product cutout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MAX_SIDE, MIN_SIDE

# ----------------------------------------------------------------------------- io helpers


def load_rgb(path) -> np.ndarray:
    """float32 RGB in [0,1]."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def load_rgba(path) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    return np.asarray(im, dtype=np.float32) / 255.0


def to_pil(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def normalise_upload(src) -> Image.Image:
    """Fix EXIF orientation, validate size, cap absurdly large photos. Raises ValueError."""
    try:
        im = Image.open(src)
        im.load()
    except Exception as e:
        raise ValueError("Not a readable image (use JPEG, PNG or WebP).") from e
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB",):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        im = im.convert("RGBA")
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        raise ValueError(f"Photo is too small ({w}x{h}). Use at least {MIN_SIDE}px on the short side.")
    if max(w, h) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


# ----------------------------------------------------------------------------- geometry


@dataclass
class Item:
    cx: float
    cy: float
    width: float
    rotation: float = 0.0
    flip: bool = False
    occlude: dict | None = None  # rings: {"capsule": [x0,y0,x1,y1,r], "rect": [u0,v0,u1,v1], "always": bool}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["cx"]), float(d["cy"]), float(d["width"]), float(d.get("rotation", 0.0)),
                   bool(d.get("flip", False)), d.get("occlude"))


def item_bbox(item: Item, aspect: float):
    """Axis-aligned bbox (x0, y0, x1, y1) of the rotated cutout. aspect = h / w of the cutout."""
    w, h = item.width, item.width * aspect
    t = math.radians(item.rotation)
    bw = abs(w * math.cos(t)) + abs(h * math.sin(t))
    bh = abs(w * math.sin(t)) + abs(h * math.cos(t))
    return item.cx - bw / 2, item.cy - bh / 2, item.cx + bw / 2, item.cy + bh / 2


def union_bbox(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def context_crop(boxes, img_w, img_h, pad_factor=1.0, min_side=384):
    """Square-ish crop around the jewellery with enough context for lighting and shadows."""
    x0, y0, x1, y1 = union_bbox(boxes)
    bw, bh = x1 - x0, y1 - y0
    pad = max(pad_factor * max(bw, bh), 0.08 * min(img_w, img_h))
    side = max(bw, bh) + 2 * pad
    side = max(side, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = min(side if bw <= bh else max(side, bw + 2 * pad), img_w)
    h = min(side, img_h)
    left = int(round(min(max(cx - w / 2, 0), img_w - w)))
    top = int(round(min(max(cy - h / 2, 0), img_h - h)))
    return left, top, int(round(w)), int(round(h))


# ----------------------------------------------------------------------------- cutout rendering


def premultiply(rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def place_patch(cutout_pm: np.ndarray, item: Item, k: float = 1.0, origin=(0.0, 0.0)):
    """Resample the premultiplied cutout for one placement.

    Returns (patch_pm, x0, y0): the transformed premultiplied RGBA patch and its integer top-left
    in the target canvas, where the canvas is the upload scaled by k and shifted by -origin*k.
    A single affine transform preserves detail; strong reductions use an area prefilter.
    """
    H, W = cutout_pm.shape[:2]
    tw = max(item.width * k, 1.0)
    # Keep two samples per destination pixel before the final transform. At native
    # resolution this is a single affine resample, rather than resize + rotate.
    s = tw / W
    prefilter = min(1.0, 2 * s)
    nw, nh = max(int(round(W * prefilter)), 1), max(int(round(H * prefilter)), 1)
    small = (cv2.resize(cutout_pm, (nw, nh), interpolation=cv2.INTER_AREA)
             if prefilter < 1 else cutout_pm)
    if item.flip:
        small = small[:, ::-1].copy()

    t = math.radians(item.rotation)
    th = tw * H / W
    bw = abs(tw * math.cos(t)) + abs(th * math.sin(t))
    bh = abs(tw * math.sin(t)) + abs(th * math.cos(t))
    cx = (item.cx - origin[0]) * k
    cy = (item.cy - origin[1]) * k
    x0 = int(math.floor(cx - bw / 2)) - 2
    y0 = int(math.floor(cy - bh / 2)) - 2
    ow, oh = int(math.ceil(bw)) + 5, int(math.ceil(bh)) + 5
    # rotate about the small patch centre, then move that centre to (cx, cy) - (x0, y0)
    M = np.array([[math.cos(t) * tw / nw, -math.sin(t) * th / nh, 0],
                  [math.sin(t) * tw / nw, math.cos(t) * th / nh, 0]], np.float64)
    # Pixel centres live at integer coordinates; placement centres refer to edges.
    M[:, 2] = [cx - x0 - 0.5, cy - y0 - 0.5] - M[:, :2] @ [(nw - 1) / 2, (nh - 1) / 2]
    patch = cv2.warpAffine(
        small, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    patch = np.clip(patch, 0, 1)
    patch[..., :3] = np.minimum(patch[..., :3], patch[..., 3:4])
    return patch, x0, y0


def _overlap(canvas_shape, patch_shape, x0, y0):
    H, W = canvas_shape[:2]
    h, w = patch_shape[:2]
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, W), min(y0 + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))


def composite_pm(canvas: np.ndarray, patch_pm: np.ndarray, x0: int, y0: int, gain=None):
    """Over-composite a premultiplied patch. gain: optional (h,w,3) multiplier aligned to the patch."""
    ov = _overlap(canvas.shape, patch_pm.shape, x0, y0)
    if ov is None:
        return
    cs, ps = ov
    p = patch_pm[ps]
    a = p[..., 3:4]
    rgb = p[..., :3]
    if gain is not None:
        straight = rgb / np.maximum(a, 1e-6)
        g = np.clip(gain[ps], 0.5, 1.6)
        # Exposure in a bounded tone curve, not multiply-and-clip: polished
        # highlights retain their differences instead of becoming flat white.
        rgb = straight * g / (1 + straight * (g - 1)) * a
    canvas[cs] = np.clip(canvas[cs] * (1 - a) + rgb, 0, 1)


def apply_contact_shadow(canvas, patch_pm, x0, y0, tw, opacity):
    """Soft drop shadow under the jewellery: blurred alpha, nudged down, multiplied into the skin."""
    if opacity <= 0:
        return
    a = patch_pm[..., 3]
    pad = int(max(4, 0.08 * tw))
    a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    sigma = max(1.0, 0.025 * tw)
    sh = cv2.GaussianBlur(a, (0, 0), sigma)
    dy = int(round(0.02 * tw)) + 1
    sh = np.roll(sh, dy, axis=0)
    sh3 = np.repeat((1 - opacity * sh)[..., None], 3, axis=2)
    ov = _overlap(canvas.shape, sh3.shape, x0 - pad, y0 - pad)
    if ov is None:
        return
    cs, ps = ov
    canvas[cs] *= sh3[ps]


# ----------------------------------------------------------------------------- AI context


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def alignment_score(pre: np.ndarray, gen: np.ndarray, jmask: np.ndarray) -> float:
    """Normalised cross-correlation of the two images away from the jewellery (1 = same framing)."""
    size = (96, 96)
    a = cv2.resize(luminance(pre), size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(luminance(gen), size, interpolation=cv2.INTER_AREA)
    m = cv2.resize((jmask < 0.05).astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.99
    a, b = a[m], b[m]
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-6
    return float((a * b).sum() / den)


def local_alignment(pre: np.ndarray, gen: np.ndarray, jmask: np.ndarray, jewel_px: float) -> float:
    """Correlation of model output vs input right around the jewellery (did the model keep the piece where it was?)."""
    r = int(max(3, 0.5 * jewel_px))
    near = cv2.dilate((jmask > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    if near.sum() < 50:
        return 1.0
    a = cv2.GaussianBlur(luminance(pre), (0, 0), 1.5)[near]
    b = cv2.GaussianBlur(luminance(gen), (0, 0), 1.5)[near]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def estimate_relight(pre: np.ndarray, gen: np.ndarray, jmask: np.ndarray, jewel_px: float, strength: float):
    """How did the model re-light the jewellery? Returns (channel_gain[3], spatial_lum_gain map).

    Only low-frequency information is taken (a colour cast + a smooth shading map), so applying
    it to the original cutout cannot invent or erase stones, links or engraving.
    """
    m = cv2.erode((jmask > 0.6).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    ones = np.ones(pre.shape[:2], np.float32)
    if m.sum() < 60 or strength <= 0:
        return np.ones(3, np.float32), ones
    eps = 1e-3
    g = np.array([(np.median(gen[..., c][m]) + eps) / (np.median(pre[..., c][m]) + eps) for c in range(3)], np.float32)
    g = np.clip(g, 0.5, 1.6)
    g = 1 + strength * (g - 1)

    sigma = max(2.0, 0.2 * jewel_px)
    mf = m.astype(np.float32)
    lp = cv2.GaussianBlur(luminance(pre) * mf, (0, 0), sigma) + eps
    lq = cv2.GaussianBlur(luminance(gen) * mf, (0, 0), sigma) + eps
    w = cv2.GaussianBlur(mf, (0, 0), sigma)
    ratio = np.where(w > 0.05, lq / lp, 1.0)
    gl = float(np.dot([0.2126, 0.7152, 0.0722], g))
    spatial = np.clip(ratio / max(gl, eps), 0.75, 1.25)
    spatial = 1 + strength * (spatial - 1)
    return g, spatial.astype(np.float32)


def scene_light(img_crop: np.ndarray, alpha: np.ndarray, jewel_px: float, strength: float = 1.0):
    """No-AI light match from the photo itself, measured on the skin / clothes right around the jewellery.

    Returns (channel_gain[3], spatial gain map) like estimate_relight: the colour cast of the local light
    (product photos are shot under neutral studio light), the exposure, and the light's direction across
    the piece. All low-frequency, so the product's texture is untouched.
    """
    eps = 1e-3
    r = int(max(4, 0.6 * jewel_px))
    near = cv2.dilate((alpha > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ring = (near > 0) & (alpha < 0.05)
    if ring.sum() < 50 or strength <= 0:
        return np.ones(3, np.float32), np.ones(alpha.shape, np.float32)
    # white-patch estimate on the whole context crop: the brightest surfaces show the light's colour and
    # strength (a dark sweater next to the piece is dark material, not dim light)
    lumc = luminance(img_crop)
    hi = lumc >= np.percentile(lumc, 97)
    illum = img_crop[hi].mean(0) + eps
    cast = illum / illum.mean()
    exposure = float(np.clip((float(np.dot(illum, [0.2126, 0.7152, 0.0722])) / 0.92) ** 0.6, 0.6, 1.05))
    g = np.clip(cast, 0.6, 1.5) ** 0.7 * exposure
    g = (1 + strength * (g - 1)).astype(np.float32)
    # light direction: plane fit of the surrounding luminance, normalised around the jewellery centre
    L = cv2.GaussianBlur(luminance(img_crop), (0, 0), max(2.0, 0.3 * jewel_px))
    ys, xs = np.nonzero(ring)
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, L[ys, xs], rcond=None)
    yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]].astype(np.float32)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    ay, ax = np.nonzero(alpha > 0.5)
    centre = float(coef[0] * ax.mean() + coef[1] * ay.mean() + coef[2]) if len(ax) else float(L.mean())
    spatial = np.clip(plane / max(centre, eps), 0.8, 1.2)
    spatial = 1 + strength * (spatial - 1)
    return g, spatial.astype(np.float32)


def photo_grain(img_crop: np.ndarray) -> float:
    """Robust std of the photo's fine noise, so pasted pixels can carry the same grain."""
    lum = luminance(img_crop)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.2)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def shading_map(pre, gen, jmask, jewel_px, strength):
    """Darkening the model added around the jewellery (contact shadows, occlusion), as a multiplier."""
    eps = 1e-3
    sigma = max(1.5, 0.03 * jewel_px)
    ratio = (cv2.GaussianBlur(luminance(gen), (0, 0), sigma) + eps) / (cv2.GaussianBlur(luminance(pre), (0, 0), sigma) + eps)
    ratio = np.clip(ratio, 0.78, 1.0)
    r = int(max(3, 0.35 * jewel_px))
    ring = cv2.dilate((jmask > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ring = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), r / 2)
    # Borrow only subtle contact darkening outside the visible jewellery.
    ring *= 1 - np.clip(jmask, 0, 1)
    return (1 + strength * ring * (ratio - 1)).astype(np.float32)


def pixel_blend_map(shape, feather_frac=0.08):
    """Weight for pasting model pixels (pendant chains etc.): crop interior, feathered at the border."""
    h, w = shape
    f = max(4, int(feather_frac * min(h, w)))
    m = np.zeros((h, w), np.float32)
    m[f:h - f, f:w - f] = 1
    m = cv2.GaussianBlur(m, (0, 0), f / 2)
    return m


# ----------------------------------------------------------------------------- top level


def occlusion_keep(cut_shape, item: Item, k, origin, patch_shape, x0, y0):
    """1 where the piece stays visible, 0 where something is in front of it (finger, face, hair)."""
    h, w = patch_shape[:2]
    if "visible" in item.occlude:
        # raster visibility in upload coords (from the AI's drawing): sample it at the patch pixels
        vis = item.occlude["visible"]
        ox, oy = item.occlude["origin"]
        xs = ((np.arange(w, dtype=np.float32) + x0 + 0.5) / k + origin[0] - ox - 0.5)
        ys = ((np.arange(h, dtype=np.float32) + y0 + 0.5) / k + origin[1] - oy - 0.5)
        mx, my = np.meshgrid(xs, ys)
        return cv2.remap(vis, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    H, W = cut_shape[:2]
    uu, vv = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
    uv, _, _ = place_patch(np.dstack([uu, vv, np.zeros_like(uu), np.ones_like(uu)]), item, k, origin)
    h, w = patch_shape[:2]
    uv = uv[:h, :w]
    a = np.maximum(uv[..., 3], 1e-4)
    u, v = uv[..., 0] / a, uv[..., 1] / a
    u0, v0, u1, v1 = item.occlude["rect"]
    in_rect = ((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)).astype(np.float32)
    if item.occlude.get("polygon"):
        poly = (np.asarray(item.occlude["polygon"], np.float32) - [origin[0], origin[1]]) * k - [x0, y0]
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [np.round(poly * 4).astype(np.int32)], 1, lineType=cv2.LINE_AA, shift=2)
        hide = in_rect * inside.astype(np.float32)
    elif item.occlude.get("always"):
        hide = in_rect
    else:
        ax, ay, bx, by, r = item.occlude["capsule"]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        X = (xs + x0) / k + origin[0]
        Y = (ys + y0) / k + origin[1]
        dx, dy = bx - ax, by - ay
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / (dx * dx + dy * dy + 1e-6), 0, 1)
        dist = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        inside = np.clip((r - dist) * k + 0.5, 0, 1)  # ~1px soft edge at output scale
        hide = in_rect * inside
    hide = cv2.GaussianBlur(hide, (0, 0), max(0.6, 0.004 * item.width * k))
    return 1 - np.clip(hide, 0, 1)


def render_layer(cutout_pm, items, aspect, region, k, gains=None, shadow_opacity=0.22):
    """Render jewellery over `base` for an upload-space region scaled by k.

    region: (x, y, w, h) in upload pixels. gains: callable(region, k) -> (H,W,3) multiplier or None.
    Returns a function that composites onto a canvas of size (h*k, w*k)."""

    def apply(canvas):
        gain_canvas = gains(region, k, canvas.shape) if gains else None
        patches = []
        for it in items:
            bx0, by0, bx1, by1 = item_bbox(it, aspect)
            margin = max(8 / k, it.width * 0.15)
            if (bx1 + margin < region[0] or bx0 - margin > region[0] + canvas.shape[1] / k
                    or by1 + margin < region[1] or by0 - margin > region[1] + canvas.shape[0] / k):
                continue
            patch, x0, y0 = place_patch(cutout_pm, it, k, origin=(region[0], region[1]))
            if it.occlude:
                patch = patch * occlusion_keep(cutout_pm.shape, it, k, (region[0], region[1]), patch.shape, x0, y0)[..., None]
            patches.append((patch, x0, y0, it))
        for patch, x0, y0, it in patches:
            apply_contact_shadow(canvas, patch, x0, y0, it.width * k, shadow_opacity)
        for patch, x0, y0, it in patches:
            g = None
            if gain_canvas is not None:
                g = np.ones(patch.shape[:2] + (3,), np.float32)
                ov = _overlap(canvas.shape, patch.shape, x0, y0)
                if ov:
                    cs, ps = ov
                    g[ps] = gain_canvas[cs]
            composite_pm(canvas, patch, x0, y0, g)
        return canvas

    return apply


# ----------------------------------------------------------------------------- necklace chain


def dominant_metal_color(cut: np.ndarray) -> np.ndarray:
    """Median colour of the brighter, saturated (metal) pixels of the cutout, RGB in [0,1]."""
    a = cut[..., 3] > 0.9
    rgb = cut[..., :3][a]
    if len(rgb) < 50:
        return np.array([0.83, 0.68, 0.36], np.float32)
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sel = (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.45)
    pick = rgb[sel] if sel.sum() > 50 else rgb
    return np.median(pick, axis=0).astype(np.float32)


def _resample(path: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return path
    u = np.arange(0, s[-1], step)
    return np.stack([np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1])], axis=1)


def draw_chain(canvas, chain: dict, color, region, k):
    """Procedural cable chain along chain['points'] (upload px) with link pitch from chain['thickness'].

    Drawn supersampled so it reads as real links in the close-up tiles and as a fine chain at 1x.
    The chain is synthetic (the catalogue has pendant photos only); the pendant itself stays detail-locked.
    """
    rx, ry = region[0], region[1]
    pts = (np.asarray(chain["points"], np.float32) - [rx, ry]) * k
    t = float(chain.get("thickness", 2.0)) * k
    H, W = canvas.shape[:2]
    pad = 4 * t + 4
    x0, y0 = int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad))
    x1, y1 = int(min(W, pts[:, 0].max() + pad)), int(min(H, pts[:, 1].max() + pad))
    if x1 <= x0 or y1 <= y0:
        return
    ss = 4 if t < 8 else 2
    bw, bh = (x1 - x0) * ss, (y1 - y0) * ss
    q = (pts - [x0, y0]) * ss
    tt = t * ss
    pitch = 1.3 * tt
    path = _resample(q, pitch / 2)
    body = np.zeros((bh, bw), np.float32)
    shine = np.zeros((bh, bw), np.float32)
    lw = max(1, int(round(0.32 * tt)))
    for i in range(0, len(path) - 2, 2):
        c = (path[i] + path[i + 2]) / 2
        d = path[i + 2] - path[i]
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        L = max(1.0, 0.72 * pitch)
        if (i // 2) % 2 == 0:  # link seen face-on: an oval ring
            axes = (int(round(L)), max(1, int(round(0.42 * tt))))
            cv2.ellipse(body, (int(c[0]), int(c[1])), axes, ang, 0, 360, 1.0, lw, cv2.LINE_AA)
            cv2.ellipse(shine, (int(c[0]), int(c[1])), axes, ang, 200, 300, 1.0, max(1, lw // 2), cv2.LINE_AA)
        else:  # link seen edge-on: a short bar
            u = d / (np.linalg.norm(d) + 1e-6) * L
            p0, p1 = c - u, c + u
            cv2.line(body, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, max(1, int(round(0.4 * tt))), cv2.LINE_AA)
            cv2.line(shine, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, max(1, lw // 2), cv2.LINE_AA)
    shadow = cv2.GaussianBlur(np.roll(body, int(0.5 * tt) + 1, axis=0), (0, 0), max(1.0, 0.6 * tt))
    size = (x1 - x0, y1 - y0)
    body = cv2.resize(body, size, interpolation=cv2.INTER_AREA)
    shine = cv2.resize(shine, size, interpolation=cv2.INTER_AREA)
    shadow = cv2.resize(shadow, size, interpolation=cv2.INTER_AREA)
    col = np.asarray(color, np.float32)
    rgb = col * 0.72 + shine[..., None] * np.minimum(col * 1.5, 1.0) * 0.6
    reg = canvas[y0:y1, x0:x1]
    reg *= (1 - 0.3 * np.clip(shadow, 0, 1))[..., None]
    a = np.clip(body, 0, 1)[..., None]
    canvas[y0:y1, x0:x1] = np.clip(reg * (1 - a) + np.clip(rgb, 0, 1) * a, 0, 1)
