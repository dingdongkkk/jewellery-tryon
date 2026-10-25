"""Automatic placement from body landmarks (MediaPipe Tasks, runs locally on CPU, no quota).

earring -> face mesh: the earlobe sits just outside the face contour, below the tragus level
pendant -> face mesh (chin) + pose (shoulders): pendant hangs below the suprasternal notch,
           with a chain path from both sides of the neck
ring    -> hand landmarks: ring finger, between the knuckle (MCP) and the middle joint (PIP)

Real-world scale comes from the iris (about 11.7 mm across) for faces and from knuckle spacing for hands,
combined with the product's approximate width in mm from the catalogue.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODELS = Path(__file__).resolve().parent.parent / "models"
IRIS_MM = 11.7
DETECT_SIDE = 1600

_lock = threading.RLock()
_models: dict = {}


class NotFound(Exception):
    pass


def _model(kind):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python import BaseOptions, vision

    with _lock:
        if kind not in _models:
            if kind == "face":
                _models[kind] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")), num_faces=1,
                    output_facial_transformation_matrixes=True))
            elif kind == "pose":
                _models[kind] = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_full.task"))))
            elif kind == "hand":
                _models[kind] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")), num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
        return _models[kind]


def _detect(kind, im: Image.Image):
    import mediapipe as mp

    small = im.copy()
    small.thumbnail((DETECT_SIDE, DETECT_SIDE), Image.LANCZOS)
    mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(small)))
    with _lock:
        return _model(kind).detect(mi)


def _pts(landmarks, W, H):
    return np.array([[p.x * W, p.y * H] for p in landmarks], np.float32)


def _center_from_anchor(anchor_pt, anchor, width, aspect, rotation=0.0):
    """Item centre such that the product's normalised anchor lands on anchor_pt."""
    ax, ay = anchor
    dx, dy = (0.5 - ax) * width, (0.5 - ay) * width * aspect
    t = math.radians(rotation)
    return (anchor_pt[0] + dx * math.cos(t) - dy * math.sin(t), anchor_pt[1] + dx * math.sin(t) + dy * math.cos(t))


def _face(im, with_yaw=False):
    W, H = im.size
    res = _detect("face", im)
    ox = oy = 0
    PW, PH = W, H
    if not res.face_landmarks:
        # faces that fill the frame are often missed: retry with the photo shrunk onto a larger canvas
        for pad in (0.35, 0.8):
            PW, PH = int(W * (1 + 2 * pad)), int(H * (1 + 2 * pad))
            canvas = Image.new("RGB", (PW, PH), (128, 128, 128))
            ox, oy = int(W * pad), int(H * pad)
            canvas.paste(im, (ox, oy))
            res = _detect("face", canvas)
            if res.face_landmarks:
                break
    if not res.face_landmarks:
        raise NotFound("No face found. Use a photo where the face is clearly visible.")
    f = _pts(res.face_landmarks[0], PW, PH) - [ox, oy]
    # A turned head foreshortens the iris horizontally but not vertically: use the largest iris diameter.
    iris_px = max(np.linalg.norm(f[469] - f[471]), np.linalg.norm(f[470] - f[472]),
                  np.linalg.norm(f[474] - f[476]), np.linalg.norm(f[475] - f[477]))
    fw = float(np.linalg.norm(f[454] - f[234]))
    if not with_yaw:
        return f, iris_px / IRIS_MM, fw
    yaw = 0.0
    if res.facial_transformation_matrixes:
        R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    return f, iris_px / IRIS_MM, fw, yaw


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def _whole_photo(im, frac=0.06):
    """No face found: send the whole photo and let the AI find the ears (it places the earrings itself)."""
    W, H = im.size
    return {"items": [{"cx": W / 2, "cy": H / 2, "width": frac * min(W, H), "rotation": 0.0, "flip": False}],
            "whole_photo": True, "detected": {"ears": "not found; the AI will find the visible ears", "whole_photo": True}}


def place_earrings(im, product):
    try:
        f, ppm, fw, yaw = _face(im, with_yaw=True)
    except NotFound:
        return _whole_photo(im)
    d = product["dimensions"]
    aspect = d["cutout_h"] / d["cutout_w"]
    width = (d.get("approx_width_mm") or 25) * ppm
    nose = f[1]
    fh = float(np.linalg.norm(f[152] - f[10]))   # forehead to chin: not shrunk by turning the head
    up = (f[10] - f[152]) / max(fh, 1e-6)
    down = -up
    items, sides = [], []
    centre = f[FACE_OVAL].mean(0)
    oval = [[float(v) for v in centre + 0.97 * (f[i] - centre)] for i in FACE_OVAL]
    turned = abs(yaw) > 30
    if not turned:
        # front view: the lobe sits just outside the face contour at ear level; head pitch shifts it
        across = (f[454] - f[234]) / fw
        down_f = np.array([-across[1], across[0]])
        pitch = float(np.dot(f[1] - (f[234] + f[454]) / 2, down_f)) / fw
        lobe_drop = float(np.clip(0.464 - 1.67 * pitch, 0.0, 0.22))
        candidates = [(234, -1, "right ear", f[234] - 0.04 * fw * across + lobe_drop * fw * down_f),
                      (454, 1, "left ear", f[454] + 0.04 * fw * across + lobe_drop * fw * down_f)]
    else:
        # turned head: only the ear on the far side from the nose is visible; it sits behind the face contour
        idx = 234 if abs(f[234][0] - nose[0]) > abs(f[454][0] - nose[0]) else 454
        back = f[idx] - nose
        back = back - np.dot(back, up) * up
        back /= np.linalg.norm(back) + 1e-6
        lobe = f[idx] + 0.07 * fh * back + 0.29 * fh * down
        candidates = [(idx, -1 if idx == 234 else 1, "right ear" if idx == 234 else "left ear", lobe)]
    for idx, sign, name, lobe in candidates:
        anchor = list(product["anchors"]["attach"])
        if sign > 0:
            anchor[0] = 1 - anchor[0]
        cx, cy = _center_from_anchor(lobe, anchor, width, aspect)  # earrings hang with gravity
        item = {"cx": cx, "cy": cy, "width": width, "rotation": 0.0, "flip": sign > 0}
        if not turned:  # in a front view the cheek and jaw are in front of the earlobe
            item["occlude"] = {"polygon": oval, "rect": [0, 0, 1, 1], "always": True}
        items.append(item)
        sides.append(name)
    return {"items": items, "occluder": None if turned else oval,
            "detected": {"ears": sides, "yaw_deg": round(yaw, 1), "view": "profile" if turned else "front",
                         "px_per_mm": round(float(ppm), 3), "face_height_px": round(fh)}}


def place_pendant(im, product):
    W, H = im.size
    f, ppm, fw = _face(im)
    pose = _detect("pose", im)
    chin = f[152]
    if pose.pose_landmarks:
        p = _pts(pose.pose_landmarks[0], W, H)
        sh_mid = (p[11] + p[12]) / 2
        sh_w = float(np.linalg.norm(p[11] - p[12]))
    else:  # no shoulders in frame: estimate from face size
        sh_mid = chin + np.array([0, 0.75 * fw])
        sh_w = 2.4 * fw
    notch = chin + 0.3 * (sh_mid - chin)                 # base of the neck (suprasternal notch)
    drop = max(55 * ppm, 0.12 * sh_w)                   # pendant hangs ~5-6 cm below the notch
    bail = notch + np.array([0, drop])
    d = product["dimensions"]
    aspect = d["cutout_h"] / d["cutout_w"]
    width = (d.get("approx_width_mm") or 30) * ppm
    cx, cy = _center_from_anchor(bail, product["anchors"]["attach"], width, aspect)
    if cy + width * aspect / 2 > H:
        raise NotFound("The chest isn't in the photo. Use a photo that shows your neck and upper chest.")
    # chain: from both sides of the neck, sagging down to the bail
    half_neck = 0.36 * fw
    side_y = notch[1] - 0.12 * fw
    pts = []
    for sx in (-1, 1):
        start = np.array([notch[0] + sx * half_neck, side_y])
        ctrl = np.array([bail[0] + sx * 0.55 * half_neck, bail[1] - 0.15 * (bail[1] - side_y)])
        t = np.linspace(0, 1, 40)[:, None]
        seg = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * bail
        pts.append(seg if sx < 0 else seg[::-1])
    path = np.concatenate([pts[0], pts[1][1:]])
    chain = {"points": path.round(1).tolist(), "thickness": max(1.4, 1.1 * ppm)}
    return {
        "items": [{"cx": cx, "cy": cy, "width": width, "rotation": 0.0, "flip": False}],
        "chain": chain,
        "detected": {"neck_base": notch.round(1).tolist(), "px_per_mm": round(float(ppm), 3)},
    }


def _hands(im):
    """Hand landmarks in image pixels. The palm detector misses hands that fill the frame, so retry with the
    photo shrunk onto a larger neutral canvas."""
    W, H = im.size
    res = _detect("hand", im)
    if res.hand_landmarks:
        return [_pts(h, W, H) for h in res.hand_landmarks]
    for pad in (0.5, 1.0):
        PW, PH = int(W * (1 + 2 * pad)), int(H * (1 + 2 * pad))
        canvas = Image.new("RGB", (PW, PH), (128, 128, 128))
        ox, oy = int(W * pad), int(H * pad)
        canvas.paste(im, (ox, oy))
        res = _detect("hand", canvas)
        if res.hand_landmarks:
            return [_pts(h, PW, PH) - [ox, oy] for h in res.hand_landmarks]
    return []


def _finger_width(arr, p, along, max_px):
    """Measure the finger across, at p, by walking both ways until the colour leaves the finger's own colour."""
    import cv2

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    n = np.array([-along[1], along[0]], np.float32)
    n /= np.linalg.norm(n) + 1e-6
    x, y = int(p[0]), int(p[1])
    if not (2 <= x < W - 2 and 2 <= y < H - 2):
        return None
    ref = lab[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).mean(0)
    ext = []
    for sgn in (-1, 1):
        d = 0
        for d in range(1, int(max_px / 2)):
            q = p + sgn * d * n
            qx, qy = int(round(q[0])), int(round(q[1]))
            if not (0 <= qx < W and 0 <= qy < H):
                break
            c = lab[qy, qx]
            if abs(c[1] - ref[1]) + abs(c[2] - ref[2]) > 14 or abs(c[0] - ref[0]) > 45:
                break
        ext.append(d)
    return float(sum(ext))


def _pose_hand(im):
    """Rough ring position from the body-pose model (wrist, index and pinky points). It still finds hands that
    hold something, when the hand landmarker doesn't. Returns (ring point, finger width, finger angle) or None."""
    W, H = im.size
    pose = _detect("pose", im)
    if not pose.pose_landmarks:
        return None
    p = pose.pose_landmarks[0]
    best = None
    for wrist, pinky, index in ((15, 17, 19), (16, 18, 20)):
        vis = min(p[wrist].visibility, p[pinky].visibility, p[index].visibility)
        if vis < 0.5:
            continue
        w_ = np.array([p[wrist].x * W, p[wrist].y * H])
        k_ = np.array([p[pinky].x * W, p[pinky].y * H])
        i_ = np.array([p[index].x * W, p[index].y * H])
        knuckles = (k_ + i_) / 2
        hand_w = float(np.linalg.norm(i_ - k_))
        along = knuckles - w_
        # the ring finger sits between the middle of the knuckles and the pinky, a little past the knuckle line
        point = knuckles + 0.25 * (k_ - i_) + 0.15 * along
        cand = (point, max(0.28 * hand_w, 6.0), along, vis)
        if best is None or vis > best[3]:
            best = cand
    return best


def place_ring(im, product):
    hands = _hands(im)
    if not hands:
        fallback = _pose_hand(im)
        if fallback is None:
            # Close crops of hands holding things defeat both detectors. The AI places the ring itself, so send it
            # the whole photo with a rough centre hint instead of failing.
            W, H = im.size
            width = 0.08 * min(W, H)
            return {"items": [{"cx": W / 2, "cy": H / 2, "width": width, "rotation": 0.0, "flip": False}],
                    "whole_photo": True,
                    "detected": {"finger": "not found; the AI will find the ring finger", "whole_photo": True}}
        pos, finger_w, v, _ = fallback
        d = product["dimensions"]
        width = finger_w * product["anchors"].get("ring", {}).get("width_over_finger", 1.25)
        rotation = math.degrees(math.atan2(v[1], v[0])) + 90
        return {"items": [{"cx": float(pos[0]), "cy": float(pos[1]), "width": float(width), "rotation": rotation,
                           "flip": False}],
                "detected": {"finger": "ring finger (rough, from body pose)", "finger_width_px": round(float(finger_w), 1)}}
    h = max(hands, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
    mcp, pip = h[13], h[14]
    knuckle = (np.linalg.norm(h[13] - h[9]) + np.linalg.norm(h[13] - h[17])) / 2
    phalanx = float(np.linalg.norm(pip - mcp))
    guess = max(0.82 * knuckle, 0.38 * phalanx)  # knuckle spacing shrinks on a curled / foreshortened hand
    pos = mcp + 0.5 * (pip - mcp)
    measured = _finger_width(np.asarray(im), pos, pip - mcp, 2.5 * guess)
    finger_w = max(measured, guess) if measured and measured < 1.8 * guess else guess
    d = product["dimensions"]
    ring_cfg = product["anchors"].get("ring", {})
    width = finger_w * ring_cfg.get("width_over_finger", 1.25)
    v = pip - mcp
    rotation = math.degrees(math.atan2(v[1], v[0])) + 90  # band across the finger, head towards the fingertip
    aspect = d["cutout_h"] / d["cutout_w"]
    cx, cy = _center_from_anchor(pos, ring_cfg.get("finger_anchor", [0.5, 0.4]), width, aspect, rotation)
    seg_a = mcp - 0.2 * v
    seg_b = pip
    item = {
        "cx": cx, "cy": cy, "width": width, "rotation": rotation, "flip": False,
        # the finger hides the back part of the band
        "occlude": {
            "capsule": [*seg_a.tolist(), *seg_b.tolist(), finger_w / 2],
            "rect": ring_cfg.get("hide_rect", [0, 0.62, 1, 1]),
            "always": bool(ring_cfg.get("hide_always", False)),
        },
    }
    return {"items": [item], "detected": {"finger": "ring finger", "finger_width_px": round(float(finger_w), 1)}}


def _plain(v):
    """numpy scalars / arrays -> JSON-friendly Python values."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def autoplace(image_path, product) -> dict:
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    cat = product["category"]
    fn = {"earring": place_earrings, "pendant": place_pendant, "ring": place_ring}.get(cat)
    if fn is None:
        raise NotFound(f"Automatic placement isn't available for {cat} yet.")
    return _plain(fn(im, product))
