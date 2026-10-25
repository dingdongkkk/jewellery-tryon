"""fal.ai providers (paid per image, no daily cap). Needs FAL_KEY in backend/.env.

Endpoints and inputs checked against fal's API pages on 2026-10-25:
* fal-ai/nano-banana/edit        prompt, image_urls[], num_images, seed, aspect_ratio, output_format, sync_mode
                                 -> {images: [{url, ...}], description}            (~$0.04 / image)
* fal-ai/qwen-image-edit-2511    prompt, image_urls[], negative_prompt, image_size, num_inference_steps,
                                 guidance_scale, seed, num_images, output_format, acceleration, sync_mode
                                 -> {images: [{url, width, height}], seed, ...}    (~$0.03 / 1024px image)

Images are sent inline as data URIs and sync_mode=True returns the result inline, so the user's photo is not
put in fal's file storage and the result is not kept in fal's request history.
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from ..config import FAL_KEY
from .base import GenerationResult, ProviderError, NoImageGeneratedError, QuotaExceededError, TransientProviderError

ENDPOINTS = {
    "fal-nano-banana": "fal-ai/nano-banana/edit",
    "fal-nano-banana-2": "fal-ai/nano-banana-2/edit",  # ~$0.08, used as the escalation for hard photos
    "fal-qwen-edit": "fal-ai/qwen-image-edit-2511",
}


NANO_RATIOS = {"21:9": 21 / 9, "16:9": 16 / 9, "3:2": 3 / 2, "4:3": 4 / 3, "5:4": 5 / 4, "1:1": 1.0,
               "4:5": 4 / 5, "3:4": 3 / 4, "2:3": 2 / 3, "9:16": 9 / 16}


def _pad_to_ratio(img: Image.Image):
    """Nano Banana only outputs fixed aspect ratios. Pad the input (edge-replicated) to the nearest one so
    the output keeps the same framing; returns (padded image, ratio name, box of the original inside it)."""
    import numpy as np

    w, h = img.size
    name = min(NANO_RATIOS, key=lambda k: abs(np.log(NANO_RATIOS[k] / (w / h))))
    r = NANO_RATIOS[name]
    if w / h < r:
        W, H = int(round(h * r)), h
    else:
        W, H = w, int(round(w / r))
    x0, y0 = (W - w) // 2, (H - h) // 2
    arr = np.asarray(img.convert("RGB"))
    padded = np.pad(arr, ((y0, H - h - y0), (x0, W - w - x0), (0, 0)), mode="edge")
    return Image.fromarray(padded), name, (x0, y0, x0 + w, y0 + h)


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load(url: str) -> Image.Image:
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def _classify(exc: Exception) -> ProviderError:
    status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    low = msg.lower()
    if "no_media_generated" in low:
        return NoImageGeneratedError("The AI provider returned no image. Local lighting was used for this piece.")
    if status == 401 or "unauthorized" in low or "invalid key" in low:
        return ProviderError("The AI service rejected the API key. Check FAL_KEY in backend/.env.")
    if status in (402, 403) and ("balance" in low or "credit" in low or "locked" in low or "exhausted" in low):
        return QuotaExceededError("The AI service account balance is used up. Add credit to continue.")
    if status in (429, 500, 502, 503, 504) or "timeout" in low or "connect" in low:
        return TransientProviderError(f"AI service: {msg}")
    if status == 422:
        return ProviderError("The AI service refused this photo (its content filter can reject revealing or "
                             "low-neckline photos). Try a different photo.")
    # Validation responses can echo inline input images. Do not send those data
    # URIs into the job database, browser error banner or worker logs.
    return ProviderError(f"The AI service could not complete the request (status {status or 'unknown'}). Press Generate again.")


class FalProvider:
    def __init__(self, name: str):
        self.name = name
        self.endpoint = ENDPOINTS[name]

    def _arguments(self, person_image, jewellery_reference, prompt, seed, settings, aspect=None):
        images = [_data_url(person_image)]
        if jewellery_reference is not None:
            images.append(_data_url(jewellery_reference))
        args = {"prompt": prompt, "image_urls": images, "num_images": 1, "seed": int(seed),
                "output_format": "png", "sync_mode": True}
        if self.name in ("fal-nano-banana", "fal-nano-banana-2"):
            args["aspect_ratio"] = aspect or "auto"
            if self.name == "fal-nano-banana-2":
                args["thinking_level"] = "high"  # follows tricky instructions (palm-up hands, hidden ears) better
        else:
            w, h = person_image.size
            args.update({
                "image_size": {"width": w, "height": h},
                "num_inference_steps": int(settings.get("fal_steps", 28)),
                "guidance_scale": float(settings.get("fal_guidance", 4.5)),
                "acceleration": settings.get("fal_acceleration", "regular"),
                "negative_prompt": "extra jewellery, extra earrings, duplicate, blurry, deformed ear",
            })
        return args

    def verify(self, image, category):
        return verify_worn(image, category)

    def generate_tryon(self, person_image, jewellery_reference, prompt, seed, settings):
        if not FAL_KEY:
            raise ProviderError("AI try-on isn't configured on this server (FAL_KEY is not set).")
        os.environ.setdefault("FAL_KEY", FAL_KEY)
        import fal_client

        box, aspect, send = None, None, person_image
        if self.name in ("fal-nano-banana", "fal-nano-banana-2"):
            send, aspect, box = _pad_to_ratio(person_image)
        try:
            res = fal_client.subscribe(
                self.endpoint,
                arguments=self._arguments(send, jewellery_reference, prompt, seed, settings, aspect),
                client_timeout=240,
            )
        except Exception as e:
            raise _classify(e) from e
        images = (res or {}).get("images") or []
        if not images:
            raise NoImageGeneratedError("The AI provider returned no image. Local lighting was used for this piece.")
        try:
            img = _load(images[0]["url"])
        except Exception as e:
            raise TransientProviderError(f"could not read the AI result: {e}") from e
        if box is not None:  # undo the padding: same framing as the image we were given
            img = img.resize(send.size, Image.LANCZOS).crop(box)
        return GenerationResult(img, int(res.get("seed", seed)), {"endpoint": self.endpoint, "description": res.get("description")})


VERIFY_QUESTIONS = {
    # multiple choice works much better than yes/no (tested on a ring lying in a palm)
    "ring": ("Where exactly is the ring in this photo? Choose one: (a) worn on a finger, its band going around the "
             "finger, (b) lying or resting on the palm or skin, not around any finger, (c) held or on an object, "
             "(d) floating, (e) there is no ring in this photo."),
    "earring": ("Where exactly is the earring in this photo? Choose one: (a) hanging from the earlobe of a real, "
                "visible ear, (b) floating in the air, not attached to an ear, (c) on the face, cheek, lips, neck or "
                "hair instead of an ear, (d) attached to an ear that looks invented or distorted, (e) there is no "
                "earring in this photo."),
    "pendant": ("Where exactly is the pendant in this photo? Choose one: (a) hanging on a chain that goes around the "
                "neck, resting on the chest or clothing, (b) floating or without a chain, (c) somewhere a necklace "
                "could not be (on the face, arm or background), (e) there is no pendant in this photo."),
}


def verify_worn(image: Image.Image, category: str, model: str = "google/gemini-2.5-flash") -> dict:
    """Cheap vision-model check (about $0.0002) that the jewellery in a close-up crop is actually worn.
    Returns {"worn": bool|None, "answer", "reason", "cost"}; worn=None means the check couldn't run (don't block)."""
    if not FAL_KEY:
        return {"worn": None, "reason": "no FAL_KEY"}
    os.environ.setdefault("FAL_KEY", FAL_KEY)
    import json as _json

    import fal_client

    im = image.convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    q = VERIFY_QUESTIONS.get(category, VERIFY_QUESTIONS["earring"])
    prompt = q + ' Reply with JSON only: {"answer": "a|b|c|d|e", "reason": "<one short sentence>"}'
    try:
        res = fal_client.subscribe("openrouter/router/vision", arguments={
            "prompt": prompt, "model": model, "image_urls": [url], "temperature": 0, "max_tokens": 150,
        }, client_timeout=60)
    except Exception as e:  # never block a result because the checker is unavailable
        return {"worn": None, "reason": f"check unavailable: {str(e)[:120]}"}
    text = (res or {}).get("output", "")
    cost = ((res or {}).get("usage") or {}).get("cost")
    try:
        s_, e_ = text.index("{"), text.rindex("}") + 1
        d = _json.loads(text[s_:e_])
        ans = str(d.get("answer", "")).strip().lower()[:1]
        # (e) "none in this crop" is inconclusive (the crop may have landed on something else the model changed)
        worn = True if ans == "a" else (None if ans == "e" else False)
        return {"worn": worn, "answer": ans, "reason": str(d.get("reason", ""))[:200], "cost": cost}
    except Exception:
        return {"worn": None, "reason": f"unreadable answer: {text[:120]}", "cost": cost}
