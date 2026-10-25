"""Qwen-Image-Edit-2511 on free Hugging Face ZeroGPU Spaces, via gradio_client.

Endpoints were discovered with Client(space).view_api() on 2026-10-25:

* prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast  /edit_image
    (images_b64_json: str, prompt: str, lora_adapter: str, seed: int, randomize_seed: bool,
     guidance_scale: float, steps: int) -> {"image": "data:image/png;base64,...", "seed", "status"}
  Rapid-AIO distilled transformer (4 steps). Output long side is 1024.
* Qwen/Qwen-Image-Edit-2511  /infer
    (images: gallery, prompt, seed, randomize_seed, true_guidance_scale, num_inference_steps,
     height, width, rewrite_prompt) -> (gallery, seed)
  Full model, ~40-50 steps, reserves 180 s of GPU per call (costly on a free quota).
"""
import base64
import io
import json
import threading

from gradio_client import Client, handle_file
from PIL import Image

from ..config import HF_TOKEN, QWEN_FAST_SPACE, QWEN_OFFICIAL_SPACE
from .base import GenerationResult, ProviderError, classify_exception

_clients: dict[str, Client] = {}
_lock = threading.Lock()


def _client(space: str) -> Client:
    with _lock:
        if space not in _clients:
            try:
                _clients[space] = Client(space, token=HF_TOKEN, verbose=False)
            except Exception as e:  # Space asleep / network
                raise classify_exception(e) from e
        return _clients[space]


def _drop_client(space: str):
    with _lock:
        _clients.pop(space, None)


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class QwenFastProvider:
    name = "qwen-fast"

    def generate_tryon(self, person_image, jewellery_reference, prompt, seed, settings):
        images = [_to_data_url(person_image)]
        if settings.get("send_reference", True) and jewellery_reference is not None:
            images.append(_to_data_url(jewellery_reference))
        lora = settings.get("lora_adapter", "Anything2Real")
        try:
            res = _client(QWEN_FAST_SPACE).predict(
                json.dumps(images),
                prompt,
                lora,
                int(seed),
                False,
                float(settings.get("guidance_scale", 1.0)),
                int(settings.get("steps", 4)),
                api_name="/edit_image",
            )
        except Exception as e:
            _drop_client(QWEN_FAST_SPACE)
            raise classify_exception(e) from e
        if not isinstance(res, dict) or res.get("status") != "success" or not res.get("image"):
            msg = res.get("message") if isinstance(res, dict) else str(res)
            raise ProviderError(f"Qwen Space returned no image: {msg}")
        data = res["image"].split(",", 1)[-1]
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        return GenerationResult(img, int(res.get("seed", seed)), {"space": QWEN_FAST_SPACE, "lora": lora})


class QwenOfficialProvider:
    name = "qwen-official"

    def generate_tryon(self, person_image, jewellery_reference, prompt, seed, settings):
        import tempfile

        tmp = []
        try:
            for im in [person_image] + ([jewellery_reference] if settings.get("send_reference", True) else []):
                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                im.convert("RGB").save(f, format="PNG")
                f.close()
                tmp.append(f.name)
            w, h = person_image.size
            k = 1024 / max(w, h)
            res = _client(QWEN_OFFICIAL_SPACE).predict(
                [{"image": handle_file(p), "caption": None} for p in tmp],
                prompt,
                int(seed),
                False,
                float(settings.get("guidance_scale", 4.0)),
                int(settings.get("steps", 20)),
                int(h * k) // 16 * 16,
                int(w * k) // 16 * 16,
                False,
                api_name="/infer",
            )
        except Exception as e:
            _drop_client(QWEN_OFFICIAL_SPACE)
            raise classify_exception(e) from e
        gallery, out_seed = res
        item = gallery[0]["image"]
        path = item["path"] if isinstance(item, dict) else item
        img = Image.open(path).convert("RGB")
        return GenerationResult(img, int(out_seed), {"space": QWEN_OFFICIAL_SPACE})
