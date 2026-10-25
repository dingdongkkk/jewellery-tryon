"""Provider adapter interface.

A provider turns (person crop with the jewellery pre-placed, product reference, prompt) into a
generated image of the same framing. The result is only used as lighting / shadow context:
the final jewellery pixels always come from the original product cutout.
"""
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


class ProviderError(Exception):
    """Permanent failure (bad input, model error)."""


class NoImageGeneratedError(ProviderError):
    """Provider completed the request without an image; local lighting is usable."""


class TransientProviderError(ProviderError):
    """Network hiccup, Space restarting, queue full: safe to retry."""


class QuotaExceededError(ProviderError):
    """Hugging Face ZeroGPU daily quota used up."""


@dataclass
class GenerationResult:
    image: Image.Image
    seed: int
    info: dict = field(default_factory=dict)


class TryOnProvider(Protocol):
    name: str

    def generate_tryon(
        self,
        person_image: Image.Image,
        jewellery_reference: Image.Image,
        prompt: str,
        seed: int,
        settings: dict,
    ) -> GenerationResult: ...


QUOTA_MARKERS = (
    "exceeded your gpu quota",
    "exceeded your free gpu quota",
    "zerogpu quota",
    "gpu quota",
    "quota exceeded",
    "you have exceeded",
    "runs limit",
)
TRANSIENT_MARKERS = (
    "connection",
    "timed out",
    "timeout",
    "temporarily",
    "503",
    "502",
    "504",
    "queue is full",
    "space is sleeping",
    "building",
    "restarting",
    "remote end closed",
    "read error",
    "ssl",
    "no gpu was available",
    "gpu task aborted",
)


def classify_exception(exc: Exception) -> ProviderError:
    """Map gradio_client / httpx errors onto our three error kinds."""
    if isinstance(exc, ProviderError):
        return exc
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if any(m in low for m in QUOTA_MARKERS):
        return QuotaExceededError(str(exc))
    if any(m in low for m in TRANSIENT_MARKERS) or type(exc).__name__ in {
        "ConnectError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
        "ReadError",
        "TimeoutError",
        "ConnectionError",
    }:
        return TransientProviderError(msg)
    return ProviderError(msg)
