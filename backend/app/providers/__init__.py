from .base import (  # noqa: F401
    GenerationResult,
    ProviderError,
    QuotaExceededError,
    TransientProviderError,
    TryOnProvider,
)


def get_provider(name: str):
    """Add new adapters here (e.g. a Replicate provider) without touching the pipeline."""
    if name == "qwen-fast":
        from .qwen_hf import QwenFastProvider

        return QwenFastProvider()
    if name == "qwen-official":
        from .qwen_hf import QwenOfficialProvider

        return QwenOfficialProvider()
    if name in ("fal-nano-banana", "fal-nano-banana-2", "fal-qwen-edit"):
        from .fal import FalProvider

        return FalProvider(name)
    if name == "none":
        return None
    raise ValueError(f"unknown provider {name!r}")
