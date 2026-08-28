"""pyomnivoice - Python bindings for omnivoice.cpp.

OmniVoice zero-shot TTS / voice cloning (646 languages, 24 kHz mono),
running on CPU, CUDA, ROCm, Metal or Vulkan through GGML.
"""

from .core import (
    FRAME_RATE,
    PROFILES,
    SAMPLE_RATE,
    Audio,
    OmniVoice,
    OmniVoiceError,
    Voice,
    read_wav_24k,
)

__all__ = [
    "OmniVoice",
    "Voice",
    "Audio",
    "OmniVoiceError",
    "read_wav_24k",
    "PROFILES",
    "SAMPLE_RATE",
    "FRAME_RATE",
]
