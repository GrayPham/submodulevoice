"""ctypes binding for the omnivoice.cpp public C ABI (ov_* symbols).

Mirrors src/omnivoice.h 1:1. Nothing here is meant to be called directly --
use pyomnivoice.OmniVoice.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_uint64,
    c_void_p,
)
from pathlib import Path

OV_ABI_VERSION = 3

OV_STATUS = {
    0: "OK",
    -1: "INVALID_PARAMS",
    -2: "INSTRUCT_INVALID",
    -3: "GENERATE_FAILED",
    -4: "OOM",
    -5: "CANCELLED",
}

LOG_LEVEL = {0: "DEBUG", 1: "INFO", 2: "WARN", 3: "ERROR"}


# --------------------------------------------------------------------- structs


class ov_audio(Structure):
    _fields_ = [
        ("samples", POINTER(c_float)),
        ("n_samples", c_int),
        ("sample_rate", c_int),
        ("channels", c_int),
    ]


class ov_init_params(Structure):
    _fields_ = [
        ("abi_version", c_int),
        ("model_path", c_char_p),
        ("codec_path", c_char_p),
        ("use_fa", c_bool),
        ("clamp_fp16", c_bool),
    ]


ov_cancel_cb = CFUNCTYPE(c_bool, c_void_p)
ov_audio_chunk_cb = CFUNCTYPE(c_bool, POINTER(c_float), c_int, c_void_p)
ov_log_cb = CFUNCTYPE(None, c_int, c_char_p, c_void_p)


class ov_tts_params(Structure):
    _fields_ = [
        ("abi_version", c_int),
        # text / language / voice design
        ("text", c_char_p),
        ("lang", c_char_p),
        ("instruct", c_char_p),
        # duration + long form chunking
        ("T_override", c_int),
        ("chunk_duration_sec", c_float),
        ("chunk_threshold_sec", c_float),
        ("denoise", c_bool),
        ("preprocess_prompt", c_bool),
        # MaskGIT sampler
        ("mg_num_step", c_int),
        ("mg_guidance_scale", c_float),
        ("mg_t_shift", c_float),
        ("mg_layer_penalty_factor", c_float),
        ("mg_position_temperature", c_float),
        ("mg_class_temperature", c_float),
        ("mg_seed", c_uint64),
        # voice reference (tokens XOR raw pcm)
        ("ref_audio_tokens", POINTER(c_int32)),
        ("ref_T", c_int),
        ("ref_audio_24k", POINTER(c_float)),
        ("ref_n_samples", c_int),
        ("ref_text", c_char_p),
        ("dump_dir", c_char_p),
        # callbacks
        ("cancel", ov_cancel_cb),
        ("cancel_user_data", c_void_p),
        ("on_chunk", ov_audio_chunk_cb),
        ("on_chunk_user_data", c_void_p),
        # tail field, abi_version >= 3
        ("postproc", c_bool),
    ]


class ov_voice_ref(Structure):
    _fields_ = [
        ("ref_codes", POINTER(c_int32)),
        ("ref_T", c_int),
        ("num_codebooks", c_int),
    ]


# --------------------------------------------------------------------- loading

_LIBNAME = {
    "win32": "omnivoice.dll",
    "darwin": "libomnivoice.dylib",
}.get(sys.platform, "libomnivoice.so")

_ROOT = Path(__file__).resolve().parent.parent

# Where a build-win.cmd / buildcuda.sh run drops the shared library.
_CANDIDATE_DIRS = [
    _ROOT / "omnivoice.cpp" / "build-cuda",
    _ROOT / "omnivoice.cpp" / "build-cuda" / "bin",
    _ROOT / "omnivoice.cpp" / "build-cpu",
    _ROOT / "omnivoice.cpp" / "build-cpu" / "bin",
    _ROOT / "omnivoice.cpp" / "build" / "bin" / "Release",
    _ROOT / "omnivoice.cpp" / "build" / "bin",
    _ROOT / "omnivoice.cpp" / "build",
    Path(__file__).resolve().parent / "lib",
]


def find_library(explicit: str | os.PathLike | None = None) -> Path:
    """Locate omnivoice.dll / libomnivoice.so.

    Order: explicit argument, $OMNIVOICE_LIB, then the standard build dirs.
    CUDA build wins over the CPU build when both are present.
    """
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            p = p / _LIBNAME
        if not p.exists():
            raise FileNotFoundError(f"omnivoice library not found at {p}")
        return p

    env = os.environ.get("OMNIVOICE_LIB")
    if env:
        return find_library(env)

    for d in _CANDIDATE_DIRS:
        p = d / _LIBNAME
        if p.exists():
            return p

    raise FileNotFoundError(
        f"{_LIBNAME} not found. Build it first:\n"
        f"    build-win.cmd cpu      (or: build-win.cmd cuda)\n"
        f"or point $OMNIVOICE_LIB at the directory holding {_LIBNAME}."
    )


_lib = None
_lib_dir: Path | None = None


def load(explicit: str | os.PathLike | None = None):
    """Load the shared library once and declare every prototype."""
    global _lib, _lib_dir
    if _lib is not None:
        return _lib

    path = find_library(explicit)
    _lib_dir = path.parent

    # ggml.dll / ggml-base.dll / ggml-cpu-*.dll sit next to it.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_lib_dir))

    lib = ctypes.CDLL(str(path))

    lib.ov_version.restype = c_char_p
    lib.ov_version.argtypes = []

    lib.ov_last_error.restype = c_char_p
    lib.ov_last_error.argtypes = []

    lib.ov_audio_free.restype = None
    lib.ov_audio_free.argtypes = [POINTER(ov_audio)]

    lib.ov_init_default_params.restype = None
    lib.ov_init_default_params.argtypes = [POINTER(ov_init_params)]

    lib.ov_init.restype = c_void_p
    lib.ov_init.argtypes = [POINTER(ov_init_params)]

    lib.ov_free.restype = None
    lib.ov_free.argtypes = [c_void_p]

    lib.ov_log_set.restype = None
    lib.ov_log_set.argtypes = [ov_log_cb, c_void_p]

    lib.ov_tts_default_params.restype = None
    lib.ov_tts_default_params.argtypes = [POINTER(ov_tts_params)]

    lib.ov_synthesize.restype = c_int
    lib.ov_synthesize.argtypes = [c_void_p, POINTER(ov_tts_params), POINTER(ov_audio)]

    lib.ov_duration_sec_to_tokens.restype = c_int
    lib.ov_duration_sec_to_tokens.argtypes = [c_void_p, c_float]

    lib.ov_num_codebooks.restype = c_int
    lib.ov_num_codebooks.argtypes = [c_void_p]

    lib.ov_extract_voice_ref.restype = c_int
    lib.ov_extract_voice_ref.argtypes = [
        c_void_p,
        POINTER(c_float),
        c_int,
        POINTER(ov_voice_ref),
    ]

    lib.ov_voice_ref_free.restype = None
    lib.ov_voice_ref_free.argtypes = [POINTER(ov_voice_ref)]

    _lib = lib
    return lib


def lib_dir() -> Path:
    if _lib_dir is None:
        raise RuntimeError("library not loaded yet")
    return _lib_dir
