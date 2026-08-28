"""High level Python API over omnivoice.cpp.

    from pyomnivoice import OmniVoice

    tts = OmniVoice(backend="cpu")                 # or "cuda" / "auto"
    voice = tts.load_voice("ref.wav", "transcript of ref.wav")
    audio = tts.say("Xin chao...", voice=voice, lang="Vietnamese")
    audio.save("out.wav")
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import time
import wave
from ctypes import POINTER, byref, c_float, c_int32
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import _ffi
from ._ffi import (
    ov_audio,
    ov_audio_chunk_cb,
    ov_cancel_cb,
    ov_init_params,
    ov_log_cb,
    ov_tts_params,
    ov_voice_ref,
)

SAMPLE_RATE = 24_000
FRAME_RATE = 25.0  # codec frames per second (hop 960 @ 24 kHz)

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = _ROOT / "omnivoice.cpp" / "models"

# (backbone, codec) presets, smallest first.
PROFILES = {
    # Nhỏ nhất, dành cho card 2 GB: codec cũng lượng tử hoá 4-bit.
    "tiny": ("omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q4_K_M.gguf"),
    "lite": ("omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q8_0.gguf"),
    "balanced": ("omnivoice-base-Q8_0.gguf", "omnivoice-tokenizer-Q8_0.gguf"),
    "quality": ("omnivoice-base-Q8_0.gguf", "omnivoice-tokenizer-F32.gguf"),
    "reference": ("omnivoice-base-BF16.gguf", "omnivoice-tokenizer-F32.gguf"),
}


class OmniVoiceError(RuntimeError):
    pass


# ------------------------------------------------------------------ audio i/o


@dataclass
class Audio:
    """Mono float32 PCM."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def save(self, path: str | os.PathLike, bits: int = 16) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x = np.clip(self.samples, -1.0, 1.0)
        if bits == 16:
            data = (x * 32767.0).astype("<i2")
        elif bits == 24:
            i32 = (x * 8388607.0).astype("<i4")
            data = i32.view(np.uint8).reshape(-1, 4)[:, :3].copy()
        elif bits == 32:
            data = x.astype("<f4")
        else:
            raise ValueError("bits must be 16, 24 or 32")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(bits // 8)
            w.setframerate(self.sample_rate)
            w.writeframes(data.tobytes())
        return path


def read_wav_24k(path: str | os.PathLike) -> np.ndarray:
    """Decode any audio file to mono float32 @ 24 kHz.

    Uses soundfile+soxr when installed, otherwise falls back to the stdlib
    wave module plus linear interpolation so the package works on a bare
    Python install.
    """
    path = str(path)
    try:
        import soundfile as sf

        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        with contextlib.closing(wave.open(path, "rb")) as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise OmniVoiceError(f"unsupported wav sample width: {sw}")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)

    if sr != SAMPLE_RATE:
        try:
            import soxr

            x = soxr.resample(x, sr, SAMPLE_RATE, quality="VHQ")
        except Exception:
            n = int(round(len(x) * SAMPLE_RATE / sr))
            x = np.interp(
                np.linspace(0.0, len(x) - 1.0, n, dtype=np.float64),
                np.arange(len(x), dtype=np.float64),
                x.astype(np.float64),
            )
    return np.ascontiguousarray(x, dtype=np.float32)


# ------------------------------------------------------------------- voice ref


@dataclass
class Voice:
    """A reusable voice reference: RVQ codes + the reference transcript.

    Built once with OmniVoice.load_voice(), then reused for every sentence:
    the codec encode (HuBERT + DAC + RVQ) is skipped on later calls.
    """

    codes: np.ndarray  # int32 [K, T]
    text: str

    @property
    def n_frames(self) -> int:
        return int(self.codes.shape[1])

    def save(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        np.savez(path, codes=self.codes, text=np.array(self.text))
        return path

    @staticmethod
    def load(path: str | os.PathLike) -> "Voice":
        z = np.load(str(path), allow_pickle=False)
        return Voice(codes=z["codes"].astype(np.int32), text=str(z["text"]))


# ----------------------------------------------------------------------- main


class OmniVoice:
    """OmniVoice TTS: 646 languages, zero-shot voice cloning, 24 kHz mono.

    Parameters
    ----------
    profile : "lite" | "balanced" | "quality" | "reference"
        Model size preset. "lite" (Q4_K_M backbone + Q8_0 codec, ~660 MB
        on disk) is the one for weak machines; "quality" matches the
        INT4-backbone + FP32-Higgs-tokenizer pairing most launchers ship.
    backend : "auto" | "cpu" | "cuda" | "vulkan" | explicit ggml device name
        "auto" picks the best device present. "cpu" forces CPU even when a
        GPU is available. Must be decided before the first synthesis.
    """

    def __init__(
        self,
        model: str | os.PathLike | None = None,
        codec: str | os.PathLike | None = None,
        *,
        profile: str = "lite",
        backend: str = "auto",
        models_dir: str | os.PathLike | None = None,
        lib: str | os.PathLike | None = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
        verbose: bool = False,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        mdir = Path(models_dir) if models_dir else _MODELS
        dflt_model, dflt_codec = PROFILES[profile]
        self.model_path = Path(model) if model else mdir / dflt_model
        self.codec_path = Path(codec) if codec else mdir / dflt_codec
        for p in (self.model_path, self.codec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing. Download the GGUFs with:\n"
                    f"    python -m pyomnivoice.download"
                )

        self._lib = _ffi.load(lib)
        self.verbose = verbose
        self._logs: list[str] = []
        self._install_log_cb()

        # ggml reads GGML_BACKEND at backend init time, inside ov_init.
        dev = {"auto": None, "cpu": "CPU", "cuda": "CUDA0", "vulkan": "Vulkan0"}.get(
            backend, backend
        )
        prev = os.environ.get("GGML_BACKEND")
        if dev is None:
            os.environ.pop("GGML_BACKEND", None)
        else:
            os.environ["GGML_BACKEND"] = dev

        ip = ov_init_params()
        self._lib.ov_init_default_params(byref(ip))
        ip.model_path = str(self.model_path).encode("utf-8")
        ip.codec_path = str(self.codec_path).encode("utf-8")
        ip.use_fa = use_fa
        ip.clamp_fp16 = clamp_fp16

        t0 = time.perf_counter()
        # ggml_backend_load_all() scans the cwd for ggml-cpu-*.dll variants.
        cwd = os.getcwd()
        try:
            os.chdir(_ffi.lib_dir())
            self._ctx = self._lib.ov_init(byref(ip))
        finally:
            os.chdir(cwd)
            if prev is None:
                os.environ.pop("GGML_BACKEND", None)
            else:
                os.environ["GGML_BACKEND"] = prev
        self.load_time = time.perf_counter() - t0

        if not self._ctx:
            raise OmniVoiceError(
                self._lib.ov_last_error().decode("utf-8", "replace")
                or "ov_init failed"
            )
        self.backend = self._detect_backend()

    # -- lifecycle ---------------------------------------------------------

    def _install_log_cb(self) -> None:
        def _cb(level: int, msg: bytes, _user):  # noqa: ANN001
            text = msg.decode("utf-8", "replace").rstrip()
            self._logs.append(text)
            if self.verbose:
                print(text, flush=True)

        self._log_cb = ov_log_cb(_cb)  # keep a ref alive
        self._lib.ov_log_set(self._log_cb, None)

    def _detect_backend(self) -> str:
        for line in self._logs:
            if "backend:" in line:
                return line.split("backend:")[1].split("(")[0].strip()
        return "unknown"

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.ov_free(self._ctx)
            self._ctx = None

    def __enter__(self) -> "OmniVoice":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def version(self) -> str:
        return self._lib.ov_version().decode()

    @property
    def num_codebooks(self) -> int:
        return int(self._lib.ov_num_codebooks(self._ctx))

    def duration_to_frames(self, seconds: float) -> int:
        return int(self._lib.ov_duration_sec_to_tokens(self._ctx, c_float(seconds)))

    # -- voice cloning -----------------------------------------------------

    def load_voice(self, ref_wav: str | os.PathLike, ref_text: str) -> Voice:
        """Encode a reference WAV into reusable RVQ codes.

        ref_wav: 3-10 s of clean speech, any sample rate/channel count.
        ref_text: its exact transcript.
        """
        pcm = read_wav_24k(ref_wav)
        out = ov_voice_ref()
        rc = self._lib.ov_extract_voice_ref(
            self._ctx,
            pcm.ctypes.data_as(POINTER(c_float)),
            len(pcm),
            byref(out),
        )
        self._check(rc, "ov_extract_voice_ref")
        n = out.num_codebooks * out.ref_T
        codes = np.ctypeslib.as_array(out.ref_codes, shape=(n,)).copy()
        codes = codes.reshape(out.num_codebooks, out.ref_T).astype(np.int32)
        self._lib.ov_voice_ref_free(byref(out))
        return Voice(codes=codes, text=ref_text)

    # -- synthesis ---------------------------------------------------------

    def say(
        self,
        text: str,
        *,
        voice: Voice | None = None,
        ref_wav: str | os.PathLike | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        lang: str = "None",
        duration: float | None = None,
        steps: int = 32,
        seed: int | None = None,
        guidance_scale: float | None = None,
        chunk_duration: float = 15.0,
        chunk_threshold: float = 30.0,
        denoise: bool = True,
        preprocess_prompt: bool = True,
        postproc: bool = True,
        on_chunk: Callable[[np.ndarray], bool] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Audio:
        """Synthesise `text`.

        Voice selection, in priority order:
          voice=Voice(...)            pre-encoded clone, cheapest
          ref_wav= + ref_text=        clone straight from a WAV
          instruct="female, young adult, moderate pitch"   voice design
          nothing                     auto voice
        """
        p = ov_tts_params()
        self._lib.ov_tts_default_params(byref(p))

        # keep every buffer referenced until ov_synthesize returns
        keep: list[object] = []

        p.text = text.encode("utf-8")
        p.lang = (lang or "None").encode("utf-8")
        if instruct:
            p.instruct = instruct.encode("utf-8")
        p.chunk_duration_sec = chunk_duration
        p.chunk_threshold_sec = chunk_threshold
        p.denoise = denoise
        p.preprocess_prompt = preprocess_prompt
        p.postproc = postproc
        p.mg_num_step = steps
        if guidance_scale is not None:
            p.mg_guidance_scale = guidance_scale
        if seed is not None:
            p.mg_seed = seed
        if duration is not None:
            p.T_override = self.duration_to_frames(duration)

        if voice is not None:
            codes = np.ascontiguousarray(voice.codes, dtype=np.int32)
            keep.append(codes)
            p.ref_audio_tokens = codes.ctypes.data_as(POINTER(c_int32))
            p.ref_T = int(codes.shape[1])
            p.ref_text = voice.text.encode("utf-8")
        elif ref_wav is not None:
            if not ref_text:
                raise ValueError("ref_text is required together with ref_wav")
            pcm = read_wav_24k(ref_wav)
            keep.append(pcm)
            p.ref_audio_24k = pcm.ctypes.data_as(POINTER(c_float))
            p.ref_n_samples = len(pcm)
            p.ref_text = ref_text.encode("utf-8")

        chunks: list[np.ndarray] = []
        if on_chunk is not None:
            def _chunk_cb(ptr, n, _user):  # noqa: ANN001
                buf = np.ctypeslib.as_array(ptr, shape=(n,)).copy()
                chunks.append(buf)
                return bool(on_chunk(buf))

            cb = ov_audio_chunk_cb(_chunk_cb)
            keep.append(cb)
            p.on_chunk = cb

        if cancel is not None:
            def _cancel_cb(_user):  # noqa: ANN001
                return bool(cancel())

            ccb = ov_cancel_cb(_cancel_cb)
            keep.append(ccb)
            p.cancel = ccb

        out = ov_audio()
        t0 = time.perf_counter()
        rc = self._lib.ov_synthesize(self._ctx, byref(p), byref(out))
        wall = time.perf_counter() - t0
        self._check(rc, "ov_synthesize")
        del keep

        if out.n_samples > 0:
            pcm = np.ctypeslib.as_array(out.samples, shape=(out.n_samples,)).copy()
            sr = out.sample_rate
            self._lib.ov_audio_free(byref(out))
        elif chunks:
            pcm = np.concatenate(chunks)
            sr = SAMPLE_RATE
        else:
            pcm = np.zeros(0, dtype=np.float32)
            sr = SAMPLE_RATE

        audio = Audio(samples=pcm, sample_rate=sr)
        self.last_wall = wall
        self.last_rtf = wall / audio.duration if audio.duration else float("nan")
        return audio

    def stream(
        self, text: str, **kw
    ) -> Iterable[np.ndarray]:
        """Yield audio chunks as they are produced (generator wrapper)."""
        import queue
        import threading

        q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=8)

        def worker() -> None:
            try:
                self.say(text, on_chunk=lambda buf: (q.put(buf), True)[1], **kw)
            finally:
                q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()

    def say_many(
        self, texts: Sequence[str], **kw
    ) -> list[Audio]:
        return [self.say(t, **kw) for t in texts]

    def dub_srt(
        self,
        srt_path: str | os.PathLike,
        *,
        voice: Voice | None = None,
        lang: str = "None",
        steps: int = 32,
        seed: int | None = None,
        instruct: str | None = None,
        progress: Callable[[int, int, "SrtCue"], None] | None = None,
        **kw,
    ) -> Audio:
        """Dub an .srt onto an absolute timeline, ready to mux onto the video.

        Each cue is forced to its exact slot length (T_override) with output
        post-processing off, so nothing drifts; gaps between cues stay silent.
        """
        from .srt import assemble, read_srt

        cues = read_srt(srt_path)
        if not cues:
            raise OmniVoiceError(f"no usable cues in {srt_path}")

        segments = []
        for i, cue in enumerate(cues):
            if progress:
                progress(i + 1, len(cues), cue)
            audio = self.say(
                cue.text,
                voice=voice,
                lang=lang,
                instruct=instruct,
                steps=steps,
                seed=seed,
                duration=cue.slot,
                chunk_duration=0.0,  # single shot, the slot is the duration
                postproc=False,
                **kw,
            )
            segments.append((cue, audio.samples))

        return Audio(assemble(segments, SAMPLE_RATE), SAMPLE_RATE)

    # -- misc --------------------------------------------------------------

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = self._lib.ov_last_error().decode("utf-8", "replace")
            raise OmniVoiceError(f"{what} failed ({_ffi.OV_STATUS.get(rc, rc)}): {msg}")
