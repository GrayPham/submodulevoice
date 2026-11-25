"""Server tổng hợp giọng nói Fish Audio Drama-3 (44.1 kHz, INT4 < 2.8 GB VRAM).

    python remote/drama3_server.py --port 8770 --key BIMAT
    python remote/drama3_server.py --engine-dir /content/drama3_engine --models-dir /content/drama3_models --vram-mode 2.8gb

Giao thức HTTP giữ tương thích với `remote/server.py` và `remote/client.py`:
  GET  /health     -> trạng thái GPU, model, sample_rate (44100), workers, formats
  GET  /voices     -> danh sách voice_id đã đăng ký
  POST /voice      -> trích xuất VQ codes 1 lần từ WAV mẫu + transcript
  POST /tts        -> tổng hợp một đoạn -> audio/wav (44.1 kHz)
  POST /tts_batch  -> tổng hợp lô nhiều đoạn -> application/zip (wav/flac/opus/vorbis)
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import io
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DRAMA3_SAMPLE_RATE = 44_100
MAX_TEXT_CHARS = 28_000
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.90
DEFAULT_TOP_K = 4097
MIN_REF_SECONDS = 0.5

try:
    import soundfile as _sf
except Exception:
    _sf = None

FORMATS = {
    "wav": (None, None, "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "vorbis": ("OGG", "VORBIS", "audio/ogg"),
}


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,compute_cap,"
                "utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
        n, tot, free, cc, ug, um = [x.strip() for x in out.split(",")]
        return {
            "name": n,
            "vram_total_mib": int(tot),
            "vram_free_mib": int(free),
            "compute_cap": cc,
            "util_gpu_pct": int(ug),
            "util_mem_pct": int(um),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def host_ram_mib() -> tuple[int, int]:
    """Trả về (total_mib, avail_mib) của RAM hệ thống."""
    if sys.platform.startswith("linux"):
        try:
            vals = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    num = v.strip().split()[0]
                    if num.isdigit():
                        vals[k] = int(num) // 1024
            return vals.get("MemTotal", 0), vals.get("MemAvailable", vals.get("MemFree", 0))
        except Exception:
            return 0, 0
    if os.name == "nt":
        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys // (1024**2)), int(st.ullAvailPhys // (1024**2))
    return 0, 0


def _pick_free_port(preferred: int = 18199) -> int:
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return int(s.getsockname()[1])
        except OSError:
            continue
    return preferred


def _find_binary(engine_dir: Path, base_name: str) -> Path | None:
    name = f"{base_name}.exe" if os.name == "nt" else base_name
    direct = engine_dir / name
    if direct.is_file():
        return direct
    for p in engine_dir.rglob(name):
        if p.is_file():
            return p
    return None


def find_drama3_model(models_dir: Path, explicit_model: str | None = None) -> Path:
    if explicit_model:
        p = Path(explicit_model).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy model GGUF: {p}")
        return p

    models_dir = Path(models_dir).resolve()
    candidates = [
        g for g in models_dir.rglob("*.gguf")
        if g.is_file() and "dac" not in g.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy file .gguf Drama-3 nào trong {models_dir}. "
            "Hãy chạy `python remote/prepare_drama3_int4.py` trước."
        )

    for g in candidates:
        if g.name == "drama-3-slim-int4-q2k.gguf":
            return g
    for g in candidates:
        low = g.name.lower()
        if "slim-int4" in low or "int4" in low:
            return g
    for g in candidates:
        if "slim" in g.name.lower():
            return g
    return candidates[0]


def collect_lib_dirs(engine_dir: Path) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    pat = "*.dll" if os.name == "nt" else "*.so*"
    for p in engine_dir.rglob(pat):
        d = str(p.parent.resolve())
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    if sys.platform.startswith("linux"):
        for cand in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ):
            if os.path.isdir(cand) and cand not in seen:
                seen.add(cand)
                dirs.append(cand)
    return dirs


def select_vram_options(
    mode: str,
    gpu_free_mib: int,
    ram_total_mib: int,
    ram_avail_mib: int,
    is_slim: bool = True,
) -> tuple[str, dict]:
    """Chọn cấu hình session_options phù hợp nhất với VRAM GPU & RAM hệ thống."""
    chosen = (mode or "auto").lower()
    if chosen == "auto":
        if gpu_free_mib >= 20_000:
            chosen = "gpu"
        elif gpu_free_mib >= 11_000:
            chosen = "t4"
        else:
            chosen = "2.8gb"

    # Nếu bật chế độ offload CPU nhưng RAM máy nhỏ (vd Colab Free 12.7 GB),
    # bật disk_experts để đọc theo nhu cầu, tránh Linux OOM killer.
    low_host_ram = 0 < ram_avail_mib < 11_000 or (0 < ram_total_mib < 14_500)

    if chosen == "2.3gb":
        opts = {
            "cpu_expert_layers": 40,
            "disk_experts": bool(low_host_ram),
            "codec_backend": "cpu",
            "codec_async_decode": True,
            "codec_batch_frames": 8,
            "backbone_prefill_cache_mb": 0,
            "tower_prefill_cache_mb": 0,
            "context_length": 1536,
            "perf_mode": "cuda_perf",
        }
    elif chosen == "3.3gb":
        opts = {
            "cpu_expert_layers": 40,
            "disk_experts": bool(low_host_ram),
            "codec_backend": "same",
            "codec_async_decode": True,
            "codec_batch_frames": 4,
            "backbone_prefill_cache_mb": 0,
            "tower_prefill_cache_mb": 0,
            "context_length": 1536,
            "perf_mode": "cuda_perf",
        }
    elif chosen == "t4":
        # T4 15 GB VRAM + ~12.7 GB RAM:
        # Đặt 34-36 tầng expert ra CPU/disk để VRAM chỉ quanh ~6-8 GB (dư ~7 GB
        # chống OOM khi sinh câu dài), đồng thời giữ System RAM dưới ~8 GB.
        # Tầng expert đẩy ra CPU được GIỮ TRONG RAM máy (model 15.88 GiB, 40 tầng
        # ~ 13-14 GiB), nên không thể đẩy hết trên Colab 12.7 GB RAM. Docs audio.cpp
        # khuyến nghị 24 tầng cho GPU 16 GB (~8 GiB RAM, VRAM ~8-9 GB).
        # RAM chật -> đọc expert từ disk để khỏi bị OOM killer.
        ram_ample = ram_avail_mib >= 9_000
        opts = {
            "cpu_expert_layers": 24 if ram_ample else 28,
            "disk_experts": not ram_ample and bool(low_host_ram),
            "codec_backend": "cpu",
            "codec_async_decode": True,
            "codec_batch_frames": 8,
            "backbone_prefill_cache_mb": 0,
            "tower_prefill_cache_mb": 0,
            "context_length": 1536,
            "perf_mode": "cuda_perf",
        }
    elif chosen == "gpu":
        opts = {
            "cpu_expert_layers": 4,
            "disk_experts": False,
            "codec_backend": "same",
            "codec_async_decode": True,
            "codec_batch_frames": 4,
            "backbone_prefill_cache_mb": 128,
            "tower_prefill_cache_mb": 128,
            "context_length": 2048,
            "perf_mode": "cuda_perf",
        }
    else:  # "2.8gb" chuẩn mặc định < 2.8 GB VRAM
        opts = {
            "cpu_expert_layers": 38,
            "disk_experts": bool(low_host_ram),
            "codec_backend": "cpu",
            "codec_async_decode": True,
            "codec_batch_frames": 8,
            "backbone_prefill_cache_mb": 0,
            "tower_prefill_cache_mb": 0,
            "context_length": 1536,
            "perf_mode": "cuda_perf",
        }

    if opts.get("disk_experts"):
        opts["expert_cache_size"] = 16
    return chosen, opts


_MOE_KEYS = {
    "cpu_expert_layers": "moe_cpu_expert_layers",
    "disk_experts": "moe_disk_experts",
    "expert_cache_size": "moe_expert_cache_size",
}


def canonical_session_options(opts: dict) -> dict:
    """Đổi sang tên khoá chuẩn của audio.cpp v0.9.6 (docs/models/fish_drama3.md):
    điều khiển MoE dùng `moe_*`, tuỳ chọn riêng của Drama-3 dùng tiền tố `fish_drama3.`.
    Thêm chunk cho reference-encoder để giảm VRAM cấp phát khi encode giọng mẫu."""
    out: dict = {}
    for k, v in opts.items():
        out[_MOE_KEYS.get(k) or f"fish_drama3.{k}"] = v
    out.setdefault("fish_drama3.codec_encoder_chunk_samples", 65536)
    return out


_VOICE_AUDIO_KEYS = (
    "wav_b64", "audio_b64", "wav_base64", "audio_base64", "ref_audio_b64",
    "reference_audio_b64", "file_b64", "audio", "wav", "data",
)


def pick_voice_audio(d: dict) -> str | None:
    """Lấy chuỗi base64 của wav mẫu từ body /voice; chịu được vài tên trường và data-URI."""
    for k in _VOICE_AUDIO_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v:
            if v.startswith("data:") and "," in v:
                v = v.split(",", 1)[1]
            return v
    return None


def pick_instruct(d: dict) -> str | None:
    """Cảm xúc/phong cách: chấp nhận instruct | emotion | style | direction."""
    for k in ("instruct", "emotion", "style", "direction"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def strip_structural_tokens(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<\|.*?\|>", "", text).strip()


# Mô tả tiếng Việt -> thẻ tiếng Anh (model hiểu thẻ tiếng Anh chắc hơn). Thứ tự = ưu tiên hiển thị.
_VI_EMOTION = [
    ("nhẹ nhàng", "gentle, soft voice"), ("dịu dàng", "gentle, soft voice"),
    ("êm ái", "soft voice"), ("êm", "soft voice"), ("thì thầm", "whispering"),
    ("thầm thì", "whispering"), ("buồn", "sad"), ("khóc", "crying"),
    ("nghẹn ngào", "sobbing"), ("giận dữ", "angry"), ("giận", "angry"),
    ("tức giận", "angry"), ("hét", "shouting"), ("quát", "shouting"),
    ("vui vẻ", "cheerful"), ("vui", "cheerful"), ("hào hứng", "excited"),
    ("phấn khích", "excited"), ("cười", "laughing"), ("sợ hãi", "scared"),
    ("sợ", "scared"), ("run rẩy", "trembling"), ("mệt mỏi", "tired"),
    ("thở dài", "sighing"), ("lạnh lùng", "cold"), ("nghiêm túc", "serious"),
    ("chậm rãi", "slow, calm"), ("chậm", "slow"), ("nhanh", "fast"),
    ("trầm", "low, deep voice"), ("ấm áp", "warm"), ("ấm", "warm"),
    ("tự tin", "confident"), ("mỉa mai", "sarcastic"), ("kể chuyện", "storytelling"),
    ("trìu mến", "affectionate"), ("lo lắng", "anxious"), ("ngạc nhiên", "surprised"),
    ("hồi hộp", "nervous"), ("bình tĩnh", "calm"), ("thân mật", "intimate, close"),
]


def vi_to_tags(note: str) -> str:
    low = note.lower()
    found: list[str] = []
    for vi, en in _VI_EMOTION:
        if re.search(rf"(?<!\w){re.escape(vi)}(?!\w)", low):
            for t in en.split(", "):
                if t not in found:
                    found.append(t)
    return ", ".join(found) if found else note


def build_directed_text(text: str, instruct: str | None) -> str:
    clean_text = strip_structural_tokens(text)
    note = strip_structural_tokens(instruct or "")
    note = re.sub(r"[\[\]<>|]", "", note)
    note = re.sub(r"\s+", " ", note).strip()[:140]
    if not note:
        return clean_text
    note = vi_to_tags(note)
    # Drama-3 hiểu thẻ ngoặc vuông nội dòng (vd [angry], [whispering], [sighs]).
    # Một dòng duy nhất: tách thành nhiều dòng sẽ bị gắn <|speaker:0|> riêng và bị ĐỌC TO.
    clean_text = re.sub(r"\s*\n+\s*", " ", clean_text)
    return f"[{note}] {clean_text}"


def format_speaker_tagged_text(raw_text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n+", raw_text.strip()) if p.strip()]
    tagged = []
    for p in paragraphs:
        if "<|speaker:" in p:
            tagged.append(p)
        else:
            tagged.append(f"<|speaker:0|> {p}")
    return "\n".join(tagged) if tagged else "<|speaker:0|> "


def apply_edge_fade_44k(
    wav: np.ndarray,
    sr: int = DRAMA3_SAMPLE_RATE,
    fade_ms: float = 12.0,
    pad_ms: float = 25.0,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.size == 0:
        return wav
    fade_len = min(int(sr * fade_ms / 1000.0), wav.size // 4)
    if fade_len > 1:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        wav = wav.copy()
        wav[:fade_len] *= ramp
        wav[-fade_len:] *= ramp[::-1]
    pad_len = int(sr * pad_ms / 1000.0)
    if pad_len > 0:
        pad = np.zeros(pad_len, dtype=np.float32)
        wav = np.concatenate([pad, wav, pad])
    return np.ascontiguousarray(wav, dtype=np.float32)


def _resample_audio(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    try:
        import soxr

        return np.ascontiguousarray(
            soxr.resample(np.ascontiguousarray(x, dtype=np.float32), src_sr, dst_sr),
            dtype=np.float32,
        )
    except ImportError:
        n = int(round(len(x) * dst_sr / src_sr))
        return np.ascontiguousarray(
            np.interp(
                np.linspace(0.0, len(x) - 1.0, n, dtype=np.float64),
                np.arange(len(x), dtype=np.float64),
                x.astype(np.float64),
            ),
            dtype=np.float32,
        )


def read_wav_bytes_44k(b: bytes) -> np.ndarray:
    """Giải mã WAV/audio mẫu -> mono float32 44.1 kHz, chuẩn hoá -1.5 dBFS cho VQ encoder."""
    if _sf is not None:
        x, sr = _sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
    else:
        with wave.open(io.BytesIO(b), "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError("Chỉ nhận WAV 16-bit hoặc 32-bit")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)

    x = _resample_audio(np.ascontiguousarray(x, dtype=np.float32), int(sr), DRAMA3_SAMPLE_RATE)
    if len(x) < int(DRAMA3_SAMPLE_RATE * MIN_REF_SECONDS):
        raise ValueError(
            f"Giọng mẫu quá ngắn ({len(x) / DRAMA3_SAMPLE_RATE:.2f}s), cần ít nhất {MIN_REF_SECONDS}s"
        )
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk > 1e-4:
        x = np.ascontiguousarray(x * (0.84 / pk), dtype=np.float32)
    return x


def wav_bytes(pcm: np.ndarray, sr: int = DRAMA3_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def decode_wav_bytes(b: bytes) -> tuple[np.ndarray, int]:
    if _sf is not None:
        x, sr = _sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), int(sr)
    with wave.open(io.BytesIO(b), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def encode_audio(pcm: np.ndarray, fmt: str, sr: int = DRAMA3_SAMPLE_RATE) -> tuple[bytes, str]:
    fmt = (fmt or "wav").lower()
    if fmt == "wav" or _sf is None or fmt not in FORMATS:
        return wav_bytes(pcm, sr=sr), "wav"
    container, subtype, _ = FORMATS[fmt]
    buf = io.BytesIO()
    _sf.write(buf, pcm, sr, format=container, subtype=subtype)
    return buf.getvalue(), ("flac" if fmt == "flac" else "ogg")


class Drama3Engine:
    """Giữ một tiến trình audiocpp_server nội bộ nạp sẵn Drama-3 INT4 và xử lý tác vụ."""

    def __init__(
        self,
        server_bin: Path,
        model_path: Path,
        engine_dir: Path,
        session_options: dict,
        threads: int = 6,
        backend: str = "cuda",
        load_mode: str = "streaming",
    ) -> None:
        self.server_bin = Path(server_bin).resolve()
        self.model_path = Path(model_path).resolve()
        self.engine_dir = Path(engine_dir).resolve()
        self.session_options = dict(session_options)
        self.threads = int(threads)
        self.backend = backend
        self.load_mode = load_mode
        self.port = _pick_free_port(18199)
        self.model_id = "fish_drama3_int4"
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._env = self._build_env()
        self._start_and_load()

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        lib_dirs = collect_lib_dirs(self.engine_dir)
        if sys.platform.startswith("linux") and lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
            )
        elif os.name == "nt" and lib_dirs:
            env["PATH"] = os.pathsep.join(lib_dirs + [env.get("PATH", "")])
        return env

    def _http_json(self, path: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=("POST" if data else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"audiocpp_server HTTP {e.code} ({path}): {body[:500]}") from e

    def _http_sse_task(self, payload: dict, timeout: float = 900.0) -> dict:
        url = f"http://127.0.0.1:{self.port}/v1/tasks/stream"
        req_body = {**payload, "stream_format": "sse"}
        req = urllib.request.Request(
            url,
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                done_result = None
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    ev = json.loads(data_str)
                    ev_type = ev.get("type")
                    if ev_type == "error":
                        err = ev.get("error") or {}
                        raise RuntimeError(err.get("message") or str(ev))
                    if ev_type == "task.stream.done":
                        done_result = ev.get("result")
                        break
                if done_result is None:
                    raise RuntimeError("audiocpp_server stream kết thúc nhưng không có task.stream.done")
                return done_result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"audiocpp_server SSE HTTP {e.code}: {body[:500]}") from e

    def _run_task(self, request_obj: dict, timeout: float = 900.0) -> dict:
        payload = {
            "model": self.model_id,
            "request": request_obj,
            "include_artifacts": True,
        }
        if self.load_mode == "offline":
            return self._http_json("/v1/tasks/run", payload, timeout=timeout)
        return self._http_sse_task(payload, timeout=timeout)

    def _start_and_load(self) -> None:
        self.close()
        self.port = _pick_free_port(self.port)
        cmd = [
            str(self.server_bin),
            "--ui",
            "--ui-management",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--backend", self.backend,
            "--threads", str(self.threads),
            "--busy-timeout-ms", "600000",
        ]
        print(
            f"[drama3] Bật audiocpp_server nội bộ (cổng {self.port}, mode={self.load_mode}, "
            f"opts={self.session_options})...",
            flush=True,
        )
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.server_bin.parent),
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _pump(p: subprocess.Popen) -> None:
            assert p.stdout is not None
            for line in p.stdout:
                s = line.rstrip()
                sl = s.lower()
                if s and any(
                    w in sl
                    for w in ("error", "load", "listening", "alloc", "vram", "mib", "expert", "option", "warn", "fail")
                ):
                    print(f"[audiocpp] {s}", flush=True)

        threading.Thread(target=_pump, args=(self.proc,), daemon=True).start()

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"audiocpp_server thoát sớm (rc={self.proc.returncode})")
            try:
                self._http_json("/health", timeout=2.0)
                break
            except Exception:
                time.sleep(0.4)
        else:
            raise RuntimeError("audiocpp_server không mở cổng /health sau 60s")

        t0 = time.monotonic()
        print(f"[drama3] Đang nạp model {self.model_path.name}...", flush=True)
        load_body = {
            "id": self.model_id,
            "family": "fish_drama3",
            "task": "clon",
            "mode": self.load_mode,
            "path": str(self.model_path),
            "session_options": canonical_session_options(self.session_options),
        }
        legacy_opts = dict(self.session_options)
        attempts = [
            (load_body["mode"], load_body["session_options"]),
            (load_body["mode"], legacy_opts),
        ]
        if self.load_mode == "streaming":
            attempts.append(("offline", legacy_opts))
        last_exc: Exception | None = None
        for i, (mode, sopts) in enumerate(attempts):
            load_body["mode"] = mode
            load_body["session_options"] = sopts
            try:
                if i:
                    print(f"[drama3] Thử nạp lại (mode={mode}, opts={sopts})", flush=True)
                self._http_json("/v1/models/load", load_body, timeout=600.0)
                self.load_mode = mode
                self.applied_options = sopts
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[drama3] Nạp lỗi ({exc}) với opts={sopts}", flush=True)
        if last_exc is not None:
            raise last_exc
        try:
            ms = self._http_json("/v1/models?include_session_options=true", None, timeout=10.0)
            print(f"[drama3] Engine báo session_options đang áp dụng: {json.dumps(ms)[:1200]}", flush=True)
        except Exception:  # noqa: BLE001
            pass
        print(f"[drama3] Nạp model xong sau {time.monotonic() - t0:.1f}s (mode={self.load_mode}).", flush=True)
        _gi = gpu_info()
        if "vram_total_mib" in _gi:
            print(
                f"[drama3] VRAM sau nạp: dùng {_gi['vram_total_mib'] - _gi['vram_free_mib']} / "
                f"{_gi['vram_total_mib']} MiB | opts={self.session_options}",
                flush=True,
            )

    def ensure_alive(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._http_json("/health", timeout=2.0)
                return
            except Exception:
                pass
        print("[drama3] Tiến trình nội bộ bị ngắt -> đang khởi động lại...", flush=True)
        self._start_and_load()

    def encode_reference(self, ref_wav_path: Path, ref_text: str) -> str:
        with self.lock:
            self.ensure_alive()
            res = self._run_task(
                {
                    "text": "hi",
                    "voice_ref": str(ref_wav_path.resolve()),
                    "reference_text": ref_text.strip(),
                    "options": {"encode_reference_only": True},
                },
                timeout=300.0,
            )
            artifacts = res.get("artifacts") or []
            if not artifacts or "payload" not in artifacts[0]:
                raise RuntimeError(f"Không trích xuất được VQ codes từ phản hồi: {list(res.keys())}")
            art = json.loads(base64.b64decode(artifacts[0]["payload"]).decode("utf-8"))
            return json.dumps(
                [{"text": ref_text.strip() or art.get("text", ""), "vq_codes": art["vq_codes"]}],
                ensure_ascii=False,
            )

    def synthesize(
        self,
        text: str,
        vq_json: str | None,
        instruct: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        seed: int = 42,
        timeout: float = 900.0,
    ) -> tuple[np.ndarray, int, float]:
        directed = build_directed_text(text, instruct)
        tagged = format_speaker_tagged_text(directed)
        req_opts: dict = {
            "text_chunk_size": 380,
            "text_chunk_mode": "tag_aware",
            "max_tokens": 900,
            "skip_empty_chunks": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "seed": int(seed),
        }
        if vq_json:
            req_opts["references"] = vq_json

        with self.lock:
            self.ensure_alive()
            print(f"[drama3] >> Bắt đầu render đoạn {len(text)} ký tự "
                  f"(có giọng mẫu={'có' if vq_json else 'không'})...", flush=True)
            t0 = time.monotonic()
            res = self._run_task({"text": tagged, "options": req_opts}, timeout=timeout)
            wall = time.monotonic() - t0
            print(f"[drama3] << Render xong sau {wall:.1f}s", flush=True)

        _gi = gpu_info()
        if "vram_total_mib" in _gi:
            print(
                f"[drama3] VRAM sau render: dùng {_gi['vram_total_mib'] - _gi['vram_free_mib']} / "
                f"{_gi['vram_total_mib']} MiB",
                flush=True,
            )

        audio_b64 = res.get("audio")
        if not audio_b64:
            raise RuntimeError(f"audiocpp_server không trả về audio: {list(res.keys())}")
        wav_raw = base64.b64decode(audio_b64)
        pcm, sr = decode_wav_bytes(wav_raw)
        if sr != DRAMA3_SAMPLE_RATE:
            pcm = _resample_audio(pcm, sr, DRAMA3_SAMPLE_RATE)
            sr = DRAMA3_SAMPLE_RATE
        pcm = apply_edge_fade_44k(pcm, sr=sr)
        return pcm, sr, wall

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            except Exception:
                pass
            self.proc = None


@dataclass
class Job:
    text: str
    voice_id: str | None
    instruct: str | None
    temperature: float
    top_p: float
    top_k: int
    seed: int
    done: threading.Event = field(default_factory=threading.Event)
    audio: np.ndarray | None = None
    error: str | None = None
    wall: float = 0.0
    queued_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0


class Drama3Pool:
    def __init__(
        self,
        engine: Drama3Engine,
        vram_mode: str,
        cache_dir: Path,
        default_vq_json: str | None = None,
    ) -> None:
        self.engine = engine
        self.vram_mode = vram_mode
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_vq_json = default_vq_json
        self.q: "queue.Queue[Job | None]" = queue.Queue()
        self.voices: dict[str, tuple[str, str]] = {}
        self.lock = threading.Lock()
        self.busy = 0
        self.served = 0
        self.failed = 0
        self.audio_sec = 0.0
        self.synth_sec = 0.0
        self.started = time.time()

        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def add_voice(self, name: str, pcm_44k: np.ndarray, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Fish Drama-3 yêu cầu lời thoại mẫu (text) khớp với file giọng mẫu.")

        pcm_bytes = wav_bytes(pcm_44k, sr=DRAMA3_SAMPLE_RATE)
        sig = hashlib.sha1(pcm_bytes + b"|" + clean_text.encode("utf-8")).hexdigest()[:16]
        cache_file = self.cache_dir / f"vq_{sig}.json"

        if cache_file.exists():
            vq_json = cache_file.read_text(encoding="utf-8")
        else:
            tmp_wav = self.cache_dir / f"_ref_{sig}.wav"
            tmp_wav.write_bytes(pcm_bytes)
            try:
                vq_json = self.engine.encode_reference(tmp_wav, clean_text)
                cache_file.write_text(vq_json, encoding="utf-8")
            finally:
                tmp_wav.unlink(missing_ok=True)

        vid = uuid.uuid4().hex[:12]
        with self.lock:
            self.voices[vid] = (name, vq_json)
        print(f"[drama3] Đăng ký giọng '{name}' -> {vid} (cache={sig})", flush=True)
        return vid

    def submit(self, job: Job) -> None:
        self.q.put(job)

    def _run_worker(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            with self.lock:
                self.busy += 1
            job.started_at = time.perf_counter()
            try:
                vq_json = self.default_vq_json
                if job.voice_id:
                    with self.lock:
                        entry = self.voices.get(job.voice_id)
                    if entry is None:
                        raise KeyError(f"voice_id khong ton tai: {job.voice_id}")
                    vq_json = entry[1]
                pcm, sr, wall = self.engine.synthesize(
                    text=job.text,
                    vq_json=vq_json,
                    instruct=job.instruct,
                    temperature=job.temperature,
                    top_p=job.top_p,
                    top_k=job.top_k,
                    seed=job.seed,
                )
                dur = len(pcm) / sr
                job.audio = pcm
                job.wall = wall
                with self.lock:
                    self.served += 1
                    self.audio_sec += dur
                    self.synth_sec += wall
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
                with self.lock:
                    self.failed += 1
            finally:
                with self.lock:
                    self.busy -= 1
                job.done.set()

    def stats(self) -> dict:
        with self.lock:
            return {
                "engine": "fish_drama3",
                "model": self.engine.model_path.name,
                "vram_mode": self.vram_mode,
                "cpu_expert_layers": self.engine.session_options.get("cpu_expert_layers"),
                "sample_rate": DRAMA3_SAMPLE_RATE,
                "workers": 1,
                "busy": self.busy,
                "queued": self.q.qsize(),
                "served": self.served,
                "failed": self.failed,
                "audio_sec": round(self.audio_sec, 1),
                "synth_sec": round(self.synth_sec, 1),
                "xrt_aggregate": (
                    round(self.audio_sec / self.synth_sec, 2) if self.synth_sec else None
                ),
                "uptime_sec": round(time.time() - self.started),
                "voices": {k: v[0] for k, v in self.voices.items()},
                "backend": self.engine.backend.upper(),
                "formats": (sorted(FORMATS) if _sf else ["wav"]),
            }


POOL: Drama3Pool | None = None
API_KEY = ""

# ── Việc chạy nền (async) ────────────────────────────────────────────────────
#
# VÌ SAO CÓ: đường hầm Cloudflare cắt mọi request không kịp trả lời trong ~100 giây
# (HTTP 524 / 502 kèm trang HTML). Drama-3 INT4 trên T4 đọc chậm (~10 giây máy cho 1 giây
# tiếng), một đoạn 25 giây tiếng mất gần 5 phút → request bị cắt, server vẫn đọc xong rồi
# ghi kết quả vào socket đã chết: tốn cả GPU mà không ai nhận được gì.
#
# CÁCH: client gửi POST /tts hoặc /tts_batch kèm `"async": true` → server trả NGAY
# 202 {"job_id": ...}; client hỏi GET /job/<id> mỗi vài giây (mỗi lần chỉ vài mili-giây):
#   202 {"status": "running"}   — đang làm
#   200 + nội dung thật         — xong (wav / zip, y hệt bản đồng bộ)
#   4xx/5xx + {"error": ...}    — hỏng (y hệt bản đồng bộ)
#   410                         — không có job này (server khởi động lại / quá hạn)
# Server cũ không biết `async` thì cứ trả đồng bộ như trước — client tự nhận ra.
ASYNC_JOBS: dict[str, dict] = {}
ASYNC_LOCK = threading.Lock()
ASYNC_KEEP_SEC = 1800.0


def _async_gc() -> None:
    now = time.time()
    with ASYNC_LOCK:
        for jid in [j for j, v in ASYNC_JOBS.items() if v["finished"] and now - v["finished"] > ASYNC_KEEP_SEC]:
            ASYNC_JOBS.pop(jid, None)


class _Shadow:
    """Thay cho Handler khi chạy nền: gom phản hồi của route vào bộ nhớ thay vì ghi ra socket."""

    _is_shadow = True
    path = ""

    def __init__(self, route: str, body: dict):
        self.path = route
        self._parsed = body
        self.code = 200
        self.headers_out: dict[str, str] = {}
        self.wfile = io.BytesIO()

    def _auth(self) -> bool:
        return True

    def _body(self) -> dict:
        return self._parsed

    def send_response(self, code, *_a) -> None:
        self.code = int(code)

    def send_header(self, k, v) -> None:
        self.headers_out[str(k)] = str(v)

    def end_headers(self) -> None:
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.code = code
        self.headers_out["Content-Type"] = ctype
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _run_async(job_id: str, route: str, body: dict) -> None:
    shadow = _Shadow(route, body)
    try:
        Handler.do_POST(shadow)  # type: ignore[arg-type]
        res = {
            "code": shadow.code,
            "headers": shadow.headers_out,
            "body": shadow.wfile.getvalue(),
        }
    except Exception as e:  # noqa: BLE001
        res = {
            "code": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode("utf-8"),
        }
    with ASYNC_LOCK:
        ASYNC_JOBS[job_id].update(result=res, finished=time.time())
    print(f"[drama3] job {job_id} xong (HTTP {res['code']}, {len(res['body'])} byte)", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "drama3-remote/1"

    def log_message(self, fmt, *args):
        code = str(args[1]) if len(args) > 1 else ""
        if not code.startswith("2"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _auth(self) -> bool:
        if not API_KEY or self.headers.get("X-API-Key") == API_KEY:
            return True
        self._json(401, {"error": "thieu hoac sai X-API-Key"})
        return False

    def _body(self) -> dict:
        parsed = getattr(self, "_parsed", None)
        if parsed is not None:
            return parsed
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _start_async(self, route: str, d: dict) -> None:
        """Nhận việc, trả 202 + job_id ngay; làm ở luồng nền (xem ASYNC_JOBS)."""
        _async_gc()
        job_id = uuid.uuid4().hex[:12]
        body = {k: v for k, v in d.items() if k != "async"}
        with ASYNC_LOCK:
            ASYNC_JOBS[job_id] = {"route": route, "started": time.time(), "finished": 0.0, "result": None}
        threading.Thread(target=_run_async, args=(job_id, route, body), daemon=True).start()
        print(f"[drama3] job {job_id} nhận ({route}) — chạy nền", flush=True)
        self._json(202, {"job_id": job_id})

    def _job_status(self, job_id: str) -> None:
        with ASYNC_LOCK:
            j = ASYNC_JOBS.get(job_id)
        if j is None:
            self._json(410, {"error": "khong co job nay (server khoi dong lai hoac qua han)"})
        elif j["result"] is None:
            self._json(202, {"status": "running", "elapsed": round(time.time() - j["started"], 1)})
        else:
            r = j["result"]
            self.send_response(r["code"])
            for k, v in r["headers"].items():
                if k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(r["body"])))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(r["body"])

    def do_OPTIONS(self):  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self):  # noqa: N802
        assert POOL is not None
        route = self.path.split("?")[0].rstrip("/")
        if route == "/health":
            self._json(200, {"ok": True, "gpu": gpu_info(), "async_jobs": True, **POOL.stats()})
        elif route.startswith("/job/"):
            if self._auth():
                self._job_status(route[len("/job/"):])
        elif route == "/voices":
            if self._auth():
                self._json(200, {"voices": POOL.stats()["voices"]})
        else:
            self._json(404, {"error": "khong co route nay"})

    def do_POST(self):  # noqa: N802
        if not self._auth():
            return
        assert POOL is not None
        route = self.path.split("?")[0].rstrip("/")
        try:
            if route in ("/tts", "/tts_batch") and not getattr(self, "_is_shadow", False):
                d = self._body()
                if d.get("async"):
                    self._start_async(route, d)
                    return
                self._parsed = d  # route đồng bộ đọc lại qua _body()
            if route == "/voice":
                d = self._body()
                b64 = pick_voice_audio(d) if isinstance(d, dict) else None
                if b64 is None:
                    # Báo rõ app đã gửi những trường nào thay vì KeyError -> 500 khó hiểu.
                    got = sorted(d.keys()) if isinstance(d, dict) else type(d).__name__
                    print(f"[voice] thiếu wav_b64; app gửi các trường: {got}", flush=True)
                    self._json(
                        400,
                        {"error": "thieu 'wav_b64' (wav base64)", "received_keys": got},
                    )
                    return
                pcm = read_wav_bytes_44k(base64.b64decode(b64))
                ref_text = (
                    d.get("text") or d.get("ref_text") or d.get("reference_text")
                    or d.get("transcript") or ""
                )
                vid = POOL.add_voice(d.get("name", "voice"), pcm, ref_text)
                self._json(
                    200,
                    {"voice_id": vid, "seconds": round(len(pcm) / DRAMA3_SAMPLE_RATE, 2)},
                )

            elif route == "/tts":
                d = self._body()
                if not d.get("text"):
                    self._json(400, {"error": "thieu 'text'"})
                    return
                if len(d["text"]) > MAX_TEXT_CHARS:
                    self._json(
                        413,
                        {
                            "error": "text qua dai",
                            "max_chars": MAX_TEXT_CHARS,
                            "got_chars": len(d["text"]),
                        },
                    )
                    return
                job = Job(
                    text=d["text"],
                    voice_id=d.get("voice_id"),
                    instruct=pick_instruct(d),
                    temperature=float(d.get("temperature", DEFAULT_TEMPERATURE)),
                    top_p=float(d.get("top_p", DEFAULT_TOP_P)),
                    top_k=int(d.get("top_k", DEFAULT_TOP_K)),
                    seed=int(d.get("seed", 42)),
                )
                POOL.submit(job)
                if not job.done.wait(timeout=float(d.get("timeout", 900))):
                    self._json(504, {"error": "qua han cho"})
                    return
                if job.error:
                    self._json(500, {"error": job.error})
                    return
                b = wav_bytes(job.audio, sr=DRAMA3_SAMPLE_RATE)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("X-Synth-Seconds", f"{job.wall:.3f}")
                self.send_header("X-Queue-Seconds", f"{job.started_at - job.queued_at:.3f}")
                self.send_header("X-Audio-Seconds", f"{len(job.audio) / DRAMA3_SAMPLE_RATE:.3f}")
                self.send_header("X-Sample-Rate", str(DRAMA3_SAMPLE_RATE))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b)

            elif route == "/tts_batch":
                d = self._body()
                items = d.get("items") or []
                if not items:
                    self._json(400, {"error": "thieu 'items'"})
                    return
                fmt = (d.get("format") or "wav").lower()
                toolong = [
                    (it.get("i"), len(it.get("text", "")))
                    for it in items
                    if len(it.get("text", "")) > MAX_TEXT_CHARS
                ]
                if toolong:
                    self._json(
                        413,
                        {
                            "error": "co item text qua dai",
                            "max_chars": MAX_TEXT_CHARS,
                            "too_long": toolong,
                        },
                    )
                    return

                jobs: list[tuple[object, Job]] = []
                for it in items:
                    j = Job(
                        text=it["text"],
                        voice_id=d.get("voice_id"),
                        instruct=pick_instruct(it) or pick_instruct(d),
                        temperature=float(d.get("temperature", DEFAULT_TEMPERATURE)),
                        top_p=float(d.get("top_p", DEFAULT_TOP_P)),
                        top_k=int(d.get("top_k", DEFAULT_TOP_K)),
                        seed=int(d.get("seed", 42)),
                    )
                    jobs.append((it.get("i"), j))
                    POOL.submit(j)

                deadline = time.time() + float(d.get("timeout", 1800))
                buf = io.BytesIO()
                meta = {"format": fmt, "sample_rate": DRAMA3_SAMPLE_RATE, "items": []}
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                    for idx, j in jobs:
                        if not j.done.wait(timeout=max(1.0, deadline - time.time())):
                            self._json(504, {"error": f"qua han cho o item {idx}"})
                            return
                        if j.error:
                            meta["items"].append({"i": idx, "error": j.error})
                            continue
                        blob, ext = encode_audio(j.audio, fmt, sr=DRAMA3_SAMPLE_RATE)
                        z.writestr(f"{idx}.{ext}", blob)
                        meta["items"].append(
                            {
                                "i": idx,
                                "file": f"{idx}.{ext}",
                                "synth": round(j.wall, 3),
                                "audio": round(len(j.audio) / DRAMA3_SAMPLE_RATE, 3),
                            }
                        )
                    z.writestr("meta.json", json.dumps(meta, ensure_ascii=False))

                b = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("X-Items", str(len(jobs)))
                self.send_header("X-Sample-Rate", str(DRAMA3_SAMPLE_RATE))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b)

            else:
                self._json(404, {"error": "khong co route nay"})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    global POOL, API_KEY
    default_eng = "/content/drama3_engine" if sys.platform.startswith("linux") else str(
        Path(__file__).resolve().parent.parent / "drama3_engine"
    )
    default_mod = "/content/drama3_models" if sys.platform.startswith("linux") else str(
        Path(__file__).resolve().parent.parent / "drama3_models"
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--key", default=None, help="API key; bỏ trống sẽ tự sinh")
    ap.add_argument("--engine-dir", default=default_eng)
    ap.add_argument("--models-dir", default=default_mod)
    ap.add_argument("--model", default=None, help="Đường dẫn thẳng tới file .gguf")
    ap.add_argument(
        "--vram-mode",
        default="auto",
        choices=["auto", "2.8gb", "2.3gb", "3.3gb", "t4", "gpu"],
        help="Cấu hình phân bổ VRAM/RAM cho Fish Drama-3 INT4",
    )
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument(
        "--cpu-expert-layers",
        type=int,
        default=-1,
        help="Ghi đè số tầng expert (0-40) chạy trên CPU. -1 = theo profile (t4: 24). "
        "Ít tầng hơn = nhanh hơn nhưng tốn VRAM hơn, đồng thời giảm RAM máy.",
    )
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--default-vq", default=None, help="File JSON VQ cache mặc định nếu gọi /tts không voice_id")
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    server_bin = _find_binary(engine_dir, "audiocpp_server")
    if not server_bin:
        raise SystemExit(
            f"Không tìm thấy audiocpp_server trong {engine_dir}. "
            "Hãy chạy `python remote/prepare_drama3_int4.py` trước."
        )

    model_path = find_drama3_model(models_dir, args.model)
    info = gpu_info()
    ram_total, ram_avail = host_ram_mib()
    gpu_free = int(info.get("vram_free_mib") or 0)
    print(f"[server] GPU: {info}", flush=True)
    print(f"[server] Host RAM: avail {ram_avail} MiB / total {ram_total} MiB", flush=True)

    is_slim = "slim" in model_path.name.lower() or "int4" in model_path.name.lower()
    chosen_mode, sess_opts = select_vram_options(
        args.vram_mode,
        gpu_free_mib=gpu_free,
        ram_total_mib=ram_total,
        ram_avail_mib=ram_avail,
        is_slim=is_slim,
    )
    if args.cpu_expert_layers >= 0:
        sess_opts["cpu_expert_layers"] = min(40, args.cpu_expert_layers)
        print(f"[server] Ghi đè cpu_expert_layers={sess_opts['cpu_expert_layers']}", flush=True)
    print(f"[server] Drama-3 profile='{chosen_mode}', model='{model_path.name}'", flush=True)

    default_vq_json = None
    if args.default_vq and Path(args.default_vq).is_file():
        default_vq_json = Path(args.default_vq).read_text(encoding="utf-8")

    API_KEY = args.key or secrets.token_urlsafe(12)
    engine = Drama3Engine(
        server_bin=server_bin,
        model_path=model_path,
        engine_dir=engine_dir,
        session_options=sess_opts,
        threads=args.threads,
        backend=args.backend,
    )
    POOL = Drama3Pool(
        engine=engine,
        vram_mode=chosen_mode,
        cache_dir=models_dir / "cached_voices",
        default_vq_json=default_vq_json,
    )

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print("=" * 66, flush=True)
    print(f"  Fish Drama-3 Server sẵn sàng  http://{args.host}:{args.port}", flush=True)
    print(f"  Model                         {model_path.name}", flush=True)
    print(f"  Profile                       {chosen_mode}", flush=True)
    print(f"  Sample rate                   {DRAMA3_SAMPLE_RATE} Hz", flush=True)
    print(f"  API key                       {API_KEY}", flush=True)
    print("=" * 66, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] dừng.", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()

