"""Server tổng hợp giọng nói chạy trên GPU từ xa (Colab), nhiều worker + hàng đợi.

    python remote/server.py --workers 4 --port 8770 --key BIMAT

Thiết kế đi thẳng từ số đo thật (`examples/bench_parallel.py`, RTX 4000 Ada):

    worker   tăng tốc   VRAM       audio so với chạy 1 luồng
      1        1.00x    1443 MiB   mốc chuẩn
      2        1.80x    2452 MiB   giống hệt từng byte
      4        2.60x    4802 MiB   giống hệt từng byte
      6        0.99x    7140 MiB   giống hệt từng byte
      8        0.94x    9440 MiB   giống hệt từng byte

Hai điều rút ra, đã đưa thẳng vào thiết kế:

  - Chạy song song KHÔNG đổi chất lượng. Mỗi worker giữ một `ov_context`
    riêng; băm SHA1 nội dung audio khớp 100% với bản chạy tuần tự cùng seed.
    Nhờ vậy chia luồng thoải mái mà giọng không đổi.
  - Quá 4 worker thì CHẬM ĐI chứ không nhanh thêm: GPU đã bão hoà, thêm
    worker chỉ tốn VRAM. Mặc định 4, và server tự hạ xuống nếu VRAM không đủ.

Chỉ dùng thư viện chuẩn, không phải cài thêm gì trên Colab.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import Audio, OmniVoice, SAMPLE_RATE, Voice  # noqa: E402

VRAM_PER_WORKER_MIB = 1250   # đo được: worker đầu 1443 MiB, mỗi worker sau ~1200
MAX_USEFUL_WORKERS = 4       # quá mức này throughput đi xuống


@dataclass
class Job:
    text: str
    voice_id: str | None
    lang: str
    steps: int
    seed: int
    instruct: str | None
    done: threading.Event = field(default_factory=threading.Event)
    audio: np.ndarray | None = None
    error: str | None = None
    wall: float = 0.0
    queued_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0


class Pool:
    """N engine độc lập, một hàng đợi chung, phục vụ theo thứ tự đến trước."""

    def __init__(self, n: int, profile: str, models_dir: Path | None) -> None:
        self.q: "queue.Queue[Job | None]" = queue.Queue()
        self.voices: dict[str, tuple[str, Voice]] = {}
        self.lock = threading.Lock()
        self.busy = 0
        self.served = 0
        self.failed = 0
        self.audio_sec = 0.0
        self.synth_sec = 0.0
        self.started = time.time()

        print(f"[server] nap {n} engine, profile {profile} ...", flush=True)
        t0 = time.perf_counter()
        self.engines = [
            OmniVoice(profile=profile, backend="cuda", models_dir=models_dir)
            for _ in range(n)
        ]
        # Mỗi engine có ggml context/scheduler RIÊNG, KHÔNG an toàn khi hai luồng
        # dùng cùng lúc. add_voice() chạy engine 0 từ luồng HTTP, còn worker 0
        # cũng chạy engine 0 từ hàng đợi — nếu /voice chen vào lúc worker 0 đang
        # tổng hợp thì ggml sập (GGML_ASSERT(buffer) / segfault). Khoá riêng từng
        # engine để mỗi context chỉ một luồng dùng tại một thời điểm.
        self.engine_locks = [threading.Lock() for _ in range(n)]
        self.backend = self.engines[0].backend
        print(f"[server] xong sau {time.perf_counter() - t0:.1f}s, backend {self.backend}",
              flush=True)
        if "CUDA" not in (self.backend or "").upper():
            raise SystemExit("Backend khong phai CUDA. Server nay chi chay GPU.")

        self.threads = [threading.Thread(target=self._worker, args=(i,), daemon=True)
                        for i in range(n)]
        for t in self.threads:
            t.start()

    def add_voice(self, name: str, pcm: np.ndarray, text: str) -> str:
        """Mã hoá giọng mẫu MỘT lần trên engine 0.

        Mã RVQ chỉ phụ thuộc codec nên mọi worker dùng chung được. Voice là
        mảng numpy chỉ đọc, chia sẻ giữa các luồng an toàn.
        """
        vid = uuid.uuid4().hex[:12]
        tmp = Path(tempfile.gettempdir()) / f"ref-{vid}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        Audio(pcm).save(tmp)
        try:
            with self.engine_locks[0]:
                voice = self.engines[0].load_voice(tmp, text)
        finally:
            tmp.unlink(missing_ok=True)
        with self.lock:
            self.voices[vid] = (name, voice)
        print(f"[server] giong '{name}' -> {vid} ({voice.n_frames} frame)", flush=True)
        return vid

    def submit(self, job: Job) -> None:
        self.q.put(job)

    def _worker(self, wid: int) -> None:
        eng = self.engines[wid]
        while True:
            job = self.q.get()
            if job is None:
                return
            with self.lock:
                self.busy += 1
            job.started_at = time.perf_counter()
            try:
                voice = None
                if job.voice_id:
                    with self.lock:
                        entry = self.voices.get(job.voice_id)
                    if entry is None:
                        raise KeyError(f"voice_id khong ton tai: {job.voice_id}")
                    voice = entry[1]
                with self.engine_locks[wid]:
                    a = eng.say(job.text, voice=voice, instruct=job.instruct,
                                lang=job.lang, steps=job.steps, seed=job.seed)
                job.audio = a.samples
                job.wall = eng.last_wall
                with self.lock:
                    self.served += 1
                    self.audio_sec += a.duration
                    self.synth_sec += eng.last_wall
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
                "workers": len(self.engines),
                "busy": self.busy,
                "queued": self.q.qsize(),
                "served": self.served,
                "failed": self.failed,
                "audio_sec": round(self.audio_sec, 1),
                "synth_sec": round(self.synth_sec, 1),
                "xrt_aggregate": (round(self.audio_sec / self.synth_sec, 2)
                                  if self.synth_sec else None),
                "uptime_sec": round(time.time() - self.started),
                "voices": {k: v[0] for k, v in self.voices.items()},
                "backend": self.backend,
                "formats": (sorted(FORMATS) if _sf else ["wav"]),
            }


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,compute_cap,"
             "utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        n, tot, free, cc, ug, um = [x.strip() for x in out.split(",")]
        # utilization.gpu la % thoi gian CO kernel dang chay — day moi la thuoc
        # do tai tinh toan. "GPU RAM" tren bang Resources cua Colab chi la BO
        # NHO, con trong khong co nghia la con suc tinh.
        return {"name": n, "vram_total_mib": int(tot),
                "vram_free_mib": int(free), "compute_cap": cc,
                "util_gpu_pct": int(ug), "util_mem_pct": int(um)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# soundfile khong bat buoc. Thieu no thi van chay, chi la tra ve WAV tho.
try:
    import soundfile as _sf
except Exception:
    _sf = None

# Do tren giong da sinh, 19.4s audio:
#   WAV 909 KB | FLAC 478 KB (1.9x, khong mat du lieu) | Opus 84 KB (10.8x)
# Duong ham cloudflared bi gioi han bang thong nen day la cho an tien nhat.
FORMATS = {
    "wav":  (None, None, "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "vorbis": ("OGG", "VORBIS", "audio/ogg"),
}


def encode_audio(pcm: np.ndarray, fmt: str) -> tuple[bytes, str]:
    """Ma hoa PCM theo dinh dang yeu cau. Tra ve (bytes, duoi file)."""
    fmt = (fmt or "wav").lower()
    if fmt == "wav" or _sf is None or fmt not in FORMATS:
        return wav_bytes(pcm), "wav"
    container, subtype, _ = FORMATS[fmt]
    buf = io.BytesIO()
    _sf.write(buf, pcm, SAMPLE_RATE, format=container, subtype=subtype)
    return buf.getvalue(), ("flac" if fmt == "flac" else "ogg")


def wav_bytes(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def read_wav_bytes(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("chi nhan WAV 16-bit")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(x) * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


POOL: Pool | None = None
API_KEY = ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "omnivoice-remote/1"

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
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_OPTIONS(self):  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self):  # noqa: N802
        assert POOL is not None
        route = self.path.split("?")[0].rstrip("/")
        if route == "/health":
            self._json(200, {"ok": True, "gpu": gpu_info(), **POOL.stats()})
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
            if route == "/voice":
                d = self._body()
                pcm = read_wav_bytes(base64.b64decode(d["wav_b64"]))
                vid = POOL.add_voice(d.get("name", "voice"), pcm, d["text"])
                self._json(200, {"voice_id": vid,
                                 "seconds": round(len(pcm) / SAMPLE_RATE, 2)})

            elif route == "/tts":
                d = self._body()
                if not d.get("text"):
                    self._json(400, {"error": "thieu 'text'"})
                    return
                job = Job(text=d["text"], voice_id=d.get("voice_id"),
                          lang=d.get("lang", "None"), steps=int(d.get("steps", 16)),
                          seed=int(d.get("seed", 42)), instruct=d.get("instruct"))
                POOL.submit(job)
                if not job.done.wait(timeout=float(d.get("timeout", 900))):
                    self._json(504, {"error": "qua han cho"})
                    return
                if job.error:
                    self._json(500, {"error": job.error})
                    return
                b = wav_bytes(job.audio)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("X-Synth-Seconds", f"{job.wall:.3f}")
                self.send_header("X-Queue-Seconds", f"{job.started_at - job.queued_at:.3f}")
                self.send_header("X-Audio-Seconds", f"{len(job.audio) / SAMPLE_RATE:.3f}")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b)
            elif route == "/tts_batch":
                # Gom nhieu doan vao MOT request.
                #
                # Do that qua cloudflared: moi request ton ~2.9s do tre co
                # dinh bat ke dai ngan. Gui 33 doan rieng le la tra 33 lan do
                # tre do. Gom lai thi tra mot lan, va quan trong hon: ca 4
                # worker cung lam mot lo nen GPU khong nam khong cho mang.
                d = self._body()
                items = d.get("items") or []
                if not items:
                    self._json(400, {"error": "thieu 'items'"})
                    return
                fmt = (d.get("format") or "wav").lower()
                jobs = []
                for it in items:
                    j = Job(text=it["text"], voice_id=d.get("voice_id"),
                            lang=d.get("lang", "None"), steps=int(d.get("steps", 16)),
                            seed=int(d.get("seed", 42)), instruct=d.get("instruct"))
                    jobs.append((it.get("i"), j))
                    POOL.submit(j)

                deadline = time.time() + float(d.get("timeout", 1800))
                buf = io.BytesIO()
                meta = {"format": fmt, "items": []}
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                    for idx, j in jobs:
                        if not j.done.wait(timeout=max(1, deadline - time.time())):
                            self._json(504, {"error": f"qua han cho o item {idx}"})
                            return
                        if j.error:
                            meta["items"].append({"i": idx, "error": j.error})
                            continue
                        blob, ext = encode_audio(j.audio, fmt)
                        z.writestr(f"{idx}.{ext}", blob)
                        meta["items"].append({
                            "i": idx, "file": f"{idx}.{ext}",
                            "synth": round(j.wall, 3),
                            "audio": round(len(j.audio) / SAMPLE_RATE, 3),
                        })
                    z.writestr("meta.json", json.dumps(meta, ensure_ascii=False))

                b = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("X-Items", str(len(jobs)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b)

            else:
                self._json(404, {"error": "khong co route nay"})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    global POOL, API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--profile", default="lite", help="lite = INT4")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--key", default=None, help="API key; bo trong se tu sinh")
    ap.add_argument("--allow-oversubscribe", action="store_true",
                    help="cho phep vuot 4 worker du do duoc la cham hon")
    args = ap.parse_args()

    info = gpu_info()
    print(f"[server] GPU: {info}", flush=True)

    n = args.workers
    if not args.allow_oversubscribe and n > MAX_USEFUL_WORKERS:
        print(f"[server] {n} worker vuot muc huu ich, ha ve {MAX_USEFUL_WORKERS}. "
              f"Do duoc: 6 worker = 0.99x, 8 worker = 0.94x so voi 4.", flush=True)
        n = MAX_USEFUL_WORKERS
    free = info.get("vram_free_mib")
    if free:
        fit = max(1, (free - 400) // VRAM_PER_WORKER_MIB)
        if n > fit:
            print(f"[server] VRAM trong {free} MiB chi du {fit} worker, "
                  f"ha tu {n} xuong {fit}.", flush=True)
            n = fit

    API_KEY = args.key or secrets.token_urlsafe(12)
    POOL = Pool(n, args.profile, Path(args.models_dir) if args.models_dir else None)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print("=" * 66, flush=True)
    print(f"  Server san sang   http://{args.host}:{args.port}", flush=True)
    print(f"  Worker            {n}", flush=True)
    print(f"  API key           {API_KEY}", flush=True)
    print("=" * 66, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] dung.", flush=True)


if __name__ == "__main__":
    main()
