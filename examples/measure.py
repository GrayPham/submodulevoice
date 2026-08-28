"""Đo thời gian thực thi + VRAM + RAM của một lần render.

    python examples/measure.py --profile lite    --script scripts/kichban_5k.txt
    python examples/measure.py --profile quality --script scripts/kichban_5k.txt
    python examples/measure.py --profile lite --backend cpu --script scripts/kichban_5k.txt

Cách đo VRAM. Driver Windows ở chế độ WDDM không báo VRAM theo từng process
(`nvidia-smi --query-compute-apps=used_gpu_memory` trả về N/A), nên script này
chạy render trong một process con rồi poll **tổng** `memory.used` của mọi GPU,
lấy baseline trước khi chạy và đỉnh trong khi chạy. Delta là phần của mình,
với điều kiện máy không có job GPU nào khác chen vào — script in cả baseline
để bạn tự kiểm tra điều đó.

RAM đo bằng RSS đỉnh của process con qua psutil, gồm cả phần model nằm ở host.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
POLL = 0.2  # giây


def gpu_mem() -> list[int]:
    """memory.used (MiB) của từng GPU theo thứ tự nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        return []


class Sampler(threading.Thread):
    """Poll VRAM tổng + RSS của process con, giữ đỉnh."""

    def __init__(self, proc: subprocess.Popen, n_gpu: int) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.peak_gpu = [0] * n_gpu
        self.peak_rss = 0
        self.samples = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        try:
            ps = psutil.Process(self.proc.pid)
        except psutil.Error:
            return
        while not self.stop_flag.is_set():
            mem = gpu_mem()
            for i, v in enumerate(mem[: len(self.peak_gpu)]):
                self.peak_gpu[i] = max(self.peak_gpu[i], v)
            try:
                rss = ps.memory_info().rss
                for ch in ps.children(recursive=True):
                    rss += ch.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except psutil.Error:
                pass
            self.samples += 1
            self.stop_flag.wait(POLL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(ROOT / "scripts" / "kichban_5k.txt"))
    ap.add_argument("--ref-wav", default=str(
        ROOT / "output" / "refs3" / "FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav"))
    ap.add_argument("--profile", default="lite")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="paragraph", choices=["paragraph", "auto"])
    ap.add_argument("--keep-parts", action="store_true", help="giữ WAV từng đoạn để so sánh")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    out = args.out or str(ROOT / "output" / "measure" / f"{args.profile}-{args.backend}.wav")

    names = gpu_names()
    base = gpu_mem()
    print("GPU trên máy:")
    for i, n in enumerate(names):
        b = base[i] if i < len(base) else 0
        print(f"  [{i}] {n}   đang dùng {b} MiB")
    print()

    cmd = [
        sys.executable, str(ROOT / "examples" / "longform.py"), args.script,
        "--ref-wav", args.ref_wav, "--profile", args.profile,
        "--backend", args.backend, "--steps", str(args.steps),
        "--lang", args.lang,
        "--seed", str(args.seed), "--mode", args.mode, "-o", out,
    ]
    if args.keep_parts:
        cmd.append("--keep-parts")

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    )
    sampler = Sampler(proc, len(base))
    sampler.start()

    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if re.search(r"MaskGIT|Dedup|GGUF|Loaded|WeightCtx|Registered|Prompt\]|graph|Qwen3|ggml_cuda|Device \d", line):
            continue
        tail.append(line)
    proc.wait()
    wall = time.perf_counter() - t0
    sampler.stop_flag.set()
    sampler.join(timeout=2)

    print("\n".join(tail))

    # Thời lượng audio ra, đọc lại từ file cho chắc.
    import wave
    try:
        with wave.open(out, "rb") as w:
            dur = w.getnframes() / w.getframerate()
    except Exception:
        dur = 0.0

    print()
    print("=" * 58)
    print(f"profile        : {args.profile}   backend: {args.backend}   steps: {args.steps}")
    print(f"tổng thời gian : {wall:.1f}s  (gồm cả khởi động Python + nạp model)")
    print(f"audio ra       : {dur:.1f}s  ({dur / 60:.2f} phút)")
    if wall > 0:
        print(f"tốc độ         : {dur / wall:.2f}x realtime (tính cả overhead)")
    print(f"RAM đỉnh       : {sampler.peak_rss / 2**20:.0f} MiB")
    if base:
        print(f"VRAM ({sampler.samples} mẫu, mỗi {POLL}s):")
        for i, n in enumerate(names):
            b = base[i] if i < len(base) else 0
            p = sampler.peak_gpu[i] if i < len(sampler.peak_gpu) else 0
            mark = "  <-- GPU đang dùng" if p - b > 200 else ""
            print(f"  [{i}] baseline {b:>6d} MiB   đỉnh {p:>6d} MiB   delta {p - b:>+6d} MiB{mark}")
    print("=" * 58)


if __name__ == "__main__":
    main()
