"""Đo xem chạy nhiều context song song trên MỘT GPU có lợi không.

    python examples/bench_parallel.py --workers 1 2 3 4

Hai câu hỏi, một thí nghiệm:

  1. ĐÚNG KHÔNG. Gọi ov_synthesize đồng thời trên nhiều ov_context riêng biệt
     có chạy được không, và kết quả có giống hệt khi chạy tuần tự không?
     So bằng băm nội dung audio với bản chạy một luồng, cùng seed.
  2. NHANH KHÔNG. GPU vẫn chia sẻ SM, nên lợi ích chỉ đến từ việc lấp khoảng
     trống lúc codec decode và chuyển dữ liệu về host. Có thể rất ít.

Đây là số liệu quyết định kiến trúc server: đặt bao nhiêu worker cho vừa.
"""

from __future__ import annotations

import argparse
import hashlib
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import OmniVoice  # noqa: E402
from pyomnivoice.refprep import load_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "output" / "refs3" / "FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav"


def gpu_used() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.splitlines()[0].strip())
    except Exception:
        return 0


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, gpu_used())
            self.stop_flag.wait(0.25)


def run_round(n_workers: int, jobs: list[tuple[int, str]], profile: str,
              lang: str, steps: int) -> tuple[float, int, dict[int, str]]:
    base = gpu_used()
    sampler = Sampler()

    # Mỗi worker giữ MỘT context riêng. Dùng chung một context giữa các luồng
    # là chuyện khác hẳn và không an toàn — ở đây cố tình tách hẳn ra.
    engines = [OmniVoice(profile=profile, backend="cuda") for _ in range(n_workers)]
    refs = [load_reference(REF) for _ in range(n_workers)]
    voices = [r.as_voice(e) for e, r in zip(engines, refs)]

    q: "queue.Queue[tuple[int, str] | None]" = queue.Queue()
    for j in jobs:
        q.put(j)
    for _ in range(n_workers):
        q.put(None)

    digests: dict[int, str] = {}
    lock = threading.Lock()
    errors: list[str] = []

    def worker(wid: int) -> None:
        while True:
            item = q.get()
            if item is None:
                return
            idx, text = item
            try:
                a = engines[wid].say(text, voice=voices[wid], lang=lang, steps=steps, seed=42)
                d = hashlib.sha1(np.ascontiguousarray(a.samples).tobytes()).hexdigest()[:12]
                with lock:
                    digests[idx] = d
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"worker {wid} job {idx}: {type(e).__name__}: {e}")

    sampler.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    sampler.stop_flag.set()
    sampler.join(timeout=2)

    for e in engines:
        e.close()
    if errors:
        print("  LỖI:", *errors[:5], sep="\n    ")
    return wall, sampler.peak - base, digests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--script", default=str(ROOT / "scripts" / "kichban_pt.txt"))
    ap.add_argument("--repeat", type=int, default=4, help="lặp kịch bản cho đủ việc")
    ap.add_argument("--profile", default="lite")
    ap.add_argument("--lang", default="Portuguese")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    paras = [re.sub(r"\s+", " ", p.strip())
             for p in re.split(r"\n\s*\n", Path(args.script).read_text(encoding="utf-8")) if p.strip()]
    jobs = [(i, t) for i, t in enumerate(paras * args.repeat)]
    print(f"{len(jobs)} công việc, {sum(len(t) for _, t in jobs)} ký tự, profile {args.profile}\n")

    ref_digests: dict[int, str] | None = None
    print(f"{'worker':>7s} {'giây':>9s} {'throughput':>12s} {'tăng tốc':>10s} "
          f"{'VRAM':>9s} {'/worker':>9s} {'giống 1 luồng':>15s}")
    print("-" * 78)
    base_wall = None
    for n in args.workers:
        wall, vram, digests = run_round(n, jobs, args.profile, args.lang, args.steps)
        if ref_digests is None:
            ref_digests = digests
            same = "(mốc chuẩn)"
        else:
            ok = sum(1 for k, v in digests.items() if ref_digests.get(k) == v)
            same = f"{ok}/{len(digests)}"
        if base_wall is None:
            base_wall = wall
        print(f"{n:>7d} {wall:>8.1f}s {len(jobs) / wall:>9.2f} job/s "
              f"{base_wall / wall:>9.2f}x {vram:>7d} MiB {vram // max(n, 1):>6d} MiB {same:>15s}")


if __name__ == "__main__":
    main()
