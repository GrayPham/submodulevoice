"""Test độ ổn định: 4 ngôn ngữ, mỗi ngôn ngữ 1 giờ audio, chạy đồng thời.

    python examples/stress_4lang.py --url https://xxx.trycloudflare.com --key KEY

Câu hỏi cần trả lời: 4 giờ audio có dựng xong dưới 1 giờ không?

Tính từ số đo (3.85x realtime trên T4, 4 worker): 14.400s ÷ 3.85 = ~62 phút.
Sát mốc, nên phải đo chứ không suy.

Mỗi ngôn ngữ là MỘT tiến trình client riêng, đúng nghĩa "4 luồng độc lập".
Mỗi client chỉ giữ 1 request đang bay (mỗi request gói 8 đoạn), nên tổng cộng
có 4 kết nối HTTP cùng lúc và 32 việc nằm trong hàng đợi của server — vừa đủ
để 4 worker luôn có việc mà không làm nghẽn đường hầm.

Đường hầm cloudflared hay đứt. Client ghi từng đoạn ra đĩa nên nếu đứt giữa
chừng, chạy lại đúng lệnh này với URL mới là nó làm tiếp phần còn thiếu.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "output" / "refs3" / "FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav"

LANGS = [
    ("vi", "stress_vi.txt", "Vietnamese"),
    ("pt", "stress_pt.txt", "Portuguese"),
    ("en", "stress_en.txt", "English"),
    ("es", "stress_es.txt", "Spanish"),
]


def hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Monitor(threading.Thread):
    """Lấy mẫu server suốt bài chạy: tải GPU, worker bận, hàng đợi, lỗi."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url.rstrip("/") + "/health"
        self.samples: list[dict] = []
        self.drops = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=10) as r:
                    h = json.load(r)
                g = h.get("gpu", {})
                self.samples.append({
                    "t": time.time(), "util": g.get("util_gpu_pct", -1),
                    "busy": h.get("busy", 0), "queued": h.get("queued", 0),
                    "served": h.get("served", 0), "failed": h.get("failed", 0),
                    "vram_free": g.get("vram_free_mib", 0),
                })
            except Exception:
                self.drops += 1
            self.stop_flag.wait(5)

    def summary(self) -> dict:
        if not self.samples:
            return {"mau": 0, "mat_ket_noi": self.drops}
        u = [s["util"] for s in self.samples if s["util"] >= 0]
        b = [s["busy"] for s in self.samples]
        q = [s["queued"] for s in self.samples]
        return {
            "mau": len(self.samples),
            "mat_ket_noi": self.drops,
            "util_tb": round(sum(u) / len(u), 1) if u else None,
            "util_min": min(u) if u else None,
            "busy_tb": round(sum(b) / len(b), 2),
            "queued_max": max(q),
            "served": self.samples[-1]["served"],
            "failed": self.samples[-1]["failed"],
            "vram_free_min": min(s["vram_free"] for s in self.samples),
        }


def run_one(code: str, script: str, lang: str, args, out: dict) -> None:
    o = ROOT / "output" / "stress" / f"{code}.wav"
    cmd = [
        sys.executable, str(ROOT / "remote" / "client.py"),
        "--url", args.url, "--key", args.key,
        "--script", str(ROOT / "scripts" / script),
        "--ref", str(REF), "--lang", lang,
        "--batch", str(args.batch), "--format", args.format,
        "--concurrency", "1", "--retries", "4",
        "-o", str(o),
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out[code] = {
        "wall": time.perf_counter() - t0,
        "rc": p.returncode,
        "tail": "\n".join(l for l in ((p.stdout or "") + (p.stderr or "")).splitlines()
                          if l.strip())[-900:],
        "wav": o,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--format", default="flac")
    args = ap.parse_args()

    for _c, s, _l in LANGS:
        if not (ROOT / "scripts" / s).exists():
            raise SystemExit(f"thiếu scripts/{s} — chạy: python scripts/make_stress.py")

    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=30) as r:
        h = json.load(r)
    g = h["gpu"]
    print(f"server : {g['name']} | {h['workers']} worker | VRAM {g['vram_free_mib']} MiB trống")
    print(f"bài    : 4 ngôn ngữ x ~1 giờ audio, 4 tiến trình client độc lập")
    print(f"dự kiến: ~62 phút (từ mốc 3.85x realtime đã đo)\n")

    mon = Monitor(args.url)
    mon.start()

    out: dict = {}
    t0 = time.perf_counter()
    ths = [threading.Thread(target=run_one, args=(c, s, l, args, out))
           for c, s, l in LANGS]
    for t in ths:
        t.start()

    # In tiến độ định kỳ để biết nó còn sống
    while any(t.is_alive() for t in ths):
        time.sleep(60)
        el = time.perf_counter() - t0
        done = sum(1 for c, _, _ in LANGS if c in out)
        s = mon.samples[-1] if mon.samples else {}
        print(f"  [{hms(el)}] xong {done}/4 ngôn ngữ | server da lam {s.get('served', '?')} viec"
              f" | util {s.get('util', '?')}% | hang doi {s.get('queued', '?')}", flush=True)

    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    mon.stop_flag.set()
    mon.join(timeout=3)

    import wave
    print(f"\n{'ngôn ngữ':<12s}{'thời gian':>12s}{'audio':>12s}{'xRT':>8s}  trạng thái")
    print("-" * 60)
    total_audio = 0.0
    ok = 0
    for code, _s, lang in LANGS:
        r = out.get(code, {})
        dur = 0.0
        try:
            with wave.open(str(r["wav"]), "rb") as w:
                dur = w.getnframes() / w.getframerate()
        except Exception:
            pass
        total_audio += dur
        good = r.get("rc") == 0 and dur > 0
        ok += good
        print(f"{lang:<12s}{hms(r.get('wall', 0)):>12s}{hms(dur):>12s}"
              f"{(dur / r['wall'] if r.get('wall') and dur else 0):>7.2f}x"
              f"  {'OK' if good else 'HỎNG (rc=' + str(r.get('rc')) + ')'}")

    print("-" * 60)
    print(f"{'TỔNG':<12s}{hms(wall):>12s}{hms(total_audio):>12s}"
          f"{(total_audio / wall if wall else 0):>7.2f}x")
    print()
    print(f"  {ok}/4 ngôn ngữ hoàn tất")
    print(f"  DƯỚI 1 GIỜ: {'CÓ' if wall < 3600 and ok == 4 else 'KHÔNG'}"
          f"  ({hms(wall)})")
    print()
    print("  server trong suốt bài chạy:")
    for k, v in mon.summary().items():
        print(f"    {k:16s} {v}")

    if ok < 4:
        print("\n  Ngôn ngữ hỏng thường do đường hầm đứt giữa chừng.")
        print("  Lấy URL mới rồi chạy lại đúng lệnh này — client làm tiếp phần còn thiếu.")
        for code, _s, lang in LANGS:
            r = out.get(code, {})
            if r.get("rc") != 0:
                print(f"\n  --- {lang} ---\n{r.get('tail', '')[-400:]}")


if __name__ == "__main__":
    main()
