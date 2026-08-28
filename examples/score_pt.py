"""Chấm độ chính xác câu chữ cho một lần render dài, chạy ASR trên GPU.

    python examples/score_pt.py output/pt1h/pt1h.wav scripts/kichban_pt_1h.txt

Đọc từng file đoạn (`*-NNN.wav` do longform --keep-parts sinh ra), cho Whisper
nghe lại rồi so với kịch bản gốc. In CER trung vị, CER cao nhất, và CER theo
từng phần tư để biết chất lượng có tụt dần về cuối hay không.

Whisper tiếng Bồ tự viết số nói thành chữ số ("seiscentos" -> "600"), nên bản
đã khớp cách viết số mới là con số phản ánh đúng chất lượng đọc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_quant import cer  # noqa: E402

NUM_ALIGN = [("seiscentos", "600"), ("mil duzentos e trinta e quatro", "1234")]


def align_numbers(t: str) -> str:
    for w, d in NUM_ALIGN:
        t = t.replace(w, d)
    return t


def sort_parts(paths):
    """Sắp theo số thứ tự, không theo chuỗi.

    Sort chuỗi đặt '-100' ngay sau '-10', nên với hơn 99 đoạn thì text bị so
    lệch với audio và CER vọt lên 70-80% dù audio hoàn toàn bình thường.
    """
    import re as _re

    def key(p):
        m = _re.search(r"-(\d+)\.wav$", p.name)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file wav tổng, các đoạn nằm cạnh dạng <stem>-NNN.wav")
    ap.add_argument("script")
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N đoạn đầu, 0 = tất cả")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    full = Path(args.wav)
    parts = sort_parts(full.parent.glob(f"{full.stem}-*.wav"))
    paras = [re.sub(r"\s+", " ", p.strip())
             for p in re.split(r"\n\s*\n", Path(args.script).read_text(encoding="utf-8")) if p.strip()]
    if args.limit:
        parts, paras = parts[: args.limit], paras[: args.limit]
    if len(parts) != len(paras):
        raise SystemExit(f"lệch số đoạn: {len(parts)} wav vs {len(paras)} đoạn text")

    from faster_whisper import WhisperModel

    ct = "float16" if args.device == "cuda" else "int8"
    print(f"Whisper {args.model} trên {args.device}, {len(parts)} đoạn\n")
    m = WhisperModel(args.model, device=args.device, compute_type=ct)

    rows, t0 = [], time.perf_counter()
    for i, (w, src) in enumerate(zip(parts, paras), 1):
        segs, info = m.transcribe(str(w), language=args.lang, vad_filter=True, beam_size=5)
        heard = " ".join(s.text.strip() for s in segs)
        rows.append({
            "i": i,
            "cer": cer(src, heard),
            "cer_aligned": cer(align_numbers(src), heard),
            "chars": len(src),
        })
        if i % 25 == 0 or i == len(parts):
            done = time.perf_counter() - t0
            print(f"  {i}/{len(parts)} đoạn  ({done:.0f}s, còn ~{done / i * (len(parts) - i):.0f}s)")

    ca = np.array([r["cer_aligned"] for r in rows])
    craw = np.array([r["cer"] for r in rows])
    q = np.array_split(ca, 4)

    print(f"\n{'':<22s}{'thô':>10s}{'khớp số':>12s}")
    print("-" * 44)
    print(f"{'CER trung vị':<22s}{np.median(craw) * 100:9.2f}%{np.median(ca) * 100:11.2f}%")
    print(f"{'CER trung bình':<22s}{craw.mean() * 100:9.2f}%{ca.mean() * 100:11.2f}%")
    print(f"{'CER cao nhất':<22s}{craw.max() * 100:9.2f}%{ca.max() * 100:11.2f}%")
    print(f"\nCER (khớp số) theo từng phần tư — xem có tụt dần không:")
    for k, part in enumerate(q, 1):
        print(f"  phần {k}/4  ({len(part):>3d} đoạn)  trung vị {np.median(part) * 100:.2f}%  "
              f"cao nhất {part.max() * 100:.2f}%")
    drift = np.median(q[-1]) - np.median(q[0])
    print(f"\nChênh phần cuối so với phần đầu: {drift * 100:+.2f} điểm phần trăm")
    bad = [r for r in rows if r["cer_aligned"] > 0.05]
    print(f"Đoạn có CER > 5%: {len(bad)}/{len(rows)}"
          + (f" (đoạn {', '.join(str(r['i']) for r in bad[:10])})" if bad else ""))

    out = Path(args.out) if args.out else full.parent / "cham-diem.json"
    out.write_text(json.dumps({
        "wav": str(full), "n": len(rows), "model": args.model,
        "cer_median": float(np.median(ca)), "cer_mean": float(ca.mean()),
        "cer_max": float(ca.max()), "cer_raw_median": float(np.median(craw)),
        "quartile_median": [float(np.median(x)) for x in q],
        "drift_pp": float(drift * 100), "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
