"""Chấm độ chính xác bằng cách lấy mẫu từng cửa sổ, cho kịch bản lặp.

    python examples/score_windows.py

Vì sao không so cả file một lượt: kịch bản một giờ được tạo bằng cách lặp lại
kịch bản gốc vài chục lần. Thuật toán so chuỗi toàn cục được phép ghép lần lặp
thứ 5 của bản nghe với lần lặp thứ 7 của kịch bản — vẫn hợp lệ về mặt toán học,
chi phí thấp hơn — rồi báo ra những khoảng "mất chữ" hàng trăm ký tự không hề
tồn tại. Đã kiểm chứng: chỗ bị báo mất 297 ký tự thì audio có đủ nội dung.

Cách ở đây: cắt vài cửa sổ ngắn rải đều theo thời lượng, cho Whisper nghe từng
cửa sổ, rồi dóng cục bộ vào kịch bản để tìm đúng chỗ nó đang đọc. Cửa sổ ngắn
hơn một chu kỳ lặp nên không còn chỗ cho việc ghép nhầm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_quant import norm_text  # noqa: E402
from score_stress import spell_numbers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LANGS = [("vi", "Việt"), ("pt", "Bồ Đào Nha"), ("en", "Anh"), ("es", "Tây Ban Nha")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dir", default=str(ROOT / "output" / "stress"))
    ap.add_argument("--windows", type=int, default=8, help="số cửa sổ mỗi ngôn ngữ")
    ap.add_argument("--seconds", type=int, default=90, help="độ dài mỗi cửa sổ")
    args = ap.parse_args()

    import soundfile as sf
    from faster_whisper import WhisperModel
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    ct = "float16" if args.device == "cuda" else "int8"
    print(f"Whisper {args.model} trên {args.device} — "
          f"{args.windows} cửa sổ x {args.seconds}s mỗi ngôn ngữ\n")
    m = WhisperModel(args.model, device=args.device, compute_type=ct)
    tmp = Path(args.dir) / "_cuaso.wav"

    out = []
    for code, ten in LANGS:
        wav = Path(args.dir) / f"{code}.wav"
        txt = ROOT / "scripts" / f"stress_{code}.txt"
        if not wav.exists():
            print(f"{ten:<14s} thiếu {wav}")
            continue

        paras = [re.sub(r"\s+", " ", p.strip())
                 for p in re.split(r"\n\s*\n", txt.read_text(encoding="utf-8")) if p.strip()]
        pool = norm_text(spell_numbers(" ".join(paras), code))

        w = wave.open(str(wav), "rb")
        sr, total = w.getframerate(), w.getnframes()
        dur = total / sr
        # Rải đều, chừa hai đầu để không rơi vào khoảng lặng mở/kết file.
        starts = np.linspace(dur * 0.03, dur * 0.93 - args.seconds, args.windows)

        cers = []
        for k, st in enumerate(starts, 1):
            w.setpos(int(st * sr))
            pcm = np.frombuffer(w.readframes(int(args.seconds * sr)), dtype=np.int16)
            sf.write(tmp, pcm.astype(np.float32) / 32768.0, sr)
            segs, _ = m.transcribe(str(tmp), language=code, beam_size=5,
                                   condition_on_previous_text=False)
            hyp = norm_text(spell_numbers(" ".join(s.text.strip() for s in segs), code))
            if len(hyp) < 50:
                continue
            # Dóng cục bộ: tìm đoạn kịch bản khớp nhất với những gì nghe được.
            al = fuzz.partial_ratio_alignment(hyp, pool)
            sub = pool[al.dest_start:al.dest_end]
            c = Levenshtein.normalized_distance(sub, hyp)
            cers.append(c)
            print(f"{ten:<14s} cửa sổ {k}/{args.windows} tại {int(st) // 60}:{int(st) % 60:02d}"
                  f"  CER {c * 100:5.2f}%  ({len(hyp)} ký tự)")

        if cers:
            a = np.array(cers)
            print(f"{ten:<14s} >>> trung vị {np.median(a) * 100:.2f}%  "
                  f"trung bình {a.mean() * 100:.2f}%  cao nhất {a.max() * 100:.2f}%\n")
            out.append({"lang": code, "ten": ten, "n": len(cers),
                        "cer_median": float(np.median(a)), "cer_mean": float(a.mean()),
                        "cer_max": float(a.max()), "cers": [float(x) for x in a]})
        w.close()

    tmp.unlink(missing_ok=True)
    if out:
        print("-" * 56)
        print(f"{'ngôn ngữ':<14s}{'CER trung vị':>14s}{'trung bình':>13s}{'cao nhất':>12s}")
        print("-" * 56)
        for r in out:
            print(f"{r['ten']:<14s}{r['cer_median'] * 100:13.2f}%{r['cer_mean'] * 100:12.2f}%"
                  f"{r['cer_max'] * 100:11.2f}%")
        print("-" * 56)
        avg = sum(r["cer_median"] for r in out) / len(out)
        print(f"{'trung bình':<14s}{avg * 100:13.2f}%")
        p = Path(args.dir) / "cham-diem-cua-so.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{p}")


if __name__ == "__main__":
    main()
