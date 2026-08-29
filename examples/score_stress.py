"""Chấm độ chính xác câu chữ cho bài test 4 ngôn ngữ x 1 giờ.

    python examples/score_stress.py

Client xoá các đoạn sau khi ghép, chỉ còn file tổng, nên không chấm từng đoạn
như score_pt.py được. Ở đây cho Whisper nghe cả file một giờ rồi so nguyên văn
với kịch bản gốc.

Cách này thật ra đo đúng thứ cần đo hơn: nó bắt được cả lỗi phát âm lẫn lỗi mất
đoạn, lặp đoạn, đọc lẫn câu của giọng mẫu — những lỗi mà chấm từng đoạn rời rạc
sẽ bỏ sót vì đoạn nào cũng khớp với chính nó.

Khoảng cách sửa dùng rapidfuzz (bit song song), vì DP thuần Python trên chuỗi
60.000 ký tự là 3,6 tỷ ô — không chạy nổi.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_quant import norm_text  # noqa: E402


def spell_numbers(t: str, lang: str) -> str:
    """Đổi mọi chữ số thành chữ, áp cho CẢ kịch bản lẫn bản nghe được.

    Whisper viết số nói thành chữ số ("six hundred and twelve" -> "612"), nên
    so thẳng thì mỗi con số bị tính là hơn hai chục ký tự sai dù máy đọc đúng.
    Kịch bản một giờ là bản lặp nên mỗi con số bị nhân lên vài chục lần — đủ
    để thổi CER tiếng Anh từ dưới 2% lên 8%.
    """
    from num2words import num2words

    def rep(m):
        try:
            return " " + num2words(int(m.group()), lang=lang) + " "
        except Exception:
            return m.group()

    return re.sub(r"\d+", rep, t)

ROOT = Path(__file__).resolve().parent.parent

LANGS = [("vi", "Việt"), ("pt", "Bồ Đào Nha"), ("en", "Anh"), ("es", "Tây Ban Nha")]


def hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def quarter_errors(ref: str, hyp: str):
    """Lỗi rơi vào phần tư nào của kịch bản — xem chất lượng có tụt về cuối không.

    Chiếu opcode về không gian ký tự của kịch bản gốc: xoá/thay tính theo độ dài
    đoạn bị ảnh hưởng, chèn tính 1 điểm tại chỗ chèn.
    """
    from rapidfuzz.distance import Levenshtein

    n = len(ref)
    bucket = [0] * 4
    worst_del = worst_ins = (0, 0)
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal":
            continue
        span = op.src_end - op.src_start
        k = min(3, op.src_start * 4 // max(n, 1))
        if op.tag == "insert":
            grow = op.dest_end - op.dest_start
            bucket[k] += grow
            if grow > worst_ins[0]:
                worst_ins = (grow, op.src_start)
        else:
            bucket[k] += span
            if op.tag == "delete" and span > worst_del[0]:
                worst_del = (span, op.src_start)
    size = [len(x) for x in (ref[: n // 4], ref[n // 4: n // 2],
                             ref[n // 2: 3 * n // 4], ref[3 * n // 4:])]
    return [b / max(s, 1) for b, s in zip(bucket, size)], worst_del, worst_ins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dir", default=str(ROOT / "output" / "stress"))
    ap.add_argument("--re-asr", action="store_true", help="nghe lại từ đầu, bỏ qua bản đã lưu")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from rapidfuzz.distance import Levenshtein

    ct = "float16" if args.device == "cuda" else "int8"
    print(f"Whisper {args.model} trên {args.device}\n")
    m = WhisperModel(args.model, device=args.device, compute_type=ct)

    out = []
    for code, ten in LANGS:
        wav = Path(args.dir) / f"{code}.wav"
        txt = ROOT / "scripts" / f"stress_{code}.txt"
        if not wav.exists():
            print(f"{ten:<14s} thiếu {wav}")
            continue

        paras = [re.sub(r"\s+", " ", p.strip())
                 for p in re.split(r"\n\s*\n", txt.read_text(encoding="utf-8")) if p.strip()]
        cache = Path(args.dir) / f"{code}-nghe-lai.txt"
        t0 = time.perf_counter()
        if cache.exists() and not args.re_asr:
            heard = cache.read_text(encoding="utf-8")
            dur = 0.0
        else:
            segs, info = m.transcribe(str(wav), language=code, vad_filter=True, beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            dur = info.duration
            cache.write_text(heard, encoding="utf-8")
        asr_s = time.perf_counter() - t0
        info_dur = dur

        ref = norm_text(spell_numbers(" ".join(paras), code))
        hyp = norm_text(spell_numbers(heard, code))
        ref_raw = norm_text(" ".join(paras))
        hyp_raw = norm_text(heard)
        cer_raw = Levenshtein.normalized_distance(ref_raw, hyp_raw)

        cer = Levenshtein.normalized_distance(ref, hyp)
        wer = Levenshtein.normalized_distance(ref.split(), hyp.split())
        q, wdel, wins = quarter_errors(ref, hyp)

        print(f"{ten:<14s} nghe lại {hms(info_dur)} mất {asr_s:.0f}s")
        print(f"{'':14s} CER {cer * 100:5.2f}% (chưa khớp số {cer_raw * 100:.2f}%)"
              f"   WER {wer * 100:5.2f}%   "
              f"kịch bản {len(ref)} ký tự, nghe được {len(hyp)}")
        print(f"{'':14s} theo phần tư: " + "  ".join(f"{x * 100:.2f}%" for x in q)
              + f"   (lệch cuối-đầu {(q[3] - q[0]) * 100:+.2f} đpt)")
        if wdel[0] > 200:
            print(f"{'':14s} ! mất một mạch {wdel[0]} ký tự quanh vị trí {wdel[1]}")
        if wins[0] > 200:
            print(f"{'':14s} ! thừa một mạch {wins[0]} ký tự quanh vị trí {wins[1]}")
        print()

        out.append({"lang": code, "ten": ten, "cer": cer, "wer": wer,
                    "ref_chars": len(ref), "hyp_chars": len(hyp),
                    "cer_raw": cer_raw, "audio_s": info_dur, "asr_s": asr_s,
                    "quartile_cer": q, "worst_delete": wdel, "worst_insert": wins,
                    "heard_head": heard[:400]})

    if out:
        print("-" * 58)
        print(f"{'ngôn ngữ':<14s}{'CER':>9s}{'WER':>9s}{'lệch cuối-đầu':>16s}")
        print("-" * 58)
        for r in out:
            d = (r["quartile_cer"][3] - r["quartile_cer"][0]) * 100
            print(f"{r['ten']:<14s}{r['cer'] * 100:8.2f}%{r['wer'] * 100:8.2f}%{d:15.2f}đpt")
        print("-" * 58)
        avg = sum(r["cer"] for r in out) / len(out)
        print(f"{'trung bình':<14s}{avg * 100:8.2f}%")

        p = Path(args.dir) / "cham-diem-4ngonngu.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{p}")


if __name__ == "__main__":
    main()
