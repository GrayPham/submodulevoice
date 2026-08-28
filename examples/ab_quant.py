"""So sánh hai lần render cùng text, khác mức lượng tử hoá.

    python examples/ab_quant.py output/int4/kichban_5k_int4.wav \
                                output/q8/kichban_5k_q8.wav \
                                --script scripts/kichban_5k.txt

Cách đo. Hai bản render là hai lần lấy mẫu ngẫu nhiên khác nhau, nên **không**
so được bằng tương quan sóng âm — hai bản đọc đúng y hệt vẫn cho cosine gần 0
vì pha và nhịp khác nhau. Ba phép đo dưới đây thì không phụ thuộc chuyện đó:

  CER      Đưa audio qua ASR rồi so với text gốc (character error rate).
           Đây là phép đo trực tiếp cho câu hỏi "có bỏ chữ, có đọc sai
           không". CER của ASR trên audio thật cũng không bằng 0, nên điều
           đáng xem là **hiệu số giữa hai bản**, không phải giá trị tuyệt đối.
  LTAS     Phổ trung bình dài hạn. Không phụ thuộc thời điểm, nên đo được
           chất giọng / âm sắc có bị đổi hay không. Cosine gần 1.0 = cùng giọng.
  im lặng  Tỉ lệ mẫu gần 0. Lệch nhiều = một bản đã bỏ chữ hoặc chèn thêm.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
import urllib.parse
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SR = 24_000


# ------------------------------------------------------------------ audio i/o


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n, sw = w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if sw == 4:
        return np.frombuffer(raw, "<f4").astype(np.float32)
    raise ValueError(f"sample width {sw} chưa hỗ trợ")


def silence_ratio(x: np.ndarray) -> float:
    return float((np.abs(x) < 1e-4).mean())


def ltas(x: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """Long-term average spectrum, chuẩn hoá — âm sắc, không phụ thuộc nhịp."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[:: n_fft // 2]
    spec = np.abs(np.fft.rfft(frames * win, axis=1)).mean(axis=0)
    spec = np.log10(spec + 1e-8)
    spec -= spec.mean()
    n = np.linalg.norm(spec)
    return spec / n if n else spec


def ltas_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(ltas(a) @ ltas(b))


# -------------------------------------------------------------------- text/CER


def norm_text(s: str) -> str:
    """Bỏ dấu câu, gộp khoảng trắng, về chữ thường, giữ dấu tiếng Việt (NFC)."""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = norm_text(ref), norm_text(hyp)
    return edit_distance(r, h) / max(len(r), 1)


_ASR = {}


def transcribe(path: Path, lang: str, model_size: str) -> str:
    if model_size not in _ASR:
        from faster_whisper import WhisperModel

        _ASR[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = _ASR[model_size].transcribe(str(path), language=lang, vad_filter=True, beam_size=5)
    return " ".join(s.text.strip() for s in segs)


# ------------------------------------------------------------------------ html

CSS = """
:root { --bg:#fbfaf8; --fg:#1c1a17; --mut:#6b6560; --line:#e3ded7; --card:#fff;
  --accent:#8a5a2b; --warn:#a8442a; --ok:#3d6b3f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#161513; --fg:#eeebe6; --mut:#a39c94; --line:#302d29; --card:#1e1d1a;
  --accent:#d9a066; --warn:#e0806a; --ok:#8fc191; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }
.note { color:var(--mut); margin:0 0 1.5rem; }
.note code { background:var(--line); padding:.1em .35em; border-radius:3px; }
h2 { font-size:.95rem; text-transform:uppercase; letter-spacing:.06em;
  color:var(--mut); margin:2rem 0 .8rem; font-weight:600; }
.full, .card { background:var(--card); border:1px solid var(--line);
  border-radius:8px; padding:.85rem 1rem; margin-bottom:.6rem; }
.full .lab { font-weight:600; margin-bottom:.4rem; }
.hdr { display:flex; gap:.75rem; align-items:baseline; flex-wrap:wrap; margin-bottom:.5rem; }
.hdr .n { font-weight:700; color:var(--accent); min-width:2rem; }
.hdr .m { color:var(--mut); font-size:.85rem; font-variant-numeric:tabular-nums; }
.hdr .m b { color:var(--fg); }
.hdr .flag { color:var(--warn); font-weight:600; font-size:.85rem; }
.pair { display:grid; grid-template-columns:5.5rem 1fr; gap:.45rem .75rem; align-items:center; }
.pair span { color:var(--mut); font-size:.8rem; font-weight:600; }
audio { width:100%; height:32px; }
.txt { color:var(--mut); font-size:.85rem; margin-top:.6rem;
  border-left:2px solid var(--line); padding-left:.7rem; }
table { border-collapse:collapse; width:100%; font-size:.88rem; }
th,td { text-align:right; padding:.35rem .6rem; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; }
th:first-child, td:first-child { text-align:left; }
tfoot td { font-weight:700; border-bottom:none; }
.good { color:var(--ok); } .bad { color:var(--warn); }
@media (max-width:640px) { .pair { grid-template-columns:1fr; } }
"""


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
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="INT4 (Q4_K_M)")
    ap.add_argument("--label-b", default="Q8_0")
    ap.add_argument("--script", default=None)
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--no-asr", action="store_true", help="bỏ CER, chỉ đo LTAS + im lặng")
    ap.add_argument("--cer-warn", type=float, default=0.15)
    ap.add_argument("--ltas-warn", type=float, default=0.95)
    ap.add_argument("-o", "--out", default=str(ROOT / "output" / "ab_quant.html"))
    args = ap.parse_args()

    pa, pb = Path(args.a).resolve(), Path(args.b).resolve()
    out = Path(args.out).resolve()
    parts_a = sort_parts(pa.parent.glob(f"{pa.stem}-*.wav"))
    parts_b = sort_parts(pb.parent.glob(f"{pb.stem}-*.wav"))
    if not parts_a or len(parts_a) != len(parts_b):
        raise SystemExit(f"số đoạn lệch nhau: {len(parts_a)} vs {len(parts_b)}")

    paras: list[str] = []
    if args.script:
        t = Path(args.script).read_text(encoding="utf-8")
        paras = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", t) if p.strip()]

    do_asr = not args.no_asr and bool(paras)
    hdr = f"{'#':>3s} {'giây A':>8s} {'giây B':>8s} {'LTAS':>7s} {'im lặng A/B':>13s}"
    if do_asr:
        hdr += f" {'CER A':>7s} {'CER B':>7s} {'Δ':>7s}"
    print(hdr)
    print("-" * len(hdr))

    rows, cards = [], []
    for i, (fa, fb) in enumerate(zip(parts_a, parts_b), 1):
        a, b = read_wav(fa), read_wav(fb)
        da, db = len(a) / SR, len(b) / SR
        lt = ltas_cos(a, b)
        sa, sb = silence_ratio(a), silence_ratio(b)

        ca = cb = None
        ha = hb = ""
        if do_asr and i <= len(paras):
            ha, hb = transcribe(fa, args.lang, args.asr_model), transcribe(fb, args.lang, args.asr_model)
            ca, cb = cer(paras[i - 1], ha), cer(paras[i - 1], hb)

        flag = []
        if lt < args.ltas_warn:
            flag.append("âm sắc lệch")
        if abs(sa - sb) > 0.10:
            flag.append("im lặng lệch")
        if ca is not None and ca > args.cer_warn:
            flag.append(f"{args.label_a} CER cao")
        if cb is not None and cb > args.cer_warn:
            flag.append(f"{args.label_b} CER cao")

        rows.append((i, da, db, lt, sa, sb, ca, cb, flag, ha, hb))
        line = f"{i:>3d} {da:>7.2f}s {db:>7.2f}s {lt:>7.3f} {sa * 100:>5.1f}%/{sb * 100:>5.1f}%"
        if ca is not None:
            line += f" {ca * 100:>6.1f}% {cb * 100:>6.1f}% {(ca - cb) * 100:>+6.1f}%"
        print(line + ("  " + " · ".join(flag) if flag else ""))

        rel_a = urllib.parse.quote(fa.relative_to(out.parent).as_posix())
        rel_b = urllib.parse.quote(fb.relative_to(out.parent).as_posix())
        cls_a = "bad" if (ca is not None and ca > args.cer_warn) else "good"
        cls_b = "bad" if (cb is not None and cb > args.cer_warn) else "good"
        metrics = [
            f'<span class="m">LTAS <b class="{"bad" if lt < args.ltas_warn else "good"}">{lt:.3f}</b></span>',
            f'<span class="m">{da:.2f}s / {db:.2f}s</span>',
            f'<span class="m">im lặng {sa * 100:.1f}% / {sb * 100:.1f}%</span>',
        ]
        if ca is not None:
            metrics.insert(
                0,
                f'<span class="m">CER <b class="{cls_a}">{ca * 100:.1f}%</b>'
                f' / <b class="{cls_b}">{cb * 100:.1f}%</b></span>',
            )
        body = [
            '<div class="card"><div class="hdr">',
            f'<span class="n">{i}</span>',
            *metrics,
            f'<span class="flag">{" · ".join(flag)}</span>' if flag else "",
            '</div><div class="pair">',
            f'<span>{html.escape(args.label_a)}</span>',
            f'<audio controls preload="none" src="{rel_a}"></audio>',
            f'<span>{html.escape(args.label_b)}</span>',
            f'<audio controls preload="none" src="{rel_b}"></audio>',
            "</div>",
        ]
        if i <= len(paras):
            body.append(f'<div class="txt"><b>gốc:</b> {html.escape(paras[i - 1])}</div>')
        if ha:
            body.append(f'<div class="txt"><b>ASR {html.escape(args.label_a)}:</b> {html.escape(ha)}</div>')
        if hb:
            body.append(f'<div class="txt"><b>ASR {html.escape(args.label_b)}:</b> {html.escape(hb)}</div>')
        body.append("</div>")
        cards.append("".join(body))

    lt_all = [r[3] for r in rows]
    print("-" * len(hdr))
    print(f"LTAS   : min {min(lt_all):.3f}  trung vị {np.median(lt_all):.3f}")
    if do_asr:
        ca_all = [r[6] for r in rows if r[6] is not None]
        cb_all = [r[7] for r in rows if r[7] is not None]
        print(
            f"CER {args.label_a:<12s}: trung vị {np.median(ca_all) * 100:.2f}%  "
            f"max {max(ca_all) * 100:.2f}%\n"
            f"CER {args.label_b:<12s}: trung vị {np.median(cb_all) * 100:.2f}%  "
            f"max {max(cb_all) * 100:.2f}%\n"
            f"chênh trung vị: {(np.median(ca_all) - np.median(cb_all)) * 100:+.2f} điểm phần trăm"
        )
    print(f"số đoạn bị gắn cờ: {sum(1 for r in rows if r[8])}/{len(rows)}")

    cols = ["đoạn", "giây A", "giây B", "LTAS", "im lặng A", "im lặng B"]
    if do_asr:
        cols += ["CER A", "CER B"]
    tbl = ["<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"]
    for i, da, db, lt, sa, sb, ca, cb, flag, _, _ in rows:
        c = "bad" if flag else ""
        cells = [
            f"{i}",
            f"{da:.2f}",
            f"{db:.2f}",
            f'<span class="{c}">{lt:.3f}</span>',
            f"{sa * 100:.1f}%",
            f"{sb * 100:.1f}%",
        ]
        if do_asr:
            cells += [f"{ca * 100:.1f}%", f"{cb * 100:.1f}%"]
        tbl.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
    foot = ["trung vị", "", "", f"{np.median(lt_all):.3f}", "", ""]
    if do_asr:
        foot += [
            f"{np.median([r[6] for r in rows if r[6] is not None]) * 100:.1f}%",
            f"{np.median([r[7] for r in rows if r[7] is not None]) * 100:.1f}%",
        ]
    tbl.append("<tfoot><tr>" + "".join(f"<td>{v}</td>" for v in foot) + "</tr></tfoot></tbody></table>")

    note = (
        "Cùng text, cùng giọng mẫu, cùng seed, khác mức lượng tử hoá. "
        "<code>CER</code> là sai số ký tự khi cho ASR nghe lại rồi so với text gốc — "
        "ASR tự nó cũng sai, nên hãy xem <b>hiệu số giữa hai cột</b>, không phải giá trị "
        "tuyệt đối. <code>LTAS</code> là phổ trung bình dài hạn, gần 1.0 nghĩa là cùng "
        "âm sắc. Sóng âm hai bản không tương quan là chuyện bình thường: đây là hai lần "
        "lấy mẫu ngẫu nhiên khác nhau, không phải hai bản sao."
    )
    doc = "\n".join(
        [
            f"<title>{html.escape(args.label_a)} vs {html.escape(args.label_b)}</title>",
            f"<style>{CSS}</style>",
            '<div class="wrap">',
            f"<h1>{html.escape(args.label_a)} vs {html.escape(args.label_b)}</h1>",
            f'<p class="note">{note}</p>',
            "<h2>Bản đầy đủ</h2>",
            f'<div class="full"><div class="lab">{html.escape(args.label_a)}</div>'
            f'<audio controls preload="none" src="{urllib.parse.quote(pa.relative_to(out.parent).as_posix())}"></audio></div>',
            f'<div class="full"><div class="lab">{html.escape(args.label_b)}</div>'
            f'<audio controls preload="none" src="{urllib.parse.quote(pb.relative_to(out.parent).as_posix())}"></audio></div>',
            "<h2>Số liệu từng đoạn</h2>",
            "".join(tbl),
            "<h2>Nghe đối chiếu từng đoạn</h2>",
            "".join(cards),
            "</div>",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
