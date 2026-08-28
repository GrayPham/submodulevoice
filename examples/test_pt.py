"""Thử clone giọng sang tiếng Bồ Đào Nha Brazil.

    python examples/test_pt.py

Không có sẵn giọng mẫu tiếng Bồ trên máy, nên đây là bài thử CLONE XUYÊN
NGÔN NGỮ: lấy giọng mẫu tiếng Việt hoặc tiếng Anh rồi bắt đọc tiếng Bồ. Đây
đúng là tình huống thật khi trong tay chỉ có giọng Việt mà khách cần tiếng Bồ.

Ba thứ được đo, vì tai người không đọc được ba thứ đó cùng lúc:
  CER       cho ASR nghe lại rồi so với kịch bản gốc — đo mức đọc đúng chữ.
  ngôn ngữ  để Whisper TỰ đoán ngôn ngữ. Nếu nó đoán ra 'pt' với xác suất
            cao thì đầu ra thật sự nghe như tiếng Bồ, không phải tiếng Việt
            đọc chữ Bồ.
  thời lượng so với kịch bản, để biết có bị đọc dồn hay bỏ chữ không.
"""

from __future__ import annotations

import html
import re
import sys
import urllib.parse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_quant import cer  # noqa: E402
from pyomnivoice import Audio, OmniVoice, SAMPLE_RATE  # noqa: E402
from pyomnivoice.refprep import load_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "pt"
SCRIPT = ROOT / "scripts" / "kichban_pt.txt"

REF_VI = ROOT / "output" / "refs3" / "FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav"
REF_EN = ROOT / "omnivoice.cpp" / "examples" / "freeman.wav"
REF_EN_TXT = ROOT / "omnivoice.cpp" / "examples" / "freeman.txt"

# Whisper tiếng Bồ tự chuyển số nói thành chữ số ("seiscentos" -> "600"), nên so
# thẳng với kịch bản gốc sẽ tính nhầm thành lỗi đọc. Khớp lại cách viết trước khi
# tính CER, và in cả hai con số để thấy phần nào là lỗi thật.
NUM_ALIGN = [("seiscentos", "600"), ("mil duzentos e trinta e quatro", "1234")]


def align_numbers(t: str) -> str:
    for word, digit in NUM_ALIGN:
        t = t.replace(word, digit)
    return t


def paragraphs() -> list[str]:
    t = SCRIPT.read_text(encoding="utf-8")
    return [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", t) if p.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paras = paragraphs()
    full = " ".join(paras)
    print(f"kịch bản: {len(paras)} đoạn, {sum(len(p) for p in paras)} ký tự\n")

    from faster_whisper import WhisperModel

    asr = WhisperModel("medium", device="cpu", compute_type="int8")

    tts = OmniVoice(profile="quality", backend="cuda")
    print(f"backend {tts.backend}  |  {tts.model_path.name} + {tts.codec_path.name}\n")

    ref_vi = load_reference(REF_VI)
    voice_vi = ref_vi.as_voice(tts)
    voice_en = tts.load_voice(REF_EN, REF_EN_TXT.read_text(encoding="utf-8").strip())

    arms = [
        ("vi-ref", "Giọng mẫu tiếng Việt (6.0s)", dict(voice=voice_vi), "Portuguese"),
        ("en-ref", "Giọng mẫu tiếng Anh (17.3s)", dict(voice=voice_en), "Portuguese"),
        ("design", "Voice design, không giọng mẫu", dict(instruct="male, young adult, moderate pitch"), "Portuguese"),
        ("vi-nolang", "Giọng Việt, KHÔNG khai báo ngôn ngữ", dict(voice=voice_vi), "None"),
    ]

    print(f"{'nhánh':<12s} {'giây':>7s} {'audio':>8s} {'ký tự/s':>8s} "
          f"{'CER thô':>8s} {'CER số đã khớp':>15s} {'ASR đoán':>12s}")
    print("-" * 78)
    rows = []
    for key, label, kw, lang in arms:
        pieces, gap = [], np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
        wall = 0.0
        for p in paras:
            a = tts.say(p, lang=lang, steps=16, seed=42, **kw)
            wall += tts.last_wall
            if pieces:
                pieces.append(gap)
            pieces.append(a.samples)
        audio = Audio(np.concatenate(pieces), SAMPLE_RATE)
        wav = OUT / f"pt-{key}.wav"
        audio.save(wav)

        segs, info = asr.transcribe(str(wav), language="pt", vad_filter=True, beam_size=5)
        heard = " ".join(s.text.strip() for s in segs)
        c = cer(full, heard)
        c_aligned = cer(align_numbers(full), heard)
        # lần hai để Whisper tự đoán ngôn ngữ
        _s2, info2 = asr.transcribe(str(wav), vad_filter=True, beam_size=1)
        det = f"{info2.language} {info2.language_probability * 100:.0f}%"

        rows.append({"key": key, "label": label, "wav": wav.name, "wall": wall,
                     "dur": audio.duration, "cer": c, "cer_aligned": c_aligned,
                     "cps": len(full) / audio.duration, "det": det, "heard": heard})
        print(f"{key:<12s} {wall:>6.2f}s {audio.duration:>7.2f}s "
              f"{len(full) / audio.duration:>8.1f} {c * 100:>7.2f}% "
              f"{c_aligned * 100:>14.2f}% {det:>12s}")

    tts.close()

    css = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#6b6560;--line:#e3ded7;--card:#fff;--accent:#8a5a2b;--bad:#a8442a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161513;--fg:#eeebe6;
--mut:#a39c94;--line:#302d29;--card:#1e1d1a;--accent:#d9a066;--bad:#e0806a}}
*{box-sizing:border-box}body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",sans-serif}.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .35rem}.note{color:var(--mut);margin:0 0 1.6rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.85rem 1rem;margin-bottom:.6rem}
.hdr{display:flex;gap:.8rem;align-items:baseline;flex-wrap:wrap;margin-bottom:.5rem}
.hdr b.n{color:var(--accent)}.m{color:var(--mut);font-size:.85rem;font-variant-numeric:tabular-nums}
audio{width:100%;height:34px}.txt{color:var(--mut);font-size:.85rem;margin-top:.5rem;
border-left:2px solid var(--line);padding-left:.7rem}
"""
    doc = ["<title>Voice clone tiếng Bồ Brazil</title>", f"<style>{css}</style>",
           '<div class="wrap">', "<h1>Clone xuyên ngôn ngữ sang tiếng Bồ Brazil</h1>",
           '<p class="note">Cùng kịch bản, cùng seed, khác giọng mẫu. <b>ASR đoán</b> là ngôn '
           'ngữ mà Whisper tự nhận ra khi nghe — đây mới là thước đo đầu ra có thật sự nghe '
           'như tiếng Bồ hay không, còn CER chỉ đo đọc đúng chữ. CER "thô" bị thổi lên vì '
           'Whisper tiếng Bồ tự viết số nói thành chữ số; con số đứng trước là sau khi khớp '
           'lại cách viết. Chú ý cột <b>ký tự/giây</b>: giọng mẫu quyết định tốc độ đọc.</p>']
    for r in rows:
        doc.append(
            '<div class="card"><div class="hdr">'
            f'<b class="n">{html.escape(r["label"])}</b>'
            f'<span class="m">CER {r["cer"] * 100:.2f}%</span>'
            f'<span class="m">ASR đoán: {html.escape(r["det"])}</span>'
            f'<span class="m">{r["dur"]:.1f}s trong {r["wall"]:.1f}s</span>'
            f'</div><audio controls preload="none" src="{urllib.parse.quote(r["wav"])}"></audio>'
            f'<div class="txt"><b>ASR nghe:</b> {html.escape(r["heard"][:400])}</div></div>')
    doc.append(f'<div class="card"><b>Kịch bản gốc</b><div class="txt">{html.escape(full)}</div></div>')
    doc.append("</div>")
    idx = OUT / "index.html"
    idx.write_text("\n".join(doc), encoding="utf-8")
    print(f"\n{idx}")


if __name__ == "__main__":
    main()
