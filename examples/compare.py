"""Tạo bảng so sánh audio + trang HTML để nghe và chọn.

Hai chế độ:

  # 1. Quét tham số: cùng một giọng, đổi steps và profile
  python examples/compare.py grid --ref giong.mp3 --steps 8 16 32 --profiles lite quality

  # 2. Thư viện giọng: cùng một câu, chạy qua mọi file audio trong thư mục
  python examples/compare.py voices --dir ../ToolEdit/assets/voice_previews

Kết quả: output/compare/grid.html và output/compare/voices.html — mở bằng
browser, bấm play từng ô để nghe, số liệu tốc độ nằm ngay cạnh.
"""

from __future__ import annotations

import argparse
import html
import sys
import urllib.parse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import OmniVoice  # noqa: E402
from pyomnivoice.refprep import prepare_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "compare"

DEFAULT_TEXT = (
    "Theo thông tin từ Sở Y tế Hà Nội, từ ngày 18 tháng 8 đến ngày 4 tháng 9, "
    "thành phố phấn đấu 100% trẻ em dưới sáu tuổi sẽ được khám sức khỏe định kỳ miễn phí."
)

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}


def render_html(title: str, note: str, groups: list[tuple[str, list[dict]]]) -> str:
    css = """
:root { --bg:#fbfaf8; --fg:#1c1a17; --mut:#6b6560; --line:#e3ded7; --card:#fff; --accent:#8a5a2b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#161513; --fg:#eeebe6; --mut:#a39c94; --line:#302d29; --card:#1e1d1a; --accent:#d9a066; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1000px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }
.note { color:var(--mut); margin:0 0 2rem; }
.note code { background:var(--line); padding:.1em .35em; border-radius:3px; }
h2 { font-size:1rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut);
  margin:2.2rem 0 .8rem; font-weight:600; }
.row { display:grid; grid-template-columns:minmax(9rem,14rem) 1fr minmax(7rem,auto);
  gap:1rem; align-items:center; padding:.7rem .9rem; background:var(--card);
  border:1px solid var(--line); border-radius:8px; margin-bottom:.5rem; }
.lab { font-weight:600; }
.sub { color:var(--mut); font-size:.8rem; font-weight:400; display:block; margin-top:.15rem;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
audio { width:100%; height:34px; }
.num { text-align:right; font-variant-numeric:tabular-nums; color:var(--mut); font-size:.85rem; }
.num b { color:var(--accent); font-size:1rem; }
@media (max-width:640px) { .row { grid-template-columns:1fr; } .num { text-align:left; } }
"""
    parts = [
        f"<title>{html.escape(title)}</title>",
        f"<style>{css}</style>",
        '<div class="wrap">',
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="note">{note}</p>',
    ]
    for gname, rows in groups:
        parts.append(f"<h2>{html.escape(gname)}</h2>")
        for r in rows:
            sub = f'<span class="sub">{html.escape(r.get("sub", ""))}</span>' if r.get("sub") else ""
            parts.append(
                '<div class="row">'
                f'<div class="lab">{html.escape(r["label"])}{sub}</div>'
                f'<div><audio controls preload="none" src="{urllib.parse.quote(r["file"])}"></audio></div>'
                f'<div class="num"><b>{r["xrt"]:.2f}x</b><br>{r["wall"]:.2f}s → {r["dur"]:.1f}s</div>'
                "</div>"
            )
    parts.append("</div>")
    return "\n".join(parts)


def mode_grid(args) -> None:  # noqa: ANN001
    OUT.mkdir(parents=True, exist_ok=True)
    ref = prepare_reference(
        args.ref, out_dir=OUT / "refs", lang="vi", model_size=args.asr_model
    )
    print(f"giọng mẫu {ref.wav_path.name} ({ref.duration:.2f}s)\n  {ref.text}\n")

    groups = []
    for profile in args.profiles:
        tts = OmniVoice(profile=profile, backend=args.backend)
        voice = ref.as_voice(tts)
        rows = []
        for steps in args.steps:
            name = f"{profile}-{steps}steps.wav"
            audio = tts.say(args.text, voice=voice, lang=args.lang, steps=steps, seed=args.seed)
            audio.save(OUT / name)
            rows.append(
                {
                    "label": f"{steps} steps",
                    "sub": tts.model_path.name,
                    "file": name,
                    "wall": tts.last_wall,
                    "dur": audio.duration,
                    "xrt": audio.duration / tts.last_wall,
                }
            )
            print(f"  {profile:9s} {steps:>2d} steps  {tts.last_wall:6.2f}s  {audio.duration / tts.last_wall:5.2f}x")
        groups.append((f"profile {profile} — backend {tts.backend}", rows))
        tts.close()

    note = (
        f"Cùng một câu, cùng giọng mẫu <code>{html.escape(ref.wav_path.name)}</code>, seed "
        f"{args.seed}. Nghe từ trên xuống: chỗ nào bắt đầu nghe méo thì đó là giới hạn "
        "steps bạn có thể hạ xuống."
    )
    idx = OUT / "grid.html"
    idx.write_text(render_html("So sánh steps / profile", note, groups), encoding="utf-8")
    print(f"\n{idx}")


def mode_voices(args) -> None:  # noqa: ANN001
    OUT.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.dir)
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"không có file audio nào trong {src_dir}")
    print(f"{len(files)} giọng mẫu từ {src_dir}\n")

    tts = OmniVoice(profile=args.profile, backend=args.backend)
    rows = []
    for i, f in enumerate(files, 1):
        try:
            ref = prepare_reference(f, out_dir=OUT / "refs", lang="vi", model_size=args.asr_model)
            voice = ref.as_voice(tts)
            audio = tts.say(args.text, voice=voice, lang=args.lang, steps=args.steps, seed=args.seed)
        except Exception as e:  # một file lỗi không được làm chết cả loạt
            print(f"  [{i}/{len(files)}] {f.name}: {type(e).__name__} {e}")
            continue
        name = f"voice-{f.stem}.wav"
        audio.save(OUT / name)
        rows.append(
            {
                "label": f.stem.replace("nghitts__vi_", "").replace("_", " "),
                "sub": ref.text[:90],
                "file": name,
                "wall": tts.last_wall,
                "dur": audio.duration,
                "xrt": audio.duration / tts.last_wall,
            }
        )
        print(f"  [{i}/{len(files)}] {f.stem:38s} {tts.last_wall:6.2f}s  {audio.duration / tts.last_wall:5.2f}x")
    tts.close()

    note = (
        "Cùng một câu, mỗi hàng là một giọng mẫu khác nhau. Dòng chữ nhỏ là transcript "
        "mà ASR nghe được từ file mẫu — <b>nếu transcript sai nhiều thì giọng nhân bản "
        "cũng kém</b>, sửa lại file <code>.txt</code> tương ứng trong "
        "<code>output/compare/refs/</code> rồi chạy lại."
    )
    idx = OUT / "voices.html"
    idx.write_text(render_html("Thư viện giọng tiếng Việt", note, [("giọng", rows)]), encoding="utf-8")
    print(f"\n{idx}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--text", default=DEFAULT_TEXT)
    common.add_argument("--lang", default="Vietnamese")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"])
    common.add_argument("--asr-model", default="small", help="tiny|base|small|medium|large-v3")

    g = sub.add_parser("grid", parents=[common])
    g.add_argument("--ref", required=True)
    g.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32])
    g.add_argument("--profiles", nargs="+", default=["lite", "quality"])

    v = sub.add_parser("voices", parents=[common])
    v.add_argument("--dir", required=True)
    v.add_argument("--steps", type=int, default=16)
    v.add_argument("--profile", default="lite")
    v.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()
    t0 = time.perf_counter()
    (mode_grid if args.mode == "grid" else mode_voices)(args)
    print(f"[{time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main()
