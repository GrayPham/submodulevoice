"""Lồng tiếng một file .srt theo đúng timeline, giữ giọng clone.

    python examples/dub_srt.py phim.srt --ref-wav giong.wav --ref-text "..." -o dub.wav

WAV kết quả là 24 kHz mono, mỗi cue nằm đúng slot thời gian của nó, nên ghép
thẳng vào video được:

    ffmpeg -i phim.mp4 -i dub.wav -c:v copy -map 0:v -map 1:a phim_long_tieng.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import OmniVoice  # noqa: E402
from pyomnivoice.srt import read_srt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("srt")
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--ref-text", required=True, help="transcript của ref-wav, hoặc đường dẫn file .txt")
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"])
    ap.add_argument("--profile", default="lite", choices=["lite", "balanced", "quality", "reference"])
    ap.add_argument("-o", "--out", default="dub.wav")
    args = ap.parse_args()

    ref_text = args.ref_text
    if Path(ref_text).is_file():
        ref_text = Path(ref_text).read_text(encoding="utf-8").strip()

    cues = read_srt(args.srt)
    total = sum(c.slot for c in cues)
    print(f"{len(cues)} cue, tổng {total:.1f}s timeline")

    tts = OmniVoice(profile=args.profile, backend=args.backend)
    print(f"backend: {tts.backend}")
    voice = tts.load_voice(args.ref_wav, ref_text)

    t0 = time.perf_counter()

    def show(i: int, n: int, cue) -> None:  # noqa: ANN001
        print(f"  [{i}/{n}] {cue.t0:7.2f}s +{cue.slot:5.2f}s  {cue.text[:52]}")

    audio = tts.dub_srt(
        args.srt, voice=voice, lang=args.lang, steps=args.steps, seed=42, progress=show
    )
    wall = time.perf_counter() - t0
    audio.save(args.out)
    print(f"\n{args.out}  |  {audio.duration:.1f}s trong {wall:.1f}s = {audio.duration / wall:.2f}x realtime")
    tts.close()


if __name__ == "__main__":
    main()
