"""Voice clone demo + realtime-factor benchmark, pure Python.

    python examples/demo_clone.py --backend cpu  --profile lite
    python examples/demo_clone.py --backend cuda --profile quality
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import OmniVoice  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF_WAV = ROOT / "omnivoice.cpp" / "examples" / "freeman.wav"
REF_TXT = ROOT / "omnivoice.cpp" / "examples" / "freeman.txt"

TEXT_VI = (
    "Theo thong tin tu So Y te Ha Noi, tu ngay muoi tam thang tam den ngay bon thang chin, "
    "thanh pho phan dau mot tram phan tram tre em duoi sau tuoi trong cac co so giao duc mam non "
    "va hoc sinh tu sau tuoi den duoi muoi tam tuoi trong cac co so giao duc pho thong tren dia ban "
    "thanh pho Ha Noi se duoc kham suc khoe dinh ky mien phi."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["auto", "cpu", "cuda", "vulkan"])
    ap.add_argument("--profile", default="lite", choices=["lite", "balanced", "quality", "reference"])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--text", default=TEXT_VI)
    ap.add_argument("--ref-wav", default=str(REF_WAV))
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("-o", "--out", default=str(ROOT / "output" / "clone.wav"))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref_text = args.ref_text
    if ref_text is None:
        ref_text = REF_TXT.read_text(encoding="utf-8").strip()

    tts = OmniVoice(profile=args.profile, backend=args.backend, verbose=args.verbose)
    print(f"omnivoice   : {tts.version}")
    print(f"backend     : {tts.backend}")
    print(f"model       : {tts.model_path.name} + {tts.codec_path.name}")
    print(f"load        : {tts.load_time:.2f} s")
    print(f"codebooks   : {tts.num_codebooks}")

    t0 = time.perf_counter()
    voice = tts.load_voice(args.ref_wav, ref_text)
    print(f"ref encode  : {time.perf_counter() - t0:.2f} s  ({voice.n_frames} frames)")

    for i in range(args.runs):
        audio = tts.say(args.text, voice=voice, lang=args.lang, steps=args.steps, seed=42)
        out = Path(args.out)
        if args.runs > 1:
            out = out.with_name(f"{out.stem}-{i + 1}{out.suffix}")
        audio.save(out)
        print(
            f"run {i + 1}       : {tts.last_wall:6.2f} s wall  ->  "
            f"{audio.duration:6.2f} s audio  =  "
            f"{audio.duration / tts.last_wall:5.2f}x realtime  (RTF {tts.last_rtf:.3f})"
        )
        print(f"              {out}")

    tts.close()


if __name__ == "__main__":
    main()
