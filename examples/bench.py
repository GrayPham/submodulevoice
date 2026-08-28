"""Sweep the knobs that actually move the needle and print realtime factors.

    python examples/bench.py --backend cpu
    python examples/bench.py --backend cuda

Cost model: MaskGIT is NOT autoregressive. Every one of the `steps`
iterations runs a full bidirectional forward over (reference frames +
target frames). So wall time scales with
    steps x forward(ref_frames + target_frames)
which makes step count and reference length the two biggest levers,
ahead of quantisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import OmniVoice, read_wav_24k  # noqa: E402
from pyomnivoice.core import Audio, SAMPLE_RATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF_WAV = ROOT / "omnivoice.cpp" / "examples" / "freeman.wav"
REF_TXT = ROOT / "omnivoice.cpp" / "examples" / "freeman.txt"

# ~6 s of the reference, with the matching transcript slice.
REF_SHORT_TEXT = (
    "If you go into different cultures, they have different concepts of creation. "
    "They have their own creation story"
)

TEXT = (
    "Theo thong tin tu So Y te Ha Noi, tu ngay muoi tam thang tam den ngay bon thang chin, "
    "thanh pho phan dau mot tram phan tram tre em duoi sau tuoi trong cac co so giao duc mam non "
    "va hoc sinh tu sau tuoi den duoi muoi tam tuoi trong cac co so giao duc pho thong tren dia ban "
    "thanh pho Ha Noi se duoc kham suc khoe dinh ky mien phi."
)


def make_short_ref(seconds: float) -> Path:
    pcm = read_wav_24k(REF_WAV)[: int(seconds * SAMPLE_RATE)]
    out = ROOT / "output" / f"ref-{seconds:g}s.wav"
    Audio(pcm).save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["auto", "cpu", "cuda", "vulkan"])
    ap.add_argument("--profile", default="lite", choices=["lite", "balanced", "quality", "reference"])
    ap.add_argument("--lang", default="Vietnamese")
    args = ap.parse_args()

    ref_long = str(REF_WAV)
    ref_long_text = REF_TXT.read_text(encoding="utf-8").strip()
    ref_short = str(make_short_ref(6.0))

    tts = OmniVoice(profile=args.profile, backend=args.backend)
    print(f"backend {tts.backend} | {tts.model_path.name} + {tts.codec_path.name}\n")

    v_long = tts.load_voice(ref_long, ref_long_text)
    v_short = tts.load_voice(ref_short, REF_SHORT_TEXT)

    cases = [
        ("ref 17s, 32 steps, single shot", v_long, 32, 30.0),
        ("ref  6s, 32 steps, single shot", v_short, 32, 30.0),
        ("ref  6s, 16 steps, single shot", v_short, 16, 30.0),
        ("ref  6s,  8 steps, single shot", v_short, 8, 30.0),
        ("ref  6s, 16 steps, chunk 10s", v_short, 16, 8.0),
    ]

    print(f"{'case':34s} {'wall':>8s} {'audio':>8s} {'xRT':>7s}")
    print("-" * 60)
    for name, voice, steps, thr in cases:
        audio = tts.say(
            TEXT,
            voice=voice,
            lang=args.lang,
            steps=steps,
            seed=42,
            chunk_threshold=thr,
            chunk_duration=10.0,
        )
        audio.save(ROOT / "output" / f"bench-{name.replace(' ', '_').replace(',', '')}.wav")
        xrt = audio.duration / tts.last_wall
        print(f"{name:34s} {tts.last_wall:7.2f}s {audio.duration:7.2f}s {xrt:6.2f}x")

    tts.close()


if __name__ == "__main__":
    main()
