"""Đọc một kịch bản dài, đo tốc độ, xuất WAV + báo cáo từng đoạn.

    python examples/longform.py scripts/kichban_dai.txt \
        --ref ../ToolEdit/assets/voice_previews/nghitts__vi_Minh_Quang.mp3

Hai chế độ:

  --mode paragraph  (mặc định)  Tách theo dòng trống, tổng hợp từng đoạn rồi
                                nối lại với khoảng lặng. Kiểm soát được nhịp,
                                làm lại từng đoạn được, bộ nhớ có giới hạn,
                                và biết đoạn nào chậm.
  --mode auto                   Một lệnh say() duy nhất, để bộ chunker trong
                                C++ tự cắt theo dấu câu. Ngắn gọn hơn nhưng
                                mất khả năng can thiệp giữa chừng.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyomnivoice import Audio, OmniVoice, SAMPLE_RATE  # noqa: E402
from pyomnivoice.refprep import load_reference, prepare_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [re.sub(r"\s+", " ", p) for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="file .txt, các đoạn cách nhau bằng dòng trống")
    ap.add_argument("--ref", help="file audio giọng mẫu (mp3/wav/...), sẽ tự cắt + transcribe")
    ap.add_argument("--ref-wav", help="WAV mẫu đã chuẩn bị sẵn (cần .txt cùng tên)")
    ap.add_argument("--instruct", help="thay vì clone, mô tả giọng: 'male, young adult, moderate pitch'")
    ap.add_argument("--asr-model", default="medium", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--mode", default="paragraph", choices=["paragraph", "auto"])
    ap.add_argument("--pause", type=float, default=0.45, help="khoảng lặng giữa các đoạn (giây)")
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "vulkan"])
    ap.add_argument("--profile", default="lite", choices=["lite", "balanced", "quality", "reference"])
    ap.add_argument("--chunk-duration", type=float, default=15.0)
    ap.add_argument("--chunk-threshold", type=float, default=30.0)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--keep-parts", action="store_true", help="giữ WAV riêng của từng đoạn")
    args = ap.parse_args()

    script_path = Path(args.script)
    text = script_path.read_text(encoding="utf-8")
    paras = split_paragraphs(text)
    n_chars = sum(len(p) for p in paras)
    print(f"kịch bản : {script_path.name}  |  {len(paras)} đoạn, {n_chars} ký tự")

    out_path = Path(args.out) if args.out else ROOT / "output" / f"{script_path.stem}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tts = OmniVoice(profile=args.profile, backend=args.backend)
    print(f"backend  : {tts.backend}  |  {tts.model_path.name} + {tts.codec_path.name}")

    voice = None
    if args.ref_wav:
        ref = load_reference(args.ref_wav)
        voice = ref.as_voice(tts)
        print(f"giọng mẫu: {ref.wav_path.name} ({ref.duration:.2f}s)")
        print(f"           {ref.text}")
    elif args.ref:
        t0 = time.perf_counter()
        ref = prepare_reference(
            args.ref, out_dir=ROOT / "output" / "refs", lang="vi", model_size=args.asr_model
        )
        voice = ref.as_voice(tts)
        print(f"giọng mẫu: {ref.wav_path.name} ({ref.duration:.2f}s, ASR {time.perf_counter() - t0:.1f}s)")
        print(f"           {ref.text}")
        print(f"           ^ kiểm tra lại câu này, sai chữ nào thì sửa trong {ref.txt_path.name}")
    elif args.instruct:
        print(f"giọng    : voice design '{args.instruct}'")
    else:
        print("giọng    : auto voice (không có mẫu)")

    common = dict(
        voice=voice,
        instruct=args.instruct,
        lang=args.lang,
        steps=args.steps,
        seed=args.seed,
    )

    t_start = time.perf_counter()

    if args.mode == "auto":
        audio = tts.say(
            " ".join(paras),
            chunk_duration=args.chunk_duration,
            chunk_threshold=args.chunk_threshold,
            **common,
        )
        pieces = [audio.samples]
        rows = [("toàn bộ", tts.last_wall, audio.duration)]
    else:
        gap = np.zeros(int(args.pause * SAMPLE_RATE), dtype=np.float32)
        pieces, rows = [], []
        print(f"\n{'#':>3s} {'ký tự':>6s} {'wall':>8s} {'audio':>8s} {'xRT':>7s}")
        print("-" * 40)
        for i, para in enumerate(paras, 1):
            a = tts.say(
                para,
                chunk_duration=args.chunk_duration,
                chunk_threshold=args.chunk_threshold,
                **common,
            )
            if pieces:
                pieces.append(gap)
            pieces.append(a.samples)
            rows.append((f"đoạn {i}", tts.last_wall, a.duration))
            print(
                f"{i:>3d} {len(para):>6d} {tts.last_wall:>7.2f}s {a.duration:>7.2f}s "
                f"{a.duration / tts.last_wall:>6.2f}x"
            )
            if args.keep_parts:
                w = len(str(len(paras)))
                Audio(a.samples).save(out_path.with_name(f"{out_path.stem}-{i:0{w}d}.wav"))

    total_wall = time.perf_counter() - t_start
    final = Audio(np.concatenate(pieces), SAMPLE_RATE)
    final.save(out_path)

    synth_wall = sum(r[1] for r in rows)
    print("-" * 40)
    print(f"tổng     : {final.duration:.1f}s audio trong {total_wall:.1f}s")
    print(f"           tổng hợp {synth_wall:.1f}s = {final.duration / synth_wall:.2f}x realtime")
    print(f"           {out_path}")

    peak = float(np.abs(final.samples).max())
    clipped = int((np.abs(final.samples) >= 0.999).sum())
    silence = float((np.abs(final.samples) < 1e-4).mean())
    print(f"kiểm tra : peak {peak:.3f} | mẫu clip {clipped} | tỉ lệ im lặng {silence * 100:.1f}%")
    if peak < 0.1:
        print("           ⚠ tín hiệu quá nhỏ, kiểm tra lại giọng mẫu")
    if silence > 0.35:
        print("           ⚠ im lặng nhiều bất thường, có thể model đã bỏ chữ")

    tts.close()


if __name__ == "__main__":
    main()
