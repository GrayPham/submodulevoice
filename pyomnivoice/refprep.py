"""Chuẩn bị giọng mẫu từ một file audio bất kỳ.

Voice clone cần WAV mẫu **và** transcript khớp chính xác. Module này lo cả hai:
cắt một đoạn 4-8 giây sạch từ file nguồn (mp3/wav/m4a...) và tự lấy transcript
bằng faster-whisper, cắt đúng ranh giới segment nên chữ luôn khớp tiếng.

    from pyomnivoice.refprep import prepare_reference
    ref = prepare_reference("giong_mau.mp3", lang="vi")
    print(ref.wav_path, ref.text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import SAMPLE_RATE, Audio, read_wav_24k

# Reference dài quá thì mỗi bước MaskGIT phải xử lý lại toàn bộ frame tham
# chiếu; ngắn quá thì không đủ đặc trưng giọng. 4-8 giây là vùng hợp lý.
MIN_SEC = 3.0
TARGET_SEC = 6.0
MAX_SEC = 9.0

_MODEL_CACHE: dict[tuple[str, str], object] = {}


@dataclass
class Reference:
    wav_path: Path
    text: str
    duration: float
    source: Path

    @property
    def txt_path(self) -> Path:
        return self.wav_path.with_suffix(".txt")

    def as_voice(self, tts):  # noqa: ANN001
        """Encode thành Voice dùng lại được."""
        return tts.load_voice(self.wav_path, self.text)

    def write_txt(self) -> Path:
        """Ghi transcript ra file .txt cạnh WAV để sửa tay."""
        self.txt_path.write_text(self.text, encoding="utf-8")
        return self.txt_path


def load_reference(wav_path: str | os.PathLike) -> Reference:
    """Nạp lại reference đã chuẩn bị, ưu tiên transcript đã sửa tay trong .txt."""
    wav_path = Path(wav_path)
    txt = wav_path.with_suffix(".txt")
    if not txt.exists():
        raise FileNotFoundError(f"thiếu transcript {txt}")
    pcm = read_wav_24k(wav_path)
    return Reference(
        wav_path,
        txt.read_text(encoding="utf-8").strip(),
        len(pcm) / SAMPLE_RATE,
        wav_path,
    )


def _load_whisper(model_size: str, device: str):  # noqa: ANN001
    key = (model_size, device)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        compute = "int8" if device == "cpu" else "float16"
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute)
    return _MODEL_CACHE[key]


def _similar(a: str, b: str) -> float:
    """Tỉ lệ giống nhau giữa hai chuỗi, dùng để phát hiện transcript lệch."""
    import difflib

    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _edge_silence(x: np.ndarray, thresh_db: float = -45.0) -> tuple[float, float]:
    """Số giây gần-im-lặng ở đầu và cuối. Chỉ để cảnh báo, không cắt gì.

    Việc trim thật do `ov_extract_voice_ref` phía C++ làm, giống hệt CLI. Cắt ở
    đây sẽ làm audio lệch với transcript và mô hình sẽ chèn chữ thừa vào đầu
    đoạn sinh ra.
    """
    if x.size == 0:
        return 0.0, 0.0
    win = 480  # 20 ms
    n = (len(x) // win) * win
    if n == 0:
        return 0.0, 0.0
    rms = np.sqrt((x[:n].reshape(-1, win) ** 2).mean(axis=1) + 1e-12)
    peak = rms.max()
    if peak <= 0:
        return len(x) / SAMPLE_RATE, len(x) / SAMPLE_RATE
    loud = rms > peak * (10 ** (thresh_db / 20.0))
    if not loud.any():
        return len(x) / SAMPLE_RATE, len(x) / SAMPLE_RATE
    lead = int(np.argmax(loud)) * win / SAMPLE_RATE
    tail = int(np.argmax(loud[::-1])) * win / SAMPLE_RATE
    return lead, tail


def prepare_reference(
    source: str | os.PathLike,
    *,
    out_dir: str | os.PathLike | None = None,
    lang: str = "vi",
    target_sec: float = TARGET_SEC,
    max_sec: float = MAX_SEC,
    model_size: str = "medium",
    device: str = "cpu",
    text: str | None = None,
) -> Reference:
    """Cắt và transcribe một đoạn giọng mẫu.

    Audio được cắt **đúng mốc thời gian ASR trên file gốc**, không qua bất kỳ
    bước trim nào. Đây là điểm sống còn: nếu audio và transcript lệch nhau, mô
    hình sẽ đẩy phần chữ thừa ra đầu đoạn sinh ra. Việc trim im lặng và RMS
    auto-gain đã được `ov_extract_voice_ref` làm ở phía C++, đúng như CLI.

    `text` cho sẵn thì bỏ ASR và dùng nguyên file làm reference — không cắt,
    vì cắt sẽ làm transcript không còn khớp tiếng.
    """
    source = Path(source)
    out_dir = Path(out_dir) if out_dir else source.parent / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm = read_wav_24k(source)
    out = out_dir / f"{source.stem}-ref.wav"

    if text is not None:
        dur = len(pcm) / SAMPLE_RATE
        if dur > max_sec * 2:
            raise ValueError(
                f"{source.name} dài {dur:.1f}s. Khi tự cho `text` thì file được dùng "
                f"nguyên vẹn (cắt sẽ làm lệch transcript) — hãy cắt sẵn file xuống "
                f"khoảng {target_sec:.0f}s kèm transcript đúng của đoạn đó."
            )
        Audio(pcm).save(out)
        ref = Reference(out, text.strip(), dur, source)
        ref.write_txt()
        return ref

    model = _load_whisper(model_size, device)
    segments, _info = model.transcribe(
        str(source), language=lang, vad_filter=True, beam_size=5
    )
    segs = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    if not segs:
        raise RuntimeError(f"faster-whisper không nhận được lời nào trong {source}")

    # Chọn dãy segment liên tiếp có tổng thời lượng gần target_sec nhất mà vẫn
    # <= max_sec. Cắt đúng ranh giới segment nên transcript khớp tiếng.
    best = None
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            t0, t1 = segs[i][0], segs[j][1]
            dur = t1 - t0
            if dur > max_sec:
                break
            if dur < MIN_SEC:
                continue
            score = abs(dur - target_sec)
            if best is None or score < best[0]:
                best = (score, i, j)
    if best is None:
        # Không có dãy nào vào khoảng [MIN_SEC, max_sec]: lấy segment đầu.
        best = (0.0, 0, 0)

    _, i, j = best
    t0, t1 = segs[i][0], segs[j][1]
    transcript = " ".join(s[2] for s in segs[i : j + 1])

    # Nới nhẹ hai đầu để không cắt mất phụ âm cuối, nhưng không được chạm vào
    # segment kế bên — chạm vào là lại lệch chữ với tiếng.
    pad_head = min(0.06, t0 - (segs[i - 1][1] if i > 0 else 0.0))
    next_start = segs[j + 1][0] if j + 1 < len(segs) else len(pcm) / SAMPLE_RATE
    pad_tail = min(0.12, max(0.0, next_start - t1))
    t0 = max(0.0, t0 - max(0.0, pad_head))
    t1 = min(len(pcm) / SAMPLE_RATE, t1 + pad_tail)

    clip = pcm[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]
    if clip.size == 0:
        raise RuntimeError(f"cắt ra đoạn rỗng từ {source} ({t0:.2f}-{t1:.2f}s)")

    Audio(clip).save(out)

    # Transcribe LẠI chính đoạn đã cắt, và dùng kết quả đó làm transcript.
    #
    # Đây là bước quan trọng nhất của cả module. Mốc segment mà ASR trả về khi
    # nghe cả file không khớp tuyệt đối với nội dung 6 giây được cắt ra: chữ
    # cuối của segment có thể nằm vắt qua ranh giới, và faster-whisper cũng
    # không cho kết quả hoàn toàn tất định. Nếu transcript dài hơn tiếng dù chỉ
    # hai chữ, mô hình coi hai chữ đó là "chưa đọc" và phát chúng ra ở ĐẦU mọi
    # đoạn sinh ra sau này. Lấy đúng những gì ASR nghe được từ chính đoạn cắt
    # thì text và tiếng khớp nhau theo cấu trúc, không thể lệch.
    heard_segs, _ = model.transcribe(str(out), language=lang, vad_filter=True, beam_size=5)
    heard = " ".join(s.text.strip() for s in heard_segs if s.text.strip()).strip()

    if not heard:
        print(f"[refprep] ⚠ {out.name}: ASR không nghe được gì trong đoạn đã cắt, "
              f"dùng transcript từ file gốc.")
        heard = transcript

    sim = _similar(transcript, heard)
    if sim < 0.85:
        print(f"[refprep] transcript chỉnh theo đoạn đã cắt (khớp {sim * 100:.0f}%):")
        print(f"           theo file gốc : {transcript}")
        print(f"           theo đoạn cắt : {heard}")

    ref = Reference(out, heard, len(clip) / SAMPLE_RATE, source)
    ref.write_txt()

    lead, tail = _edge_silence(clip)
    if tail > 0.4:
        print(f"[refprep] ⚠ {out.name}: im lặng ở cuối {tail:.2f}s — cân nhắc cắt ngắn lại.")
    return ref
