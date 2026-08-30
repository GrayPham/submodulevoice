#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""LVC Voice 6 — CLIENT THAM CHIẾU (cách gọi server ĐÚNG để ra giọng SẠCH).

Vì sao có file này
------------------
App đang bị "giật / lẫn lời mẫu" khi clone qua server, DÙ ref audio tốt. Nguyên
nhân KHÔNG phải server, mà là: app gửi **lời mẫu (ref transcript) RỖNG** lên
server (khi profile không kèm lời mẫu, hoặc khi cho rằng lời mẫu "không khớp"),
với giả định "server tự ASR". Nhưng server INT4 này **KHÔNG có ASR**. Nhận rỗng
thì model canh sai -> RÒ nguyên văn lời trong audio ref vào bài đọc + LẶP/chèn
chữ ở đầu = tiếng "giật".

Bốn nguyên tắc để KHÔNG bị (file này minh hoạ đủ cả bốn):

  1. LUÔN gửi lời mẫu khớp audio ref. TUYỆT ĐỐI không gửi rỗng.
  2. Không có/nghi sai lời mẫu -> TỰ ASR ref bằng faster-whisper rồi gửi cái đó
     (đúng việc mà app tưởng server làm hộ — làm ở client mới chắc).
  3. Soi tỉ lệ lời mẫu (>= 4 ký tự/giây; lời nói thật ~12-16). Lệch -> ASR lại,
     KHÔNG gửi rỗng.
  4. Chia text THEO CÂU, gộp tới ~800 ký tự/lần (dưới ngưỡng ~100s của Cloudflare
     quick-tunnel), gửi từng mẩu steps=32, rồi nối — KHÔNG cắt giữa câu.

Đây chính là quy trình đã cho ra `adam_online_FIXED.wav` (sạch, không giật).

Cách chạy
---------
    pip install requests soundfile numpy faster-whisper

    python voice6_reference_client.py \
        --url  https://xxx.trycloudflare.com \
        --key  API_KEY \
        --ref  "Adam vip.wav" \
        --ref-text "" \                # để trống -> script TỰ ASR (không gửi rỗng lên server)
        --script kichban.txt \         # hoặc --text "..."
        --out  out.wav \
        --verify                       # (tuỳ chọn) ASR lại output để kiểm rò/lặp

Đội app đọc hàm `resolve_ref_text()` + `register_voice()` là thấy đúng chỗ cần
sửa: thay vì `ref_text = ""` thì gọi ASR cục bộ rồi gửi.
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import time
import wave

import numpy as np
import requests
import soundfile as sf

SERVER_SAMPLE_RATE = 24000
# Ngưỡng an toàn cho Cloudflare quick-tunnel: 1 request phải xong dưới ~100s.
# ~800 ký tự -> ~45s audio -> ~25-50s tổng hợp trên T4, thừa biên.
DEFAULT_MAX_CHARS = 800
DEFAULT_STEPS = 32           # 16 cũng gần như y hệt (nhanh gấp đôi); 8 thì giật.
MIN_CHARS_PER_SEC = 4.0      # dưới mức này gần như chắc chắn lời mẫu sai/thiếu.


# ─────────────────────────── ASR ref (mấu chốt) ───────────────────────────
_WHISPER = None


def _whisper():
    """Nạp faster-whisper một lần. Ưu tiên GPU, không có thì CPU."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        try:
            _WHISPER = WhisperModel("small", device="cuda", compute_type="float16")
        except Exception:
            _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    return _WHISPER


def asr_reference(ref_wav_path: str, lang: str = "vi") -> str:
    """Phiên âm audio ref -> lời mẫu khớp tiếng. Đây là thứ app QUÊN làm."""
    segs, _ = _whisper().transcribe(ref_wav_path, language=lang, beam_size=5)
    return " ".join(s.text.strip() for s in segs).strip()


def resolve_ref_text(ref_wav_path: str, ref_text: str, lang: str = "vi") -> str:
    """Trả về lời mẫu ĐÚNG để gửi — KHÔNG BAO GIỜ trả rỗng.

    Logic mà app cần áp dụng (thay cho `ref_text = ""`):
      - Rỗng             -> ASR.
      - Tỉ lệ ký tự/giây quá thấp (nghi sai/thiếu) -> ASR lại.
      - ASR vẫn ra rỗng  -> báo lỗi rõ, KHÔNG âm thầm gửi rỗng.
    """
    dur = sf.info(ref_wav_path).duration
    text = (ref_text or "").strip()
    reason = None
    if not text:
        reason = "lời mẫu để trống"
    else:
        rate = len(text) / max(dur, 0.1)
        if rate < MIN_CHARS_PER_SEC:
            reason = f"lời mẫu nghi sai (chỉ {rate:.1f} ký tự/giây < {MIN_CHARS_PER_SEC})"
    if reason:
        print(f"  [ref] {reason} -> tự ASR ref bằng whisper...")
        text = asr_reference(ref_wav_path, lang=lang)
        if not text:
            raise SystemExit(
                "  [ref] ASR không ra chữ nào. DỪNG — gửi rỗng lên server sẽ giật. "
                "Hãy nhập tay lời mẫu đúng của file ref.")
        print(f"  [ref] ASR ra ({len(text)} ký tự): {text!r}")
    else:
        print(f"  [ref] dùng lời mẫu sẵn ({len(text)} ký tự, {len(text)/dur:.1f} ký tự/giây).")
    return text


# ─────────────────────────── Chia câu ───────────────────────────
def split_by_sentence(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Gộp trọn câu tới `max_chars`, KHÔNG cắt giữa câu."""
    parts: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?…。！？])\s+", text.strip()):
        if not sent:
            continue
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                parts.append(cur)
            cur = sent
            if len(sent) > max_chars:
                # Một câu dài hơn cả trần: vẫn gửi NGUYÊN (đừng cắt giữa câu),
                # chỉ cảnh báo vì có thể chạm ngưỡng thời gian của tunnel.
                print(f"  [chia] 1 câu dài {len(sent)} ký tự > {max_chars} — gửi nguyên câu.")
    if cur:
        parts.append(cur)
    return parts


# ─────────────────────────── Gọi server ───────────────────────────
def _wav16_b64(audio: np.ndarray, sr: int) -> str:
    """PCM float -> WAV 16-bit mono base64 (server yêu cầu WAV 16-bit)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def register_voice(url: str, key: str, ref_wav_path: str, ref_text: str) -> str:
    """Đăng ký giọng mẫu -> voice_id. `ref_text` PHẢI khác rỗng (xem resolve_ref_text)."""
    assert ref_text.strip(), "KHÔNG được đăng ký với lời mẫu rỗng — sẽ giật."
    audio, sr = sf.read(ref_wav_path, dtype="float32", always_2d=False)
    r = requests.post(
        f"{url}/voice",
        headers={"X-API-Key": key},
        json={"name": "ref", "text": ref_text, "wav_b64": _wav16_b64(audio, sr)},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["voice_id"]


def tts_one(url: str, key: str, voice_id: str, text: str, lang: str,
            steps: int, seed: int, retries: int = 2) -> np.ndarray:
    """Gọi /tts cho MỘT mẩu, trả mảng float32 24kHz. Thử lại lỗi tạm."""
    last = None
    for attempt in range(1, retries + 2):
        try:
            r = requests.post(
                f"{url}/tts",
                headers={"X-API-Key": key},
                json={"text": text, "voice_id": voice_id, "lang": lang,
                      "steps": steps, "seed": seed},
                timeout=150,
            )
            if r.status_code == 200:
                data, _ = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
                return data if data.ndim == 1 else data.mean(axis=1)
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        print(f"    (thử {attempt} lỗi: {last})")
        time.sleep(3)
    raise SystemExit(f"  [tts] mẩu thất bại sau {retries + 1} lần: {last}")


# ─────────────────────────── Kiểm tra output (tuỳ chọn) ───────────────────────────
def verify_clean(audio: np.ndarray, ref_text: str, lang: str = "vi") -> None:
    """ASR lại output để KHÁCH QUAN kiểm rò lời ref + lặp. Chỉ để yên tâm."""
    sf.write("_verify_tmp.wav", audio, SERVER_SAMPLE_RATE, subtype="PCM_16")
    try:
        segs, _ = _whisper().transcribe("_verify_tmp.wav", language=lang, beam_size=5)
        out = " ".join(s.text.strip() for s in segs).strip()
    finally:
        import os
        os.remove("_verify_tmp.wav")
    words = re.findall(r"\w+", out.lower())
    tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    rep = (1 - len(set(tri)) / len(tri)) if tri else 0.0
    ref_words = set(re.findall(r"\w+", ref_text.lower()))
    # rò = từ hiếm của lời mẫu (không phải từ phổ thông) xuất hiện lại trong output
    common = {"và", "của", "là", "có", "không", "các", "một", "cho", "với", "được"}
    bleed = sorted((ref_words & set(words)) - common)
    print(f"  [verify] lặp 3-gram: {rep*100:.0f}%  |  nghi rò từ ref: {bleed or 'không'}")
    print(f"  [verify] ASR 160 ký tự đầu: {out[:160]}")


# ─────────────────────────── main ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="LVC Voice 6 — client tham chiếu (gọi ĐÚNG).")
    ap.add_argument("--url", required=True, help="API URL server (từ ô 3 Colab).")
    ap.add_argument("--key", required=True, help="API key.")
    ap.add_argument("--ref", required=True, help="File audio giọng mẫu (wav/mp3...).")
    ap.add_argument("--ref-text", default="", help="Lời mẫu của ref. Để TRỐNG -> script tự ASR.")
    ap.add_argument("--text", default=None, help="Văn bản cần đọc (hoặc dùng --script).")
    ap.add_argument("--script", default=None, help="File .txt văn bản cần đọc.")
    ap.add_argument("--out", default="out.wav", help="File audio kết quả.")
    ap.add_argument("--lang", default="Vietnamese", help="Ngôn ngữ đọc (tên đầy đủ).")
    ap.add_argument("--asr-lang", default="vi", help="Mã ngôn ngữ để ASR ref (vi/en...).")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Số bước MaskGIT (32 khuyến nghị).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Trần ký tự/mẩu.")
    ap.add_argument("--gap", type=float, default=0.12, help="Giây lặng chèn giữa các mẩu.")
    ap.add_argument("--verify", action="store_true", help="ASR lại output kiểm rò/lặp.")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    if args.script:
        with open(args.script, encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        return int(ap.error("cần --text hoặc --script"))

    print("=== LVC Voice 6 — client tham chiếu ===")
    # 0) health
    h = requests.get(f"{url}/health", headers={"X-API-Key": args.key}, timeout=30)
    print(f"[0] health: {h.status_code} {h.json().get('gpu', {}).get('name', '') if h.ok else h.text[:80]}")
    if not h.ok:
        return 1

    # 1) LỜI MẪU ĐÚNG (không bao giờ rỗng) — đây là mấu chốt chống giật
    print("[1] chuẩn bị lời mẫu (ref transcript):")
    ref_text = resolve_ref_text(args.ref, args.ref_text, lang=args.asr_lang)

    # 2) đăng ký giọng
    vid = register_voice(url, args.key, args.ref, ref_text)
    print(f"[2] đăng ký giọng -> voice_id {vid}")

    # 3) chia theo câu
    parts = split_by_sentence(text, args.max_chars)
    print(f"[3] văn bản {len(text)} ký tự -> {len(parts)} mẩu (cắt theo câu, <= {args.max_chars})")

    # 4) tổng hợp từng mẩu + nối
    pieces, t0 = [], time.time()
    for i, p in enumerate(parts, 1):
        ts = time.time()
        d = tts_one(url, args.key, vid, p, args.lang, args.steps, args.seed)
        pieces.append(d)
        print(f"    mẩu {i}/{len(parts)}: {len(p)} ký tự -> {len(d)/SERVER_SAMPLE_RATE:.1f}s "
              f"(wall {time.time()-ts:.0f}s)")
    gap = np.zeros(int(args.gap * SERVER_SAMPLE_RATE), dtype=np.float32)
    full = np.concatenate([x for pr in pieces for x in (pr, gap)][:-1]) if pieces else np.zeros(0, np.float32)
    sf.write(args.out, full, SERVER_SAMPLE_RATE, subtype="PCM_16")
    print(f"[4] XONG -> {args.out}: {len(full)/SERVER_SAMPLE_RATE:.1f}s audio | "
          f"tổng wall {time.time()-t0:.0f}s | đỉnh {np.abs(full).max():.3f}")

    # 5) (tuỳ chọn) tự kiểm
    if args.verify:
        print("[5] kiểm khách quan (ASR lại output):")
        verify_clean(full, ref_text, lang=args.asr_lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
