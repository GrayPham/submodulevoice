"""Client gọi lên server GPU từ xa, chạy nhiều luồng, ghép lại đúng thứ tự.

    python remote/client.py --url https://xxx.trycloudflare.com --key BIMAT \
        --script scripts/kichban_pt.txt \
        --ref output/refs3/FDown...-ref.wav --lang Portuguese \
        --concurrency 4 -o output/remote/ket-qua.wav

Cách hoạt động:

  - Đăng ký giọng mẫu MỘT lần. Server mã hoá RVQ rồi giữ lại, các lần gọi sau
    chỉ gửi `voice_id` — không phải đẩy file WAV qua mạng mỗi câu.
  - Mỗi đoạn văn là một request độc lập, gửi song song `--concurrency` luồng.
    Kết quả về không theo thứ tự nên được xếp lại theo chỉ số đoạn trước khi
    ghép, vì vậy thứ tự đầu ra luôn đúng dù mạng trả về lộn xộn.
  - Chất lượng không đổi khi chia luồng: đã kiểm chứng bằng băm SHA1 trong
    `examples/bench_parallel.py`, 4 worker cho audio giống hệt 1 worker.

Đặt `--concurrency` bằng đúng số worker của server. Cao hơn chỉ làm request
nằm chờ trong hàng đợi, không nhanh thêm.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 24_000


def http(url: str, key: str, path: str, payload: dict | None = None,
         timeout: float = 900) -> tuple[bytes, dict]:
    req = urllib.request.Request(url.rstrip("/") + path)
    if key:
        req.add_header("X-API-Key", key)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
        # Ha het ten header ve chu thuong. cloudflared proxy qua HTTP/2, ma
        # HTTP/2 BAT BUOC ten header viet thuong -> "X-Synth-Seconds" den noi
        # thanh "x-synth-seconds". Goi thang localhost bang HTTP/1.1 thi giu
        # nguyen hoa thuong nen khong lo ra, chi hong khi di qua duong ham.
        return r.read(), {k.lower(): v for k, v in r.headers.items()}


def read_wav(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0


def save_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL cloudflared/ngrok của server")
    ap.add_argument("--key", default="", help="API key server in ra")
    ap.add_argument("--script", required=True)
    ap.add_argument("--ref", help="WAV giọng mẫu 16-bit (kèm .txt cùng tên)")
    ap.add_argument("--ref-text", help="transcript giọng mẫu, hoặc để tự đọc file .txt")
    ap.add_argument("--instruct", help="dùng voice design thay cho giọng mẫu")
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=0, help="0 = lấy đúng số worker server")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.45)
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="lam lai tu dau, bo qua cac doan da co san")
    ap.add_argument("-o", "--out", default="output/remote/ket-qua.wav")
    args = ap.parse_args()

    body, _ = http(args.url, args.key, "/health", timeout=30)
    health = json.loads(body)
    gpu = health.get("gpu", {})
    print(f"server   : {gpu.get('name', '?')}  |  {health['workers']} worker  |  "
          f"backend {health.get('backend')}")
    print(f"           VRAM {gpu.get('vram_free_mib')}/{gpu.get('vram_total_mib')} MiB trống")

    conc = args.concurrency or health["workers"]
    if conc > health["workers"]:
        print(f"           (giảm concurrency {conc} -> {health['workers']}: "
              f"cao hơn số worker chỉ nằm chờ hàng đợi)")
        conc = health["workers"]

    voice_id = None
    if args.ref:
        ref = Path(args.ref)
        text = args.ref_text
        if text is None:
            txt = ref.with_suffix(".txt")
            if not txt.exists():
                raise SystemExit(f"thiếu {txt}, hoặc truyền --ref-text")
            text = txt.read_text(encoding="utf-8").strip()
        payload = {"name": ref.stem, "text": text,
                   "wav_b64": base64.b64encode(ref.read_bytes()).decode("ascii")}
        body, _ = http(args.url, args.key, "/voice", payload, timeout=120)
        voice_id = json.loads(body)["voice_id"]
        print(f"giọng mẫu: {ref.name} -> voice_id {voice_id}")
        print(f"           {text[:100]}")
    elif args.instruct:
        print(f"giọng    : voice design '{args.instruct}'")

    paras = [re.sub(r"\s+", " ", p.strip())
             for p in re.split(r"\n\s*\n", Path(args.script).read_text(encoding="utf-8"))
             if p.strip()]
    print(f"kịch bản : {len(paras)} đoạn, {sum(len(p) for p in paras)} ký tự")
    # Mỗi đoạn được ghi ra đĩa NGAY khi xong. Colab đứt giữa chừng thì chạy lại
    # đúng lệnh này là nó bỏ qua phần đã có và làm tiếp — không mất công GPU đã
    # chạy. Vân tay gồm kịch bản + tham số giọng, nên đổi bất cứ thứ gì là các
    # đoạn cũ bị coi như không dùng được, tránh ghép nhầm hai lần chạy khác nhau.
    out = Path(args.out)
    parts_dir = out.parent / f"{out.stem}.parts"
    fp = hashlib.sha1(json.dumps({
        "paras": paras, "lang": args.lang, "steps": args.steps, "seed": args.seed,
        "instruct": args.instruct, "ref": Path(args.ref).name if args.ref else None,
    }, ensure_ascii=False).encode()).hexdigest()[:16]
    manifest = parts_dir / "manifest.json"

    results: dict[int, np.ndarray] = {}
    if args.no_resume and parts_dir.exists():
        for f in parts_dir.glob("*"):
            f.unlink()
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if old.get("fingerprint") == fp:
            for f in sorted(parts_dir.glob("part-*.wav")):
                idx = int(f.stem.split("-")[1]) - 1
                if 0 <= idx < len(paras):
                    results[idx] = read_wav(f.read_bytes())
            if results:
                print(f"tiếp tục : đã có {len(results)}/{len(paras)} đoạn từ lần chạy trước")
        else:
            print("làm lại  : kịch bản hoặc tham số đã đổi, bỏ các đoạn cũ")
            for f in parts_dir.glob("*"):
                f.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"fingerprint": fp, "n": len(paras)}), encoding="utf-8")
    todo = [(k, t) for k, t in enumerate(paras) if k not in results]
    print(f"gửi      : {len(todo)} đoạn qua {conc} luồng song song")
    meta: dict[int, tuple[float, float, float]] = {}
    lock = threading.Lock()
    done_n = [0]
    t0 = time.perf_counter()

    def one(idx_text: tuple[int, str]) -> None:
        idx, text = idx_text
        payload = {"text": text, "voice_id": voice_id, "lang": args.lang,
                   "steps": args.steps, "seed": args.seed, "instruct": args.instruct}
        last = None
        for attempt in range(args.retries + 1):
            try:
                b, hdr = http(args.url, args.key, "/tts", payload)
                pcm = read_wav(b)
                save_wav(parts_dir / f"part-{idx + 1:06d}.wav", pcm)
                with lock:
                    results[idx] = pcm
                    meta[idx] = (float(hdr.get("x-synth-seconds", 0)),
                                 float(hdr.get("x-queue-seconds", 0)),
                                 len(pcm) / SAMPLE_RATE)
                    done_n[0] += 1
                    el = time.perf_counter() - t0
                    print(f"\r  {done_n[0]}/{len(todo)} đoạn  |  {el:.0f}s trôi qua  "
                          f"|  còn ~{el / done_n[0] * (len(todo) - done_n[0]):.0f}s   ",
                          end="", flush=True)
                return
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        with lock:
            print(f"\n  đoạn {idx + 1} HỎNG sau {args.retries + 1} lần: {last}")

    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(one, todo))
    wall = time.perf_counter() - t0
    print()

    missing = [i for i in range(len(paras)) if i not in results]
    if missing:
        print(f"\nTHIẾU {len(missing)} đoạn: {[i + 1 for i in missing][:10]}")
        print("Không ghép file để bạn không nhận nhầm bản thiếu. Chạy lại.")
        raise SystemExit(1)

    gap = np.zeros(int(args.pause * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    w = len(str(len(paras)))
    for i in range(len(paras)):
        if pieces:
            pieces.append(gap)
        pieces.append(results[i])
        if args.keep_parts:
            save_wav(out.with_name(f"{out.stem}-{i + 1:0{w}d}.wav"), results[i])
    final = np.concatenate(pieces)
    save_wav(out, final)

    audio = len(final) / SAMPLE_RATE
    synth = sum(m[0] for m in meta.values())
    queued = sum(m[1] for m in meta.values())
    print(f"\ntổng     : {audio:.1f}s audio trong {wall:.1f}s = {audio / wall:.2f}x realtime")
    print(f"           GPU tính {synth:.1f}s, chờ hàng đợi {queued:.1f}s")
    print(f"           tăng tốc nhờ chia luồng: {synth / wall:.2f}x so với gọi tuần tự")
    print(f"           {out}")


if __name__ == "__main__":
    main()
