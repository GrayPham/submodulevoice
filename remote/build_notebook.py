"""Sinh notebook Colab tự chứa từ mã nguồn thật.

    python remote/build_notebook.py

Notebook nhúng thẳng nội dung `pyomnivoice/` và `remote/server.py` vào các ô
`%%writefile`, nên trên Colab không phải upload gì, cũng không cần repo public.
Sinh từ mã nguồn để không bao giờ có hai bản lệch nhau — sửa server.py rồi
chạy lại lệnh này là notebook cập nhật theo.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "OmniVoice_Server_Colab.ipynb"

EMBED = [
    ("pyomnivoice/__init__.py", ROOT / "pyomnivoice" / "__init__.py"),
    ("pyomnivoice/_ffi.py", ROOT / "pyomnivoice" / "_ffi.py"),
    ("pyomnivoice/core.py", ROOT / "pyomnivoice" / "core.py"),
    ("pyomnivoice/srt.py", ROOT / "pyomnivoice" / "srt.py"),
    ("server.py", ROOT / "remote" / "server.py"),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


CELL_INTRO = """\
# OmniVoice — Server GPU trên Colab

Chạy tổng hợp giọng nói trên GPU Colab, máy ở nhà gọi lên qua API.

**Chỉ có client/server.** Không có giao diện, không có trình quản lý dự án —
đúng ba việc: nhiều worker INT4, hàng đợi, trả kết quả về client.

## Chạy theo thứ tự

1. Kiểm tra GPU
2. Build omnivoice.cpp (lần đầu 8–15 phút, cache lại được vào Drive)
3. Tải model INT4 (~660 MB)
4. Ghi mã Python
5. Khởi động server
6. Mở đường hầm ra ngoài → lấy URL + API key

Rồi ở máy mình:

```
python remote/client.py --url <URL> --key <KEY> ^
    --script scripts/kichban_pt.txt ^
    --ref output/refs3/FDown...-ref.wav --lang Portuguese ^
    --concurrency 4 -o output/remote/ket-qua.wav
```

## Số worker đặt bao nhiêu

Đo thật trên RTX 4000 Ada (`examples/bench_parallel.py`):

| worker | tăng tốc | VRAM | audio so với 1 luồng |
|---|---|---|---|
| 1 | 1.00x | 1443 MiB | mốc chuẩn |
| 2 | 1.80x | 2452 MiB | giống hệt từng byte |
| **4** | **2.60x** | 4802 MiB | giống hệt từng byte |
| 6 | 0.99x | 7140 MiB | giống hệt từng byte |
| 8 | 0.94x | 9440 MiB | giống hệt từng byte |

**4 là điểm tối ưu.** Quá 4 thì chậm đi chứ không nhanh thêm — GPU đã bão hoà.
Chia luồng **không đổi chất lượng**: băm SHA1 nội dung audio khớp 100% với bản
chạy tuần tự. T4 của Colab yếu hơn RTX 4000 Ada nên tốc độ tuyệt đối thấp hơn,
nhưng hình dạng đường cong thì giữ nguyên.

## Lưu ý

Phiên Colab tự ngắt sau vài giờ và mọi thứ trong `/content` mất theo. Bật
cache vào Drive ở ô số 2 thì lần sau khỏi build lại.
"""

CELL_GPU = """\
# ── 1. Kiểm tra GPU ─────────────────────────────────────────────────────────
import shutil, subprocess, sys

if not shutil.which("nvidia-smi"):
    raise SystemExit(
        "Runtime này KHÔNG có GPU.\\n"
        "Sửa: menu Runtime -> Change runtime type -> T4 GPU, rồi chạy lại từ đầu.\\n"
        "Server này chỉ chạy CUDA, không có đường lui về CPU."
    )

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
cc = subprocess.run(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip().splitlines()[0]
CUDA_ARCH = cc.replace(".", "")
print(f"compute capability {cc} -> build riêng cho sm_{CUDA_ARCH} (nhanh hơn build đa kiến trúc)")
"""

CELL_BUILD = """\
# ── 2. Build omnivoice.cpp ──────────────────────────────────────────────────
# Lần đầu 8-15 phút. Bật USE_DRIVE_CACHE để lần sau khỏi build lại.
import os, subprocess, shutil
from pathlib import Path

USE_DRIVE_CACHE = False   # True -> mount Drive, cache thư mục build
DRIVE_CACHE = "/content/drive/MyDrive/omnivoice-build"

SRC = Path("/content/omnivoice.cpp")
BUILD = SRC / "build"
LIB = BUILD / "libomnivoice.so"

if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount("/content/drive")

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode:
        raise SystemExit(f"lệnh thất bại: {cmd}")

if LIB.exists():
    print("đã có bản build sẵn.")
elif USE_DRIVE_CACHE and Path(DRIVE_CACHE, "libomnivoice.so").exists():
    print("khôi phục build từ Drive ...")
    SRC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DRIVE_CACHE, BUILD, dirs_exist_ok=True)
else:
    if not SRC.exists():
        sh("git clone --recurse-submodules --depth 1 "
           "https://github.com/ServeurpersoCom/omnivoice.cpp.git /content/omnivoice.cpp")
    sh(f"cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
       f"-DOMNIVOICE_SHARED=ON -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}", cwd=SRC)
    sh(f"cmake --build build -j$(nproc)", cwd=SRC)
    if USE_DRIVE_CACHE:
        Path(DRIVE_CACHE).mkdir(parents=True, exist_ok=True)
        for f in BUILD.glob("*.so"):
            shutil.copy2(f, Path(DRIVE_CACHE, f.name))
        print("đã lưu build vào Drive.")

assert LIB.exists(), "không thấy libomnivoice.so sau khi build"
os.environ["OMNIVOICE_LIB"] = str(BUILD)
print("thư viện:", LIB)
print(sorted(p.name for p in BUILD.glob("*.so")))
"""

CELL_MODELS = """\
# ── 3. Tải model INT4 (~660 MB) ─────────────────────────────────────────────
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS = Path("/content/models"); MODELS.mkdir(exist_ok=True)
for f in ["omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q8_0.gguf"]:
    if (MODELS / f).exists():
        print("[có sẵn]", f)
    else:
        print("[tải]", f)
        hf_hub_download("Serveurperso/OmniVoice-GGUF", f, local_dir=str(MODELS))
print(sorted(p.name for p in MODELS.glob("*.gguf")))
"""

CELL_START = """\
# ── 5. Khởi động server ─────────────────────────────────────────────────────
import json, os, secrets, subprocess, time, urllib.request

PORT = 8770
WORKERS = 4          # 4 là điểm tối ưu đo được; server tự hạ nếu VRAM không đủ
API_KEY = secrets.token_urlsafe(12)
LOG = "/content/server.log"

subprocess.run(f"kill -9 $(lsof -t -i:{PORT}) 2>/dev/null || true", shell=True)
time.sleep(1)

env = dict(os.environ, OMNIVOICE_LIB="/content/omnivoice.cpp/build",
           PYTHONUNBUFFERED="1")
with open(LOG, "wb") as f:
    subprocess.Popen(
        ["python", "/content/app/server.py", "--workers", str(WORKERS),
         "--port", str(PORT), "--key", API_KEY,
         "--models-dir", "/content/models", "--profile", "lite"],
        stdout=f, stderr=subprocess.STDOUT, env=env, cwd="/content/app")

for i in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            h = json.load(r)
        print(json.dumps(h, ensure_ascii=False, indent=2))
        print(f"\\nSERVER SẴN SÀNG — {h['workers']} worker")
        break
    except Exception:
        time.sleep(2)
else:
    print(open(LOG, encoding="utf-8", errors="replace").read()[-4000:])
    raise SystemExit("server không lên. Xem log ở trên.")
"""

CELL_TUNNEL = """\
# ── 6. Mở đường hầm ra ngoài ────────────────────────────────────────────────
# cloudflared: không cần đăng ký tài khoản, URL dùng ngay.
import re, subprocess, time
from pathlib import Path

if not Path("/content/cloudflared").exists():
    subprocess.run(
        "wget -q -O /content/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 && chmod +x /content/cloudflared", shell=True, check=True)

subprocess.run("pkill -f cloudflared || true", shell=True)
subprocess.Popen(
    f"/content/cloudflared tunnel --url http://127.0.0.1:{PORT} --no-autoupdate "
    f"> /content/cloudflared.log 2>&1", shell=True)

URL = None
for _ in range(40):
    time.sleep(2)
    log = Path("/content/cloudflared.log").read_text(errors="replace")
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", log)
    if m:
        URL = m.group(0)
        break

if not URL:
    print(Path("/content/cloudflared.log").read_text(errors="replace")[-3000:])
    raise SystemExit("không lấy được URL đường hầm")

print("=" * 70)
print("  CHÉP HAI DÒNG NÀY VỀ MÁY MÌNH")
print("=" * 70)
print(f"  URL      {URL}")
print(f"  API key  {API_KEY}")
print("=" * 70)
print()
print("Lệnh chạy ở máy mình:")
print()
print(f"  python remote/client.py --url {URL} --key {API_KEY} \\\\")
print( "      --script scripts/kichban_pt.txt \\\\")
print( "      --ref output/refs3/FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav \\\\")
print( "      --lang Portuguese --concurrency 4 -o output/remote/ket-qua.wav")
"""

CELL_TEST = """\
# ── 7. Tự kiểm tra (tuỳ chọn) ───────────────────────────────────────────────
# Gửi thử một câu qua chính đường hầm, đo độ trễ khứ hồi.
import json, time, urllib.request

t0 = time.perf_counter()
req = urllib.request.Request(URL + "/tts", method="POST",
    data=json.dumps({"text": "Xin chào, đây là bài kiểm tra kết nối.",
                     "lang": "Vietnamese", "steps": 16}).encode(),
    headers={"Content-Type": "application/json", "X-API-Key": API_KEY})
with urllib.request.urlopen(req, timeout=300) as r:
    wav = r.read()
    synth = float(r.headers.get("X-Synth-Seconds", 0))
    audio = float(r.headers.get("X-Audio-Seconds", 0))
rtt = time.perf_counter() - t0

open("/content/test.wav", "wb").write(wav)
print(f"audio {audio:.2f}s | GPU tính {synth:.2f}s | khứ hồi qua đường hầm {rtt:.2f}s")
print(f"phần mạng chiếm {rtt - synth:.2f}s")
from IPython.display import Audio, display
display(Audio("/content/test.wav"))
"""


def main() -> None:
    cells = [md(CELL_INTRO), code(CELL_GPU), code(CELL_BUILD), code(CELL_MODELS)]

    write = ["# ── 4. Ghi mã Python ────────────────────────────────────────────────────────",
             "# Nhúng sẵn trong notebook, không phải upload gì.",
             "from pathlib import Path", "",
             "APP = Path('/content/app'); APP.mkdir(exist_ok=True)",
             "(APP / 'pyomnivoice').mkdir(exist_ok=True)", "", "FILES = {}"]
    for name, path in EMBED:
        src = path.read_text(encoding="utf-8")
        write.append(f"FILES[{name!r}] = {src!r}")
    write += ["", "for name, body in FILES.items():",
              "    p = APP / name",
              "    p.parent.mkdir(parents=True, exist_ok=True)",
              "    p.write_text(body, encoding='utf-8')",
              "    print(f'{len(body):7d} ký tự  {name}')"]
    cells.append(code("\n".join(write)))

    cells += [code(CELL_START), code(CELL_TUNNEL), code(CELL_TEST)]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    size = OUT.stat().st_size
    print(f"{OUT}  ({size / 1024:.0f} KB, {len(cells)} ô)")


if __name__ == "__main__":
    main()
