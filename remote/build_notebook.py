"""Sinh notebook Colab cho server GPU từ xa.

    python remote/build_notebook.py --repo https://github.com/ban/submodulevoice.git

Notebook chỉ làm bốn việc: clone repo này, build omnivoice.cpp, tải model, bật
server. Mã nguồn không nhúng vào notebook — sửa code, push lên git, chạy lại ô
số 2 trên Colab là có bản mới.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "OmniVoice_Server_Colab.ipynb"
DEFAULT_REPO = "https://github.com/CHUA-DAT-REPO/submodulevoice.git"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


INTRO = """\
# OmniVoice — Server GPU trên Colab

Tổng hợp giọng nói chạy trên GPU Colab, máy ở nhà gọi lên qua API.

**Chỉ có client/server.** Không giao diện, không quản lý dự án — đúng ba việc:
nhiều worker INT4, hàng đợi, trả kết quả về client.

## Chạy lần lượt từ trên xuống

| ô | việc | lần đầu | lần sau |
|---|---|---|---|
| 1 | kiểm tra GPU | vài giây | vài giây |
| 2 | clone repo + build omnivoice.cpp | **8–15 phút** | vài giây nếu cache Drive |
| 3 | tải model INT4 (~660 MB) | 1–2 phút | vài giây |
| 4 | bật server | ~30 giây | ~30 giây |
| 5 | mở đường hầm, lấy URL + key | ~20 giây | ~20 giây |

Xong ô 5 thì chép URL với API key về máy mình rồi chạy `remote/client.py`.

## Số worker đặt bao nhiêu

Số đo thật trên RTX 4000 Ada (`examples/bench_parallel.py`):

| worker | tăng tốc | VRAM | audio so với 1 luồng |
|---|---|---|---|
| 1 | 1.00x | 1443 MiB | mốc chuẩn |
| 2 | 1.80x | 2452 MiB | giống hệt từng byte |
| **4** | **2.60x** | 4802 MiB | giống hệt từng byte |
| 6 | 0.99x | 7140 MiB | giống hệt từng byte |
| 8 | 0.94x | 9440 MiB | giống hệt từng byte |

**4 là điểm tối ưu** — quá 4 thì chậm đi chứ không nhanh thêm, GPU đã bão hoà.
Chia luồng **không đổi chất lượng**: băm SHA1 nội dung audio khớp 100% với bản
chạy tuần tự cùng seed. T4 của Colab yếu hơn nên tốc độ tuyệt đối thấp hơn,
nhưng hình dạng đường cong giữ nguyên. Server tự hạ số worker nếu VRAM không đủ.

## Cần biết trước

- Phiên Colab tự ngắt sau vài giờ, mọi thứ trong `/content` mất theo. Bật
  `USE_DRIVE_CACHE` ở ô 2 thì lần sau khỏi build lại.
- URL cloudflared đổi mỗi lần chạy lại ô 5.
- Server chỉ chạy CUDA, không có đường lui về CPU — không có GPU là dừng ngay.
"""

CELL_GPU = """\
# ── 1. Kiểm tra GPU ─────────────────────────────────────────────────────────
import shutil, subprocess

if not shutil.which("nvidia-smi"):
    raise SystemExit(
        "Runtime nay KHONG co GPU.\\n"
        "Sua: menu Runtime -> Change runtime type -> T4 GPU, roi chay lai tu dau."
    )

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True).stdout.strip().splitlines()[0]
CUDA_ARCH = cc.replace(".", "")
print(f"compute capability {cc} -> build rieng cho sm_{CUDA_ARCH}, nhanh hon build da kien truc")
"""

CELL_BUILD = """\
# ── 2. Clone repo + build omnivoice.cpp ─────────────────────────────────────
import os, shutil, subprocess
from pathlib import Path

REPO_URL   = "%%REPO%%"
BRANCH     = "master"
GIT_TOKEN  = ""        # chi can khi repo dat private
USE_DRIVE_CACHE = False
DRIVE_CACHE = "/content/drive/MyDrive/omnivoice-build"

APP  = Path("/content/submodulevoice")
SRC  = Path("/content/omnivoice.cpp")
BUILD = SRC / "build"
LIB  = BUILD / "libomnivoice.so"

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    if subprocess.run(cmd, shell=True, cwd=cwd).returncode:
        raise SystemExit(f"that bai: {cmd}")

if USE_DRIVE_CACHE:
    from google.colab import drive; drive.mount("/content/drive")

# --- ma nguon cua minh -------------------------------------------------------
url = REPO_URL
if GIT_TOKEN:
    url = REPO_URL.replace("https://", f"https://{GIT_TOKEN}@")
if APP.exists():
    sh(f"git -C {APP} fetch --depth 1 origin {BRANCH} && "
       f"git -C {APP} reset --hard origin/{BRANCH}")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {url} {APP}")

sh("pip install -q numpy huggingface_hub")

# --- runtime C++ -------------------------------------------------------------
if LIB.exists():
    print("da co ban build san.")
elif USE_DRIVE_CACHE and Path(DRIVE_CACHE, "libomnivoice.so").exists():
    print("khoi phuc build tu Drive ...")
    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DRIVE_CACHE, BUILD, dirs_exist_ok=True)
else:
    if not SRC.exists():
        sh("git clone --recurse-submodules --depth 1 "
           "https://github.com/ServeurpersoCom/omnivoice.cpp.git /content/omnivoice.cpp")
    sh(f"cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
       f"-DOMNIVOICE_SHARED=ON -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}", cwd=SRC)
    sh("cmake --build build -j$(nproc)", cwd=SRC)
    if USE_DRIVE_CACHE:
        Path(DRIVE_CACHE).mkdir(parents=True, exist_ok=True)
        for f in BUILD.glob("*.so"):
            shutil.copy2(f, Path(DRIVE_CACHE, f.name))
        print("da luu build vao Drive.")

assert LIB.exists(), "khong thay libomnivoice.so sau khi build"
os.environ["OMNIVOICE_LIB"] = str(BUILD)
print("thu vien:", LIB)
print(sorted(p.name for p in BUILD.glob("*.so")))
"""

CELL_MODELS = """\
# ── 3. Tải model INT4 (~660 MB) ─────────────────────────────────────────────
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS = Path("/content/models"); MODELS.mkdir(exist_ok=True)
for f in ["omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q8_0.gguf"]:
    if (MODELS / f).exists():
        print("[co san]", f)
    else:
        print("[tai]", f, flush=True)
        hf_hub_download("Serveurperso/OmniVoice-GGUF", f, local_dir=str(MODELS))
print(sorted(p.name for p in MODELS.glob("*.gguf")))
"""

CELL_START = """\
# ── 4. Bật server ───────────────────────────────────────────────────────────
import json, os, secrets, subprocess, time, urllib.request

PORT    = 8770
WORKERS = 4          # diem toi uu do duoc; server tu ha neu VRAM khong du
API_KEY = secrets.token_urlsafe(12)
LOG     = "/content/server.log"

subprocess.run(f"kill -9 $(lsof -t -i:{PORT}) 2>/dev/null || true", shell=True)
time.sleep(1)

env = dict(os.environ, OMNIVOICE_LIB="/content/omnivoice.cpp/build", PYTHONUNBUFFERED="1")
with open(LOG, "wb") as f:
    subprocess.Popen(
        ["python", "remote/server.py", "--workers", str(WORKERS), "--port", str(PORT),
         "--key", API_KEY, "--models-dir", "/content/models", "--profile", "lite"],
        stdout=f, stderr=subprocess.STDOUT, env=env, cwd="/content/submodulevoice")

for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            h = json.load(r)
        print(json.dumps(h, ensure_ascii=False, indent=2))
        print(f"\\nSERVER SAN SANG - {h['workers']} worker")
        break
    except Exception:
        time.sleep(2)
else:
    print(open(LOG, encoding="utf-8", errors="replace").read()[-4000:])
    raise SystemExit("server khong len, xem log o tren.")
"""

CELL_TUNNEL = """\
# ── 5. Mở đường hầm, lấy URL + API key ──────────────────────────────────────
import re, subprocess, time
from pathlib import Path

if not Path("/content/cloudflared").exists():
    subprocess.run(
        "wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/"
        "releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared",
        shell=True, check=True)

subprocess.run("pkill -f cloudflared || true", shell=True)
subprocess.Popen(f"/content/cloudflared tunnel --url http://127.0.0.1:{PORT} "
                 f"--no-autoupdate > /content/cloudflared.log 2>&1", shell=True)

URL = None
for _ in range(40):
    time.sleep(2)
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",
                  Path("/content/cloudflared.log").read_text(errors="replace"))
    if m:
        URL = m.group(0); break

if not URL:
    print(Path("/content/cloudflared.log").read_text(errors="replace")[-3000:])
    raise SystemExit("khong lay duoc URL duong ham")

print("=" * 72)
print("  CHEP VE MAY MINH")
print("=" * 72)
print(f"  URL      {URL}")
print(f"  API key  {API_KEY}")
print("=" * 72)
print("\\nLenh chay o may minh:\\n")
print(f"  python remote/client.py --url {URL} --key {API_KEY} \\\\")
print( "      --script scripts/kichban_pt.txt \\\\")
print( "      --ref output/refs3/FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav \\\\")
print( "      --lang Portuguese --concurrency 4 -o output/remote/ket-qua.wav")
"""

CELL_TEST = """\
# ── 6. Tự kiểm tra (tuỳ chọn) ───────────────────────────────────────────────
# Gui thu mot cau qua chinh duong ham, tach thoi gian GPU va thoi gian mang.
import json, time, urllib.request
from IPython.display import Audio, display

t0 = time.perf_counter()
req = urllib.request.Request(URL + "/tts", method="POST",
    data=json.dumps({"text": "Xin chao, day la bai kiem tra ket noi.",
                     "lang": "Vietnamese", "steps": 16}).encode(),
    headers={"Content-Type": "application/json", "X-API-Key": API_KEY})
with urllib.request.urlopen(req, timeout=300) as r:
    wav = r.read()
    synth = float(r.headers.get("X-Synth-Seconds", 0))
    audio = float(r.headers.get("X-Audio-Seconds", 0))
rtt = time.perf_counter() - t0

open("/content/test.wav", "wb").write(wav)
print(f"audio {audio:.2f}s | GPU tinh {synth:.2f}s | khu hoi {rtt:.2f}s "
      f"| phan mang {rtt - synth:.2f}s")
display(Audio("/content/test.wav"))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="URL git của repo này")
    args = ap.parse_args()

    cells = [
        md(INTRO),
        code(CELL_GPU),
        code(CELL_BUILD.replace("%%REPO%%", args.repo)),
        code(CELL_MODELS),
        code(CELL_START),
        code(CELL_TUNNEL),
        code(CELL_TEST),
    ]
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
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} ô)")
    print(f"repo trong notebook: {args.repo}")
    if "CHUA-DAT-REPO" in args.repo:
        print("\n  ^ chưa có URL repo thật. Chạy lại với --repo <url> sau khi tạo repo.")


if __name__ == "__main__":
    main()
