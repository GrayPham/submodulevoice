"""Sinh notebook Colab cho server Fish Audio Drama-3 INT4 (< 2.8 GB VRAM, 44.1 kHz).

    python remote/build_drama3_notebook.py --repo https://github.com/GrayPham/submodulevoice.git

Notebook làm 7 bước trên Colab GPU:
  1. Kiểm tra GPU, RAM hệ thống & dung lượng ổ đĩa.
  2. Clone repo submodulevoice + cài audio.cpp CUDA 12.8 cho Colab.
  3. Tải Fish Drama-3 từ HuggingFace (Jmica/Drama-3-GGUF) và đóng gói ra bản
     INT4 chuẩn (`drama-3-slim-int4-q2k.gguf`, 13.27 GiB), hoặc lấy từ Drive.
  4. Bật `remote/drama3_server.py` (tương thích API /voice, /tts, /tts_batch).
  5. Mở đường hầm cloudflared và in URL + API key.
  6. Tự kiểm tra /health và tổng hợp thử tiếng Việt (nếu có WAV mẫu).
  7. Watchdog giữ phiên Colab sống & tự khôi phục server/tunnel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "Fish_Drama3_Server_Colab.ipynb"
DEFAULT_REPO = "https://github.com/GrayPham/submodulevoice.git"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": t.splitlines(keepends=True),
    }


INTRO = """\
# Fish Audio Drama-3 (INT4) — Server GPU trên Colab

Server tổng hợp giọng nói **Fish Audio Drama 3 (256 Experts, 44.1 kHz)** bản
lượng tử hoá **INT4 chuẩn (`drama-3-slim-int4-q2k.gguf`, 13.27 GiB)** trên Colab.

- Tự tải bản gốc `drama-3-backbone-8bit-experts-Q2_K.gguf` (15.88 GiB) từ
  HuggingFace `Jmica/Drama-3-GGUF`, giải nén sidecar và đóng gói thành
  `drama-3-slim-int4-q2k.gguf` (13.27 GiB, `< 2.8 GB VRAM` ở chế độ `2.8gb`,
  hoặc tự cân đối cho T4 ở chế độ `auto`) ngay trên mạng Colab, rồi xoá bản gốc
  để tiết kiệm ổ đĩa.
- Giao thức HTTP giống `remote/server.py` (`/health`, `/voices`, `/voice`,
  `/tts`, `/tts_batch`), xuất audio **44.100 Hz**.

## Chạy lần lượt từ ô 1 → ô 7
"""

CELL_ENV = """\
# ── 1. Kiểm tra GPU, RAM & Ổ đĩa ────────────────────────────────────────────
import shutil, subprocess

if not shutil.which("nvidia-smi"):
    raise SystemExit(
        "Runtime này KHÔNG có GPU.\\n"
        "Sửa: Runtime -> Change runtime type -> T4 GPU (hoặc L4/A100), rồi chạy lại."
    )

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("=== RAM & Ổ đĩa (/content) ===")
print(subprocess.run("free -h && echo '---' && df -h /content", shell=True,
                     capture_output=True, text=True).stdout)
"""

CELL_SETUP = """\
# ── 2. Clone repo + Tải engine audio.cpp CUDA 12.8 ──────────────────────────
import subprocess
from pathlib import Path

REPO_URL = "%%REPO%%"
BRANCH   = "master"
APP      = Path("/content/submodulevoice")
ENG_DIR  = Path("/content/drama3_engine")
MOD_DIR  = Path("/content/drama3_models")

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        raise SystemExit(f"Thất bại (rc={p.returncode}): {cmd}")

if APP.exists():
    sh(f"git -C {APP} fetch --depth 1 origin {BRANCH} && "
       f"git -C {APP} reset --hard origin/{BRANCH}")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {APP}")

sh("pip install -q httpx numpy soundfile soxr")
sh(f"python remote/prepare_drama3_int4.py --engine-dir {ENG_DIR} --models-dir {MOD_DIR} --engine-only",
   cwd=str(APP))
"""

CELL_MODEL = """\
#@title 3. Tải & Đóng gói Drama-3 INT4 (drama-3-slim-int4-q2k.gguf) { display-mode: "form" }
#@markdown Bật `USE_DRIVE_CACHE` nếu muốn lưu bản INT4 (13.27 GiB) lên Google Drive cho lần sau.
USE_DRIVE_CACHE = False  #@param {type:"boolean"}
DRIVE_MODEL_DIR = "/content/drive/MyDrive/drama3-models"  #@param {type:"string"}
HF_TOKEN = ""  #@param {type:"string"}
REMOVE_ORIG_AFTER_PACK = True  #@param {type:"boolean"}

import os, shutil
from pathlib import Path

INT4_NAME = "drama-3-slim-int4-q2k.gguf"
MOD_DIR.mkdir(parents=True, exist_ok=True)
int4_local = MOD_DIR / INT4_NAME

if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount("/content/drive")
    drive_dir = Path(DRIVE_MODEL_DIR)
    drive_dir.mkdir(parents=True, exist_ok=True)
    drive_file = drive_dir / INT4_NAME
    if not int4_local.exists() and drive_file.exists() and drive_file.stat().st_size > 12 * 1024**3:
        print(f"[Drive] Đang chép {INT4_NAME} từ Drive về /content...", flush=True)
        shutil.copy2(drive_file, int4_local)

if not (int4_local.exists() and int4_local.stat().st_size > 12 * 1024**3):
    cmd = (
        f"python remote/prepare_drama3_int4.py "
        f"--engine-dir {ENG_DIR} --models-dir {MOD_DIR} --workers 8"
    )
    if REMOVE_ORIG_AFTER_PACK:
        cmd += " --remove-orig"
    if HF_TOKEN.strip():
        os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    sh(cmd, cwd=str(APP))

    if USE_DRIVE_CACHE and int4_local.exists():
        drive_file = Path(DRIVE_MODEL_DIR) / INT4_NAME
        if not drive_file.exists():
            print(f"[Drive] Đang sao lưu {INT4_NAME} lên Google Drive...", flush=True)
            shutil.copy2(int4_local, drive_file)
else:
    print(f"[ok] Đã có sẵn {int4_local} ({int4_local.stat().st_size / (1024**3):.2f} GiB)")
"""

CELL_START = """\
#@title 4. Bật Server Fish Drama-3 { display-mode: "form" }
#@markdown `auto`: tự chọn tối ưu theo GPU (T4 dùng ~8.8 GB VRAM; card < 8 GB dùng < 2.8 GB VRAM).
VRAM_MODE = "auto"  #@param ["auto", "2.8gb", "t4", "2.3gb", "3.3gb", "gpu"]
PORT = 8770  #@param {type:"integer"}
THREADS = 6  #@param {type:"integer"}

import json, os, secrets, subprocess, time, urllib.request

LOG = "/content/drama3_server.log"
KEY_FILE = "/content/drama3_api_key.txt"
if os.path.exists(KEY_FILE):
    API_KEY = open(KEY_FILE).read().strip()
else:
    API_KEY = secrets.token_urlsafe(12)
    open(KEY_FILE, "w").write(API_KEY)

subprocess.run(f"kill -9 $(lsof -t -i:{PORT}) 2>/dev/null || true", shell=True)
subprocess.run("pkill -f audiocpp_server || true", shell=True)
time.sleep(1)

env = dict(os.environ, PYTHONUNBUFFERED="1")
with open(LOG, "wb") as f:
    subprocess.Popen(
        [
            "python", "remote/drama3_server.py",
            "--port", str(PORT),
            "--key", API_KEY,
            "--engine-dir", str(ENG_DIR),
            "--models-dir", str(MOD_DIR),
            "--vram-mode", VRAM_MODE,
            "--threads", str(THREADS),
        ],
        stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(APP),
        start_new_session=True,
    )

for _ in range(240):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            h = json.load(r)
        print(json.dumps(h, ensure_ascii=False, indent=2))
        print(f"\\nFISH DRAMA-3 SERVER SẴN SÀNG  (44.1 kHz | mode={h.get('vram_mode')})")
        print(f"API key: {API_KEY}")
        break
    except Exception:
        time.sleep(2)
else:
    print(open(LOG, encoding="utf-8", errors="replace").read()[-4000:])
    raise SystemExit("Server không lên sau 480s — xem log phía trên.")
"""

CELL_TUNNEL = """\
# ── 5. Mở đường hầm Cloudflared, lấy URL + API key ──────────────────────────
import re, subprocess, time
from pathlib import Path

if not Path("/content/cloudflared").exists():
    subprocess.run(
        "wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/"
        "releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared",
        shell=True, check=True,
    )

subprocess.run("pkill -f cloudflared || true", shell=True)
subprocess.Popen(
    f"/content/cloudflared tunnel --url http://127.0.0.1:{PORT} "
    "--no-autoupdate > /content/cloudflared.log 2>&1",
    shell=True, start_new_session=True,
)

URL = None
for _ in range(40):
    time.sleep(2)
    m = re.search(
        r"https://[a-z0-9-]+\\.trycloudflare\\.com",
        Path("/content/cloudflared.log").read_text(errors="replace"),
    )
    if m:
        URL = m.group(0)
        break

if not URL:
    print(Path("/content/cloudflared.log").read_text(errors="replace")[-3000:])
    raise SystemExit("Không lấy được URL đường hầm cloudflared")

print("=" * 72)
print("  CHÉP VỀ MÁY MÌNH (FISH DRAMA-3 INT4)")
print("=" * 72)
print(f"  URL      {URL}")
print(f"  API key  {API_KEY}")
print("=" * 72)
print("\\nLệnh gọi từ máy bạn (tự động nhận 44.1 kHz):\\n")
print(f"  python remote/client.py --url {URL} --key {API_KEY} \\\\")
print("      --script scripts/kichban_dai.txt \\\\")
print("      --ref path/to/giong_mau.wav --ref-text 'lời thoại chuẩn của file mẫu' \\\\")
print("      --concurrency 1 --batch 4 -o output/drama3_out.wav")
"""

CELL_TEST = """\
#@title 6. Tự kiểm tra kết nối & Test giọng mẫu (tuỳ chọn) { display-mode: "form" }
#@markdown Điền đường dẫn file WAV mẫu + lời thoại trên Colab để test `/voice` + `/tts`.
#@markdown Để trống `TEST_REF_WAV` sẽ chỉ kiểm tra `/health` qua local + Cloudflared.
TEST_REF_WAV = ""  #@param {type:"string"}
TEST_REF_TEXT = ""  #@param {type:"string"}
TEST_TEXT = "Xin chào mọi người! [chuckles] Đây là bài kiểm tra giọng nói Fish Audio Drama 3 chạy trên Colab."  #@param {type:"string"}

import base64, json, os, urllib.request
from IPython.display import Audio, display

for label, base in (("127.0.0.1", f"http://127.0.0.1:{PORT}"), ("Cloudflared", URL)):
    with urllib.request.urlopen(base + "/health", timeout=20) as r:
        h = json.load(r)
    print(f"[health {label}] ok={h.get('ok')} | model={h.get('model')} | sr={h.get('sample_rate')}")

if TEST_REF_WAV.strip() and os.path.isfile(TEST_REF_WAV.strip()):
    hdrs = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    wav_b64 = base64.b64encode(open(TEST_REF_WAV.strip(), "rb").read()).decode("ascii")
    req_v = urllib.request.Request(
        f"{URL}/voice",
        data=json.dumps({"name": "test_ref", "text": TEST_REF_TEXT.strip(), "wav_b64": wav_b64}).encode(),
        headers=hdrs, method="POST",
    )
    with urllib.request.urlopen(req_v, timeout=180) as r:
        vid = json.load(r)["voice_id"]
    print("Đăng ký voice_id =", vid)

    req_t = urllib.request.Request(
        f"{URL}/tts",
        data=json.dumps({"text": TEST_TEXT, "voice_id": vid}).encode(),
        headers=hdrs, method="POST",
    )
    with urllib.request.urlopen(req_t, timeout=300) as r:
        out_wav = r.read()
        print(f"Xong: audio={r.headers.get('X-Audio-Seconds')}s | synth={r.headers.get('X-Synth-Seconds')}s")
    open("/content/drama3_test.wav", "wb").write(out_wav)
    display(Audio("/content/drama3_test.wav"))
else:
    print("Bỏ qua test tổng hợp vì chưa nhập TEST_REF_WAV (Fish Drama-3 cần 1 file giọng mẫu).")
"""

CELL_WATCH = """\
# ── 7. Giữ phiên sống + tự dựng lại khi hỏng (ĐỂ Ô NÀY CHẠY LIÊN TỤC) ──────
import json, os, re, subprocess, time, urllib.request
from datetime import datetime
from pathlib import Path

CHECK_EVERY = 60
last_served = -1

def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return json.load(r)
    except Exception:
        return None

def tunnel_alive() -> bool:
    if subprocess.run("pgrep -f cloudflared", shell=True, capture_output=True).returncode != 0:
        return False
    if not URL:
        return False
    try:
        with urllib.request.urlopen(URL + "/health", timeout=15) as r:
            return r.status == 200
    except Exception:
        return False

print(f"Theo dõi mỗi {CHECK_EVERY}s. Để ô này chạy liên tục.\\n", flush=True)
while True:
    h = health()
    ts = datetime.now().strftime("%H:%M:%S")
    if h is None:
        print(f"[{ts}] SERVER CHẾT -> dựng lại...", flush=True)
        subprocess.run("pkill -f audiocpp_server || true", shell=True)
        with open(LOG, "ab") as f:
            subprocess.Popen(
                [
                    "python", "remote/drama3_server.py",
                    "--port", str(PORT), "--key", API_KEY,
                    "--engine-dir", str(ENG_DIR), "--models-dir", str(MOD_DIR),
                    "--vram-mode", VRAM_MODE, "--threads", str(THREADS),
                ],
                stdout=f, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                cwd=str(APP), start_new_session=True,
            )
        time.sleep(45)
        print(f"[{ts}] {'đã lên lại' if health() else 'VẪN CHƯA LÊN, xem ' + LOG}", flush=True)
    else:
        d = h["served"] - last_served if last_served >= 0 else h["served"]
        last_served = h["served"]
        gb = h.get("gpu", {})
        print(
            f"[{ts}] busy={h['busy']} | queue={h['queued']} | done={h['served']} (+{d}) | "
            f"err={h['failed']} | VRAM free={gb.get('vram_free_mib')} MiB",
            flush=True,
        )

    if not tunnel_alive():
        print(f"[{ts}] ĐƯỜNG HẦM ĐỨT -> mở lại...", flush=True)
        subprocess.run("pkill -f cloudflared || true", shell=True)
        time.sleep(2)
        subprocess.Popen(
            f"/content/cloudflared tunnel --url http://127.0.0.1:{PORT} "
            "--no-autoupdate > /content/cloudflared.log 2>&1",
            shell=True, start_new_session=True,
        )
        time.sleep(20)
        m = re.search(
            r"https://[a-z0-9-]+\\.trycloudflare\\.com",
            Path("/content/cloudflared.log").read_text(errors="replace"),
        )
        if m and m.group(0) != URL:
            URL = m.group(0)
            print(f"[{ts}] URL MỚI: {URL}", flush=True)

    time.sleep(CHECK_EVERY)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cells = [
        md(INTRO),
        code(CELL_ENV),
        code(CELL_SETUP.replace("%%REPO%%", args.repo)),
        code(CELL_MODEL),
        code(CELL_START),
        code(CELL_TUNNEL),
        code(CELL_TEST),
        code(CELL_WATCH),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out} ({out.stat().st_size / 1024:.0f} KB, {len(cells)} ô)")


if __name__ == "__main__":
    main()

