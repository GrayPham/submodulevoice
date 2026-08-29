"""Sinh HAI notebook cho luồng khách: build loader (.so) + notebook khách.

    python remote/build_client_notebooks.py --repo https://github.com/GrayPham/submodulevoice.git

Ra:
  colab/OmniVoice_Build_Loader_Colab.ipynb  — CHỈ BẠN chạy: Cython hoá
      remote/voice_loader.py -> voice_loader.*.so, đóng gói để đưa lên Release.
  colab/OmniVoice_Client_Colab.ipynb        — GỬI KHÁCH: tải loader .so đã build
      sẵn, getpass license, gọi run() -> server voice bật qua đường hầm.

Vì sao tách: kill-switch chỉ có tác dụng nếu khách nhận BINARY .so. Nếu notebook
khách tự biên dịch từ nguồn thì khách sửa bỏ kill-switch trước khi biên dịch là
xong. Nên loader phải build sẵn một lần, khách chỉ tải binary về chạy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BUILD = ROOT / "colab" / "OmniVoice_Build_Loader_Colab.ipynb"
OUT_CLIENT = ROOT / "colab" / "OmniVoice_Client_Colab.ipynb"
DEFAULT_REPO = "https://github.com/GrayPham/submodulevoice.git"
# Release chứa loader .so đã build. Notebook khách tải từ đây.
LOADER_RELEASE_TAG = "loader-linux"
LOADER_ASSET = "voice_loader-cp313-linux-x86_64.so"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ═══════════════════ NOTEBOOK 1: BUILD LOADER ═══════════════════
BUILD_INTRO = """\
# OmniVoice — Build loader bảo mật (.so)

**Chỉ bạn chạy.** Notebook này Cython hoá `remote/voice_loader.py` thành
`voice_loader.*.so` (khớp Python + nền tảng của Colab), rồi in checksum. Tải
file `.so` về, đưa lên Release `%%TAG%%` với tên `%%ASSET%%` — notebook khách
kéo từ đó.

Loader .so chứa kill-switch; giao khách dạng binary thì khó gỡ hơn nhiều so với
giao mã nguồn.
"""

BUILD_CELL = """\
# ── Build loader -> .so ───────────────────────────────────────────────
import os, sys, subprocess, hashlib
from pathlib import Path

REPO = "%%REPO%%"
BRANCH = "%%BRANCH%%"
APP = Path("/content/app")
PYTAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python Colab: {sys.version.split()[0]} (tag {PYTAG})")

def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"lỗi ({p.returncode}): {cmd}")

if APP.exists():
    sh(f"git -C {APP} fetch --depth 1 origin {BRANCH} && git -C {APP} reset --hard origin/{BRANCH}")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {APP}")
sh("pip install -q cython httpx cryptography")

# Cython hoá voice_loader.py (đứng ở gốc app để --inplace đặt .so đúng chỗ).
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

os.chdir(APP)
rel = "remote/voice_loader.py"
ext = Extension("remote.voice_loader", [rel])
dist = Distribution({"ext_modules": cythonize([ext], language_level=3,
    compiler_directives={"emit_code_comments": False}, quiet=True)})
dist.script_args = ["build_ext", "--inplace", "--build-temp", "/tmp/cyb"]
dist.parse_command_line(); dist.run_commands()

so = list((APP / "remote").glob("voice_loader*.so"))
assert so, "không sinh được .so"
src = so[0]
# Xoá nguồn + .c để không phát tán kèm.
for junk in (APP / rel, APP / "remote" / "voice_loader.c"):
    if junk.exists():
        junk.unlink()

# Đổi tên chuẩn hoá để notebook khách tải đúng một tên cố định.
out = Path("/content") / "%%ASSET%%"
out.write_bytes(src.read_bytes())
digest = hashlib.sha256(out.read_bytes()).hexdigest()
(Path("/content") / ("%%ASSET%%" + ".sha256")).write_text(f"{digest}  {out.name}\\n")

print(f"\\n=== XONG ===")
print(f"Loader : {out}  ({out.stat().st_size/1024:.0f} KB)")
print(f"SHA-256: {digest}")
print(f"\\nTải {out.name} + .sha256 về, đưa lên Release '%%TAG%%'.")
print(f"(PYTAG={PYTAG} — phải khớp Python mà notebook khách chạy.)")
"""


# ═══════════════════ NOTEBOOK 2: KHÁCH ═══════════════════
CLIENT_INTRO = """\
# 🎙️ OmniVoice — Tạo giọng nói (khách)

Chạy server voice của bạn trên GPU miễn phí của Colab. Chỉ cần **key bản quyền**.

**Cách dùng:** điền key vào ô 1 → menu **Runtime → Run all** → đợi ô 3 in ra
**API URL** và **API key**. Dùng cặp đó gọi tạo giọng từ máy bạn (xem ô cuối).

> **Bản chất lượng cao:** server mặc định **32 bước · engine TỰ chia đoạn**
> (liền mạch, không cắt vụn), nhận tới **8000 ký tự/request**. Cứ gửi cả đoạn —
> để engine tự chia cho giọng mượt. VRAM có trần nhờ guard nên gửi bài dài không
> nổ. Muốn nhanh hơn (giọng gần như không đổi) thì gửi `steps: 16`.

> Để **ô 3 chạy suốt** trong lúc dùng — tắt ô là tắt server. Mất mạng thì chạy
> lại ô 3, lấy URL mới.
"""

CLIENT_USAGE = """\
## Gọi tạo giọng từ máy bạn (chất lượng cao)

Sau khi ô 3 in ra **API URL** + **API key**, chạy đoạn dưới **trên máy bạn**
(không phải trong Colab — ô 3 đang giữ server). Mặc định **32 bước, engine tự
chia đoạn** nên chỉ cần gửi cả đoạn, KHÔNG tự cắt 250/500 nữa:

```python
import requests, base64, io, soundfile as sf

URL = "https://....trycloudflare.com"   # dán từ ô 3
KEY = "..."                              # dán từ ô 3
H = {"X-API-Key": KEY}

# 1) Đăng ký giọng mẫu: ref audio + transcript ĐÚNG của ref (rất quan trọng —
#    transcript sai/ngắn làm ước lượng độ dài loạn; để TRỐNG "" thì an toàn).
audio, sr = sf.read("ref.wav")
buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
vid = requests.post(f"{URL}/voice", json={
    "name": "toi",
    "text": "transcript ĐÚNG của file ref.wav",
    "wav_b64": base64.b64encode(buf.getvalue()).decode(),
}, timeout=180, headers=H).json()["voice_id"]

# 2) Tạo giọng: gửi CẢ đoạn (tới 8000 ký tự), server tự chia -> giọng mượt.
r = requests.post(f"{URL}/tts", json={
    "text": "Câu một... Câu hai... (đoạn dài tuỳ ý, để engine tự chia).",
    "voice_id": vid, "lang": "Vietnamese", "steps": 32,
}, timeout=900, headers=H)
open("out.wav", "wb").write(r.content)
print("Xong -> out.wav  |  synth", r.headers.get("X-Synth-Seconds"), "s")
```

**Bài dài nhiều đoạn?** Dùng `/tts_batch` với `items=[{"i":0,"text":...}, ...]`
để gộp một request (đỡ trễ mạng). Mỗi item vẫn nên là cả đoạn, không cắt vụn.
"""

CLIENT_INPUT = """\
#@title 1. Nhập key bản quyền { display-mode: "form" }
#@markdown Dán key bản quyền của bạn rồi chạy ô này.
LICENSE_KEY = ""  #@param {type:"string"}

import hashlib
LICENSE_KEY = LICENSE_KEY.strip()
assert LICENSE_KEY and not LICENSE_KEY.startswith("XXXX"), "Chưa nhập key bản quyền."

# device_id GẮN VỚI KEY, không theo phần cứng: mỗi máy Colab là một VM mới nên
# fingerprint phần cứng đổi mỗi phiên -> mỗi lần chạy ăn một slot thiết bị của
# license, vài lần là hết. Hash của key thì cố định -> mỗi key luôn là "một
# thiết bị", không churn slot.
DEVICE_ID = "colab-" + hashlib.sha256(LICENSE_KEY.encode()).hexdigest()[:24]
# MẶC ĐỊNH 1 LUỒNG: chạy tuần tự thì pool VRAM chỉ nở đúng một buffer, có trần
# cố định, KHÔNG cộng dồn qua các job -> không OOM dù nhiều câu dài. Nhiều luồng
# nhanh hơn nhưng pool nở theo số luồng song song và không co lại -> dễ cạn VRAM
# rồi crash. Đổi số này lên chỉ khi đã có guard giới hạn VRAM ở server.
WORKERS = 1  #@param {type:"integer"}
print("Key nhận rồi. device_id =", DEVICE_ID, "| WORKERS =", WORKERS)
"""

CLIENT_DEPS = """\
# ── 2. Cài thư viện + tải loader bảo mật ──────────────────────────────
import hashlib, urllib.request, importlib.util, sys, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "httpx", "cryptography"], check=True)

BASE = "%%REPO%%".replace(".git", "")
SO_URL  = f"{BASE}/releases/download/%%TAG%%/%%ASSET%%"
SHA_URL = SO_URL + ".sha256"
SO_PATH = "/content/%%ASSET%%"

print("Tải loader:", SO_URL)
urllib.request.urlretrieve(SO_URL, SO_PATH)
try:
    want = urllib.request.urlopen(SHA_URL, timeout=20).read().decode().split()[0]
    got = hashlib.sha256(open(SO_PATH, "rb").read()).hexdigest()
    assert got == want, f"checksum loader KHÔNG khớp:\\n  cần {want}\\n  được {got}"
    print("checksum loader khớp.")
except Exception as e:
    print("(bỏ qua kiểm checksum:", e, ")")

# Nạp loader .so vào như module 'voice_loader'.
spec = importlib.util.spec_from_file_location("voice_loader", SO_PATH)
voice_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voice_loader)
print("Loader nạp OK — phiên bản binary, kill-switch bên trong.")
"""

CLIENT_RUN = """\
# ── 3. Đăng nhập + bật server (ĐỂ Ô NÀY CHẠY LIÊN TỤC) ────────────────
# Loader tự: xác thực key -> tải gói mã hoá từ R2 -> giải mã trong RAM ->
# bật server voice -> mở đường hầm -> in API URL + key. Định kỳ gọi về; nếu
# key bị thu hồi thì tự tắt.
rc = voice_loader.run(LICENSE_KEY, DEVICE_ID, WORKERS)
print("Kết thúc, rc =", rc)
"""


def build_loader_nb(repo: str, branch: str) -> dict:
    def s(t):
        return (t.replace("%%REPO%%", repo).replace("%%BRANCH%%", branch)
                .replace("%%TAG%%", LOADER_RELEASE_TAG).replace("%%ASSET%%", LOADER_ASSET))
    return nb([md(s(BUILD_INTRO)), code(s(BUILD_CELL))])


def build_client_nb(repo: str, branch: str) -> dict:
    def s(t):
        return (t.replace("%%REPO%%", repo).replace("%%BRANCH%%", branch)
                .replace("%%TAG%%", LOADER_RELEASE_TAG).replace("%%ASSET%%", LOADER_ASSET))
    return nb([md(s(CLIENT_INTRO)), code(s(CLIENT_INPUT)),
               code(s(CLIENT_DEPS)), code(s(CLIENT_RUN)), md(s(CLIENT_USAGE))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default="master")
    args = ap.parse_args()

    for path, doc in ((OUT_BUILD, build_loader_nb(args.repo, args.branch)),
                      (OUT_CLIENT, build_client_nb(args.repo, args.branch))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Đã ghi {path}  ({len(doc['cells'])} ô)")


if __name__ == "__main__":
    main()
