"""Sinh notebook Colab BUILD BẢN BẢO MẬT của server voice.

    python remote/build_secure_notebook.py --repo https://github.com/ban/submodulevoice.git

Khác với build_notebook.py (sinh notebook CHẠY server, phơi nguyên mã .py),
notebook này BIÊN DỊCH mã Python thành .so bằng Cython ngay trên Colab, xoá
sạch file .py nguồn, rồi đóng gói thành một tar.gz để phát hành. Sản phẩm là
một artifact chạy được mà không kèm mã nguồn Python.

Vì sao build trên Colab chứ không build sẵn ở máy Windows:
  - .so của Cython gắn với ABI + phiên bản Python. Colab là Linux, Python của
    Colab đổi theo thời gian. Build ngay trên Colab thì .so luôn khớp máy sẽ
    chạy nó, không phải đoán nền tảng, không lo lệch phiên bản.
  - C++ runtime (omnivoice.cpp) mất ~90 phút nếu build từ nguồn, nên mặc định
    lấy bản prebuilt; Cython chỉ tốn vài giây mỗi module. Đặt BUILD_CPP=True
    nếu muốn tự build cả phần C++.

Giới hạn thành thật: mã .so vẫn là mã máy chạy trên máy khách, người quyết tâm
vẫn dịch ngược được. Cython nâng rào từ "đọc thẳng .py" lên "phải RE mã máy",
không phải bất khả xâm phạm. Chốt kiểm soát thật nằm ở license phía server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "OmniVoice_Build_Secure_Colab.ipynb"
DEFAULT_REPO = "https://github.com/CHUA-DAT-REPO/submodulevoice.git"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


INTRO = """\
# OmniVoice — Build bản bảo mật (Cython) trên Colab

Notebook này **không chạy** server. Nó **đóng gói** server thành một bản phát
hành đã giấu mã Python:

1. Lấy mã nguồn + C++ runtime
2. Cython biên dịch các module Python (`server.py`, `pyomnivoice/*`) thành `.so`
3. Xoá sạch file `.py` nguồn
4. Nén thành `omnivoice-secure-<pyver>.tar.gz` + in checksum

Bản `.tar.gz` này là thứ tải lên R2 để notebook khách chỉ việc kéo về chạy.

> **Cần GPU?** Không, trừ khi bật `BUILD_CPP=True`. Bản mặc định (prebuilt C++
> + Cython) chạy được trên runtime CPU, xong trong vài phút.
"""

# ── Cell 1: cấu hình ────────────────────────────────────────────────
CELL_CONFIG = """\
# ── 1. Cấu hình ───────────────────────────────────────────────────────
REPO_URL = "%%REPO%%"
BRANCH   = "%%BRANCH%%"

# False: lấy C++ runtime prebuilt (nhanh, vài phút).
# True : build omnivoice.cpp từ nguồn (~90 phút, cần runtime GPU).
BUILD_CPP = False

# Danh sách module Python sẽ biên dịch thành .so. Mọi thứ khách cần import mà
# ta không muốn phơi nguồn đều phải nằm ở đây.
CYTHON_TARGETS = [
    "remote/server.py",
    "pyomnivoice/__init__.py",
    "pyomnivoice/_ffi.py",
    "pyomnivoice/core.py",
    "pyomnivoice/refprep.py",
    "pyomnivoice/srt.py",
    "pyomnivoice/download.py",
]

# File .py để nguyên (điểm vào khách gọi, không chứa logic đáng giấu).
KEEP_PY = ["remote/client.py"]

import os, sys, subprocess, shutil, hashlib, textwrap
from pathlib import Path

APP  = Path("/content/app")
DIST = Path("/content/dist")
PYTAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python Colab : {sys.version.split()[0]}  (tag {PYTAG})")
print(f"Repo         : {REPO_URL} @ {BRANCH}")
print(f"Build C++    : {'CÓ (từ nguồn)' if BUILD_CPP else 'không — dùng prebuilt'}")
"""

# ── Cell 2: hàm chạy lệnh có stream log ─────────────────────────────
CELL_SH = """\
# ── 2. Hàm chạy lệnh (STREAM log, không để ô treo câm) ────────────────
# Bài học cũ: subprocess.run nuốt output, Colab không thấy gì -> ô như treo
# suốt cả tiếng. Phải Popen + đọc từng dòng thì mới thấy tiến độ.
def sh(cmd, cwd=None, tail=0):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)
    buf = []
    for line in p.stdout:
        if tail:
            buf.append(line)
            buf = buf[-tail:]
            print(f"\\r{line.rstrip()[:100]:<100}", end="", flush=True)
        else:
            print(line, end="", flush=True)
    p.wait()
    if tail:
        print()
    if p.returncode != 0:
        if tail:
            print("".join(buf))
        raise RuntimeError(f"lệnh lỗi ({p.returncode}): {cmd}")
"""

# ── Cell 3: lấy mã nguồn + C++ runtime ──────────────────────────────
CELL_SOURCE = """\
# ── 3. Lấy mã nguồn + C++ runtime ─────────────────────────────────────
if APP.exists():
    sh(f"git -C {APP} fetch --depth 1 origin {BRANCH} && "
       f"git -C {APP} reset --hard origin/{BRANCH}")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {APP}")

sh("pip install -q cython numpy huggingface_hub soundfile")

CUDA_DIR = APP / "omnivoice.cpp" / "build" / "bin"
if BUILD_CPP:
    print("\\n=== Build omnivoice.cpp từ nguồn (lâu) ===")
    if not (APP / "omnivoice.cpp" / "CMakeLists.txt").exists():
        sh("git clone --recurse-submodules --depth 1 "
           "https://github.com/k2-fsa/omnivoice.cpp "
           f"{APP}/omnivoice.cpp")
    src = APP / "omnivoice.cpp"
    sh("cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
       "-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DOMNIVOICE_SHARED=ON",
       cwd=str(src))
    sh("cmake --build build -j$(nproc)", cwd=str(src), tail=1)
else:
    # Prebuilt runtime-linux từ Releases của repo. Asset đặt tên theo kiến trúc
    # GPU: ...-sm75 (T4), -sm80/86 (A100/30xx), -sm89 (40xx). Tự dò GPU rồi ghép
    # đúng hậu tố; nếu asset đúng kiến trúc không có thì lùi về sm75 (phổ biến
    # nhất trên Colab free).
    base = REPO_URL.replace(".git", "")
    rel_url = base + "/releases/download/runtime-linux"
    try:
        cc = subprocess.check_output(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
            shell=True, text=True).strip().splitlines()[0].replace(".", "")
    except Exception:
        cc = "75"
    tgz = "/content/runtime.tgz"
    got = False
    for sm in (cc, "75"):
        url = f"{rel_url}/omnivoice-linux-cuda-sm{sm}.tar.gz"
        print(f"\\n=== Tải C++ runtime prebuilt (sm{sm}) ===\\n{url}")
        rc = subprocess.call(f"curl -fL -o {tgz} {url}", shell=True)
        if rc == 0:
            got = True
            break
        print(f"  không có asset sm{sm}, thử phương án khác…")
    assert got, ("không tải được runtime prebuilt. Kiểm tra Releases có asset "
                 f"omnivoice-linux-cuda-sm{cc}.tar.gz hoặc -sm75, hoặc đặt BUILD_CPP=True")
    (APP / "omnivoice.cpp" / "build" / "bin").mkdir(parents=True, exist_ok=True)
    sh(f"tar xzf {tgz} -C {APP}/omnivoice.cpp/build/bin")

so_count = len(list(CUDA_DIR.glob("*.so*"))) if CUDA_DIR.exists() else 0
print(f"\\nC++ runtime: {so_count} file .so trong {CUDA_DIR}")
assert so_count >= 3, "thiếu .so của C++ runtime — kiểm tra lại bước này"

# Model INT4 nhúng thẳng vào gói: server tìm ở omnivoice.cpp/models (core.py).
# Gói thành ~700 MB nhưng mọi thứ đi qua một kênh có license, không phụ thuộc
# HuggingFace lúc khách chạy.
from huggingface_hub import hf_hub_download
MODELS = APP / "omnivoice.cpp" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
for f in ("omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q8_0.gguf"):
    if (MODELS / f).exists():
        print("[model có sẵn]", f)
    else:
        print("[tải model]", f, flush=True)
        hf_hub_download("Serveurperso/OmniVoice-GGUF", f, local_dir=str(MODELS))
ggufs = sorted(p.name for p in MODELS.glob("*.gguf"))
print("model:", ggufs)
assert len(ggufs) >= 2, "thiếu file model .gguf"
"""

# ── Cell 4: Cython biên dịch Python -> .so ──────────────────────────
CELL_CYTHON = """\
# ── 4. Cython: biên dịch .py -> .so, rồi XOÁ .py nguồn ────────────────
# Mỗi module dịch riêng thành .so đặt cạnh .py, sau đó xoá .py. import trong
# Python tự ưu tiên .so cùng tên nên mã chạy y hệt, chỉ khác là không còn nguồn.
#
# QUAN TRỌNG: build_ext --inplace chép .so tới đường dẫn TƯƠNG ĐỐI (vd
# remote/server...so) theo THƯ MỤC ĐANG ĐỨNG. Colab mặc định đứng ở /content,
# không phải /content/app, nên chép hụt -> FileNotFoundError. Phải chdir vào
# gốc app trước, rồi mọi đường dẫn tương đối mới khớp.
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

os.chdir(APP)
print("thư mục build:", os.getcwd())

def build_one(rel):
    if not Path(rel).exists():          # rel tương đối với cwd = APP
        print(f"  bỏ qua (không có): {rel}")
        return None
    # Tên module = đường dẫn có dấu chấm, để __init__.so nằm đúng gói.
    modname = rel[:-3].replace("/", ".")
    ext = Extension(modname, [rel])
    dist = Distribution({"ext_modules": cythonize(
        [ext], language_level=3,
        compiler_directives={"emit_code_comments": False},
        quiet=True)})
    dist.script_args = ["build_ext", "--inplace", "--build-temp", "/tmp/cybuild"]
    dist.parse_command_line()
    dist.run_commands()
    return modname

print("=== Biên dịch từng module ===")
built = []
for rel in CYTHON_TARGETS:
    print(f"→ {rel}", flush=True)
    if build_one(rel):
        built.append(rel)

# Xoá .py nguồn + .c trung gian của những module ĐÃ dịch. Giữ .py trong KEEP_PY.
print("\\n=== Xoá mã nguồn đã biên dịch ===")
for rel in built:
    for junk in (APP / rel, APP / (rel[:-3] + ".c")):
        if junk.exists():
            junk.unlink()
            print(f"  xoá {junk.relative_to(APP)}")

# Kiểm chứng: mỗi target phải còn ĐÚNG một .so và KHÔNG còn .py.
print("\\n=== Kiểm chứng ===")
ok = True
for rel in built:
    d, stem = (APP / rel).parent, Path(rel).stem
    so = list(d.glob(f"{stem}*.so"))
    py = (APP / rel).exists()
    mark = "OK " if (so and not py) else "!! "
    if not (so and not py):
        ok = False
    print(f"  {mark}{rel}: {len(so)} .so, còn .py: {py}")
assert ok, "có module chưa thành .so hoặc còn sót .py — DỪNG, đừng phát hành bản hở"
print("\\nTất cả module đã thành .so, không còn .py nguồn.")
"""

# ── Cell 5: kiểm tra bản .so chạy được ──────────────────────────────
CELL_SMOKE = """\
# ── 5. Kiểm tra bản đã biên dịch còn chạy (import thật, không phơi nguồn) ─
# Chạy trong tiến trình con với cwd = APP để import lấy đúng .so vừa tạo.
test = textwrap.dedent('''
    import sys
    sys.path.insert(0, ".")
    import importlib, pathlib
    m = importlib.import_module("remote.server")
    src = pathlib.Path(m.__file__)
    assert src.suffix == ".so", f"server nạp từ {src.suffix}, không phải .so"
    import pyomnivoice
    assert pathlib.Path(pyomnivoice.__file__).suffix == ".so"
    print("import OK — server.__file__ =", src.name)
    print("PROFILES:", list(getattr(m, "PROFILES", {}) or
          __import__("pyomnivoice").core.PROFILES if hasattr(pyomnivoice,"core") else {}))
''')
(APP / "_smoke.py").write_text(test)
sh(f"cd {APP} && python _smoke.py")
(APP / "_smoke.py").unlink()
"""

# ── Cell 6: đóng gói + checksum ─────────────────────────────────────
CELL_PACK = """\
# ── 6. Đóng gói bản phát hành + checksum ──────────────────────────────
DIST.mkdir(parents=True, exist_ok=True)
out = DIST / f"omnivoice-secure-{PYTAG}-linux-x86_64.tar.gz"

# Gói gọn: chỉ .so, điểm vào KEEP_PY, C++ runtime, và metadata. Không kèm .git,
# không kèm test, không kèm scripts kịch bản.
INCLUDE = ["remote", "pyomnivoice", "omnivoice.cpp/build/bin", "omnivoice.cpp/models"]
manifest = {
    "python_tag": PYTAG,
    "built_from": f"{REPO_URL}@{BRANCH}",
    "cython_modules": [r[:-3].replace("/", ".") for r in CYTHON_TARGETS],
    "entrypoints": KEEP_PY,
}
(APP / "BUILD_MANIFEST.json").write_text(__import__("json").dumps(manifest, indent=2))

# Dọn sạch mọi .py không thuộc KEEP_PY khỏi các thư mục sắp đóng gói. Đây là
# chốt chặn thật: module logic đã thành .so ở ô 4, nhưng thư mục vẫn còn các
# script phụ (bộ sinh notebook, tiện ích...) ở dạng .py. Xoá hẳn trước khi nén
# thì dù sau này thêm script nào cũng không lọt. KEEP_PY là điểm vào duy nhất
# được phép giữ nguyên nguồn.
keep = {str((APP / k).resolve()) for k in KEEP_PY}
purged = []
for top in INCLUDE:
    d = APP / top
    if not d.exists():
        continue
    for py in d.rglob("*.py"):
        if str(py.resolve()) not in keep:
            py.unlink()
            purged.append(str(py.relative_to(APP)))
if purged:
    print("Dọn .py thừa khỏi gói:")
    for p in purged:
        print("  ", p)

inc = " ".join(f"app/{p}" for p in INCLUDE) + " app/BUILD_MANIFEST.json"
excl = "--exclude='__pycache__' --exclude='*.pyc' --exclude='*.c'"
sh(f"cd /content && tar czf {out} {excl} {inc}")

# Checksum để notebook khách kiểm tra tải đủ, không hỏng giữa chừng.
digest = hashlib.sha256(out.read_bytes()).hexdigest()
(DIST / (out.name + ".sha256")).write_text(f"{digest}  {out.name}\\n")

mb = out.stat().st_size / 1024 / 1024
print(f"\\n=== XONG ===")
print(f"Bản phát hành : {out}  ({mb:.1f} MB)")
print(f"SHA-256       : {digest}")
print(f"\\nTải file này về rồi upload lên R2. Notebook khách kéo về, kiểm")
print(f"sha256 khớp, giải nén và chạy:  python -m remote.server --workers 4")
"""

# ── Cell 7: (tuỳ chọn) kiểm tra còn sót .py không ───────────────────
CELL_AUDIT = """\
# ── 7. (tuỳ chọn) Soát lại gói: có lọt .py logic nào không ─────────────
import tarfile
leaked = []
with tarfile.open(out) as t:
    for m in t.getmembers():
        if m.name.endswith(".py"):
            # chỉ KEEP_PY được phép là .py
            rel = m.name.split("app/", 1)[-1]
            if rel not in KEEP_PY:
                leaked.append(m.name)
if leaked:
    print("!! LỌT MÃ NGUỒN — các file .py này không nên có trong gói:")
    for x in leaked:
        print("   ", x)
else:
    print("Sạch: trong gói chỉ còn .so và điểm vào cho phép, không lọt .py logic.")
"""


def build(repo: str, branch: str) -> dict:
    cells = [md(INTRO)]
    for tmpl in (CELL_CONFIG, CELL_SH, CELL_SOURCE, CELL_CYTHON,
                 CELL_SMOKE, CELL_PACK, CELL_AUDIT):
        cells.append(code(tmpl.replace("%%REPO%%", repo).replace("%%BRANCH%%", branch)))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    nb = build(args.repo, args.branch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out}")
    print(f"  repo   : {args.repo}")
    print(f"  branch : {args.branch}")
    print(f"  {len(nb['cells'])} ô")


if __name__ == "__main__":
    main()
