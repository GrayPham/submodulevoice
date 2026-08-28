"""Đóng gói bộ test GPU thành một exe + dữ liệu, rồi nén lại thành một file zip.

    python packaging/build_package.py

Bố cục sinh ra trong packaging/dist/OmniVoiceGpuTest/:

    START-TEST.exe          <- bấm vào là chạy
    CHAY-PROFILE-TINY.cmd   <- chạy lại bằng model nhỏ nhất khi card 2 GB bị hụt
    runtime/                omnivoice.dll + ggml*.dll (CUDA, sm_61..sm_89)
    models/                 *.gguf
    data/                   ref.wav + ref.txt + script.txt (10.450 ký tự)
    DOC-TRUOC-KHI-CHAY.txt

Vì sao model nằm ngoài exe chứ không nhúng hết vào một file duy nhất: gộp cả
660 MB model vào onefile khiến PyInstaller phải giải nén lại toàn bộ vào %TEMP%
ở MỖI lần chạy, mất hàng phút và tốn thêm 660 MB đĩa mỗi lần. Launcher trong
ảnh chụp màn hình cũng để model ở thư mục riêng vì đúng lý do này.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
DIST = PKG / "dist"
APP = DIST / "OmniVoiceGpuTest"

BUILD = ROOT / "omnivoice.cpp" / "build-cuda"
MODELS_SRC = ROOT / "omnivoice.cpp" / "models"

# Chỉ những model bộ test dùng tới: profile lite và tiny.
MODELS = [
    "omnivoice-base-Q4_K_M.gguf",
    "omnivoice-tokenizer-Q8_0.gguf",
    "omnivoice-tokenizer-Q4_K_M.gguf",
]

# ggml-cuda.dll phụ thuộc động vào runtime CUDA. Driver NVIDIA KHÔNG cung cấp
# mấy file này — chỉ CUDA Toolkit mới có. Máy nào không cài Toolkit mà thiếu
# chúng thì ggml_backend_load_all() bỏ qua ggml-cuda.dll trong im lặng và
# ov_init báo "no GGML backend available".
CUDA_BIN = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin")
CUDA_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]

# Runtime C++ của MSVC, phòng trường hợp máy đích chưa cài VC++ Redistributable.
VC_REDIST = Path(r"C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Redist/MSVC/14.40.33807/x64/Microsoft.VC143.CRT")
VC_DLLS = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]

REF_WAV = ROOT / "output" / "refs3" / "FDown.vn_Tai_video_Facebook_MP3_9995-ref.wav"
REF_TXT = REF_WAV.with_suffix(".txt")
SCRIPT = ROOT / "scripts" / "kichban_10k.txt"

READ_ME_FILE = PKG / "readme_staff.txt"

TINY_CMD = """\
@echo off
rem Chạy lại bằng profile nhỏ nhất — codec cũng lượng tử hoá 4-bit.
rem Dùng khi profile mặc định báo thiếu VRAM trên card 2 GB.
set GPUTEST_PROFILE=tiny
"%~dp0START-TEST.exe"
"""


def check_inputs() -> None:
    missing = [str(p) for p in [BUILD / "omnivoice.dll", REF_WAV, REF_TXT, SCRIPT,
                                READ_ME_FILE] if not p.exists()]
    missing += [str(CUDA_BIN / d) for d in CUDA_DLLS if not (CUDA_BIN / d).exists()]
    missing += [str(MODELS_SRC / m) for m in MODELS if not (MODELS_SRC / m).exists()]
    if missing:
        raise SystemExit("Thiếu file đầu vào:\n  " + "\n  ".join(missing))


def build_exe() -> Path:
    work = PKG / "build"
    for d in (work, DIST):
        shutil.rmtree(d, ignore_errors=True)
    # Build trong venv riêng: Python hệ thống có package `pathlib` backport cũ
    # mà PyInstaller từ chối làm việc cùng, và gỡ nó ra là động vào môi trường
    # chung của máy.
    venv_py = PKG / ".venv" / "Scripts" / "python.exe"
    py = str(venv_py) if venv_py.exists() else sys.executable
    cmd = [
        py, "-m", "PyInstaller",
        "--onefile", "--console", "--name", "START-TEST",
        "--distpath", str(DIST), "--workpath", str(work),
        "--specpath", str(work),
        "--paths", str(ROOT),
        "--hidden-import", "pyomnivoice",
        "--hidden-import", "pyomnivoice.core",
        "--hidden-import", "pyomnivoice._ffi",
        # Những thứ chỉ dùng lúc chuẩn bị giọng mẫu, bộ test không cần.
        "--exclude-module", "faster_whisper",
        "--exclude-module", "ctranslate2",
        "--exclude-module", "torch",
        "--exclude-module", "transformers",
        "--exclude-module", "librosa",
        "--exclude-module", "scipy",
        "--exclude-module", "matplotlib",
        "--exclude-module", "soundfile",
        "--exclude-module", "PIL",
        "--exclude-module", "tkinter",
        str(PKG / "gputest.py"),
    ]
    print("  " + " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    exe = DIST / "START-TEST.exe"
    if not exe.exists():
        raise SystemExit("PyInstaller không tạo được exe")
    return exe


def assemble(exe: Path) -> None:
    APP.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exe), str(APP / "START-TEST.exe"))

    rt = APP / "runtime"
    rt.mkdir(exist_ok=True)
    for dll in sorted(BUILD.glob("*.dll")):
        shutil.copy2(dll, rt / dll.name)
    for d in CUDA_DLLS:
        shutil.copy2(CUDA_BIN / d, rt / d)
    for d in VC_DLLS:
        src = VC_REDIST / d
        if src.exists():
            shutil.copy2(src, rt / d)
        else:
            print(f"  bỏ qua {d} (không tìm thấy VC redist)")

    md = APP / "models"
    md.mkdir(exist_ok=True)
    for m in MODELS:
        shutil.copy2(MODELS_SRC / m, md / m)

    dd = APP / "data"
    dd.mkdir(exist_ok=True)
    shutil.copy2(REF_WAV, dd / "ref.wav")
    shutil.copy2(REF_TXT, dd / "ref.txt")
    shutil.copy2(SCRIPT, dd / "script.txt")

    (APP / "DOC-TRUOC-KHI-CHAY.txt").write_text(
        READ_ME_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (APP / "CHAY-PROFILE-TINY.cmd").write_text(TINY_CMD, encoding="utf-8")


def zip_it() -> Path:
    out = DIST / "OmniVoiceGpuTest.zip"
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(APP.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(APP.parent))
                total += p.stat().st_size
    print(f"  đã nén {total / 2**20:.0f} MiB -> {out.stat().st_size / 2**20:.0f} MiB")
    return out


def main() -> None:
    print("[1/4] kiểm tra đầu vào")
    check_inputs()
    print("[2/4] build exe bằng PyInstaller")
    exe = build_exe()
    print("[3/4] ghép thư mục phát hành")
    assemble(exe)
    print("[4/4] nén zip")
    z = zip_it()

    print("\nXong.")
    for p in sorted(APP.iterdir()):
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        print(f"  {size / 2**20:8.1f} MiB  {p.name}{'/' if p.is_dir() else ''}")
    print(f"\n  gói: {z}")


if __name__ == "__main__":
    main()
