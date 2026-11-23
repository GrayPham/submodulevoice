"""Tải engine audio.cpp + Fish Drama-3 GGUF và đóng gói bản chuẩn INT4 (< 2.8 GB VRAM).

    python remote/prepare_drama3_int4.py --engine-only
    python remote/prepare_drama3_int4.py --remove-orig

Trên Linux (Google Colab):
  1. Tải bản dựng sẵn audio.cpp v0.9.6 (CUDA 12.8 Colab) + CUDA runtime libs.
  2. Tải bản gốc Fish Audio Drama 3 (256 Experts):
     Jmica/Drama-3-GGUF / drama-3-backbone-8bit-experts-Q2_K.gguf (15.88 GiB)
     bằng nhiều luồng HTTP Range song song (hỗ trợ resume).
  3. Giải nén embedded sidecars (_sidecar_root) và chạy `audiocpp_gguf` với bộ
     lượng tử hoá INT4 chuẩn giữ nguyên 100% chất lượng tiếng Việt ->
     `drama-3-slim-int4-q2k.gguf` (13.27 GiB).
  4. Tuỳ chọn xoá file gốc 15.88 GiB (--remove-orig) để tiết kiệm ổ đĩa Colab.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

AUDIOCPP_TAG = "v0.9.6"
LINUX_BIN_URL = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_TAG}/"
    f"audio-{AUDIOCPP_TAG}-bin-ubuntu-x64-cuda12.8-colab.tar.gz"
)
LINUX_CUDART_URL = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_TAG}/"
    f"audio-{AUDIOCPP_TAG}-cudart-ubuntu-x64-cuda12.8.tar.gz"
)
LINUX_CPU_BIN_URL = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_TAG}/"
    f"audio-{AUDIOCPP_TAG}-bin-ubuntu-x64-cpu.tar.gz"
)
WIN_BIN_URL = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_TAG}/"
    f"audio-{AUDIOCPP_TAG}-bin-windows-x64-cuda12.4.zip"
)
WIN_CUDART_URL = (
    f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_TAG}/"
    f"audio-{AUDIOCPP_TAG}-cudart-windows-x64-cuda12.4.zip"
)
FISH_SPEC_RAW_URL = (
    f"https://raw.githubusercontent.com/0xShug0/audio.cpp/{AUDIOCPP_TAG}/"
    "model_specs/fish_drama3.json"
)

DEFAULT_HF_REPO = "Jmica/Drama-3-GGUF"
DEFAULT_ORIG_GGUF = "drama-3-backbone-8bit-experts-Q2_K.gguf"
DEFAULT_INT4_GGUF = "drama-3-slim-int4-q2k.gguf"
MIN_INT4_VALID_BYTES = 12 * 1024**3  # bản INT4 chuẩn ~13.27 GiB (14,251,982,912 bytes)


def _exe_name(base: str) -> str:
    return f"{base}.exe" if os.name == "nt" else base


def _default_engine_dir() -> Path:
    if sys.platform.startswith("linux"):
        return Path("/content/drama3_engine")
    return Path(__file__).resolve().parent.parent / "drama3_engine"


def _default_models_dir() -> Path:
    if sys.platform.startswith("linux"):
        return Path("/content/drama3_models")
    return Path(__file__).resolve().parent.parent / "drama3_models"


def _download_file(url: str, dest: Path, label: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[tai] {label or dest.name}: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lvc-drama3/1"})
    with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last_log = time.monotonic()
        while True:
            buf = r.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            now = time.monotonic()
            if total and now - last_log >= 3.0:
                pct = got * 100.0 / total
                print(
                    f"      {got / (1024**2):.1f} / {total / (1024**2):.1f} MiB ({pct:.0f}%)",
                    flush=True,
                )
                last_log = now
    os.replace(part, dest)
    return dest


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                tar.extractall(dest_dir, filter="data")
            except TypeError:
                tar.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    else:
        raise ValueError(f"Dinh dang luu tru khong ho tro: {archive_path}")


def _find_first(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None


def collect_lib_dirs(engine_dir: Path) -> list[str]:
    """Gom các thư mục chứa .so/.dll của audio.cpp và CUDA runtime."""
    dirs: list[str] = []
    seen: set[str] = set()
    pattern = "*.dll" if os.name == "nt" else "*.so*"
    for p in engine_dir.rglob(pattern):
        d = str(p.parent.resolve())
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    if sys.platform.startswith("linux"):
        for cand in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ):
            if os.path.isdir(cand) and cand not in seen:
                seen.add(cand)
                dirs.append(cand)
    return dirs


def ensure_audio_cpp_engine(engine_dir: Path, need_gguf: bool = True) -> dict[str, Path]:
    """Đảm bảo đủ audiocpp_server (CUDA Colab) và audiocpp_gguf (để nén INT4)."""
    engine_dir = Path(engine_dir).resolve()
    engine_dir.mkdir(parents=True, exist_ok=True)

    gguf_name = _exe_name("audiocpp_gguf")
    srv_name = _exe_name("audiocpp_server")

    srv_bin = _find_first(engine_dir, srv_name)
    gguf_bin = _find_first(engine_dir, gguf_name)

    if not srv_bin:
        if sys.platform.startswith("linux"):
            bin_pkg = engine_dir / "_audiocpp_bin.tar.gz"
            cudart_pkg = engine_dir / "_audiocpp_cudart.tar.gz"
            _download_file(LINUX_BIN_URL, bin_pkg, "audio.cpp Colab CUDA12.8")
            _extract_archive(bin_pkg, engine_dir)
            bin_pkg.unlink(missing_ok=True)

            _download_file(LINUX_CUDART_URL, cudart_pkg, "audio.cpp CUDA 12.8 runtime")
            _extract_archive(cudart_pkg, engine_dir)
            cudart_pkg.unlink(missing_ok=True)
        elif os.name == "nt":
            bin_pkg = engine_dir / "_audiocpp_bin.zip"
            cudart_pkg = engine_dir / "_audiocpp_cudart.zip"
            _download_file(WIN_BIN_URL, bin_pkg, "audio.cpp Windows CUDA12.4")
            _extract_archive(bin_pkg, engine_dir)
            bin_pkg.unlink(missing_ok=True)

            _download_file(WIN_CUDART_URL, cudart_pkg, "audio.cpp CUDA 12.4 runtime")
            _extract_archive(cudart_pkg, engine_dir)
            cudart_pkg.unlink(missing_ok=True)
        else:
            raise SystemExit(f"Chưa hỗ trợ tự tải engine cho nền tảng {sys.platform}.")

        srv_bin = _find_first(engine_dir, srv_name)
        gguf_bin = _find_first(engine_dir, gguf_name)

    # Gói -colab.tar.gz của audio.cpp chỉ đóng gói audiocpp_server (không kèm audiocpp_gguf).
    # Tải thêm gói CPU vào thư mục con gguf_tool/ để lấy audiocpp_gguf nén INT4.
    if need_gguf and not gguf_bin and sys.platform.startswith("linux"):
        gguf_dir = engine_dir / "gguf_tool"
        cpu_pkg = gguf_dir / "_audiocpp_cpu.tar.gz"
        _download_file(LINUX_CPU_BIN_URL, cpu_pkg, "audio.cpp CPU tools (audiocpp_gguf)")
        _extract_archive(cpu_pkg, gguf_dir)
        cpu_pkg.unlink(missing_ok=True)
        gguf_bin = _find_first(gguf_dir, gguf_name)

    if not srv_bin or (need_gguf and not gguf_bin):
        raise FileNotFoundError(
            f"Không tìm thấy đủ audiocpp_server / audiocpp_gguf trong {engine_dir}"
        )

    if not os.name == "nt":
        for p in (srv_bin, gguf_bin):
            if p is not None and p.exists():
                os.chmod(p, os.stat(p).st_mode | 0o111)

    spec_path = _find_first(engine_dir, "fish_drama3.json")
    if not spec_path:
        spec_path = engine_dir / "model_specs" / "fish_drama3.json"
        _download_file(FISH_SPEC_RAW_URL, spec_path, "fish_drama3.json spec")

    out = {"server": srv_bin, "spec": spec_path}
    if gguf_bin is not None:
        out["gguf"] = gguf_bin
    return out


def _resolve_remote_size(url: str, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, headers={**headers, "User-Agent": "lvc-drama3/1"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.headers.get("Content-Length"):
                return int(r.headers["Content-Length"])
    except Exception:
        pass
    # Fallback dùng Range bytes=0-0 (HuggingFace trả Content-Range: bytes 0-0/TOTAL)
    req2 = urllib.request.Request(
        url, headers={**headers, "User-Agent": "lvc-drama3/1", "Range": "bytes=0-0"}
    )
    with urllib.request.urlopen(req2, timeout=30) as r:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else 0


def download_orig_gguf(
    models_dir: Path,
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_ORIG_GGUF,
    hf_token: str = "",
    workers: int = 8,
) -> Path:
    """Tải song song nhiều luồng HTTP Range có resume (hoặc fallback single stream)."""
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / filename

    headers: dict[str, str] = {}
    tok = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    total_size = _resolve_remote_size(url, headers)
    if total_size <= 0:
        raise RuntimeError(f"Không lấy được kích thước file từ {url}")

    cur_base = target.stat().st_size if target.exists() else 0
    if cur_base == total_size:
        print(f"[ok] Đã có đủ bản gốc: {target} ({total_size / (1024**3):.2f} GiB)", flush=True)
        return target
    if cur_base > total_size:
        target.unlink()
        cur_base = 0

    rem = total_size - cur_base
    print(
        f"[tai] {filename}: đã có {cur_base / (1024**3):.2f} / {total_size / (1024**3):.2f} GiB, "
        f"còn {rem / (1024**3):.2f} GiB ({workers} luồng song song)...",
        flush=True,
    )

    try:
        import httpx
    except ImportError:
        httpx = None

    part_size = (rem + workers - 1) // workers
    parts: list[tuple[int, int, int, Path]] = []
    for i in range(workers):
        s_off = cur_base + i * part_size
        e_off = min(total_size - 1, s_off + part_size - 1)
        if s_off <= e_off:
            parts.append((i, s_off, e_off, models_dir / f".{filename}.part_{i}"))

    progress = [0] * len(parts)
    for idx, (_, s_off, e_off, p_file) in enumerate(parts):
        expected_len = e_off - s_off + 1
        if p_file.exists():
            sz = p_file.stat().st_size
            if sz > expected_len:
                p_file.unlink()
            else:
                progress[idx] = sz

    errors: list[str] = []

    def _worker(idx: int, start_b: int, end_b: int, out_f: Path) -> None:
        expected = end_b - start_b + 1
        for attempt in range(12):
            done_now = out_f.stat().st_size if out_f.exists() else 0
            if done_now >= expected:
                progress[idx] = expected
                return
            req_start = start_b + done_now
            req_hdrs = {**headers, "Range": f"bytes={req_start}-{end_b}", "User-Agent": "lvc-drama3/1"}
            try:
                if httpx is not None:
                    with httpx.Client(
                        follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0)
                    ) as client:
                        with client.stream("GET", url, headers=req_hdrs) as resp:
                            if resp.status_code not in (200, 206):
                                raise RuntimeError(f"HTTP {resp.status_code}")
                            with open(out_f, "ab") as fp:
                                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                                    if chunk:
                                        fp.write(chunk)
                                        done_now += len(chunk)
                                        progress[idx] = done_now
                else:
                    req = urllib.request.Request(url, headers=req_hdrs)
                    with urllib.request.urlopen(req, timeout=120) as resp, open(out_f, "ab") as fp:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            fp.write(chunk)
                            done_now += len(chunk)
                            progress[idx] = done_now
                if out_f.stat().st_size == expected:
                    return
            except Exception as exc:  # noqa: BLE001
                if attempt == 11:
                    errors.append(f"luồng {idx}: {exc}")
                time.sleep(min(2.0 * (attempt + 1), 15.0))

    t0 = time.monotonic()
    init_sum = sum(progress)
    threads = [
        threading.Thread(target=_worker, args=(idx, s_off, e_off, p_file), daemon=True)
        for idx, (_, s_off, e_off, p_file) in enumerate(parts)
    ]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(5.0)
        done_parts = sum(progress)
        total_now = cur_base + done_parts
        elapsed = max(time.monotonic() - t0, 0.1)
        speed = (done_parts - init_sum) / elapsed / (1024**2)
        pct = total_now * 100.0 / total_size
        print(
            f"[tai] {total_now / (1024**3):.2f} / {total_size / (1024**3):.2f} GiB "
            f"({pct:.1f}%) | {speed:.1f} MB/s",
            flush=True,
        )

    if errors:
        raise RuntimeError("Tải GGUF thất bại: " + "; ".join(errors))

    print(f"[ghep] Đang ghép {len(parts)} mảnh vào {target.name}...", flush=True)
    with open(target, "ab") as base_fp:
        for _, s_off, e_off, p_file in parts:
            expected = e_off - s_off + 1
            actual = p_file.stat().st_size
            if actual != expected:
                raise RuntimeError(f"Mảnh {p_file} lệch kích thước: {actual} != {expected}")
            with open(p_file, "rb") as pf:
                while True:
                    buf = pf.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    base_fp.write(buf)
            p_file.unlink()

    final_sz = target.stat().st_size
    if final_sz != total_size:
        raise RuntimeError(f"Kích thước sau ghép không khớp: {final_sz} != {total_size}")
    print(f"[ok] Tải xong bản gốc: {target} ({final_sz / (1024**3):.2f} GiB)", flush=True)
    return target


def extract_embedded_sidecars(gguf_path: Path, out_dir: Path) -> list[str]:
    """Trích xuất config.json, tokenizer.json... nhúng sẵn trong GGUF gốc."""

    def r_str(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def r_val(f, vt: int):
        fmt = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h",
            4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d",
        }
        if vt in fmt:
            return struct.unpack(fmt[vt], f.read(struct.calcsize(fmt[vt])))[0]
        if vt == 8:
            return r_str(f)
        if vt == 9:
            at, al = struct.unpack("<IQ", f.read(12))
            if at == 0:
                return f.read(al)
            return [r_val(f, at) for _ in range(al)]
        raise ValueError(f"GGUF value type không hỗ trợ: {vt}")

    with open(gguf_path, "rb") as f:
        magic, _ver, _ntensors, nkv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"File không phải GGUF hợp lệ: {gguf_path}")
        kv = {}
        for _ in range(nkv):
            k = r_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            kv[k] = r_val(f, vt)

    names = kv["audiocpp.embedded_files.names"]
    offsets = kv["audiocpp.embedded_files.offsets"]
    blob = kv["audiocpp.embedded_files.data"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names):
        dest = out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob[offsets[idx] : offsets[idx + 1]])
    print(f"[sidecar] Đã giải nén {len(names)} file nhúng vào {out_dir}", flush=True)
    return list(names)


def build_int4_keep_types() -> list[str]:
    """Công thức nén INT4 chuẩn (< 2.8 GB VRAM, giữ nguyên 100% chất lượng tiếng Việt)."""
    args: list[str] = [
        "--keep-type", "model_weights/lm_head*=q4_0",
        "--keep-type", "model_weights/model.language_model.embed_tokens*=q4_0",
        "--keep-type", "model_weights/audio_decoder_mtp.heads.*=q4_0",
    ]
    for i in range(9):
        p = f"model_weights/audio_decoder_mtp.layers.{i}"
        args += [
            "--keep-type", f"{p}.codebook_embedding*=q8_0",
            "--keep-type", f"{p}.eh_proj*=q4_0",
            "--keep-type", f"{p}.transformer_layer.attention.wo*=q4_0",
            "--keep-type", f"{p}.transformer_layer.attention.wqkv*=q4_0",
            "--keep-type", f"{p}.transformer_layer.feed_forward.*=q4_0",
        ]
    for i in range(4):
        p = f"model_weights/audio_decoder_mtp_shared.layers.{i}"
        args += [
            "--keep-type", f"{p}.semantic_embedding*=q8_0",
            "--keep-type", f"{p}.eh_proj*=q4_0",
            "--keep-type", f"{p}.transformer_layer.attention.wo*=q4_0",
            "--keep-type", f"{p}.transformer_layer.attention.wqkv*=q4_0",
            "--keep-type", f"{p}.transformer_layer.feed_forward.*=q4_0",
        ]
    for i in range(40):
        p = f"model_weights/model.language_model.layers.{i}"
        args += [
            "--keep-type", f"{p}.self_attn.q_p*=q4_0",
            "--keep-type", f"{p}.self_attn.k_p*=q4_0",
            "--keep-type", f"{p}.self_attn.v_p*=q4_0",
            "--keep-type", f"{p}.self_attn.o_p*=q4_0",
            "--keep-type", f"{p}.linear_attn.in_proj_q*=q4_0",
            "--keep-type", f"{p}.linear_attn.in_proj_z*=q4_0",
            "--keep-type", f"{p}.linear_attn.out*=q4_0",
            "--keep-type", f"{p}.mlp.shared_expert.*=q4_0",
        ]
    return args


def compress_to_int4(
    gguf_bin: Path,
    input_gguf: Path,
    output_gguf: Path,
    spec_path: Path,
    sidecar_root: Path,
    engine_dir: Path,
) -> Path:
    """Chạy audiocpp_gguf đóng gói từ bản 8-bit/Q2_K sang drama-3-slim-int4-q2k.gguf."""
    tmp_out = output_gguf.with_suffix(".gguf.tmp")
    tmp_out.unlink(missing_ok=True)

    cmd = [
        str(gguf_bin),
        "--input", str(input_gguf),
        "--output", str(tmp_out),
        "--type", "orig",
        "--family", "fish_drama3",
        "--model-spec", str(spec_path),
        "--root", str(sidecar_root),
        "--overwrite",
        *build_int4_keep_types(),
    ]

    env = dict(os.environ)
    lib_dirs = collect_lib_dirs(gguf_bin.parent)
    if sys.platform.startswith("linux") and lib_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    elif os.name == "nt" and lib_dirs:
        env["PATH"] = os.pathsep.join(lib_dirs + [env.get("PATH", "")])

    print("[nen-int4] Đang đóng gói Drama-3 INT4 bằng audiocpp_gguf...", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(gguf_bin.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail_lines.append(line)
        tail_lines = tail_lines[-40:]
        s = line.rstrip()
        if s:
            print(f"  {s}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(
            f"audiocpp_gguf lỗi (rc={proc.returncode}):\n" + "".join(tail_lines)
        )

    if not tmp_out.exists() or tmp_out.stat().st_size < MIN_INT4_VALID_BYTES:
        raise RuntimeError(f"File nén INT4 đầu ra không hợp lệ: {tmp_out}")

    os.replace(tmp_out, output_gguf)
    elapsed = time.monotonic() - t0
    sz_gib = output_gguf.stat().st_size / (1024**3)
    print(f"[ok] Đã đóng gói xong INT4 sau {elapsed:.1f}s: {output_gguf} ({sz_gib:.2f} GiB)", flush=True)
    return output_gguf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", default=str(_default_engine_dir()))
    ap.add_argument("--models-dir", default=str(_default_models_dir()))
    ap.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    ap.add_argument("--orig-name", default=DEFAULT_ORIG_GGUF)
    ap.add_argument("--int4-name", default=DEFAULT_INT4_GGUF)
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine-only", action="store_true", help="Chỉ tải & kiểm tra audio.cpp engine")
    ap.add_argument("--remove-orig", action="store_true", help="Xoá file gốc 15.88 GiB sau khi nén INT4 xong")
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    bins = ensure_audio_cpp_engine(engine_dir, need_gguf=True)
    print(f"[engine] server={bins['server']} | gguf={bins.get('gguf')}", flush=True)

    if args.engine_only:
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    int4_path = models_dir / args.int4_name
    if int4_path.exists() and int4_path.stat().st_size >= MIN_INT4_VALID_BYTES:
        print(
            f"[ok] Bản INT4 đã sẵn sàng: {int4_path} ({int4_path.stat().st_size / (1024**3):.2f} GiB)",
            flush=True,
        )
        return

    orig_path = download_orig_gguf(
        models_dir=models_dir,
        repo_id=args.repo_id,
        filename=args.orig_name,
        hf_token=args.hf_token,
        workers=args.workers,
    )
    sidecar_root = models_dir / "_sidecar_root"
    if not (sidecar_root / "config.json").exists():
        extract_embedded_sidecars(orig_path, sidecar_root)

    compress_to_int4(
        gguf_bin=bins["gguf"],
        input_gguf=orig_path,
        output_gguf=int4_path,
        spec_path=bins["spec"],
        sidecar_root=sidecar_root,
        engine_dir=engine_dir,
    )

    if args.remove_orig and orig_path.exists():
        orig_path.unlink()
        shutil.rmtree(sidecar_root, ignore_errors=True)
        print(f"[don-dep] Đã xoá bản gốc {orig_path.name} để tiết kiệm ổ đĩa.", flush=True)


if __name__ == "__main__":
    main()

