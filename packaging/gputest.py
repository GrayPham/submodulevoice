"""OmniVoice GPU Stability Test — bản debug chạy GPU, KHÔNG fallback CPU.

Bấm vào START-TEST.exe là chạy. Công cụ này không nhằm đánh giá chất lượng
giọng, mà trả lời ba câu hỏi về phần cứng:

  1. VRAM có đủ cho toàn bộ 10.000 ký tự không?
  2. Tốc độ có tụt dần khi card nóng lên không?
  3. Có đoạn nào làm chương trình chết đột ngột không?

Không có đường lui về CPU. Nếu GPU không dùng được thì chương trình DỪNG và
ghi ra nguyên nhân cụ thể: thiếu driver, driver quá cũ, kiến trúc GPU không
được build, hay VRAM không đủ.

Kiến trúc hai tiến trình: tiến trình cha chỉ giám sát và ghi báo cáo, tiến
trình con làm việc thật. ggml gọi abort() khi CUDA hết bộ nhớ, tức là tiến
trình con có thể chết ngang mà không kịp báo gì — lúc đó tiến trình cha vẫn
còn sống để ghi lại nó chết ở đoạn nào và VRAM lúc ấy còn bao nhiêu. Đó chính
là dữ liệu cần cho một bài test độ ổn định.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Kiến trúc GPU đã được biên dịch vào ggml-cuda.dll (xem build-win.cmd).
# 61 có cả PTX nên mọi card >= 6.1 đều chạy được, card cũ hơn thì không.
BUILT_ARCHS = [61, 75, 86, 89]
MIN_CC = 6.1
# CUDA 12.1 yêu cầu driver Windows >= 527.41.
MIN_DRIVER = 527.41
# Đo được trên RTX 4000 Ada: 1727 MiB cho profile lite, 2199 MiB cho quality.
VRAM_NEED_MIB = {"tiny": 1500, "lite": 1750, "quality": 2250}

SEP = "=" * 68


def _setup_console() -> None:
    """Ép console sang UTF-8.

    Console Windows mặc định là cp1252 / cp437, không encode nổi tiếng Việt:
    exe sẽ chết ngay ở dòng print đầu tiên với UnicodeEncodeError. Đổi code
    page để hiển thị đúng, và reconfigure với errors="replace" để dù code page
    có đổi không được thì cũng không bao giờ crash vì một ký tự.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_setup_console()


def app_dir() -> Path:
    """Thư mục chứa exe (khi đóng gói) hoặc chứa script (khi chạy từ nguồn).

    OMNIVOICE_TEST_ROOT cho phép trỏ sang chỗ khác — dùng khi thử từ mã nguồn
    mà không muốn nhân bản 660 MB model.
    """
    env = os.environ.get("OMNIVOICE_TEST_ROOT")
    if env:
        return Path(env).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = app_dir()
OUT_DIR = ROOT / "ket-qua"


def _slug() -> str:
    """Tên máy + thời điểm. Nhiều nhân viên gửi báo cáo về thì không đè tên nhau."""
    host = re.sub(r"[^A-Za-z0-9_-]", "-", socket.gethostname() or "may")[:32]
    return f"{host}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


RUN_SLUG = _slug()


# --------------------------------------------------------------- chẩn đoán GPU


def _smi(query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip() or f"nvidia-smi rc={out.returncode}")
    return [[c.strip() for c in ln.split(",")] for ln in out.stdout.splitlines() if ln.strip()]


def collect_gpu_info() -> dict:
    """Trả về thông tin GPU, hoặc lý do vì sao không lấy được."""
    info: dict = {"ok": False, "gpus": [], "driver": None, "error": None, "error_kind": None}
    try:
        rows = _smi("name,driver_version,compute_cap,memory.total,memory.used,memory.free")
    except FileNotFoundError:
        info["error_kind"] = "NO_DRIVER"
        info["error"] = (
            "Không tìm thấy nvidia-smi. Máy chưa cài driver NVIDIA, hoặc không có GPU NVIDIA. "
            "Cài driver tại https://www.nvidia.com/Download/index.aspx rồi khởi động lại."
        )
        return info
    except Exception as e:
        info["error_kind"] = "SMI_FAILED"
        info["error"] = f"nvidia-smi chạy lỗi: {e}"
        return info

    if not rows:
        info["error_kind"] = "NO_GPU"
        info["error"] = "nvidia-smi chạy được nhưng không liệt kê GPU nào."
        return info

    for i, r in enumerate(rows):
        def num(x, cast=float, default=None):
            try:
                return cast(x)
            except Exception:
                return default

        info["gpus"].append({
            "index": i,
            "name": r[0] if len(r) > 0 else "?",
            "driver": r[1] if len(r) > 1 else "?",
            "compute_cap": num(r[2], float) if len(r) > 2 else None,
            "vram_total_mib": num(r[3], int) if len(r) > 3 else None,
            "vram_used_mib": num(r[4], int) if len(r) > 4 else None,
            "vram_free_mib": num(r[5], int) if len(r) > 5 else None,
        })
    info["driver"] = info["gpus"][0]["driver"]
    info["ok"] = True
    return info


def check_cuda_dll() -> dict:
    """Thử nạp ggml-cuda.dll trước khi chạy.

    Đây là lỗi hay gặp nhất khi đem exe sang máy khác: ggml-cuda.dll phụ thuộc
    động vào cudart64_12.dll và cublas64_12.dll, mà driver NVIDIA KHÔNG có
    hai file đó — chỉ CUDA Toolkit mới có. Thiếu chúng thì
    ggml_backend_load_all() lặng lẽ bỏ qua backend CUDA và ov_init chỉ báo
    chung chung "no GGML backend available". Nạp thử ở đây để chỉ đúng tên
    file còn thiếu thay vì để người dùng đoán.
    """
    rt = ROOT / "runtime"
    dll = rt / "ggml-cuda.dll"
    if not dll.exists():
        return {"level": "FAIL", "code": "NO_CUDA_DLL",
                "msg": f"Thiếu {dll}. Giải nén thiếu file, hoặc đang chạy từ trong file nén."}
    import ctypes

    prev = os.getcwd()
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(rt))
        os.chdir(rt)
        ctypes.WinDLL(str(dll))
        return {"level": "OK", "code": "CUDA_DLL", "msg": "ggml-cuda.dll nạp được."}
    except OSError as e:
        have = {f.name.lower() for f in rt.glob("*.dll")}
        need = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
        thieu = [n for n in need if n.lower() not in have]
        msg = f"Không nạp được ggml-cuda.dll: {e}. "
        if thieu:
            msg += ("Thư mục runtime đang thiếu: " + ", ".join(thieu)
                    + ". Bộ cài này thiếu file, báo lại người phát hành.")
        else:
            msg += ("Đủ file nhưng vẫn không nạp được — thường là do driver NVIDIA "
                    "không nạp được nvcuda.dll. Cài lại driver rồi thử lại.")
        return {"level": "FAIL", "code": "CUDA_DLL_LOAD_FAILED", "msg": msg}
    finally:
        os.chdir(prev)


def preflight(gpu: dict, profile: str) -> list[dict]:
    """Kiểm tra trước khi chạy. Mỗi mục: level FAIL / WARN / OK + thông điệp."""
    f: list[dict] = []
    if not gpu["ok"]:
        return [{"level": "FAIL", "code": gpu["error_kind"], "msg": gpu["error"]}]

    drv = gpu["driver"]
    try:
        drv_num = float(".".join(str(drv).split(".")[:2]))
    except Exception:
        drv_num = None
    if drv_num is None:
        f.append({"level": "WARN", "code": "DRIVER_UNKNOWN",
                  "msg": f"Không đọc được số hiệu driver ('{drv}'), bỏ qua kiểm tra phiên bản."})
    elif drv_num < MIN_DRIVER:
        f.append({"level": "FAIL", "code": "DRIVER_TOO_OLD",
                  "msg": (f"Driver {drv} quá cũ. Bản dựng này dùng CUDA 12.1, cần driver "
                          f">= {MIN_DRIVER}. Cập nhật driver NVIDIA rồi chạy lại.")})
    else:
        f.append({"level": "OK", "code": "DRIVER",
                  "msg": f"Driver {drv} đạt yêu cầu (cần >= {MIN_DRIVER})."})

    g = gpu["gpus"][0]
    cc = g["compute_cap"]
    if cc is None:
        f.append({"level": "WARN", "code": "CC_UNKNOWN",
                  "msg": "Driver không báo compute capability, bỏ qua kiểm tra kiến trúc."})
    elif cc < MIN_CC:
        f.append({"level": "FAIL", "code": "ARCH_UNSUPPORTED",
                  "msg": (f"{g['name']} có compute capability {cc}, dưới mức {MIN_CC}. "
                          f"ggml-cuda.dll chỉ chứa mã cho {BUILT_ARCHS} (kèm PTX 6.1), "
                          f"không có mã nào chạy được trên card này.")})
    else:
        exact = int(round(cc * 10)) in BUILT_ARCHS
        f.append({"level": "OK", "code": "ARCH",
                  "msg": (f"{g['name']} compute capability {cc} — "
                          + ("có mã dựng sẵn." if exact else "chạy qua PTX JIT, lần khởi động đầu chậm hơn."))})

    free = g["vram_free_mib"]
    need = VRAM_NEED_MIB.get(profile, 1750)
    if free is None:
        f.append({"level": "WARN", "code": "VRAM_UNKNOWN", "msg": "Không đọc được VRAM trống."})
    elif free < need * 0.75:
        f.append({"level": "FAIL", "code": "VRAM_TOO_LOW",
                  "msg": (f"VRAM trống {free} MiB, quá thấp so với mức cần khoảng {need} MiB "
                          f"cho profile '{profile}'. Đóng bớt ứng dụng dùng GPU (trình duyệt, "
                          f"game, phần mềm dựng phim) rồi chạy lại.")})
    elif free < need:
        f.append({"level": "WARN", "code": "VRAM_TIGHT",
                  "msg": (f"VRAM trống {free} MiB, dưới mức cần khoảng {need} MiB. Vẫn sẽ thử "
                          f"chạy — đây chính là trường hợp bài test muốn kiểm chứng.")})
    else:
        f.append({"level": "OK", "code": "VRAM",
                  "msg": f"VRAM trống {free}/{g['vram_total_mib']} MiB, cần khoảng {need} MiB."})

    f.append(check_cuda_dll())
    return f


# ------------------------------------------------------------------ giám sát


class VramSampler(threading.Thread):
    """Theo dõi VRAM của MỌI GPU.

    Driver Windows ở chế độ WDDM không báo VRAM theo từng tiến trình, nên cách
    duy nhất là lấy tổng đã dùng của cả card rồi trừ đi mức nền đo trước khi
    chạy. Theo dõi mọi GPU rồi chọn card có mức tăng lớn nhất, như vậy không
    cần đoán CUDA đã chọn card nào.
    """

    def __init__(self, baseline: list[int]) -> None:
        super().__init__(daemon=True)
        self.baseline = baseline
        self.peak = [b for b in baseline]
        self.min_free: list[int | None] = [None] * len(baseline)
        self.samples = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                rows = _smi("memory.used,memory.free")
                for i, r in enumerate(rows[: len(self.peak)]):
                    used, free = int(r[0]), int(r[1])
                    self.peak[i] = max(self.peak[i], used)
                    self.min_free[i] = free if self.min_free[i] is None else min(self.min_free[i], free)
                self.samples += 1
            except Exception:
                pass
            self.stop_flag.wait(0.3)

    def busiest(self) -> tuple[int, int, int, int | None]:
        """(index, VRAM tiêu thụ, đỉnh tuyệt đối, còn trống thấp nhất)."""
        if not self.peak:
            return 0, 0, 0, None
        deltas = [self.peak[i] - self.baseline[i] for i in range(len(self.peak))]
        i = max(range(len(deltas)), key=lambda k: deltas[k])
        return i, deltas[i], self.peak[i], self.min_free[i]


# -------------------------------------------------------------------- worker


def run_worker() -> int:
    """Tổng hợp thật. In ra stdout từng dòng có tiền tố @@ để tiến trình cha đọc."""
    os.environ["OMNIVOICE_LIB"] = str(ROOT / "runtime")
    sys.path.insert(0, str(ROOT))
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    def emit(kind: str, **kw) -> None:
        print("@@" + kind + " " + json.dumps(kw, ensure_ascii=False), flush=True)

    try:
        import numpy as np
        from pyomnivoice import Audio, OmniVoice, SAMPLE_RATE
    except Exception as e:
        emit("FATAL", stage="import", error=f"{type(e).__name__}: {e}")
        return 3

    profile = os.environ.get("GPUTEST_PROFILE", "lite")
    steps = int(os.environ.get("GPUTEST_STEPS", "16"))
    data = ROOT / "data"
    paras = [re.sub(r"\s+", " ", p.strip())
             for p in re.split(r"\n\s*\n", (data / "script.txt").read_text(encoding="utf-8"))
             if p.strip()]

    try:
        t0 = time.perf_counter()
        tts = OmniVoice(profile=profile, backend="cuda", models_dir=ROOT / "models")
        load_s = time.perf_counter() - t0
    except Exception as e:
        emit("FATAL", stage="init", error=f"{type(e).__name__}: {e}",
             hint=("ov_init thất bại với GGML_BACKEND=CUDA0. Bản debug này không cho phép "
                   "rơi về CPU. Nếu VRAM tiêu thụ là 0 MiB và dừng trong vài giây thì "
                   "chưa hề chạm tới GPU — nguyên nhân gần như luôn là ggml-cuda.dll "
                   "không nạp được (thiếu cudart64_12.dll / cublas64_12.dll / "
                   "cublasLt64_12.dll trong thư mục runtime), chứ không phải hết VRAM."))
        return 4

    # Chốt chặn: nếu vì lý do gì đó ggml vẫn chọn CPU thì coi như hỏng.
    if "CUDA" not in (tts.backend or "").upper():
        emit("FATAL", stage="backend", error=f"backend thực tế là '{tts.backend}', không phải CUDA",
             hint="Bản debug GPU không chấp nhận chạy bằng CPU.")
        return 5

    emit("READY", backend=tts.backend, load_s=round(load_s, 2),
         model=tts.model_path.name, codec=tts.codec_path.name,
         paras=len(paras), chars=sum(len(p) for p in paras))

    try:
        ref_txt = (data / "ref.txt").read_text(encoding="utf-8").strip()
        tv = time.perf_counter()
        voice = tts.load_voice(data / "ref.wav", ref_txt)
        emit("VOICE", frames=voice.n_frames, encode_s=round(time.perf_counter() - tv, 2),
             text=ref_txt[:120])
    except Exception as e:
        emit("FATAL", stage="voice", error=f"{type(e).__name__}: {e}")
        return 6

    pieces, gap = [], np.zeros(int(0.45 * SAMPLE_RATE), dtype=np.float32)
    synth_t0 = time.perf_counter()
    for i, para in enumerate(paras, 1):
        emit("BEGIN", i=i, n=len(paras), chars=len(para))
        try:
            a = tts.say(para, voice=voice, lang="Vietnamese", steps=steps, seed=42)
        except Exception as e:
            emit("FATAL", stage="synth", para=i, error=f"{type(e).__name__}: {e}",
                 tb=traceback.format_exc(limit=3))
            return 7
        if pieces:
            pieces.append(gap)
        pieces.append(a.samples)
        emit("DONE", i=i, wall=round(tts.last_wall, 2), audio=round(a.duration, 2),
             xrt=round(a.duration / tts.last_wall, 2))

    final = Audio(np.concatenate(pieces), SAMPLE_RATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav = OUT_DIR / "ket-qua.wav"
    final.save(wav)
    emit("FINISH", audio=round(final.duration, 2), wav=str(wav),
         synth_s=round(time.perf_counter() - synth_t0, 2),
         peak=round(float(np.abs(final.samples).max()), 3))
    tts.close()
    return 0


# ------------------------------------------------------------------ giám sát


def supervise() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = os.environ.get("GPUTEST_PROFILE", "lite")
    started = datetime.now(timezone.utc).astimezone()

    print(SEP)
    print("  OmniVoice — KIỂM TRA ĐỘ ỔN ĐỊNH GPU")
    print("  Bản debug: chỉ chạy CUDA, không có đường lui về CPU.")
    print(SEP)

    gpu = collect_gpu_info()
    print("\n[1/4] Phần cứng\n")
    print(f"  Hệ điều hành : {platform.platform()}")
    print(f"  CPU          : {platform.processor() or '?'}")
    if gpu["ok"]:
        for g in gpu["gpus"]:
            print(f"  GPU {g['index']}        : {g['name']}")
            print(f"                 driver {g['driver']}  |  compute {g['compute_cap']}  |  "
                  f"VRAM {g['vram_free_mib']}/{g['vram_total_mib']} MiB trống")
    else:
        print(f"  GPU          : KHÔNG PHÁT HIỆN ĐƯỢC")

    print("\n[2/4] Kiểm tra điều kiện\n")
    findings = preflight(gpu, profile)
    for f in findings:
        mark = {"OK": "  [ OK ]", "WARN": "  [CẢNH]", "FAIL": "  [LỖI ]"}[f["level"]]
        print(f"{mark} {f['msg']}")

    report: dict = {
        "started": started.isoformat(),
        "app_version": "gputest-1",
        "profile": profile,
        "machine": socket.gethostname(),
        "run_slug": RUN_SLUG,
        "system": {"platform": platform.platform(), "processor": platform.processor()},
        "gpu": gpu,
        "preflight": findings,
        "paragraphs": [],
        "result": None,
    }

    fails = [f for f in findings if f["level"] == "FAIL"]
    if fails:
        report["result"] = {"status": "PREFLIGHT_FAILED",
                            "codes": [f["code"] for f in fails]}
        write_report(report)
        print("\n" + SEP)
        print("  DỪNG — không đạt điều kiện chạy GPU. Xem lý do ở trên.")
        print(SEP)
        return 1

    print("\n[3/4] Tổng hợp\n")
    env = dict(os.environ, GPUTEST_WORKER="1", PYTHONIOENCODING="utf-8", GPUTEST_PROFILE=profile)
    cmd = [sys.executable] + ([] if getattr(sys, "frozen", False) else [__file__]) + ["--worker"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            env=env, cwd=str(ROOT), bufsize=1)

    baseline = [g["vram_used_mib"] or 0 for g in gpu["gpus"]] if gpu["ok"] else [0]
    report["vram_baseline_mib"] = baseline
    sampler = VramSampler(baseline)
    sampler.start()

    last_para, fatal, log_lines = 0, None, []
    t_start = time.perf_counter()
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        log_lines.append(line)
        if not line.startswith("@@"):
            continue
        kind, _, payload = line[2:].partition(" ")
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if kind == "READY":
            report["model_load_s"] = d["load_s"]
            report["model"] = d["model"]
            report["codec"] = d["codec"]
            report["chars"] = d["chars"]
            print(f"  backend {d['backend']}  |  {d['model']} + {d['codec']}")
            print(f"  nạp model {d['load_s']}s  |  {d['paras']} đoạn, {d['chars']} ký tự")
        elif kind == "VOICE":
            report["voice_frames"] = d["frames"]
            report["voice_encode_s"] = d.get("encode_s")
            print(f"  mã hoá giọng mẫu {d.get('encode_s')}s ({d['frames']} frame)\n")
            print(f"  {'#':>4s} {'giây':>9s} {'audio':>9s} {'xRT':>8s}")
            print("  " + "-" * 34)
        elif kind == "BEGIN":
            last_para = d["i"]
        elif kind == "DONE":
            report["paragraphs"].append(d)
            print(f"  {d['i']:>4d} {d['wall']:>8.2f}s {d['audio']:>8.2f}s {d['xrt']:>7.2f}x")
        elif kind == "FINISH":
            report["audio_sec"] = d["audio"]
            report["synth_sec"] = d.get("synth_s")
            report["wav"] = d["wav"]
        elif kind == "FATAL":
            fatal = d
    proc.wait()
    wall = time.perf_counter() - t_start
    sampler.stop_flag.set()
    sampler.join(timeout=2)

    idx, vram_used, vram_peak, min_free = sampler.busiest()
    report["wall_sec"] = round(wall, 1)
    report["vram_gpu_index"] = idx
    report["vram_used_by_test_mib"] = vram_used
    report["vram_peak_total_mib"] = vram_peak
    report["vram_min_free_mib"] = min_free
    report["vram_samples"] = sampler.samples
    report["worker_exit_code"] = proc.returncode
    report["log"] = log_lines[-400:]

    print("\n[4/4] Kết quả\n")
    if proc.returncode == 0 and not fatal:
        rows = report["paragraphs"]
        audio = report.get("audio_sec", 0.0)
        xrts = [r["xrt"] for r in rows]
        first, last = xrts[: max(1, len(xrts) // 4)], xrts[-max(1, len(xrts) // 4):]
        drift = (sum(last) / len(last)) / (sum(first) / len(first)) if first and last else 1.0
        report["result"] = {
            "status": "PASS",
            "xrt_min": min(xrts), "xrt_max": max(xrts),
            "xrt_first_quarter": round(sum(first) / len(first), 3),
            "xrt_last_quarter": round(sum(last) / len(last), 3),
            "drift_ratio": round(drift, 3),
        }
        report["result"]["xrt_overall"] = round(audio / max(report.get("synth_sec") or wall, 1e-9), 3)
        print(f"  ĐẠT — chạy hết {len(rows)} đoạn, không có đoạn nào lỗi\n")
        print(f"  THỜI GIAN")
        print(f"    nạp model      {report.get('model_load_s')}s")
        print(f"    mã hoá giọng   {report.get('voice_encode_s')}s")
        print(f"    tổng hợp       {report.get('synth_sec')}s")
        print(f"    tổng cộng      {wall:.1f}s  ({wall / 60:.1f} phút)")
        print(f"    audio sinh ra  {audio:.1f}s  ({audio / 60:.1f} phút)")
        print(f"    tốc độ         {report['result']['xrt_overall']:.2f}x realtime "
              f"(từng đoạn {min(xrts):.2f}x – {max(xrts):.2f}x)\n")
        print(f"  VRAM  (GPU {report['vram_gpu_index']}, mẫu 0.3s một lần, {sampler.samples} mẫu)")
        print(f"    mức nền trước khi chạy   {report['vram_baseline_mib'][report['vram_gpu_index']]} MiB")
        print(f"    đỉnh tổng trên card      {vram_peak} MiB")
        print(f"    >> bài test tiêu thụ     {vram_used} MiB")
        print(f"    còn trống thấp nhất      {min_free} MiB\n")
        if drift < 0.85:
            print(f"  ⚠ tốc độ TỤT {(1 - drift) * 100:.0f}% từ đầu đến cuối — card có thể đang "
                  f"bị hạ xung vì nóng.")
        else:
            print(f"  tốc độ ổn định từ đầu đến cuối (lệch {abs(1 - drift) * 100:.0f}%)")
        print(f"  audio: {report.get('wav', '?')}\n")
        for ln in projection_lines(report):
            print("  " + ln)
    else:
        kind = "CRASH" if not fatal else "ERROR"
        report["result"] = {
            "status": kind,
            "exit_code": proc.returncode,
            "last_paragraph": last_para,
            "fatal": fatal,
            "vram_min_free_mib": min_free,
            "vram_used_by_test_mib": vram_used,
            "wall_sec": round(wall, 1),
        }
        print(f"  KHÔNG ĐẠT — dừng ở đoạn {last_para}/33, mã thoát {proc.returncode}")
        if fatal:
            print(f"  giai đoạn : {fatal.get('stage')}")
            print(f"  lỗi       : {fatal.get('error')}")
            if fatal.get("hint"):
                print(f"  giải thích: {fatal['hint']}")
        else:
            print("  Tiến trình con chết ngang, không kịp báo lỗi. Đây gần như luôn là "
                  "CUDA out of memory: ggml gọi abort() thay vì trả lỗi.")
        print(f"  VRAM: tiêu thụ {vram_used} MiB, còn trống thấp nhất {min_free} MiB")
        print(f"  Thời gian đã chạy trước khi dừng: {wall:.1f}s")
        print("  Thử: đóng trình duyệt và các ứng dụng dùng GPU, hoặc chạy lại bằng "
              "CHAY-PROFILE-TINY.cmd để dùng model nhỏ hơn.")

    write_report(report)
    print(f"\n  Báo cáo: {OUT_DIR / f'bao-cao-{RUN_SLUG}.txt'}")
    print(f"           {OUT_DIR / f'bao-cao-{RUN_SLUG}.json'}")
    print(f"\n  >> GỬI LẠI file .json ở trên. Tên file đã có sẵn tên máy nên"
          f" nhiều máy gửi về cùng lúc cũng không trùng nhau.")
    print(SEP)
    return 0 if report["result"]["status"] == "PASS" else 1


def _hms(sec: float) -> str:
    """Giây -> '12 phút 30' cho dễ đọc."""
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} giây"
    m, ss = divmod(sec, 60)
    if m < 60:
        return f"{m} phút {ss:02d}"
    h, m = divmod(m, 60)
    return f"{h} giờ {m:02d} phút"


def projection(report: dict) -> dict | None:
    """Suy ra năng suất của chính máy này từ số liệu vừa đo.

    Hai đại lượng độc lập nhau:
      - mật độ chữ: bao nhiêu ký tự cho ra một giây audio. Phụ thuộc GIỌNG MẪU
        (bộ ước lượng thời lượng lấy tốc độ nói từ reference), không phụ thuộc
        phần cứng.
      - tốc độ realtime: phần cứng chạy nhanh gấp mấy lần thời gian thực.
    Nhân chia hai cái đó ra được thời gian cần cho bất kỳ độ dài audio nào.
    """
    audio = report.get("audio_sec")
    synth = report.get("synth_sec")
    chars = report.get("chars")
    if not audio or not synth or not chars:
        return None
    return {
        "chars_per_audio_sec": round(chars / audio, 2),
        "xrt": round(audio / synth, 3),
        "load_s": report.get("model_load_s") or 0.0,
    }


def projection_lines(report: dict) -> list[str]:
    pr = projection(report)
    if not pr:
        return []
    L = ["-- Ước tính năng suất máy này " + "-" * 36,
         f"Tốc độ đo được : {pr['xrt']:.2f}x realtime",
         f"Mật độ chữ     : {pr['chars_per_audio_sec']:.1f} ký tự cho mỗi giây audio",
         "                 (con số này theo GIỌNG MẪU, đổi giọng là đổi)",
         "",
         f"{'audio cần':>12s} {'ký tự cần':>12s} {'máy này chạy hết':>20s}"]
    for label, sec in [("1 phút", 60), ("5 phút", 300), ("10 phút", 600),
                       ("30 phút", 1800), ("1 giờ", 3600), ("2 giờ", 7200)]:
        need_chars = int(round(sec * pr["chars_per_audio_sec"]))
        wall = sec / pr["xrt"] + pr["load_s"]
        L.append(f"{label:>12s} {need_chars:>12,d} {_hms(wall):>20s}")
    L.append("")
    return L


def write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report["projection"] = projection(report)
    (OUT_DIR / f"bao-cao-{RUN_SLUG}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [SEP, "OmniVoice — BÁO CÁO KIỂM TRA GPU", SEP, ""]
    L.append(f"Máy       : {report.get('machine', '?')}")
    L.append(f"Thời điểm : {report['started']}")
    L.append(f"Hệ điều hành: {report['system']['platform']}")
    L.append(f"Profile   : {report['profile']}  ({report.get('model', '?')} + {report.get('codec', '?')})")
    L.append(f"Ký tự     : {report.get('chars', '?')}")
    g = report["gpu"]
    if g["ok"]:
        for x in g["gpus"]:
            L.append(f"GPU {x['index']}     : {x['name']}  driver {x['driver']}  "
                     f"compute {x['compute_cap']}  VRAM {x['vram_total_mib']} MiB")
    else:
        L.append(f"GPU       : KHÔNG PHÁT HIỆN — {g['error']}")
    L += ["", "-- Kiểm tra điều kiện " + "-" * 44]
    for f in report["preflight"]:
        L.append(f"[{f['level']:4s}] {f['code']}: {f['msg']}")

    r = report.get("result") or {}
    L += ["", "-- Kết quả " + "-" * 55, f"Trạng thái: {r.get('status')}"]
    if r.get("status") == "PASS":
        L += [f"Tốc độ    : {r['xrt_min']}x – {r['xrt_max']}x realtime",
              f"Đầu / cuối: {r['xrt_first_quarter']}x -> {r['xrt_last_quarter']}x "
              f"(tỉ lệ {r['drift_ratio']})",
              f"Audio     : {report.get('audio_sec')}s trong {report.get('wall_sec')}s"]
    else:
        L += [f"Mã thoát  : {r.get('exit_code')}", f"Đoạn cuối : {r.get('last_paragraph')}"]
        if r.get("fatal"):
            L += [f"Giai đoạn : {r['fatal'].get('stage')}", f"Lỗi       : {r['fatal'].get('error')}"]
            if r["fatal"].get("hint"):
                L.append(f"Giải thích: {r['fatal']['hint']}")
    L += ["", "-- Thời gian " + "-" * 53,
          f"Nạp model     : {report.get('model_load_s')} s",
          f"Mã hoá giọng  : {report.get('voice_encode_s')} s",
          f"Tổng hợp      : {report.get('synth_sec')} s",
          f"Tổng cộng     : {report.get('wall_sec')} s",
          f"Audio sinh ra : {report.get('audio_sec')} s",
          "",
          "-- VRAM " + "-" * 58,
          f"GPU dùng      : index {report.get('vram_gpu_index')}",
          f"Mức nền       : {(report.get('vram_baseline_mib') or [0])[report.get('vram_gpu_index', 0)]} MiB",
          f"Đỉnh trên card: {report.get('vram_peak_total_mib')} MiB",
          f"BÀI TEST DÙNG : {report.get('vram_used_by_test_mib')} MiB",
          f"Trống thấp nhất: {report.get('vram_min_free_mib')} MiB",
          f"Số mẫu đo     : {report.get('vram_samples')}", ""]

    L += projection_lines(report)

    if report["paragraphs"]:
        L += ["-- Từng đoạn " + "-" * 53, f"{'#':>3s} {'giây':>8s} {'audio':>8s} {'xRT':>7s}"]
        for d in report["paragraphs"]:
            L.append(f"{d['i']:>3d} {d['wall']:>7.2f}s {d['audio']:>7.2f}s {d['xrt']:>6.2f}x")
        L.append("")

    L += ["-- Nhật ký thô " + "-" * 51] + report.get("log", [])
    (OUT_DIR / f"bao-cao-{RUN_SLUG}.txt").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    if "--worker" in sys.argv or os.environ.get("GPUTEST_WORKER") == "1":
        return run_worker()
    try:
        rc = supervise()
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "loi-giam-sat.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nLỗi trong tiến trình giám sát:")
        traceback.print_exc()
        rc = 2
    if sys.stdout.isatty() or getattr(sys, "frozen", False):
        try:
            input("\nNhấn Enter để đóng cửa sổ...")
        except EOFError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
