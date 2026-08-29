"""Bootstrap bảo mật cho server voice — tải + giải mã + chạy qua license.

Đúc từ mẫu production h:/tmp/voice_secure_loader_sample.py (đã chạy thật, mirror
lvc_key_verify/scripts/test_e2e_full.py). Phần crypto/auth/kill-switch giữ
NGUYÊN vì đã kiểm chứng round-trip thật; chỉ thay phần CHẠY cho khớp gói
omnivoice: gói không phải một file thực thi mà là `python -m remote.server`,
kèm đường hầm cloudflared và vòng giữ sống.

File này sẽ được Cython biên dịch thành .so (chống sửa kill-switch), rồi notebook
khách công khai chỉ việc gọi run(). Không chứa bí mật: public key là công khai,
endpoint là công khai, license do khách tự nhập.

ĐỊNH DẠNG R2 (đã kiểm chứng): file .enc là CIPHERTEXT THUẦN, nonce lấy từ
encrypted_payload → AESGCM.decrypt(nonce, cả_file). KHÔNG cắt [12:].
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request

import httpx
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ============================ CẤU HÌNH ============================
VERIFY_BASE = "https://tool.lvcmedia.vn"
GO_BASE_FALLBACK = "https://check.lvcmedia.vn"

TOOL_CODE = "lvc-voice"
# Gói cp313 là gói HOÀN CHỈNH và phổ dụng: Cython .so + C++ CUDA runtime
# (GGML_BACKEND_DL tự dùng GPU khi có, tự lùi CPU khi không) + model INT4. Một
# gói chạy cả GPU lẫn CPU, nên chỉ cần một biến thể.
# LƯU Ý: module 'omnivoice-linux-cuda-sm75' trên R2 chỉ là bó .so runtime C++
# thô (không server.so, không model) — KHÔNG chọn, sẽ hỏng.
MODULE_VARIANTS = [
    {"module_name": "omnivoice-cp313-linux-x86_64", "version": "1.0.0", "need_gpu": False},
]

HKDF_INFO = b"license verification"     # phải KHỚP server (legacy_crypto.go)

# Public key EC P-256 của tool Voice — NHÚNG SẴN (công khai, an toàn).
TOOL_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEO0TFjsVKW4vOeKgDupoe1wDZOh44
F/BklL2ANX/DrUq1yzQRfFW0hQPI8yZRNd37Np1eho6GAWwxFaio6NP1SQ==
-----END PUBLIC KEY-----"""

HEARTBEAT_INTERVAL = 180
HTTP_TIMEOUT = 20
DOWNLOAD_TIMEOUT = 600          # gói có model, có thể ~700 MB
LOGIN_FAIL_STATUSES = (401, 403, 404)

# Tham số chạy server (khớp remote/server.py). Model nằm sẵn trong gói ở
# omnivoice.cpp/models nên KHÔNG cần --models-dir.
SERVER_MODULE = "remote.server"
SERVER_PORT = 8770
DEFAULT_WORKERS = 4
# ================================================================


class LicenseError(Exception):
    """Xác thực hỏng / thu hồi — DỪNG, không retry."""


class ModuleAccessError(Exception):
    """Lỗi tạm (mạng/parse/giải mã) — có thể retry ở heartbeat."""


class SecureVoiceLoader:
    def __init__(self, license_key: str, device_id: str, workers: int = DEFAULT_WORKERS):
        if not license_key or license_key.startswith("XXXX"):
            raise LicenseError("Chưa nhập license_key.")
        self.license_key = license_key
        self.device_id = device_id
        self.workers = workers
        self._tool_pub = serialization.load_pem_public_key(TOOL_PUBLIC_KEY_PEM.encode())
        self._stop = threading.Event()
        self._go_base = GO_BASE_FALLBACK
        self._server: subprocess.Popen | None = None
        self._tunnel: subprocess.Popen | None = None
        self._work_dir: str | None = None
        self._lock = threading.Lock()

    # ---------- PHẦN 1: ĐĂNG NHẬP ----------
    def login(self) -> list:
        try:
            r = httpx.post(
                f"{VERIFY_BASE}/api/v1/modules/list-access",
                json={"tool_code": TOOL_CODE, "license_key": self.license_key,
                      "device_id": self.device_id},
                timeout=HTTP_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise ModuleAccessError(f"Không gọi được list-access: {e}") from e

        if r.status_code in LOGIN_FAIL_STATUSES:
            raise LicenseError(f"License/tool không hợp lệ ({r.status_code}): {_detail(r)}")
        if r.status_code != 200:
            raise ModuleAccessError(f"list-access {r.status_code}: {r.text[:200]}")

        try:
            body = r.json()
            self._go_base = body.get("secure_code_server_url") or GO_BASE_FALLBACK
            modules = body.get("modules") or []
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise ModuleAccessError(f"list-access body không hợp lệ: {e}") from e

        if not modules:
            raise LicenseError("Không còn module khả dụng (có thể đã bị thu hồi).")
        return modules

    # ---------- PHẦN 2: TẢI + GIẢI MÃ (trong RAM) ----------
    def fetch_module(self, mod: dict) -> bytes:
        go_base = self._go_base
        try:
            access_ticket = mod["access_ticket"]
            eph_priv = ec.generate_private_key(ec.SECP256R1())
            eph_pub_b64 = base64.b64encode(eph_priv.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )).decode()

            r = httpx.post(
                f"{go_base}/api/v1/modules/access",
                json={"access_ticket": access_ticket, "ephemeral_pubkey_b64": eph_pub_b64},
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code in (401, 403):
                raise LicenseError(f"Ticket bị từ chối ({r.status_code}) — thu hồi/sai thiết bị.")
            if r.status_code != 200:
                raise ModuleAccessError(f"modules/access {r.status_code}: {r.text[:200]}")
            encrypted_payload = r.json()["encrypted_payload"]

            shared = eph_priv.exchange(ec.ECDH(), self._tool_pub)
            derived = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                           info=HKDF_INFO).derive(shared)
            payload = json.loads(
                Fernet(base64.urlsafe_b64encode(derived)).decrypt(encrypted_payload.encode()))

            master_key = bytes.fromhex(payload["master_key"])
            nonce = bytes.fromhex(payload["nonce"])
            cdn_url = payload["cdn_url"]
            file_hash = payload["file_hash"]
        except LicenseError:
            raise
        except (httpx.HTTPError, KeyError, ValueError, InvalidToken, json.JSONDecodeError) as e:
            raise ModuleAccessError(f"Lấy/giải encrypted_payload lỗi: {e}") from e

        try:
            r = httpx.get(cdn_url, timeout=DOWNLOAD_TIMEOUT)
        except httpx.HTTPError as e:
            raise ModuleAccessError(f"Tải R2 lỗi: {e}") from e
        if r.status_code != 200:
            raise ModuleAccessError(f"Tải R2 thất bại: {r.status_code}")
        ciphertext = r.content
        if hashlib.sha256(ciphertext).hexdigest() != file_hash:
            raise ModuleAccessError("file_hash KHÔNG khớp — file bị sửa/hỏng.")

        try:
            return AESGCM(master_key).decrypt(nonce, ciphertext, None)   # cả file là ciphertext
        except Exception as e:
            raise ModuleAccessError(f"AES-GCM decrypt lỗi: {e}") from e

    # ---------- PHẦN 2b: CHẠY SERVER (gói omnivoice) ----------
    def run_server(self, module_bytes: bytes) -> None:
        """Giải nén gói -> bật server (import trực tiếp) -> mở đường hầm -> giữ sống."""
        # KHÔNG dùng /dev/shm: trên nhiều môi trường (Colab) nó mount noexec nên
        # không nạp được .so ("failed to map segment from shared object"). Dùng
        # ổ thường; _cleanup vẫn xoá hẳn khi thu hồi/kết thúc.
        work_dir = tempfile.mkdtemp(prefix="voice_mod_")
        with self._lock:
            self._work_dir = work_dir

        with tarfile.open(fileobj=io.BytesIO(module_bytes), mode="r:gz") as tar:
            _safe_extract(tar, work_dir)
        app_dir = _find_app_dir(work_dir)
        print(f"[RUN] Giải nén -> {app_dir}: {os.listdir(app_dir)[:12]}", flush=True)

        api_key = base64.urlsafe_b64encode(os.urandom(9)).decode().rstrip("=")
        # libomnivoice.so cần libggml*.so (cùng thư mục) và, với backend CUDA,
        # cần libcudart.so.12/libcublas… của CUDA runtime. Linux không có
        # add_dll_directory nên gom hết vào LD_LIBRARY_PATH cho linker thấy:
        #   - bin của gói (libggml*)
        #   - các thư mục CUDA runtime phổ biến (nếu có)
        bin_dir = os.path.join(app_dir, "omnivoice.cpp", "build", "bin")
        ld_dirs = [bin_dir] + [d for d in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/lib/x86_64-linux-gnu",
        ) if os.path.isdir(d)]
        env = dict(os.environ, PYTHONPATH=app_dir, PYTHONUNBUFFERED="1",
                   LD_LIBRARY_PATH=os.pathsep.join(
                       ld_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
        # server.py là .so Cython — KHÔNG chạy được bằng `python -m` (runpy cần
        # code object mà extension module không phơi ra: "No code object
        # available"). Import trực tiếp rồi gọi main() thì .so chạy bình thường.
        launch = f"import {SERVER_MODULE} as s; s.main()"
        try:
            server = subprocess.Popen(
                [sys.executable, "-c", launch,
                 "--workers", str(self.workers), "--port", str(SERVER_PORT),
                 "--key", api_key],
                cwd=app_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=(os.name == "posix"),
            )
        except OSError as e:
            raise ModuleAccessError(f"Không bật được server: {e}") from e
        with self._lock:
            self._server = server
        # Bơm log server ra ngoài để thấy được tiến độ/lỗi (trước đây là hộp đen).
        def _pump():
            for line in server.stdout:
                print("[server] " + line.rstrip(), flush=True)
        threading.Thread(target=_pump, daemon=True).start()

        if not self._wait_health(server, timeout=300):
            self._kill_children()
            raise ModuleAccessError("Server không lên sau 300s — xem [server] log phía trên.")
        print(f"[RUN] Server sẵn sàng trên cổng {SERVER_PORT}", flush=True)

        public_url = self._open_tunnel()
        print("=" * 62, flush=True)
        print(f"  API URL : {public_url}", flush=True)
        print(f"  API key : {api_key}", flush=True)
        print(f"  Worker  : {self.workers}", flush=True)
        print("=" * 62, flush=True)

        # Giữ sống: chờ tới khi server chết hoặc kill-switch kích hoạt.
        while not self._stop.is_set():
            if server.poll() is not None:
                print(f"[RUN] Server thoát rc={server.returncode}", flush=True)
                break
            self._stop.wait(5)
        self._cleanup()

    def _wait_health(self, server: subprocess.Popen, timeout: int) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if server.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=2):
                    return True
            except OSError:
                time.sleep(2)
        return False

    def _open_tunnel(self) -> str:
        """Cài (nếu cần) và chạy cloudflared quick tunnel, trả URL công khai."""
        bin_path = _ensure_cloudflared()
        proc = subprocess.Popen(
            [bin_path, "tunnel", "--no-autoupdate", "--url",
             f"http://127.0.0.1:{SERVER_PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=(os.name == "posix"),
        )
        with self._lock:
            self._tunnel = proc
        deadline = time.time() + 40
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            if "trycloudflare.com" in line:
                for tok in line.split():
                    if tok.startswith("https://") and "trycloudflare.com" in tok:
                        return tok.strip()
        raise ModuleAccessError("Không lấy được URL đường hầm cloudflared.")

    # ---------- PHẦN 3: KILL-SWITCH ----------
    def _heartbeat_loop(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self.login()
                print("[KILL-SWITCH] OK — license còn hiệu lực.", flush=True)
            except LicenseError as e:
                print(f"[KILL-SWITCH] ĐÃ THU HỒI: {e} -> giết + wipe + thoát.", flush=True)
                self._revoke_shutdown()
            except Exception as e:  # fail-safe: lỗi tạm KHÔNG được giết heartbeat
                print(f"[KILL-SWITCH] lỗi tạm (bỏ qua): {type(e).__name__}: {e}", flush=True)

    def _revoke_shutdown(self):
        self._stop.set()
        self._kill_children()
        self._cleanup()
        os._exit(3)

    def _kill_children(self):
        with self._lock:
            procs = [p for p in (self._server, self._tunnel) if p]
        for child in procs:
            if child.poll() is not None:
                continue
            try:
                if os.name == "posix":
                    pgid = os.getpgid(child.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                else:
                    child.terminate()
            except (ProcessLookupError, OSError):
                pass

    def _cleanup(self):
        with self._lock:
            wd = self._work_dir
            self._work_dir = None
        if wd:
            shutil.rmtree(wd, ignore_errors=True)

    def start_kill_switch(self) -> threading.Thread:
        t = threading.Thread(target=self._heartbeat_loop, daemon=True)
        t.start()
        return t


# ============================ helpers ============================
def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text[:200]))
    except Exception:
        return r.text[:200]


def _pick_module(modules: list, name: str, version: str) -> dict | None:
    for m in modules:
        if m.get("module_name") == name and m.get("version") == version:
            return m
    same = [m for m in modules if m.get("module_name") == name]
    return same[0] if same else None


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def select_module(modules: list) -> dict:
    gpu = _has_nvidia_gpu()
    for v in MODULE_VARIANTS:
        if v["need_gpu"] and not gpu:
            continue
        m = _pick_module(modules, v["module_name"], v["version"])
        if m and "access_ticket" in m:
            return m
    # KHÔNG lùi về module lạ: các module ngoài MODULE_VARIANTS (vd bó runtime
    # thô sm75) không phải gói server chạy được, chọn nhầm sẽ hỏng khó hiểu.
    raise LicenseError(
        "License không cấp gói server hợp lệ "
        f"({[v['module_name'] for v in MODULE_VARIANTS]}).")


def _find_app_dir(work_dir: str) -> str:
    """Tar đóng với tiền tố app/, nên gói giải nén ra <work>/app. Nếu khác thì
    dò thư mục chứa remote/ (nơi có server)."""
    cand = os.path.join(work_dir, "app")
    if os.path.isdir(os.path.join(cand, "remote")):
        return cand
    for root, dirs, _ in os.walk(work_dir):
        if "remote" in dirs and os.path.isdir(os.path.join(root, "pyomnivoice")):
            return root
    return work_dir


def _safe_extract(tar: tarfile.TarFile, dest: str):
    dest_abs = os.path.abspath(dest)
    for m in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, m.name))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            raise ModuleAccessError(f"Tar chứa path nguy hiểm: {m.name}")
        if m.issym() or m.islnk():
            ln = os.path.abspath(os.path.join(dest, os.path.dirname(m.name), m.linkname))
            if not ln.startswith(dest_abs + os.sep):
                raise ModuleAccessError(f"Tar chứa link thoát thư mục: {m.name} -> {m.linkname}")
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def _ensure_cloudflared() -> str:
    """Trả đường dẫn cloudflared, tải bản Linux amd64 nếu chưa có."""
    found = shutil.which("cloudflared")
    if found:
        return found
    dest = "/tmp/cloudflared"
    if os.path.isfile(dest) and os.access(dest, os.X_OK):
        return dest
    url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
           "cloudflared-linux-amd64")
    urllib.request.urlretrieve(url, dest)
    os.chmod(dest, 0o755)
    return dest


def run(license_key: str, device_id: str, workers: int = DEFAULT_WORKERS) -> int:
    """Điểm vào cho notebook khách: đăng nhập -> tải -> giải mã -> chạy server."""
    print("=" * 62, flush=True)
    print(f"VOICE SECURE LOADER — {TOOL_CODE}", flush=True)
    print("=" * 62, flush=True)
    loader = SecureVoiceLoader(license_key, device_id, workers)

    try:
        modules = loader.login()
        mod = select_module(modules)
        print(f"[LOGIN] OK — {len(modules)} module; chọn {mod['module_name']}@{mod['version']} "
              f"(GPU={_has_nvidia_gpu()}, TTL={mod.get('ticket_expires_in_seconds')}s)", flush=True)
    except LicenseError as e:
        print(f"[LOGIN] HỎNG: {e}", flush=True)
        return 1
    except ModuleAccessError as e:
        print(f"[LOGIN] Lỗi hệ thống: {e}", flush=True)
        return 2

    loader.start_kill_switch()
    try:
        module_bytes = loader.fetch_module(mod)
        print(f"[FETCH] Giải mã xong: {len(module_bytes):,} bytes", flush=True)
        loader.run_server(module_bytes)
        return 0
    except LicenseError as e:
        print(f"[FETCH/RUN] Thu hồi: {e}", flush=True)
        return 1
    except ModuleAccessError as e:
        print(f"[FETCH/RUN] Lỗi: {e}", flush=True)
        return 2


def main():
    lic = os.environ.get("VOICE_LICENSE_KEY", "")
    dev = os.environ.get("VOICE_DEVICE_ID", "")
    workers = int(os.environ.get("VOICE_WORKERS", str(DEFAULT_WORKERS)))
    sys.exit(run(lic, dev, workers))


if __name__ == "__main__":
    main()
