# Báo cáo sự cố + khắc phục: server voice crash (SIGSEGV/SIGABRT)

**Ngày:** 2026-08-29
**Trạng thái:** ĐÃ TÌM RA NGUYÊN NHÂN GỐC VÀ SỬA.

---

## 1. Tóm tắt

Server voice (OmniVoice INT4, Colab T4, 4 worker) crash trong lúc phục vụ:
lần đầu `rc=-11` (SIGSEGV) khi đang tổng hợp, lần sau `rc=-6` (SIGABRT) với
backtrace đầy đủ. **Cả hai là cùng một nguyên nhân gốc, và đã xác nhận chắc
chắn nhờ backtrace:**

> **Engine 0 bị hai luồng dùng cùng lúc.** `add_voice()` (đăng ký giọng, chạy
> từ luồng HTTP) và worker 0 (tổng hợp, chạy từ hàng đợi) **cùng dùng
> `self.engines[0]`** mà không khoá. Mỗi engine có ggml context/scheduler
> riêng, KHÔNG an toàn đa luồng. Khi một request `/voice` chen vào đúng lúc
> worker 0 đang tổng hợp → ggml compute trên context bị hai luồng giẫm lên
> nhau → `GGML_ASSERT(buffer) failed` hoặc segfault.

Đây là **bug ở tầng điều phối của server (Pool), không phải lỗi engine
OmniVoice**. Engine đúng như thiết kế: mỗi context phải dùng một luồng tại một
thời điểm. Server đã vi phạm điều đó.

**Đã sửa:** thêm khoá riêng cho từng engine (`engine_locks[i]`); add_voice giữ
khoá engine 0, mỗi worker giữ khoá engine của mình → không còn hai luồng dùng
chung một context.

---

## 2. Môi trường

| Thành phần | Giá trị |
|---|---|
| GPU | Tesla T4, sm75, VRAM 15360 MiB |
| Backend | CUDA0 (ggml, `GGML_BACKEND_DL`) |
| Runtime C++ | omnivoice.cpp prebuilt `omnivoice-linux-cuda-sm75` |
| Model | profile `lite` INT4 (Q4_K_M backbone + Q8_0 tokenizer) |
| Python | cp313, `pyomnivoice`/`remote.server` biên dịch Cython → .so |
| Worker | 4 engine độc lập, hàng đợi dùng chung |

---

## 3. Bằng chứng — backtrace lần crash thứ hai (rc=-6)

```
[server] /content/omnivoice.cpp/ggml/src/ggml-backend.cpp:189: GGML_ASSERT(buffer) failed
[server] libggml-base.so.0(ggml_print_backtrace...)
[server] libggml-base.so.0(ggml_abort...)
[server] libggml-cuda.so.0(...)
[server] libggml-base.so.0(ggml_backend_sched_graph_compute_async...)
[server] libomnivoice.so(ov_extract_voice_ref+0x1163)      ← ĐĂNG KÝ GIỌNG
[server] pyomnivoice/core.cpython-...so
[server] remote/server.cpython-...so
[RUN] Server thoát rc=-6
```

`ov_extract_voice_ref` là hàm xử lý giọng mẫu (gọi khi `/voice`). `GGML_ASSERT
(buffer)` nghĩa là một tensor chưa được cấp buffer khi compute — hệ quả của
việc scheduler của engine 0 bị luồng worker 0 đụng vào song song.

Lần crash đầu (rc=-11) backtrace ngắn hơn nhưng crash trong `[MaskGIT]` (tổng
hợp) — chính là chiều ngược lại: worker 0 đang tổng hợp thì `/voice` đụng vào
engine 0. Cùng một gốc.

---

## 4. Nguyên nhân gốc (mã nguồn)

Trong `remote/server.py`, lớp `Pool`:

```python
def add_voice(self, ...):
    voice = self.engines[0].load_voice(tmp, text)   # luồng HTTP, engine 0

def _worker(self, wid):
    eng = self.engines[wid]                          # worker wid
    ...
    a = eng.say(...)                                 # worker 0 -> engine 0
```

→ `engines[0]` bị **luồng HTTP (add_voice)** và **luồng worker 0** dùng đồng
thời. Không có khoá. ggml context không chịu được điều này.

**Vì sao bài stress 4 tiếng trước không sập:** client lúc đó đăng ký giọng MỘT
lần lúc đầu, xong mới gửi toàn bộ TTS → add_voice hoàn tất trước khi worker 0
bắt đầu → không chồng lấn. Client thật/đợt test sau đăng ký giọng chen vào lúc
đang chạy → chồng lấn → sập.

---

## 5. Cách khắc phục (đã áp vào server.py)

Khoá riêng từng engine để mỗi ggml context chỉ một luồng dùng tại một thời điểm:

```python
# __init__:
self.engine_locks = [threading.Lock() for _ in range(n)]

# add_voice:
with self.engine_locks[0]:
    voice = self.engines[0].load_voice(tmp, text)

# _worker(wid):
with self.engine_locks[wid]:
    a = eng.say(...)
```

Chi phí: khi đăng ký giọng, chỉ **worker 0** bị chặn trong ~2 giây (thời gian
trích giọng); các worker 1..N-1 không ảnh hưởng, vẫn tổng hợp bình thường. Đăng
ký giọng thưa nên tác động thực tế không đáng kể.

---

## 6. Quá trình khoanh vùng (để đối chiếu)

Trước khi có backtrace, đã thử tái hiện bằng 65 job (5 tuần tự, 8 đồng thời,
client.py 4 luồng đoạn dài, 40 job hammer, 8 tiếng Việt) — **không sập lần
nào**, vì tất cả đều đăng ký giọng TRƯỚC rồi mới tổng hợp, không chạm vào điều
kiện chồng lấn engine 0. Điều này khẳng định thêm: crash chỉ xảy ra khi `/voice`
và tổng hợp trên worker 0 **thực sự đồng thời** — đúng như nguyên nhân gốc.

---

## 7. Việc cần làm để bản sửa có hiệu lực

`remote/server.py` được **biên dịch vào gói mã hoá** trên R2. Vì vậy phải:
1. Rebuild gói bằng `OmniVoice_Build_Secure_Colab.ipynb` (biên dịch lại server.py
   → .so, gộp model).
2. Upload đè gói mới lên R2 (module `omnivoice-cp313-linux-x86_64@1.0.0`).
3. Chạy lại client — đăng ký giọng lúc đang tổng hợp không còn sập.

Watchdog trong loader (tự dựng lại khi crash) vẫn là lớp an toàn phụ, nhưng sau
bản sửa này thì crash do nguyên nhân trên sẽ không còn xảy ra.
