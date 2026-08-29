# Báo cáo sự cố + khắc phục: server voice crash khi đăng ký giọng

**Ngày:** 2026-08-29 (cập nhật sau phát hiện của đội tool)
**Trạng thái:** Đã xác định NGUYÊN NHÂN THẬT của cú sập quan sát được, và đã vá
cả phía tool lẫn phía server.

---

## 1. Tóm tắt (đã đính chính)

Server voice (OmniVoice INT4, Colab T4, 4 worker) crash `rc=-6` (SIGABRT do
`ggml_abort`) với backtrace trỏ vào `ov_extract_voice_ref` — hàm trích giọng mẫu
(đường `/voice`), **ngay ở job đầu tiên của một server vừa khởi động**.

**Nguyên nhân thật (đội tool phát hiện, đã đối chiếu và đồng ý):**
> Client gửi **WAV 16 kHz**. Trong `read_wav_bytes` của server, nhánh
> `sr != 24000` dùng **`np.interp`** nội suy tuyến tính lên 24 kHz. Audio đó qua
> `load_voice` → `ov_extract_voice_ref` → **`GGML_ASSERT(buffer) failed`**. Đây
> là nhánh duy nhất mà bộ repro 65 job không chạm tới, vì giọng mẫu dùng để test
> đã sẵn 24 kHz nên không kích hoạt resample.

**Đính chính một chẩn đoán sai trong bản báo cáo trước:** bản trước quy cú sập
này cho việc "engine 0 bị hai luồng dùng chung". Điều đó **không đúng cho cú sập
này** — crash xảy ra ở job ĐẦU của server vừa lên, lúc chưa có synthesis nào
chạy để giẫm lên engine 0. Việc dùng chung engine 0 vẫn là **một bug tiềm ẩn
khác** (đã vá riêng, xem mục 5), nhưng không phải nguyên nhân của cú sập quan sát
được. Sai sót do bản trước suy ra tính đồng thời từ backtrace mà không có bằng
chứng; chứng cứ "job đầu của server mới" của đội tool đã bác bỏ nó.

---

## 2. Chi tiết kỹ thuật đáng chú ý

So `np.interp` với `soxr` trên cùng input 16 kHz → 24 kHz:

| | số mẫu ra | dtype | contiguous |
|---|---|---|---|
| np.interp | 28800 | float32 | có |
| soxr | 28800 | float32 | có |

→ **Kích thước/kiểu/bố cục GIỐNG HỆT nhau.** Khác biệt duy nhất là **nội dung**:
`np.interp` là nội suy tuyến tính không có bộ lọc chống chồng phổ (aliasing),
`soxr` thì có. Vì shape giống nhau mà một cái sập một cái không, **crash không
phải lỗi shape** — nó liên quan tới nội dung audio (có thể là artefact aliasing
đưa feature vào một trạng thái suy biến) hoặc một điểm yếu trong
`ov_extract_voice_ref`. Đây là manh mối quan trọng cho đội engine.

---

## 3. Backtrace (bằng chứng)

```
/content/omnivoice.cpp/ggml/src/ggml-backend.cpp:189: GGML_ASSERT(buffer) failed
libggml-base.so.0(ggml_abort...)
libggml-cuda.so.0(...)
libggml-base.so.0(ggml_backend_sched_graph_compute_async...)
libomnivoice.so(ov_extract_voice_ref+0x1163)     ← trích giọng mẫu (/voice)
pyomnivoice/core.cpython-...so
remote/server.cpython-...so
[RUN] Server thoát rc=-6                          ← SIGABRT
```

---

## 4. Khắc phục phía server (đã áp, `read_wav_bytes`)

Thay `np.interp` bằng `soxr` cho mọi tần số khác 24 kHz + chặn audio quá ngắn.
Không thể trông chờ mọi client tự resample — 16 kHz cực phổ biến (điện thoại, ghi
âm cũ), client nào gửi cũng hạ cả server. **Server phải tự bền.**

```python
if sr != SAMPLE_RATE:
    import soxr
    x = soxr.resample(np.ascontiguousarray(x, dtype=np.float32), sr, SAMPLE_RATE)
x = np.ascontiguousarray(x, dtype=np.float32)
if len(x) < SAMPLE_RATE // 2:          # < 0.5s
    raise ValueError("giọng mẫu quá ngắn, cần >= 0.5s")
```

Đã test tại chỗ: 8k/16k/24k/44.1k đều ra 24 kHz float32 sạch; audio < 0.5s trả
lỗi rõ ràng (handler bọc try/except → HTTP 500, KHÔNG abort tiến trình). Đội tool
cũng đã resample 24 kHz phía client (42/42 pass) — hai lớp bảo vệ, phía server là
lớp nền không phụ thuộc client.

---

## 5. Bug tiềm ẩn thứ hai (đã vá riêng, KHÔNG phải cú sập này)

Trong `Pool`, `add_voice()` chạy `engines[0]` từ luồng HTTP, còn worker 0 cũng
chạy `engines[0]` từ hàng đợi → nếu `/voice` chen vào lúc worker 0 đang tổng hợp
thì hai luồng giẫm lên cùng ggml context → sẽ sập. Bộ repro không kích hoạt vì
luôn đăng ký giọng trước, tổng hợp sau. Đã vá bằng **khoá riêng từng engine**
(`engine_locks[i]`). Đây là phòng ngừa chủ động, độc lập với cú sập 16 kHz.

---

## 6. Đề xuất cho đội engine (vá tận gốc)

Cú sập là do **abort cả tiến trình** khi gặp input hợp lệ về mặt định dạng nhưng
làm buffer null. Đề nghị:
1. `ov_extract_voice_ref` **kiểm tra buffer/độ dài trước khi assert**, trả mã
   lỗi thay vì `ggml_abort` — một client gửi audio lạ không nên hạ cả server.
2. Hoặc `read_wav_bytes` phía engine/SDK **từ chối thẳng tần số != 24 kHz** với
   lỗi rõ ràng, thay vì nội suy rồi crash ngầm.
3. Tìm hiểu vì sao audio cùng shape nhưng nội dung có aliasing (np.interp) lại
   làm null buffer, còn audio sạch (soxr) thì không — có thể lộ một nhánh suy
   biến trong trích đặc trưng.

---

## 7. Việc cần làm để bản sửa server có hiệu lực

`remote/server.py` nằm trong gói mã hoá trên R2, nên phải:
1. Rebuild gói bằng `OmniVoice_Build_Secure_Colab.ipynb`.
2. Upload đè lên R2 (`omnivoice-cp313-linux-x86_64@1.0.0`).
3. Chạy lại `/voice` với WAV 16 kHz — không còn sập, resample sạch.
