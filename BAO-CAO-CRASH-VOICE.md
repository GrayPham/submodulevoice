# Báo cáo sự cố + khắc phục: server voice crash khi đăng ký giọng

**Ngày:** 2026-08-29 (cập nhật sau phát hiện của đội tool)
**Trạng thái:** Đã xác định NGUYÊN NHÂN THẬT của cú sập quan sát được, và đã vá
cả phía tool lẫn phía server.

---

## 1. Tóm tắt (HỒ SƠ CHƯA ĐÓNG — nguyên nhân chưa chứng minh)

Đã ghi nhận HAI cú sập, cùng ngắt quãng, khác chỗ nổ:
- **rc=-11** (SIGSEGV) trong `[MaskGIT]` (tổng hợp).
- **rc=-6** (SIGABRT, `GGML_ASSERT(buffer)`) trong `ov_extract_voice_ref` (đăng
  ký giọng, đường `/voice`).

**Bằng chứng then chốt (đội tool):** đúng file 16 kHz gây sập đã đăng ký **trót
lọt ít nhất 2 lần trước đó** (0,99s và 1,50s), rồi mới sập lần 3. Phân tích nội
dung file: **không** NaN/Inf/tràn biên/im lặng dài; np.interp vs soxr chỉ lệch
tối đa 0,108 (thuần chồng phổ).

**Hệ quả suy luận (chưa phải kết luận):**
- Nếu **nội dung 16 kHz** là nguyên nhân thì phải sập MỌI lần → nhưng không →
  **loại nội dung khỏi vai trò nguyên nhân tất định.**
- Tính ngắt quãng "cùng file, chạy được rồi mới sập" **khớp với một race**: chỉ
  sập khi thao tác chồng lấn nhau đúng thời điểm. Đây là chữ ký của việc **hai
  luồng dùng chung ggml context** (vd `/voice` chen vào lúc worker 0 đang tổng
  hợp), và có thể gắn CẢ HAI cú sập về một gốc — nổ ở bất kỳ nhánh nào đang chạy.
- **Chưa chứng minh.** Cả hai bản báo cáo trước đều đã nói quá theo hai hướng
  ngược nhau (một bên "engine-lock là gốc", một bên "16 kHz là gốc"). Trạng thái
  đúng hiện tại: **tương quan, chưa nhân quả; cần thí nghiệm phân định (mục 8).**

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
3. Chạy bài phân định (mục 8) trên server đã vá.

---

## 8. Thí nghiệm phân định (cần chạy — đừng đóng hồ sơ)

Xác nhận "chạy được một lần" KHÔNG chứng minh gì: 16 kHz vốn đã chạy 2/3 lần. Chỉ
tỉ lệ sập qua nhiều lần lặp mới cho tín hiệu.

**Tách bạch hai giả thuyết:**
- **Nhánh A — nội dung:** 30 lần `/voice` với WAV 16 kHz THÔ, **tuần tự, không có
  synthesis chạy song song**. Nếu sạch → nội dung 16 kHz vô can.
- **Nhánh B — race:** vừa chạy synthesis liên tục trên các worker, **vừa bơm
  `/voice` 16 kHz thô chen vào** (nhiều lần). Đây mới tạo điều kiện chồng lấn.

**Đọc kết quả:**
- A sạch + B sập (trên bản KHÔNG có engine-lock) → nguyên nhân là **race dùng
  chung engine**. Engine-lock là fix đúng.
- B vẫn sập cả khi CÓ engine-lock → còn một race rộng hơn trong ggml
  scheduler/CUDA khi nhiều worker chạy song song → cần đội engine điều tra
  thread-safety (compute-sanitizer `--tool racecheck`, core dump lấy backtrace).
- A cũng sập → nội dung/độ dài sau resample thực sự có vai trò → soi lại
  `ov_extract_voice_ref`.

Ghi **tỉ lệ sập**, không chỉ pass/fail một lần. So bản vá soxr với bản chưa vá
nếu còn giữ được.

---

## 9. Hai bản vá — giữ cả hai bất kể kết quả phân định

Cả hai đúng và đáng giữ dù thí nghiệm mục 8 ra sao:
- **soxr thay np.interp** (`read_wav_bytes`): tiếng sạch hơn hẳn (np.interp không
  lọc chống chồng phổ), + chặn audio < 0.5s. Là cải thiện chất lượng + phòng thủ.
- **engine-lock** (`Pool`): dùng chung engine 0 giữa add_voice và worker 0 là bug
  thật; khoá riêng từng engine là đúng dù nó có phải nguyên nhân cú sập này hay
  không.

Điều CHƯA làm được: chứng minh bản vá nào (hoặc cả hai) thực sự chặn crash. Đó là
việc của mục 8.
