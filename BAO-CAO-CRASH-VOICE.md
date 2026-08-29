# Báo cáo sự cố: server voice crash rc=-11 (SIGSEGV)

**Ngày:** 2026-08-29
**Người test:** tích hợp loader bảo mật (Colab client/server INT4)
**Mục đích báo cáo:** bàn giao cho đội engine OmniVoice để điều tra nguyên nhân segfault.

---

## 1. Tóm tắt

Server voice (OmniVoice INT4, chạy trên Colab T4) **crash với mã thoát `rc=-11`
(SIGSEGV / segmentation fault)** trong lúc đang tổng hợp, sau khi đã phục vụ vài
job thành công. Sự cố xảy ra **một lần**, trong một phiên có nhiều worker chạy
song song.

Sau đó đã thử tái hiện bằng **65 job** trên nhiều kiểu tải và 2 ngôn ngữ —
**không tái hiện được**. Kết luận sơ bộ: đây là **crash không thường xuyên
(intermittent)**, nhiều khả năng là **race condition hiếm trong nhân CUDA của
engine tổng hợp** khi nhiều worker chạy đồng thời, chứ không phải lỗi tất định
theo input hay theo số worker.

---

## 2. Môi trường

| Thành phần | Giá trị |
|---|---|
| GPU | Tesla T4, compute capability 7.5 (sm75), VRAM 15360 MiB |
| Backend | CUDA0 (ggml, `GGML_BACKEND_DL`) |
| Runtime C++ | omnivoice.cpp, bản prebuilt `omnivoice-linux-cuda-sm75` |
| Model | profile `lite` (INT4): backbone `omnivoice-base-Q4_K_M.gguf` + tokenizer `omnivoice-tokenizer-Q8_0.gguf` |
| Python | cp313 (Colab), lớp `pyomnivoice`/`remote.server` **biên dịch Cython → .so** |
| Số worker | 4 engine độc lập (mỗi worker một `OmniVoice`), hàng đợi dùng chung |
| Cách bật | `python -c "import remote.server as s; s.main()" --workers 4` |

Ghi chú: lớp Python được **biên dịch Cython** (khác bản `.py` thuần dùng trong
các test ổn định trước đây) — xem mục 5, đây là biến số cần loại trừ.

---

## 3. Sự cố quan sát được (log gốc)

Server boot hoàn toàn bình thường (4 engine, 3.1s, VRAM trống 14913 MiB), mở
đường hầm, phục vụ được. Sau đó:

```
[server] [http] "GET / HTTP/1.1" 404 -
[server] [http] "GET /favicon.ico HTTP/1.1" 404 -
[server] giọng '_clip_810e0ca18f5c' -> 49a1f213f393 (151 frame)
[KILL-SWITCH] OK — license còn hiệu lực.
[server] [Prompt] Built: B'=2 K=8 S=421 N1=7 N2=77 Sref=151 Stgt=186 c_len=421 u_len=186 denoise=1
[server] [MaskGIT] Start: T=186 K=8 S=421 num_step=8 guidance=2.00 t_shift=0.100 layer_pen=5.00 cls_t=0.00 pos_t=5.00
[server] [Prompt] Built: B'=2 K=8 S=407 N1=7 N2=72 Sref=151 Stgt=177 c_len=407 u_len=177 denoise=1
[server] [MaskGIT] Start: T=177 ...
[RUN] Server thoát rc=-11
```

**Đặc điểm crash:**
- `rc=-11` = tiến trình bị **SIGSEGV** (không phải exception Python — nếu là
  lỗi Python sẽ có traceback; đây là crash ở tầng C/CUDA).
- Xảy ra **trong lúc MaskGIT đang chạy** (job thứ 2 vừa in `[MaskGIT] Start`,
  chưa in dòng kết thúc `[MaskGIT] Total LM forward`).
- Tham số job: `B'=2` (CFG cond+uncond), `num_step=8`, `guidance=2.00`,
  `denoise=1`, chuỗi `S=421`/`S=407`, target `Stgt=186`/`177`.
- Đang có **nhiều job chạy song song** (heartbeat kill-switch vừa fire ở ~180s,
  chứng tỏ server đã chạy một lúc và đang xử lý liên tục).

---

## 4. Các phép thử tái hiện (đã thực hiện) + kết quả

Gọi API trực tiếp qua đường hầm, cùng server đã crash được dựng lại (4 worker,
T4, cùng model INT4). Tổng **65 job, 0 crash, 0 lỗi**.

| # | Phép thử | Tham số | Kết quả |
|---|---|---|---|
| 1 | 5 request **tuần tự** (concurrency 1) | EN, ~70 ký tự, steps=16 | 5/5 OK (~3.9s/req) |
| 2 | 8 request **đồng thời** | EN, ~90 ký tự, steps=16 | 8/8 OK |
| 3 | `client.py` thật, 4 luồng | EN, 4 đoạn dài (~970 ký tự), batch 8 | 4/4 OK, 2.46x realtime |
| 4 | **Hammer**: 5 đợt × 8 đồng thời = 40 job | EN, đoạn dài, **steps=8** | 40/40 OK |
| 5 | 8 đồng thời **tiếng Việt** | VI, đoạn dài, steps=16 | 8/8 OK |

Sau tất cả: `served=65, failed=0`, VRAM trống ~10 GB, server vẫn sống.

**Đã cố ý phủ các biến nghi ngờ:**
- Tuần tự vs đồng thời → không phải do tranh chấp worker đơn thuần.
- Text ngắn vs dài (tới ~970 ký tự, chuỗi tương đương crash) → không phải do độ dài.
- `steps=8` (đúng như log crash) vs `steps=16` → không phải do số bước.
- Tải dồn 40 job liên tiếp → không phải do tích luỹ/volume trong khoảng này.
- 2 ngôn ngữ (EN + VI) → không phải do ngôn ngữ cụ thể đã thử.

---

## 5. Phân tích / giả thuyết

**Giả thuyết chính — race condition hiếm trong nhân CUDA khi chạy song song.**
Crash là SIGSEGV ở tầng C/CUDA, xảy ra đúng lúc MaskGIT chạy với nhiều worker
đồng thời, và **không tái hiện được** qua 65 job. Đây là dấu hiệu kinh điển của
một race/heisenbug: cần đúng thời điểm tương tranh giữa các luồng mới lộ. Bài
stress 4 tiếng trước đó (4 worker) sống sót, càng cho thấy xác suất crash thấp
chứ không phải lỗi chắc chắn.

**Biến số cần đội engine loại trừ — bản Cython `.so` vs bản `.py` thuần.**
Điểm KHÁC BIỆT duy nhất so với các lần chạy ổn định trước: lớp Python
(`pyomnivoice.core`, `remote.server`) lần này được **biên dịch Cython thành
.so**, còn trước là `.py` thường. Engine C++/CUDA thì **y hệt** (cùng bản
prebuilt sm75, cùng model GGUF). Về lý thuyết Cython không đổi ngữ nghĩa ctypes,
nhưng cần kiểm: liệu Cython có làm một buffer numpy (voice ref / audio) bị giải
phóng sớm (refcount/scope khác) khiến con trỏ truyền xuống C++ trỏ vào vùng đã
free → segfault, đặc biệt dưới áp lực GC khi nhiều luồng chạy? Đây là giả thuyết,
chưa chứng minh — nhưng là điểm khác biệt đáng nghi nhất.

**Điểm cần chú ý trong tham số job:** `B'=2` (CFG), `denoise=1` (có xử lý khử
nhiễu voice ref). Hai job trước crash đều `denoise=1`. Nếu đường denoise của
voice ref có nhánh cấp phát riêng, cần soi kỹ khi chạy đồng thời.

---

## 6. Thông tin đội engine cần để điều tra sâu

Để bắt được segfault, đề nghị đội engine:

1. **Chạy dưới `cuda-gdb` hoặc bật `compute-sanitizer`** (`--tool memcheck` /
   `--tool racecheck`) trên đúng cấu hình: T4 sm75, model INT4 lite, **4 worker
   đồng thời**, tải liên tục. racecheck sẽ chỉ ra data race trong kernel nếu có.
2. **Bật core dump** (`ulimit -c unlimited`) rồi phân tích backtrace của tiến
   trình khi rc=-11, để biết segfault ở hàm/kernel nào (MaskGIT? DAC? denoise?).
3. **Kiểm tính thread-safety** của đường `ov_synthesize` khi nhiều engine độc
   lập cùng gọi: có tài nguyên CUDA nào (stream, context, buffer tĩnh) bị chia
   sẻ ngầm giữa các worker không?
4. **So bản `.py` vs bản Cython `.so`** trên cùng input + cùng mức đồng thời:
   nếu bản `.py` không bao giờ crash còn `.so` thỉnh thoảng crash, khoanh vùng
   được về lớp marshalling Python (loại trừ hoặc xác nhận giả thuyết mục 5).

---

## 7. Đề xuất giảm thiểu (phía tích hợp, không chờ engine sửa)

Vì đây là crash hiếm và ở tầng engine, phía tích hợp nên làm **cho hệ thống tự
phục hồi** thay vì chờ sửa gốc:

1. **Watchdog tự khởi động lại server** khi tiến trình chết `rc != 0` mà license
   vẫn hợp lệ — biến một crash hiếm thành gián đoạn vài giây thay vì sập cả
   phiên. (Đang bổ sung vào loader.)
2. **Client đã có cơ chế resume** (ghi từng đoạn, chạy lại là làm tiếp phần
   thiếu) → mất một job giữa chừng không mất cả bài.
3. Nếu cần độ ổn định tối đa tạm thời: **giảm số worker** (ít CUDA đồng thời =
   xác suất race thấp hơn), đổi lại chậm hơn.

---

## Phụ lục — lệnh test đã dùng

- Đăng ký giọng: `POST /voice {name, text, wav_b64}` → `voice_id`
- Tổng hợp: `POST /tts {text, voice_id, lang, steps, seed, format}`
- Client thật: `python remote/client.py --url <URL> --key <KEY> --script <txt>
  --ref <wav> --lang English --batch 8 --concurrency 4`
- Giọng mẫu: FDown…9995-ref.wav (6.0s, transcript tiếng Việt 98 ký tự)
