# submodulevoice — OmniVoice TTS / voice clone cho LVC_Project2

Module voice clone chạy **hoàn toàn từ Python**, offline, không cần server,
không cần viết C++.

Đây chính là model trong ảnh chụp màn hình (launcher "Omni test" của
AIOLauncher): **OmniVoice** của Xiaomi / k2-fsa.

| Trong ảnh | Thực tế là gì |
|---|---|
| `Backbone directory: models/upstream-int4` | LM nền OmniVoice (Qwen3-0.6B, 612M tham số) lượng tử hoá 4-bit |
| `Higgs tokenizer directory: models/higgs-fp32` | Audio codec Higgs Audio v2 (HuBERT + DAC + RVQ, 8 codebook, 25 fps, 24 kHz) để FP32 |
| `32 steps` | Số bước giải mã **MaskGIT** |
| `144 frames` | Độ dài đầu ra thủ công (25 frame = 1 giây) |
| `Clone from reference WAV` + `Reference transcript` | Zero-shot voice clone: cần WAV mẫu **và** transcript của nó |
| `Official auto duration + long form` | Tự ước lượng độ dài + tự cắt chunk cho văn bản dài |
| `3.38x realtime` | Con số **trên GPU CUDA**, không phải CPU |

Launcher trong ảnh chạy ONNX Runtime. Bản này chạy **GGML** (cùng dòng với
llama.cpp) — cùng model, cùng chất lượng, nhưng chạy được CPU / CUDA / ROCm /
Metal / Vulkan và nhẹ hơn nhiều khi build.

---

## 1. Cài đặt

```bash
# 1. Lấy nguồn (đã có sẵn trong submodulevoice/omnivoice.cpp)
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git

# 2. Build (Windows, cần VS2022 Build Tools)
build-win.cmd cpu       # bản CPU  -> omnivoice.cpp\build-cpu\omnivoice.dll
build-win.cmd cuda      # bản CUDA -> omnivoice.cpp\build-cuda\omnivoice.dll

# 3. Tải model GGUF (~660 MB cho profile lite)
python -m pyomnivoice.download
```

`pyomnivoice` tự tìm `omnivoice.dll`: ưu tiên `build-cuda`, sau đó `build-cpu`.
Muốn ép dùng bản nào thì đặt biến môi trường `OMNIVOICE_LIB` trỏ tới thư mục
chứa DLL.

> **Lưu ý build CUDA trên máy này.** CUDA 12.1 + MSVC 14.41 không build được:
> `host_config.h` chặn `_MSC_VER >= 1940`, và kể cả thêm
> `-allow-unsupported-compiler` thì STL của 14.41 vẫn `static_assert` cứng
> *"expected CUDA 12.4 or newer"*. Vì vậy `build-win.cmd cuda` ghim toolset
> **v142 (MSVC 19.29)** — nvcc 12.1 chấp nhận toolset này. Cài CUDA >= 12.8 thì
> gỡ được dòng `-vcvars_ver=14.29` trong script.

---

## 2. Dùng bằng Python

```python
from pyomnivoice import OmniVoice

tts = OmniVoice(profile="lite", backend="auto")     # "cpu" | "cuda" | "auto"

# Mã hoá giọng mẫu MỘT LẦN, dùng lại cho mọi câu sau đó
voice = tts.load_voice("ref.wav", "transcript đúng của ref.wav")

audio = tts.say(
    "Xin chào, đây là giọng nói được nhân bản.",
    voice=voice,
    lang="Vietnamese",
    steps=16,
)
audio.save("out.wav")
print(f"{audio.duration:.2f}s audio trong {tts.last_wall:.2f}s")
```

### Các chế độ giọng

```python
# 1. Clone từ WAV mẫu (chính xác nhất)
tts.say(text, voice=voice, lang="Vietnamese")

# 2. Clone trực tiếp, không cache (chậm hơn vì phải encode lại mỗi lần)
tts.say(text, ref_wav="ref.wav", ref_text="...", lang="Vietnamese")

# 3. Voice design — mô tả bằng từ khoá, không cần WAV
tts.say(text, instruct="female, young adult, moderate pitch", lang="Vietnamese")

# 4. Auto voice — để model tự chọn, giữ nhất quán suốt văn bản dài
tts.say(text, lang="Vietnamese")
```

### Streaming (phát ngay khi có chunk đầu tiên)

```python
for chunk in tts.stream(long_text, voice=voice, lang="Vietnamese"):
    speaker.write(chunk)        # numpy float32 mono 24 kHz
```

### Lồng tiếng file .srt theo timeline

```python
audio = tts.dub_srt("phim.srt", voice=voice, lang="Vietnamese", steps=16)
audio.save("dub.wav")
```

Mỗi cue được ép đúng độ dài slot (`T_override`) và tắt post-processing, nên
không bị trôi; khoảng trống giữa các cue là im lặng. Ghép vào video:

```bash
ffmpeg -i phim.mp4 -i dub.wav -c:v copy -map 0:v -map 1:a phim_long_tieng.mp4
```

### Lưu / nạp lại giọng

```python
voice.save("voices/nam_mien_bac.npz")
voice = Voice.load("voices/nam_mien_bac.npz")
```

---

## 3. Test audio và kịch bản dài

### 3.1 Lấy giọng mẫu từ file audio bất kỳ

Voice clone cần WAV mẫu **và** transcript khớp chính xác. `refprep` lo cả hai:
cắt đoạn 4–8 giây sạch, tự transcribe bằng faster-whisper, cắt đúng ranh giới
segment nên chữ luôn khớp tiếng.

```python
from pyomnivoice.refprep import prepare_reference, load_reference

ref = prepare_reference("giong_mau.mp3", lang="vi", model_size="medium")
print(ref.text)          # kiểm tra lại câu này
voice = ref.as_voice(tts)
```

Transcript được ghi ra file `.txt` cạnh WAV. **Sửa tay các chữ ASR nghe sai**
rồi nạp lại bằng `load_reference("...-ref.wav")`.

> #### Vì sao `refprep` phải ASR hai lần
>
> Lỗi này tôi gặp thật và mất khá lâu mới tìm ra, nên ghi lại đây.
>
> Đầu ra bị chèn chữ lạ ở **đầu mọi đoạn** — cụ thể là hai chữ cuối của
> transcript reference, lặp lại ở mọi đoạn của kịch bản 5000 ký tự. Chạy bằng
> CLI upstream cũng y hệt, nên không phải lỗi binding Python.
>
> Nguyên nhân: mốc segment mà ASR trả về khi nghe **cả file** không khớp tuyệt
> đối với nội dung 6 giây được cắt ra. ASR trên file gốc nói segment 1 kết thúc
> ở *"...mà dễ dàng"*, nhưng transcript đã lưu lại có thêm *"lật bài ngửa"* —
> hai chữ không có trong audio. Transcript dài hơn tiếng, model coi hai chữ đó
> là **chưa đọc**, nên phát chúng ra ở đầu đoạn sinh ra. Đây là mặt còn lại của
> điều launcher trong ảnh cảnh báo: thiếu chữ thì bị bỏ chữ, thừa chữ thì bị
> chèn chữ.
>
> Cách sửa: sau khi cắt, **ASR lại chính đoạn đã cắt** và dùng kết quả đó làm
> transcript. Text và tiếng khớp nhau theo cấu trúc, không thể lệch. `refprep`
> in cảnh báo khi hai bản transcript khác nhau quá 15% để bạn biết mà kiểm tra.
>
> Bài học chung: **luôn ASR lại chính file reference và so với transcript của
> nó** trước khi tin vào một giọng mẫu. Đây là lỗi nghe vẫn trôi chảy nên rất
> khó phát hiện, và nó làm hỏng toàn bộ output chứ không chỉ một đoạn.

Whisper `medium` trên CPU mất ~40s cho một file, `small` ~15s nhưng sai nhiều
hơn ở tiếng Việt. Chỉ chạy một lần cho mỗi giọng, sau đó dùng `.txt` đã cache.

### 3.2 Nghe và so sánh

```bash
# Quét steps + profile trên cùng một câu
python examples/compare.py grid --ref giong.mp3 --steps 8 16 32

# Cùng một câu qua toàn bộ giọng mẫu trong một thư mục
python examples/compare.py voices --dir ../ToolEdit/assets/voice_previews
```

Xuất ra `output/compare/grid.html` và `output/compare/voices.html` — bấm play
từng ô, số liệu tốc độ nằm ngay cạnh. Mở bằng:

```cmd
start output\compare\voices.html
```

Dùng để chọn giọng và tìm ngưỡng `steps` thấp nhất mà tai bạn còn chấp nhận.
Trang `voices.html` in kèm transcript ASR của từng giọng mẫu — hàng nào
transcript sai nhiều thì giọng nhân bản của hàng đó cũng kém, sửa `.txt` trong
`output/compare/refs/` rồi chạy lại.

### 3.3 Chạy kịch bản dài

```bash
python examples/longform.py scripts/kichban_dai.txt --ref giong.mp3 --steps 16
```

Kịch bản là file `.txt`, các đoạn cách nhau bằng **dòng trống**. Mặc định
`--mode paragraph`: tổng hợp từng đoạn rồi nối lại với khoảng lặng
`--pause 0.45`. So với `--mode auto` (một lệnh `say()`, để chunker C++ tự cắt),
chế độ đoạn cho bạn:

- biết đoạn nào chậm, đoạn nào ra sai độ dài
- làm lại một đoạn mà không phải chạy lại cả kịch bản (`--keep-parts`)
- kiểm soát nhịp nghỉ giữa các đoạn
- bộ nhớ có giới hạn, không phụ thuộc độ dài kịch bản

Script tự in cảnh báo nếu peak quá nhỏ hoặc tỉ lệ im lặng bất thường (dấu hiệu
model đã bỏ chữ).

### 3.4 Cần nghe kiểm tra những gì

| Kiểm tra | Dấu hiệu lỗi | Cách sửa |
|---|---|---|
| Đầu đoạn có bị thiếu chữ | câu bắt đầu giữa từ | sửa transcript trong `.txt` của ref |
| Số, ngày tháng, phần trăm | đọc dồn, nghe hụt hơi | viết ra chữ, xem [3.5](#35-vì-sao-phải-viết-số-thành-chữ) |
| Chất giọng giữa các đoạn | giọng trôi, đổi người | dùng `voice=` (clone) chứ đừng dùng auto voice |
| Âm cuối câu | méo, rè | tăng `steps` lên 16 hoặc 32 |
| Ngữ điệu tiếng Việt | nghe như người nước ngoài | giọng mẫu phải là **tiếng Việt**; `instruct` không có accent Việt |

Về mục cuối: vocabulary của voice design chỉ có `male/female`, `child` →
`elderly`, `very low` → `very high pitch`, `whisper`, và các accent
american / british / australian / chinese / canadian / indian / korean /
portuguese / russian / japanese. **Không có accent tiếng Việt** — muốn giọng
Việt tự nhiên thì buộc phải clone từ WAV mẫu tiếng Việt.

### 3.5 Vì sao phải viết số thành chữ

Đây không phải mẹo cảm tính, có cơ chế rõ ràng. Model tự ước lượng độ dài đầu ra
từ text bằng bảng trọng số theo từng ký tự — xem
[`duration-estimator.h:941`](omnivoice.cpp/src/duration-estimator.h#L941):

| loại ký tự | trọng số |
|---|---:|
| chữ Latin (kể cả có dấu, dạng NFC) | 1.0 |
| **chữ số** | **3.5** |
| dấu câu | 0.5 |
| khoảng trắng | 0.2 |
| dấu tổ hợp (NFD) | 0.0 |

Một chữ số được tính bằng 3.5 chữ cái. Nhưng khi đọc ra tiếng Việt thì tốn
nhiều hơn thế:

| viết | trọng số ước lượng | trọng số khi đọc thành chữ | thiếu |
|---|---:|---:|---:|
| `18/8` | 11.0 | `mười tám tháng tám` = 18.6 | −41% |
| `100%` | 11.0 | `một trăm phần trăm` = 16.6 | −34% |

Model được cấp ít frame hơn số frame nó thực sự cần, nên phải đọc nhanh hoặc
bỏ chữ. Đo được trên cùng một nội dung, cùng seed, 16 steps:

| dạng viết | ký tự | audio ra |
|---|---:|---:|
| `18/8`, `4/9`, `100%`, `6 tuổi` | 84 | 5.23s |
| viết chữ hết | 130 | 5.85s |
| trộn: `18 tháng 8`, `100 phần trăm` | 112 | **6.52s** |

Ba file tương ứng ở `output/compare/num-{digits,written,mixed}.wav` — nghe thử
sẽ thấy ngay bản `digits` bị dồn.

Nên viết:

| Đừng viết | Viết thành |
|---|---|
| `18/8` | `ngày 18 tháng 8` |
| `100%` | `100 phần trăm` |
| `2.5 triệu` | `hai phẩy năm triệu` |
| `TP.HCM` | `Thành phố Hồ Chí Minh` |
| `1.234.567 đồng` | `một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy đồng` |

Không phải bỏ hết chữ số — dạng "trộn" đo được là tốt nhất. Cái cần bỏ là các
cụm số dày đặc không có chữ xen giữa (`18/8`, `100%`, `1.234.567`).

Cách khác nếu không muốn sửa text: truyền tay `duration=` để tự quyết định số
giây, hoặc `steps` cao hơn để model xử lý phần bị dồn tốt hơn.

`scripts/kichban_dai.txt` là ví dụ đã viết theo cách này.

### 3.6 Kiểm tra INT4 có ổn định trên văn bản dài không

```bash
# render cùng text, cùng seed, hai mức lượng tử hoá
python examples/longform.py scripts/kichban_5k.txt --ref-wav ref.wav \
    --profile lite    --steps 16 --seed 42 --keep-parts -o output/int4/x.wav
python examples/longform.py scripts/kichban_5k.txt --ref-wav ref.wav \
    --profile quality --steps 16 --seed 42 --keep-parts -o output/q8/x.wav

# so sánh
python examples/ab_quant.py output/int4/x.wav output/q8/x.wav --script scripts/kichban_5k.txt
```

**Đừng so bằng tương quan sóng âm.** Tôi thử trước và đo được cosine 0.025 —
nhìn như INT4 hỏng hoàn toàn, thực ra không phải. Hai lần render là hai lần lấy
mẫu ngẫu nhiên khác nhau: cùng đọc đúng một câu nhưng pha và nhịp nội bộ khác
nhau thì sóng âm gần như không tương quan. Độ dài thì trùng khít vì bộ ước
lượng thời lượng là hàm tất định của text.

`ab_quant.py` dùng ba phép đo không phụ thuộc chuyện đó:

| phép đo | đo cái gì | ngưỡng |
|---|---|---|
| **CER** | cho ASR nghe lại rồi so với text gốc — phát hiện bỏ chữ, đọc sai | xem **hiệu số** giữa hai bản, không phải trị tuyệt đối |
| **LTAS** | phổ trung bình dài hạn — âm sắc / chất giọng có đổi không | < 0.95 là đáng nghi |
| **im lặng** | tỉ lệ mẫu gần 0 | lệch > 10 điểm phần trăm là đáng nghi |

Kết quả xuất ra `output/ab_quant.html`: hai bản đầy đủ ở đầu trang, rồi từng
đoạn ghép đôi hai player kèm text gốc và cả hai bản ASR để đối chiếu chữ.

#### Kết quả đo được trên `scripts/kichban_5k.txt`

5246 ký tự / 16 đoạn / 309s audio (5 phút 09), giọng clone 6.00s từ một file mp3
tải về, 16 steps, seed 42:

| | INT4 (Q4_K_M) | Q8_0 |
|---|---:|---:|
| CER trung vị | **0.66%** | 0.56% |
| CER cao nhất | 3.06% | 1.83% |
| LTAS trung vị | 0.994 (min 0.991) | |
| đoạn bị gắn cờ | **0 / 16** | |

Chênh CER trung vị **+0.11 điểm phần trăm**. Không trôi dần về cuối: đoạn 16 cho
CER 0.0% ở cả hai bản. Điểm duy nhất INT4 kém hơn thật là **worst case**: 3.06%
so với 1.83%, tập trung ở ba đoạn dài nhất (10, 14, 15).

Kết luận: **INT4 ổn định, dùng được cho văn bản dài.** Trên GPU thì Q8 chỉ đắt
thêm khoảng 470 MiB VRAM và 13% thời gian, nên nếu đủ VRAM thì cứ dùng `quality`;
còn INT4 dành cho card 4 GB hoặc khi cần chạy nhiều tiến trình song song.

> **Cẩn thận khi đọc CER.** Lần đo đầu tiên tôi ra CER 3.45% / 3.39% và đã kết
> luận đó là "nhiễu ASR". Sai. Phần lớn con số đó đến từ bug reference bleed ở
> [mục 3.1](#31-lấy-giọng-mẫu-từ-file-audio-bất-kỳ) — mọi đoạn đều bị chèn hai
> chữ ở đầu, cả hai bản giống nhau nên hiệu số vẫn đúng, nhưng mức nền thì bị
> bơm lên gấp 5 lần. Sau khi sửa, CER về dưới 1%. Kết luận INT4 ≈ Q8 vẫn đứng,
> nhưng bài học là: **CER nền cao bất thường thì hãy đi tìm lỗi hệ thống trước
> khi gán cho nhiễu.** Người dùng nghe ra bug này trước khi số liệu của tôi chỉ
> ra nó.

### 3.7 Clone xuyên ngôn ngữ — thử với tiếng Bồ Brazil

```bash
python examples/test_pt.py     # -> output/pt/index.html
```

Không có giọng mẫu tiếng Bồ nên bài này lấy giọng mẫu **tiếng Việt** và
**tiếng Anh** rồi bắt đọc tiếng Bồ. Kịch bản `scripts/kichban_pt.txt`,
1172 ký tự, profile quality, 16 steps:

| nhánh | ký tự/giây | CER | ASR tự đoán ngôn ngữ |
|---|---:|---:|---|
| giọng mẫu tiếng Việt (6.0s) | 15.9 | **0.00%** | pt 99% |
| giọng mẫu tiếng Anh (17.3s) | 11.8 | 0.89% | pt 99% |
| voice design, không giọng mẫu | 16.2 | 0.98% | pt 100% |
| giọng Việt, không khai báo `lang` | 16.0 | 0.89% | pt 99% |

Ba kết luận:

1. **Clone xuyên ngôn ngữ chạy được.** Giọng mẫu tiếng Việt đọc tiếng Bồ vẫn
   được Whisper tự nhận là tiếng Bồ với 99% tin cậy, CER 0.00%. Không phải
   "tiếng Việt đọc chữ Bồ".
2. **Giọng mẫu quyết định tốc độ đọc.** Cùng một kịch bản: mẫu tiếng Việt cho
   74.2s, mẫu tiếng Anh cho 99.9s — chênh **35%**. Bộ ước lượng thời lượng lấy
   tốc độ nói từ reference. Nghĩa là tỉ lệ ký tự → giây audio ở
   [mục 4.2](#42-thời-gian-và-bộ-nhớ-cho-một-kịch-bản-5000-ký-tự) **phải đo
   lại cho từng giọng** trước khi báo giá khách.
3. Khai báo `lang="Portuguese"` hay để `"None"` gần như không khác nhau ở bài
   này — mô hình tự suy ngôn ngữ từ chính văn bản.

**Cảnh báo về cách đo CER.** Con số thô là 3.1–4.3%, nhìn như tiếng Bồ tệ hơn
tiếng Việt 5 lần. Sai. Whisper tiếng Bồ **tự viết số nói thành chữ số**:
kịch bản ghi `seiscentos`, ASR trả về `600`. Khớp lại cách viết số thì CER về
0.00–0.98%. Đây là cùng loại bẫy với vụ reference bleed ở
[mục 3.1](#31-lấy-giọng-mẫu-từ-file-audio-bất-kỳ): **CER cao bất thường thì
xem diff trước khi kết luận**.

**Điều không đo được.** CER và nhận dạng ngôn ngữ không nói được giọng nghe có
tự nhiên với tai người Brazil hay không. Giọng mẫu là người Việt/Anh, nên nhiều
khả năng vẫn nghe ra chất ngoại. Muốn chắc thì phải có người bản xứ nghe, hoặc
dùng giọng mẫu người Brazil. Vocabulary voice design cũng không có nhãn nào cho
tiếng Bồ Brazil — `portuguese accent` là nhãn mô tả **giọng Bồ khi nói tiếng
Anh**, không phải để sinh tiếng Bồ.

---

## 4. Hiệu năng — đọc kỹ phần này

MaskGIT **không phải** autoregressive. Mỗi bước trong `steps` chạy **một lượt
forward hai chiều trên toàn bộ chuỗi** (frame tham chiếu + frame đích). Nghĩa là:

```
thời gian ≈ steps × forward(ref_frames + target_frames)
```

Không có KV-cache để tiết kiệm. Đây là lý do model chỉ 612M tham số nhưng vẫn
nặng, và là lý do GPU nhanh hơn CPU rất nhiều.

### 4.1 Số đo nhanh: steps và độ dài WAV mẫu

Cùng một câu tiếng Việt, `profile="lite"` (Q4_K_M + Q8_0), `python examples/bench.py`:

**GPU** (RTX 4000 SFF Ada, sm_89, backend CUDA):

| cấu hình | wall | audio ra | tốc độ |
|---|---:|---:|---:|
| WAV mẫu 17s, 32 steps | 11.03s | 25.9s | **2.34x** |
| WAV mẫu 6s, 32 steps | 6.73s | 16.8s | **2.49x** |
| WAV mẫu 6s, 16 steps | 3.34s | 16.6s | **4.97x** |
| WAV mẫu 6s, 8 steps | 1.78s | 16.8s | **9.42x** |
| WAV mẫu 6s, 16 steps, chunk 10s | 3.54s | 16.8s | 4.75x |

Cột giữa của ảnh chụp màn hình (26.67s audio, 32 steps, 3.34x) rơi đúng vào
hàng đầu tiên — và đây là **RTX 4000 SFF Ada**, một card workstation tầm trung,
không phải card cao cấp.

Streaming: chunk âm thanh đầu tiên ra sau **1.28s**, tổng 4.44x realtime.
Kịch bản dài 13 đoạn / 3699 ký tự: **178.6s audio trong 36.6s = 4.88x**.
Lồng tiếng SRT: **7.09x**.

**`profile="quality"` gần như miễn phí trên GPU** — cùng 16 steps: lite 1.65s
vs quality 1.68s. LM forward chiếm gần hết thời gian, và Q8_0 so với Q4_K_M
không khác biệt đáng kể khi bị giới hạn bởi tính toán chứ không phải băng thông
bộ nhớ. Có GPU thì dùng `quality`.

**Trên CPU thì INT4 không nhanh hơn, mà còn chậm hơn một chút** — xem
[mục 4.3](#43-int4-trên-cpu).

**CPU** (20 thread, backend GGML CPU):

| cấu hình | wall | audio ra | tốc độ |
|---|---:|---:|---:|
| WAV mẫu 17s, 32 steps | 420.0s | 25.9s | **0.06x** |
| WAV mẫu 6s, 32 steps | 175.1s | 16.8s | **0.10x** |
| WAV mẫu 6s, 16 steps | 89.3s | 16.7s | **0.19x** |
| WAV mẫu 6s, 8 steps | 45.8s | 16.7s | **0.36x** |
| WAV mẫu 6s, 16 steps, chunk 10s | 112.2s | 16.8s | 0.15x |

Đọc bảng này:

- Rút WAV mẫu từ 17s xuống 6s: **nhanh 2.4 lần**. Chuỗi mỗi bước giảm từ
  S=1246 xuống S=704, thời gian mỗi bước từ 12.8s xuống 5.3s.
- Giảm steps 32 → 8: **nhanh thêm 3.8 lần**, tuyến tính đúng như công thức.
- **Chunk lại chậm hơn**, không nhanh hơn: mỗi chunk phải xử lý lại toàn bộ
  frame tham chiếu. Chỉ bật chunk khi văn bản thực sự dài (> 30s).
- `guidance_scale` **không phải** cần gạt: `B_prime = 2` được hardcode, cond và
  uncond luôn chạy cùng batch, hạ guidance xuống 1.0 không tiết kiệm gì.

Kết luận thẳng: **CPU không đạt được 3.38x realtime trong ảnh.** Trần thực tế
là ~0.36x, tức chậm hơn realtime gần 3 lần, và đó là ở mức 8 steps đã bắt đầu
giảm chất lượng. Chênh lệch CPU/GPU ở đây là **25–40 lần**. Con số trong ảnh là
số của GPU CUDA, không phải CPU.

Ý nghĩa thực tế cho từng ca dùng:

| ca dùng | CPU | GPU |
|---|---|---|
| Lồng tiếng hàng loạt (offline) | Được, chạy nền qua đêm | Thoải mái |
| Đọc realtime / streaming | Không khả thi | Được (chunk đầu ~1.3s) |
| Lồng tiếng theo SRT | Được nếu video ngắn | Thoải mái |

### 4.2 Thời gian và bộ nhớ cho một kịch bản 5000 ký tự

Đo bằng `python examples/measure.py --profile lite --backend cuda`. VRAM lấy
theo delta tổng `memory.used` của GPU (driver WDDM trên Windows không báo VRAM
theo process), mẫu 0.2 giây một lần. Kịch bản `scripts/kichban_5k.txt`, 5246 ký
tự / 16 đoạn, giọng clone 6.00s, 16 steps, RTX 4000 SFF Ada:

| | INT4 / CUDA | Q8_0 / CUDA | INT4 / CPU |
|---|---:|---:|---:|
| audio ra | 309.1s (5 phút 09) | 308.7s | 307.7s |
| **tổng thời gian** | **60.6s** | 68.4s | **1484.7s** (24 phút 45) |
| trong đó tổng hợp | 57.3s | 65.0s | 1481.1s |
| tốc độ | **5.11x realtime** | 4.51x | **0.21x** |
| **VRAM đỉnh** | 1727 MiB | 2199 MiB | **0** |
| RAM đỉnh | 648 MiB | 746 MiB | 1783 MiB |
| model trên đĩa | 660 MB | 1.39 GB | 660 MB |

CPU chậm hơn CUDA **24.5 lần** trên cùng kịch bản. Tốc độ rất đều qua cả 16
đoạn (0.20–0.21x, không đoạn nào tụt), nên con số này ngoại suy được: nhân số
phút audio bạn cần với khoảng 4.8 để ra thời gian máy phải chạy.

Ảnh chụp màn hình của launcher ghi "8g vram GPU". Bản INT4 này dùng
**1.7 GB VRAM** — chạy được trên card 4 GB, và còn thừa chỗ cho 2–3 tiến trình
song song trên card 8 GB.

Lưu ý về cách đọc VRAM: 1727 MiB không phải kích thước model (660 MB). Phần
chênh là context CUDA của ggml, graph buffer, và buffer attention
`B' × S × S` cho đoạn dài nhất. Đoạn càng dài thì `S` càng lớn nên VRAM tăng
theo — chia đoạn nhỏ hơn thì giảm được.

Cột CPU: VRAM đúng bằng 0, RAM đỉnh 1783 MiB vì toàn bộ weight nằm ở host thay
vì trên card.

### Năng suất: 1 giờ audio mất bao lâu

Đo thật, không ngoại suy — `scripts/kichban_1h.txt`, 62.700 ký tự, RTX 4000 Ada,
profile lite (INT4), 16 steps:

| | giá trị |
|---|---:|
| ký tự | 62.700 |
| audio ra | 3708.6s (**61.8 phút**) |
| thời gian chạy | 745.6s (**12 phút 26**) |
| tốc độ | 4.97x realtime |
| VRAM đỉnh | **1817 MiB** |
| RAM đỉnh | 1864 MiB |

Quy đổi để báo khách:

- **~16,9 ký tự cho mỗi giây audio** → 1 giờ audio ≈ **61.000 ký tự**.
  Con số này theo **giọng mẫu**, vì bộ ước lượng thời lượng lấy tốc độ nói từ
  reference. Đổi giọng mẫu là đổi tỉ lệ.
- GPU tầm trung: 1 giờ audio ≈ **12 phút máy chạy**.
- CPU (0,21x): 1 giờ audio ≈ **4 giờ 46 phút**.

Hai điều quan trọng khi chạy dài:

- **VRAM không tăng theo tổng độ dài.** Bài 10.450 ký tự dùng 1783 MiB, bài
  62.700 ký tự dùng 1817 MiB — chênh 34 MiB dù dài gấp 6 lần. VRAM chỉ phụ
  thuộc **đoạn văn dài nhất**, vì ma trận chú ý có kích thước bình phương độ
  dài chuỗi.
- **RAM thì có tăng.** 648 MiB lên 1864 MiB, vì `longform.py` giữ toàn bộ audio
  trong bộ nhớ rồi mới nối và ghi. Với 3+ giờ audio nên đổi sang ghi từng đoạn
  ra đĩa rồi ghép sau, hoặc dùng `tts.stream()`.

Bộ test đóng gói tự in bảng quy đổi này theo số đo của **chính máy chạy nó**,
nên mỗi báo cáo nhân viên gửi về đều trả lời sẵn "máy này dựng 1 giờ audio mất
bao lâu". `gather_reports.py` gom lại thành cột `1h audio`.

#### 1 giờ audio tiếng Bồ — số đo đầy đủ

`scripts/kichban_pt_1h.txt`, 57.428 ký tự, giọng mẫu tiếng Việt 6.0s,
profile lite (INT4), 16 steps, RTX 4000 Ada:

| | giá trị |
|---|---:|
| audio ra | 3628.0s (**60.5 phút**) |
| thời gian chạy | 689.6s (**11 phút 30**) |
| tốc độ | 5.26x realtime |
| **VRAM đỉnh** | **1443 MiB** |
| RAM đỉnh | 1842 MiB |
| CER trung vị | **0.00%** |
| CER trung bình | 0.79% |
| CER cao nhất | 2.18% |
| đoạn có CER > 5% | **0 / 245** |

Chất lượng **không tụt theo thời gian**: chia 245 đoạn thành bốn phần tư, cả
bốn đều có CER trung vị 0.00% và cao nhất 2.18%. Chênh phần cuối so với phần
đầu đúng bằng 0.

**VRAM tiếng Bồ thấp hơn tiếng Việt** (1443 vs 1817 MiB) không phải vì ngôn
ngữ, mà vì đoạn dài nhất của kịch bản Bồ chỉ 274 ký tự trong khi kịch bản Việt
là 476. VRAM đi theo **đoạn dài nhất**, không theo tổng độ dài — muốn giảm VRAM
thì cắt đoạn ngắn lại.

> **Lại một lỗi đo nữa, ghi lại để không lặp.** Lần chấm đầu tiên ra CER 75%.
> Nguyên nhân: `longform.py --keep-parts` đặt tên file `-01` đến `-99` rồi
> `-100`, mà sort chuỗi xếp `-100` ngay sau `-10`. Với hơn 99 đoạn thì text bị
> so lệch với audio. Đã sửa ở ba chỗ: `longform.py` đệm số 0 theo tổng số đoạn,
> `score_pt.py` và `ab_quant.py` sort theo số. Các kết quả trước đó không ảnh
> hưởng vì đều dưới 100 đoạn. Nguyên tắc rút ra vẫn thế: **con số vô lý thì
> nghi phép đo trước, đừng nghi model.**

### 4.3 INT4 trên CPU

Câu hỏi tự nhiên: nếu không có GPU thì INT4 có giúp gì không? Đo trên probe
773 ký tự (3 đoạn đầu của kịch bản), backend CPU, 16 steps:

| | INT4 (Q4_K_M) | Q8_0 |
|---|---:|---:|
| thời gian tổng hợp | 234.5s | **225.6s** |
| tốc độ | 0.20x realtime | 0.20x |
| RAM đỉnh | **1532 MiB** | 2008 MiB |
| CER | 0.13% | 0.40% |

**INT4 không nhanh hơn trên CPU — còn chậm hơn khoảng 4%.** Đây là điều trái
với trực giác thông thường "quantise nhỏ hơn thì nhanh hơn", và cũng trái với
dự đoán ban đầu của tôi ghi trong bản README trước.

Giải thích hợp lý nhất: Q4_K_M là K-quant, mỗi block phải giải nén thang đo
phân cấp và 8 sub-block trước khi nhân, trong khi Q8_0 × Q8_0 là đường đi được
tối ưu nhất trong ggml — activation dù sao cũng bị quantise về Q8_0. Trên máy
này (2 socket Xeon Gold 6138, 6 kênh RAM mỗi socket) băng thông không phải cổ
chai, nên phần giải nén thêm của K-quant không được trả lại. Trên máy băng
thông hẹp hơn — laptop một kênh RAM chẳng hạn — cân bằng có thể lệch về phía
INT4; tôi chưa đo trường hợp đó.

Vậy INT4 trên CPU mua được gì: **RAM (1532 vs 2008 MiB) và dung lượng đĩa
(660 MB vs 1.39 GB)**, không phải tốc độ. Máy CPU-only mà đủ RAM thì cứ dùng
`quality`.

#### Chất lượng CPU so với GPU

Chạy full 5246 ký tự trên CPU rồi so với bản CUDA cùng seed
(`output/ab_cpu_vs_gpu.html`):

| | INT4 / CPU | INT4 / CUDA |
|---|---:|---:|
| CER trung vị | 0.88% | 0.66% |
| CER cao nhất | 3.36% | 3.06% |
| LTAS trung vị | 0.993 (min 0.984) | |
| đoạn bị gắn cờ | 1 / 16 | |

Chênh 0.21 điểm phần trăm — **CPU đọc đúng ngang GPU**, chỉ chậm hơn. Đoạn duy
nhất bị gắn cờ là đoạn 3 (đoạn ngắn nhất, 92 ký tự): tỉ lệ im lặng 8.8% so với
19.6%, nhưng CER 0.0% ở cả hai nên không mất chữ, chỉ là cách ngắt nghỉ khác.

Tốc độ cũng không tụt dần: cả 16 đoạn đều 0.20–0.21x, đoạn cuối bằng đoạn đầu.
Không có hiện tượng nóng máy hay rò bộ nhớ trong 25 phút chạy liên tục.

#### Số thread: ggml chỉ dùng một nửa số core

`backend_cpu_n_threads()` trong
[`backend.h:29`](omnivoice.cpp/src/backend.h#L29) lấy
`hardware_concurrency() / 2`. Trên Windows, `hardware_concurrency()` chỉ thấy
**một processor group**, nên máy 2 socket × 20 core × 2 thread = 80 logical chỉ
được báo 40, và ggml chạy **20 thread** — đúng một socket. Nửa máy nằm không.

Không có cờ CLI hay biến môi trường nào để đổi; muốn sửa phải patch hàm đó rồi
build lại. Nghĩa là các con số CPU ở trên là **cận dưới** trên máy nhiều socket.

Cũng nên nhớ CPU ở đây là Xeon Gold 6138 @ 2.0 GHz, Skylake-SP đời 2017, và
AVX-512 trên kiến trúc này còn bị hạ xung khi chạy. Một CPU desktop hiện đại
ít core nhưng xung cao sẽ cho kết quả tốt hơn đáng kể — đừng lấy 0.20x làm
chuẩn cho mọi máy CPU.

### 4.4 Bốn cần gạt để tăng tốc, theo thứ tự hiệu quả

1. **`steps`** — 32 → 16 nhanh gần gấp đôi, chất lượng gần như không đổi.
   32 → 8 nhanh gấp 4, bắt đầu nghe rõ artifact.
2. **Độ dài WAV mẫu** — dùng 4–6 giây, đừng dùng 15–20 giây. Frame tham chiếu
   bị xử lý lại ở **mọi** bước MaskGIT.
3. **`profile="lite"`** — backbone Q4_K_M + codec Q8_0 (~660 MB).
4. **Chunk ngắn hơn** — `chunk_duration=10, chunk_threshold=8` giữ mỗi lượt
   forward ngắn, chi phí attention giảm theo bình phương độ dài.

---

## 5. Cấu hình tối thiểu

Số đo thật, không phải ước lượng (kịch bản 5246 ký tự, 16 steps):

| | CPU-only | GPU |
|---|---|---|
| profile lite (INT4) | RAM 1783 MiB, VRAM 0 | VRAM 1727 MiB + RAM 648 MiB |
| profile quality (Q8_0) | RAM 2008 MiB, VRAM 0 | VRAM 2199 MiB + RAM 746 MiB |
| CPU | x86-64 có SSE4.2 trở lên | bất kỳ |
| GPU | không cần | NVIDIA (CUDA), AMD/Intel (Vulkan), Apple (Metal) |
| card 4 GB có chạy được? | — | có, cả hai profile |

RAM cho `quality` trên CPU lấy từ probe 773 ký tự; các số còn lại từ full run.

Bản build dùng `GGML_CPU_ALL_VARIANTS=ON` nên cùng một bộ DLL chạy được từ CPU
đời cũ chỉ có SSE4.2 cho tới AVX-512, tự chọn lúc runtime. `GGML_BACKEND_DL=ON`
nghĩa là backend nạp động — máy không có GPU sẽ tự rơi về CPU, không crash.

---

## 6. Cấu trúc thư mục

```
submodulevoice/
├── build-win.cmd            # build CPU / CUDA trên Windows
├── omnivoice.cpp/           # upstream (git submodule được)
│   ├── build-cpu/           # omnivoice.dll + ggml*.dll (CPU)
│   ├── build-cuda/          # omnivoice.dll + ggml*.dll (CUDA)
│   └── models/              # *.gguf
├── pyomnivoice/
│   ├── __init__.py
│   ├── _ffi.py              # ctypes binding cho ABI ov_*
│   ├── core.py              # class OmniVoice, Voice, Audio
│   ├── srt.py               # parse .srt + ghép timeline
│   ├── refprep.py           # cắt + tự transcribe giọng mẫu (faster-whisper)
│   └── download.py          # tải GGUF từ HuggingFace
├── scripts/
│   ├── kichban_dai.txt      # kịch bản mẫu (13 đoạn, ~3 phút)
│   └── kichban_5k.txt       # kịch bản 5246 ký tự (16 đoạn, ~4m45s)
├── examples/
│   ├── demo_clone.py        # clone + đo tốc độ
│   ├── longform.py          # chạy kịch bản dài, báo cáo từng đoạn
│   ├── compare.py           # bảng so sánh + trang HTML để nghe
│   ├── ab_quant.py          # INT4 vs Q8: CER qua ASR + LTAS + im lặng
│   ├── measure.py           # đo thời gian + VRAM + RAM của một lần render
│   ├── dub_srt.py           # lồng tiếng .srt theo timeline
│   └── bench.py             # quét steps / độ dài WAV mẫu
└── output/
```

---

## 7. Phương án Python thuần (không build)

Nếu không muốn build gì cả:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install omnivoice
```

```python
from omnivoice import OmniVoice
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")
audio = model.generate(text="...", ref_audio="ref.wav", ref_text="...")
```

Đổi lại: tải ~2.5 GB checkpoint PyTorch, phụ thuộc torch, và trên CPU thì chậm
hơn bản GGML.

---

---

## 8. Gói test GPU cho máy khác

```bash
python packaging/build_package.py
# -> packaging/dist/OmniVoiceGpuTest.zip  (995 MiB)
```

Giải nén, bấm `START-TEST.exe`. Đọc sẵn 10.450 ký tự bằng giọng clone có sẵn,
khoảng 10 phút audio, rồi ghi báo cáo vào `ket-qua/`.

**Không có fallback CPU.** Đây là bản debug GPU: nếu CUDA không dùng được thì
dừng và ghi nguyên nhân cụ thể — thiếu driver (`NO_DRIVER`), driver quá cũ
(`DRIVER_TOO_OLD`), kiến trúc GPU không có mã (`ARCH_UNSUPPORTED`), VRAM không
đủ (`VRAM_TOO_LOW`), hoặc backend thực tế không phải CUDA. `backend="cuda"`
đặt `GGML_BACKEND=CUDA0` nên ggml không thể lặng lẽ rơi về CPU, và harness còn
kiểm tra lại tên backend sau khi nạp.

Hai tiến trình: tiến trình cha chỉ giám sát và ghi báo cáo, tiến trình con làm
việc thật. ggml gọi `abort()` khi CUDA hết bộ nhớ — tiến trình con chết ngang
không kịp báo gì, nhưng tiến trình cha vẫn sống để ghi lại nó chết ở đoạn nào
và VRAM lúc đó còn bao nhiêu. Đó chính là dữ liệu cần cho một bài test ổn định.

Báo cáo có: thời gian nạp model / mã hoá giọng / tổng hợp / tổng cộng, VRAM
mức nền và **VRAM bài test tiêu thụ** (đỉnh trừ mức nền, lấy trên GPU có mức
tăng lớn nhất nên không cần đoán CUDA chọn card nào), tốc độ từng đoạn, và
cảnh báo nếu tốc độ cuối tụt quá 15% so với đầu.

> **Bẫy lớn nhất khi đem exe sang máy khác: runtime CUDA.**
> `ggml-cuda.dll` phụ thuộc động vào `cudart64_12.dll`, `cublas64_12.dll` và
> `cublasLt64_12.dll`. **Driver NVIDIA không có mấy file này** — chỉ CUDA
> Toolkit mới có. Máy nhân viên chỉ cài driver thì `ggml_backend_load_all()`
> lặng lẽ bỏ qua backend CUDA và `ov_init` chỉ báo chung chung
> *"no GGML backend available"*, nhìn rất giống hết VRAM dù thực tế chưa hề
> chạm tới GPU. Dấu hiệu nhận biết: **VRAM tiêu thụ 0 MiB và dừng trong vài
> giây**. Gói phát hành phải kèm ba DLL đó (+561 MB) cùng VC++ runtime, và
> preflight nạp thử `ggml-cuda.dll` để chỉ đúng tên file còn thiếu.

Kiến trúc GPU: `build-win.cmd cuda` mặc định build
`61-virtual;61-real;75-real;86-real;89-real` — GTX 1050 trở lên. Một kiến trúc
đơn lẻ chỉ chạy trên đúng dòng card đó, không phát hành được.

## Nguồn

- Model: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — Apache 2.0
- Runtime: [ServeurpersoCom/omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) — MIT
- GGUF: [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF)
- Codec: [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) — Apache 2.0
