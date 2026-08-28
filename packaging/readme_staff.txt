OmniVoice — KIỂM TRA ĐỘ ỔN ĐỊNH GPU
====================================================================

LÀM THEO 3 BƯỚC

  1. GIẢI NÉN TOÀN BỘ file nén ra một thư mục.
     KHÔNG bấm chạy trực tiếp từ trong cửa sổ xem file nén của
     Windows. Chương trình cần các thư mục runtime, models, data nằm
     cạnh file exe; chạy từ trong file nén thì Windows chỉ bung mỗi
     exe ra thư mục tạm và chương trình sẽ báo thiếu file.

  2. Bấm đúp START-TEST.exe rồi để yên khoảng 3 đến 20 phút tuỳ máy.

     - Nếu Windows hiện bảng xanh "Windows protected your PC":
       bấm "More info" rồi "Run anyway". File này chưa mua chữ ký
       số nên Windows cảnh báo như với mọi exe lạ.
     - Nếu phần mềm diệt virus chặn: cho phép chạy, hoặc tạm tắt.

  3. Mở thư mục ket-qua, GỬI LẠI file bao-cao-<tên máy>-<ngày>.json
     Tên file đã có sẵn tên máy nên nhiều người gửi về không trùng.

CẦN GÌ
  - GPU NVIDIA, compute capability từ 6.1 trở lên (GTX 1050 trở lên)
  - Driver NVIDIA từ 527.41 trở lên
  - VRAM trống khoảng 1750 MiB
  - Trống khoảng 1,5 GB ổ đĩa

BÀI TEST LÀM GÌ
  Đọc sẵn 10.450 ký tự tiếng Việt bằng một giọng clone có sẵn, ra
  khoảng 10 phút audio. Mục đích không phải nghe hay dở, mà là xem
  card đồ hoạ có trụ nổi từ đầu đến cuối không.

ĐÂY LÀ BẢN DEBUG GPU
  Chương trình KHÔNG tự chuyển sang CPU khi GPU gặp vấn đề. Chạy
  không được thì nó dừng và ghi rõ nguyên nhân: thiếu driver, driver
  quá cũ, GPU quá cũ, hay VRAM không đủ.
  Báo lỗi cũng là kết quả hợp lệ — cứ gửi file json về.

NẾU BÁO THIẾU VRAM
  1. Đóng trình duyệt, game, phần mềm dựng phim. Chúng giữ VRAM.
  2. Chạy CHAY-PROFILE-TINY.cmd để dùng model nhỏ hơn (~1500 MiB),
     rồi gửi về cả hai file json.

CỬA SỔ TẮT NGAY KHÔNG KỊP ĐỌC?
  Không sao, file trong thư mục ket-qua vẫn được ghi. Tiến trình
  giám sát chạy riêng nên vẫn ghi lại được là hỏng ở đoạn nào và
  VRAM lúc đó còn bao nhiêu. Cứ gửi file json về.
