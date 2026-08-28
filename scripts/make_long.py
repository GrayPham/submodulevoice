"""Sinh kịch bản dài từ kịch bản gốc, dùng cho các bài đo 1 giờ.

    python scripts/make_long.py

Các file `*_1h.txt` chỉ là kịch bản gốc lặp lại cho đủ độ dài, nên không đưa
vào git — sinh lại trong vài mili giây. Mật độ ký tự trên giây lấy từ số đo
thật ở README mục 4.2, và phụ thuộc GIỌNG MẪU chứ không phải ngôn ngữ.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (nguồn, đích, ký tự cho mỗi giây audio đo được với giọng mẫu tương ứng)
JOBS = [
    ("kichban_10k.txt", "kichban_1h.txt", 16.9),   # giọng mẫu tiếng Việt 6.0s
    ("kichban_pt.txt", "kichban_pt_1h.txt", 15.9),  # cùng giọng, đọc tiếng Bồ
]
TARGET_SEC = 3600


def paragraphs(p: Path) -> list[str]:
    return [x.strip() for x in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")) if x.strip()]


def main() -> None:
    for src_name, dst_name, cps in JOBS:
        src = HERE / src_name
        if not src.exists():
            print(f"bỏ qua {src_name}: không có")
            continue
        paras = paragraphs(src)
        n = sum(len(re.sub(r"\s+", " ", x)) for x in paras)
        reps = -(-int(TARGET_SEC * cps) // n)
        out = paras * reps
        (HERE / dst_name).write_text("\n\n".join(out) + "\n", encoding="utf-8")
        total = sum(len(re.sub(r"\s+", " ", x)) for x in out)
        print(f"{dst_name:22s} {len(out):4d} đoạn, {total:6d} ký tự, "
              f"~{total / cps / 60:.1f} phút audio")


if __name__ == "__main__":
    main()
