"""Sinh 4 kịch bản, mỗi kịch bản đúng 1 giờ audio, cho bài test độ ổn định.

    python scripts/make_stress.py

Mật độ ký tự trên giây khác nhau theo NGÔN NGỮ và theo GIỌNG MẪU, vì bộ ước
lượng thời lượng lấy tốc độ nói từ reference. Số dưới đây đo được với cùng một
giọng mẫu tiếng Việt 6.0s:

    tiếng Việt  16.9 ký tự/giây   (đo trên kichban_10k, 614.6s audio)
    tiếng Bồ    15.9 ký tự/giây   (đo trên kichban_pt, 74.2s audio)

Tiếng Anh và Tây Ban Nha chưa đo, tạm dùng 16.5 — sai số ở đây chỉ làm độ dài
lệch vài phút, không ảnh hưởng kết luận về độ ổn định.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_SEC = 3600

JOBS = [
    ("kichban_10k.txt", "stress_vi.txt", 16.9, "Vietnamese"),
    ("kichban_pt.txt", "stress_pt.txt", 15.9, "Portuguese"),
    ("kichban_en.txt", "stress_en.txt", 16.5, "English"),
    ("kichban_es.txt", "stress_es.txt", 16.5, "Spanish"),
]


def paragraphs(p: Path) -> list[str]:
    return [x.strip() for x in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")) if x.strip()]


def main() -> None:
    print(f"{'file':<16s}{'đoạn':>7s}{'ký tự':>9s}{'audio dự kiến':>16s}")
    print("-" * 50)
    for src_name, dst_name, cps, lang in JOBS:
        paras = paragraphs(HERE / src_name)
        n = sum(len(re.sub(r"\s+", " ", x)) for x in paras)
        reps = -(-int(TARGET_SEC * cps) // n)
        out = paras * reps
        (HERE / dst_name).write_text("\n\n".join(out) + "\n", encoding="utf-8")
        tot = sum(len(re.sub(r"\s+", " ", x)) for x in out)
        print(f"{dst_name:<16s}{len(out):>7d}{tot:>9d}{tot / cps / 60:>13.1f} phút   {lang}")


if __name__ == "__main__":
    main()
