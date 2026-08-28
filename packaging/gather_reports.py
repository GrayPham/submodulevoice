"""Gộp nhiều bao-cao-*.json từ các máy nhân viên gửi về thành một bảng.

    python packaging/gather_reports.py bao-cao-thu-ve/
    python packaging/gather_reports.py bao-cao-thu-ve/ -o tong-hop.html

Bỏ tất cả file json vào một thư mục rồi chạy. Script tự đọc, xếp theo VRAM
card, và tách riêng những máy ĐẠT với những máy hỏng kèm lý do.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.rglob("bao-cao-*.json")) or sorted(d.rglob("*.json")):
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  bỏ qua {f.name}: {type(e).__name__}")
    return rows


def gpu_of(r: dict) -> dict:
    g = (r.get("gpu") or {}).get("gpus") or [{}]
    return g[min(r.get("vram_gpu_index", 0), len(g) - 1)]


def _phut_1h(r: dict) -> float | None:
    """Số phút máy này cần để dựng 1 giờ audio. Đây là con số gửi cho khách."""
    pr = r.get("projection") or {}
    xrt = pr.get("xrt")
    if not xrt:
        audio, synth = r.get("audio_sec"), r.get("synth_sec")
        if not audio or not synth:
            return None
        xrt = audio / synth
    return round((3600.0 / xrt + (pr.get("load_s") or 0)) / 60.0, 1)


def summarize(r: dict) -> dict:
    g = gpu_of(r)
    res = r.get("result") or {}
    status = res.get("status", "?")
    why = ""
    if status == "PREFLIGHT_FAILED":
        why = ", ".join(res.get("codes") or [])
    elif status in ("CRASH", "ERROR"):
        fatal = res.get("fatal") or {}
        why = (f"đoạn {res.get('last_paragraph')}, mã thoát {res.get('exit_code')}"
               + (f", {fatal.get('error')}" if fatal.get("error") else ", chết ngang (nghi CUDA OOM)"))
    return {
        "may": r.get("machine", "?"),
        "gpu": g.get("name", "?"),
        "vram_tong": g.get("vram_total_mib"),
        "cc": g.get("compute_cap"),
        "driver": g.get("driver"),
        "profile": r.get("profile"),
        "trang_thai": status,
        "vram_dung": r.get("vram_used_by_test_mib"),
        "vram_trong_min": r.get("vram_min_free_mib"),
        "giay": r.get("wall_sec"),
        "audio": r.get("audio_sec"),
        "xrt": (res.get("xrt_overall") if status == "PASS" else None),
        "troi": res.get("drift_ratio"),
        "phut_cho_1h": _phut_1h(r),
        "ly_do": why,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", default=None, help="xuất thêm bảng HTML")
    args = ap.parse_args()

    rows = [summarize(r) for r in load(Path(args.dir))]
    if not rows:
        raise SystemExit("Không tìm thấy file bao-cao-*.json nào")
    rows.sort(key=lambda x: (x["vram_tong"] or 0, x["may"]))

    ok = [r for r in rows if r["trang_thai"] == "PASS"]
    bad = [r for r in rows if r["trang_thai"] != "PASS"]

    hdr = (f"{'máy':<16s} {'GPU':<28s} {'VRAM':>6s} {'cc':>4s} {'dùng':>6s} "
           f"{'trống':>6s} {'xRT':>6s} {'1h audio':>9s}")
    print(f"\n{len(ok)}/{len(rows)} máy ĐẠT\n")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        p1h = f"{r['phut_cho_1h']:.1f}p" if r["phut_cho_1h"] else "?"
        print(f"{r['may'][:16]:<16s} {str(r['gpu'])[:28]:<28s} {r['vram_tong'] or 0:>6d} "
              f"{r['cc'] or 0:>4} {r['vram_dung'] or 0:>6d} {r['vram_trong_min'] or 0:>6d} "
              f"{r['xrt'] or 0:>6.2f} {p1h:>9s}")
    if bad:
        print(f"\n{len(bad)} máy KHÔNG ĐẠT\n")
        for r in bad:
            print(f"{r['may'][:16]:<16s} {str(r['gpu'])[:30]:<30s} "
                  f"{r['vram_tong'] or 0:>6d} MiB  [{r['trang_thai']}] {r['ly_do']}")

    if ok:
        vr = [r["vram_dung"] for r in ok if r["vram_dung"]]
        if vr:
            print(f"\nVRAM tiêu thụ: thấp nhất {min(vr)} MiB, cao nhất {max(vr)} MiB")
        thr = [r for r in ok if (r["vram_trong_min"] or 9999) < 300]
        if thr:
            print(f"Sát ngưỡng (còn trống < 300 MiB): {', '.join(r['may'] for r in thr)}")
        p = [r["phut_cho_1h"] for r in ok if r["phut_cho_1h"]]
        if p:
            print(f"Dựng 1 giờ audio: nhanh nhất {min(p):.0f} phút, chậm nhất {max(p):.0f} phút")
        dr = [r for r in ok if (r["troi"] or 1) < 0.85]
        if dr:
            print(f"Tụt tốc độ > 15% từ đầu đến cuối: {', '.join(r['may'] for r in dr)}")

    if args.out:
        cols = ["may", "gpu", "vram_tong", "cc", "driver", "profile", "trang_thai",
                "vram_dung", "vram_trong_min", "xrt", "phut_cho_1h", "giay", "ly_do"]
        t = ["<title>Tổng hợp test GPU</title>",
             "<style>body{font:14px system-ui;margin:2rem;background:#fbfaf8;color:#1c1a17}"
             "table{border-collapse:collapse;width:100%}th,td{padding:.35rem .6rem;"
             "border-bottom:1px solid #e3ded7;text-align:right}th:first-child,td:first-child,"
             "td:nth-child(2){text-align:left}.bad{color:#a8442a;font-weight:600}</style>",
             f"<h1>Tổng hợp test GPU — {len(ok)}/{len(rows)} máy đạt</h1><table><tr>"
             + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"]
        for r in rows:
            cls = "" if r["trang_thai"] == "PASS" else ' class="bad"'
            t.append("<tr>" + "".join(f"<td{cls}>{html.escape(str(r.get(c) if r.get(c) is not None else ''))}</td>"
                                      for c in cols) + "</tr>")
        t.append("</table>")
        Path(args.out).write_text("\n".join(t), encoding="utf-8")
        print(f"\n{args.out}")


if __name__ == "__main__":
    main()
