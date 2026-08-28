"""SubRip (.srt) parsing and timeline assembly.

Mirrors the --srt path of the omnivoice-tts CLI: every cue is synthesised
single-shot with T_override set to its slot length and postproc off, so the
raw decode lands at exactly the slot duration, then placed on an absolute
timeline that muxes straight onto the source video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_ARROW = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")


@dataclass
class Cue:
    index: int
    t0: float
    t1: float
    text: str

    @property
    def slot(self) -> float:
        return self.t1 - self.t0


def _stamp(h: str, m: str, s: str, ms: str) -> float:
    frac = float(ms) / (10 ** len(ms)) if ms else 0.0
    return int(h) * 3600 + int(m) * 60 + int(s) + frac


def parse_srt(text: str) -> list[Cue]:
    """Tolerant of CRLF, a UTF-8 BOM, missing index lines, '.' or ',' as the
    millisecond separator, and multi-line cue text (joined with a space)."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = 0
        if lines[0].isdigit() and len(lines) > 1 and _ARROW.search(lines[1]):
            idx = int(lines[0])
            lines = lines[1:]
        m = _ARROW.search(lines[0]) if lines else None
        if not m:
            continue
        g = m.groups()
        t0 = _stamp(*g[0:4])
        t1 = _stamp(*g[4:8])
        body = " ".join(lines[1:]).strip()
        if body and t1 > t0:
            cues.append(Cue(index=idx or len(cues) + 1, t0=t0, t1=t1, text=body))
    cues.sort(key=lambda c: c.t0)
    return cues


def read_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return parse_srt(raw)


def assemble(
    segments: list[tuple[Cue, np.ndarray]],
    sample_rate: int,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Place each synthesised segment at its cue start on a zero timeline.

    A segment is clipped at the next cue's start so an overlapping source
    stamp never bleeds into the following line; a short raised-cosine fade on
    both edges kills the click the raw (postproc-off) decode leaves behind.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    n_total = int(round(max(c.t1 for c, _ in segments) * sample_rate))
    timeline = np.zeros(n_total, dtype=np.float32)
    fade_n = int(sample_rate * fade_ms / 1000.0)

    starts = [int(round(c.t0 * sample_rate)) for c, _ in segments]
    for i, (cue, seg) in enumerate(segments):
        off = starts[i]
        limit = starts[i + 1] if i + 1 < len(starts) else n_total
        limit = min(limit, n_total)
        if off >= limit or seg.size == 0:
            continue
        n = min(len(seg), limit - off)
        chunk = seg[:n].astype(np.float32, copy=True)
        if fade_n > 0 and n > 2 * fade_n:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_n) / fade_n)
            chunk[:fade_n] *= ramp
            chunk[-fade_n:] *= ramp[::-1]
        timeline[off : off + n] = chunk
    return timeline
