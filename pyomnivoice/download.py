"""Download the OmniVoice GGUFs into omnivoice.cpp/models/.

    python -m pyomnivoice.download            # lite + quality pairs
    python -m pyomnivoice.download --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = "Serveurperso/OmniVoice-GGUF"

FILES = {
    "lite": ["omnivoice-base-Q4_K_M.gguf", "omnivoice-tokenizer-Q8_0.gguf"],
    "balanced": ["omnivoice-base-Q8_0.gguf", "omnivoice-tokenizer-Q8_0.gguf"],
    "quality": ["omnivoice-base-Q8_0.gguf", "omnivoice-tokenizer-F32.gguf"],
    "reference": ["omnivoice-base-BF16.gguf", "omnivoice-tokenizer-F32.gguf"],
}

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "omnivoice.cpp" / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=["lite", "quality"], nargs="*", choices=list(FILES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    profiles = list(FILES) if args.all else args.profile
    wanted = sorted({f for p in profiles for f in FILES[p]})
    out = Path(args.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for f in wanted:
        if (out / f).exists():
            print(f"[ok]   {f}")
            continue
        print(f"[get]  {f}")
        hf_hub_download(REPO, f, local_dir=str(out))
    print(f"\nmodels -> {out}")


if __name__ == "__main__":
    main()
