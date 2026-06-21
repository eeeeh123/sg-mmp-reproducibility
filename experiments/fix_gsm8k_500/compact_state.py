"""Compact oversized quantization state files by narrowing redundant dtypes.

This does not change quantization values. GPTQ 4-bit codes are stored as int8
instead of int64, which avoids large CPU/GPU transfer peaks on Windows.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import torch


def compact_value(key: str, value):
    if not hasattr(value, "to"):
        return value
    if key == "w_q":
        return value.to(torch.int8).contiguous()
    return value.contiguous()


def compact_file(src: Path, dst: Path) -> None:
    t0 = time.time()
    gb_src = os.path.getsize(src) / 1024**3
    if gb_src > 6 and os.environ.get("PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD") != "1":
        raise SystemExit(
            f"Refusing to load {src} ({gb_src:.2f} GB). This oversized legacy "
            "state is known to trigger native PyTorch/Windows access violations. "
            "Regenerate the quantized state with the patched GPTQ int8 code "
            "instead of compacting this file. Set PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD=1 "
            "only if you accept the crash risk."
        )
    print(f"[load] {src}", flush=True)
    state = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    out = {}
    total = len(state)
    for i, (name, qi) in enumerate(state.items(), 1):
        out[name] = {key: compact_value(key, value) for key, value in qi.items()}
        if i % 20 == 0 or i == total:
            print(f"[compact] {i}/{total}", flush=True)
            gc.collect()
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] {dst}", flush=True)
    torch.save(out, dst)
    gb = os.path.getsize(dst) / 1024**3
    print(f"[done] {dst} {gb:.2f} GB in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    compact_file(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
