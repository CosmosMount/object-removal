#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VGGT4D dynamic masks to ProPainter-ready PNG masks.")
    parser.add_argument("--src_dir", required=True, help="VGGT4D scene output directory containing dynamic_mask_*.png.")
    parser.add_argument("--dst_dir", required=True, help="Destination directory for frame-indexed masks (00000.png...).")
    parser.add_argument("--frame_dir", required=False, default="", help="Optional frame directory to align mask size to each frame.")
    parser.add_argument("--threshold", type=int, default=0, help="Binary threshold on grayscale mask values.")
    return parser.parse_args()


def extract_idx(name: str) -> int:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    mask_files = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.name.startswith("dynamic_mask_") and p.suffix.lower() == ".png"],
        key=lambda p: extract_idx(p.name),
    )
    if not mask_files:
        raise RuntimeError(f"No dynamic masks found in {src_dir}")

    frame_files = []
    if args.frame_dir:
        frame_dir = Path(args.frame_dir)
        if frame_dir.is_dir():
            frame_files = sorted(
                [
                    p for p in frame_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ],
                key=lambda p: extract_idx(p.name),
            )

    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for src_path in mask_files:
        idx = extract_idx(src_path.name)
        if idx < 0:
            continue
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if frame_files and idx < len(frame_files):
            ref = cv2.imread(str(frame_files[idx]), cv2.IMREAD_COLOR)
            if ref is not None:
                h, w = ref.shape[:2]
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.zeros_like(img, dtype=np.uint8)
        out[img > args.threshold] = 255
        dst_path = dst_dir / f"{idx:05d}.png"
        ok = cv2.imwrite(str(dst_path), out)
        if ok:
            written += 1

    if written == 0:
        raise RuntimeError("Failed to write any converted masks")

    print(f"Converted {written} masks to {dst_dir}")


if __name__ == "__main__":
    main()
