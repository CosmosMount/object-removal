#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate indexed init mask from VGGT4D dynamic masks.")
    parser.add_argument("--vggt_scene_output", required=True, help="VGGT scene output directory containing dynamic_mask_*.png.")
    parser.add_argument("--output_dir", required=True, help="Output directory for indexed init mask.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold for dynamic mask.")
    return parser.parse_args()


def _extract_idx(name: str) -> int:
    stem = os.path.splitext(name)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 10**9


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.vggt_scene_output):
        raise FileNotFoundError(f"VGGT scene output dir not found: {args.vggt_scene_output}")

    candidates = [
        n for n in os.listdir(args.vggt_scene_output)
        if n.startswith("dynamic_mask_") and n.lower().endswith(".png")
    ]
    if not candidates:
        raise RuntimeError(f"No dynamic_mask_*.png found in {args.vggt_scene_output}")

    best_name = None
    best_area = -1
    best_indexed = None
    thr = int(args.threshold)

    for name in sorted(candidates, key=_extract_idx):
        path = os.path.join(args.vggt_scene_output, name)
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        indexed = np.zeros_like(arr, dtype=np.uint8)
        indexed[arr > thr] = 1
        area = int((indexed > 0).sum())
        if area > best_area:
            best_area = area
            best_name = name
            best_indexed = indexed

    if best_name is None or best_indexed is None:
        raise RuntimeError(f"Failed to read any dynamic mask from {args.vggt_scene_output}")

    best_idx = _extract_idx(best_name)
    out_name = f"{best_idx:05d}.png"
    os.makedirs(args.output_dir, exist_ok=True)
    output_mask = os.path.join(args.output_dir, out_name)
    ok = cv2.imwrite(output_mask, best_indexed)
    if not ok:
        raise RuntimeError(f"Failed to write indexed init mask: {output_mask}")

    print(
        f"Saved indexed init mask from max-area frame {best_name} -> {output_mask}; "
        f"foreground={best_area}"
    )


if __name__ == "__main__":
    main()
