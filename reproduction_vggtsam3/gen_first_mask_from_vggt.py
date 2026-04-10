#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate indexed first-frame mask from VGGT4D dynamic mask.")
    parser.add_argument("--vggt_scene_output", required=True, help="VGGT scene output directory containing dynamic_mask_*.png.")
    parser.add_argument("--output_mask", required=True, help="Output indexed PNG path (0 background, 1 foreground).")
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

    first_name = sorted(candidates, key=_extract_idx)[0]
    first_path = os.path.join(args.vggt_scene_output, first_name)
    arr = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Failed to read dynamic mask: {first_path}")

    indexed = np.zeros_like(arr, dtype=np.uint8)
    indexed[arr > int(args.threshold)] = 1

    os.makedirs(os.path.dirname(args.output_mask), exist_ok=True)
    ok = cv2.imwrite(args.output_mask, indexed)
    if not ok:
        raise RuntimeError(f"Failed to write indexed first-frame mask: {args.output_mask}")

    print(
        f"Saved first-frame indexed mask from {first_name} -> {args.output_mask}; "
        f"foreground={int((indexed > 0).sum())}"
    )


if __name__ == "__main__":
    main()
