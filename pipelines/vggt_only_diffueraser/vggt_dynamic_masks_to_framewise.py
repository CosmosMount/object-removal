#!/usr/bin/env python3
"""Map VGGT dynamic_mask_0000.png ... to one PNG per video frame basename (for postprocess / DiffuEraser)."""
from __future__ import annotations

import argparse
import os
import re

import cv2
import numpy as np


def _list_frames(frame_dir: str) -> list[str]:
    ex = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    return sorted(f for f in os.listdir(frame_dir) if f.endswith(ex))


def _vggt_mask_count(vggt_dir: str) -> int:
    mx = -1
    for name in os.listdir(vggt_dir):
        if not (name.startswith("dynamic_mask_") and name.lower().endswith(".png")):
            continue
        m = re.match(r"dynamic_mask_(\d+)\.png$", name, re.IGNORECASE)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vggt_scene_output", required=True)
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--output_raw_dir", required=True, help="Directory for PNGs named like frames (e.g. 00000.png).")
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument(
        "--tail_policy",
        choices=("hold_last", "zeros"),
        default="hold_last",
        help="If fewer VGGT masks than frames (VGGT was capped): repeat last mask or empty masks.",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.vggt_scene_output):
        raise FileNotFoundError(args.vggt_scene_output)
    if not os.path.isdir(args.frame_dir):
        raise FileNotFoundError(args.frame_dir)

    os.makedirs(args.output_raw_dir, exist_ok=True)
    for f in os.listdir(args.output_raw_dir):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(args.output_raw_dir, f))

    frames = _list_frames(args.frame_dir)
    num_masks = _vggt_mask_count(args.vggt_scene_output)
    thr = int(args.threshold)

    for i, frame_name in enumerate(frames):
        stem = os.path.splitext(frame_name)[0]
        out_path = os.path.join(args.output_raw_dir, f"{stem}.png")
        frame_path = os.path.join(args.frame_dir, frame_name)
        ref = cv2.imread(frame_path)
        if ref is None:
            raise RuntimeError(f"Cannot read frame: {frame_path}")
        h, w = ref.shape[:2]

        if num_masks == 0:
            out = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(out_path, out)
            continue

        if i < num_masks:
            vidx = i
        elif args.tail_policy == "hold_last":
            vidx = num_masks - 1
        else:
            out = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(out_path, out)
            continue

        vpath = os.path.join(args.vggt_scene_output, f"dynamic_mask_{vidx:04d}.png")
        arr = cv2.imread(vpath, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Cannot read VGGT mask: {vpath}")
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = ((arr > thr).astype(np.uint8)) * 255
        cv2.imwrite(out_path, binary)

    print(
        f"Wrote {len(frames)} masks to {args.output_raw_dir} "
        f"(vggt_masks={num_masks}, tail_policy={args.tail_policy})"
    )


if __name__ == "__main__":
    main()
