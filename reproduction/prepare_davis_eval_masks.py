#!/usr/bin/env python3
import argparse
import os

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare indexed DAVIS evaluation masks.")
    parser.add_argument("--src_dir", required=True, help="Source mask directory.")
    parser.add_argument("--dst_dir", required=True, help="Destination DAVIS result directory.")
    parser.add_argument("--max_eval_labels", type=int, default=20, help="Max kept foreground labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    max_eval_labels = max(1, min(int(args.max_eval_labels), 255))

    png_names = sorted([n for n in os.listdir(src_dir) if n.lower().endswith(".png")])
    if not png_names:
        raise RuntimeError(f"No PNG masks found in source dir: {src_dir}")

    unique_labels = set()
    for name in png_names:
        arr = np.array(Image.open(os.path.join(src_dir, name)))
        if arr.ndim > 2:
            arr = arr[..., 0]
        vals = np.unique(arr)
        unique_labels.update(int(v) for v in vals if int(v) > 0)

    sorted_labels = sorted(unique_labels)
    if len(sorted_labels) > max_eval_labels:
        sorted_labels = sorted_labels[:max_eval_labels]

    label_map = {raw_v: i + 1 for i, raw_v in enumerate(sorted_labels)}

    os.makedirs(dst_dir, exist_ok=True)
    for name in png_names:
        arr = np.array(Image.open(os.path.join(src_dir, name)))
        if arr.ndim > 2:
            arr = arr[..., 0]
        out = np.zeros(arr.shape, dtype=np.uint8)
        for raw_v, mapped_v in label_map.items():
            out[arr == raw_v] = mapped_v
        Image.fromarray(out, mode="L").save(os.path.join(dst_dir, name))

    first_frame_path = os.path.join(dst_dir, "00000.png")
    if not os.path.isfile(first_frame_path):
        pngs = sorted([x for x in os.listdir(dst_dir) if x.lower().endswith(".png")])
        if pngs:
            Image.open(os.path.join(dst_dir, pngs[0])).save(first_frame_path)

    print(f"Prepared DAVIS masks in: {dst_dir}; mapped_labels={len(label_map)}")


if __name__ == "__main__":
    main()
