#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DAVIS eval masks with optional multi-object splitting for binary masks.")
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--max_eval_labels", type=int, default=20)
    parser.add_argument("--target_objects", type=int, default=2, help="Target tracked objects when source is binary.")
    parser.add_argument("--min_component_area", type=int, default=120, help="Ignore connected components smaller than this.")
    return parser.parse_args()


def _sorted_pngs(src_dir: str) -> List[str]:
    return sorted([n for n in os.listdir(src_dir) if n.lower().endswith(".png")])


def _read_gray(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def _extract_components(binary: np.ndarray, min_area: int) -> List[Tuple[int, np.ndarray, Tuple[float, float]]]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    comps = []
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = labels == cid
        cx, cy = float(centroids[cid][0]), float(centroids[cid][1])
        comps.append((area, mask, (cx, cy)))
    comps.sort(key=lambda x: x[0], reverse=True)
    return comps


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def split_binary_to_multi(frames: List[np.ndarray], target_objects: int, min_area: int) -> List[np.ndarray]:
    out_frames: List[np.ndarray] = []
    prev_centers: Dict[int, Tuple[float, float]] = {}

    for t, arr in enumerate(frames):
        binary = arr > 0
        comps = _extract_components(binary, min_area=min_area)
        comps = comps[: max(1, target_objects)]

        out = np.zeros(arr.shape, dtype=np.uint8)
        if not comps:
            out_frames.append(out)
            continue

        if t == 0 or not prev_centers:
            comps_sorted = sorted(comps, key=lambda x: x[2][0])
            for i, (_, comp_mask, center) in enumerate(comps_sorted, start=1):
                if i > target_objects:
                    break
                out[comp_mask] = i
                prev_centers[i] = center
            out_frames.append(out)
            continue

        assigned_curr = set()
        new_prev: Dict[int, Tuple[float, float]] = {}
        for obj_id in range(1, target_objects + 1):
            if obj_id not in prev_centers:
                continue
            best_idx = -1
            best_d = 1e30
            for j, (_, _, center) in enumerate(comps):
                if j in assigned_curr:
                    continue
                d = _dist(prev_centers[obj_id], center)
                if d < best_d:
                    best_d = d
                    best_idx = j
            if best_idx >= 0:
                _, comp_mask, center = comps[best_idx]
                out[comp_mask] = obj_id
                new_prev[obj_id] = center
                assigned_curr.add(best_idx)

        next_label = 1
        for j, (_, comp_mask, center) in enumerate(comps):
            if j in assigned_curr:
                continue
            while next_label in new_prev and next_label <= target_objects:
                next_label += 1
            if next_label > target_objects:
                break
            out[comp_mask] = next_label
            new_prev[next_label] = center
            next_label += 1

        prev_centers = new_prev
        out_frames.append(out)

    return out_frames


def remap_existing_labels(frames: List[np.ndarray], max_eval_labels: int) -> List[np.ndarray]:
    unique_labels = set()
    for arr in frames:
        vals = np.unique(arr)
        unique_labels.update(int(v) for v in vals if int(v) > 0)

    sorted_labels = sorted(unique_labels)
    if len(sorted_labels) > max_eval_labels:
        sorted_labels = sorted_labels[:max_eval_labels]

    label_map = {raw_v: i + 1 for i, raw_v in enumerate(sorted_labels)}
    out_frames: List[np.ndarray] = []
    for arr in frames:
        out = np.zeros(arr.shape, dtype=np.uint8)
        for raw_v, mapped_v in label_map.items():
            out[arr == raw_v] = mapped_v
        out_frames.append(out)
    return out_frames


def main() -> None:
    args = parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    max_eval_labels = max(1, min(int(args.max_eval_labels), 255))
    target_objects = max(1, min(int(args.target_objects), max_eval_labels))

    png_names = _sorted_pngs(src_dir)
    if not png_names:
        raise RuntimeError(f"No PNG masks found in source dir: {src_dir}")

    frames = [_read_gray(os.path.join(src_dir, n)) for n in png_names]

    all_labels = set()
    for arr in frames:
        all_labels.update(int(v) for v in np.unique(arr) if int(v) > 0)

    if len(all_labels) <= 1:
        out_frames = split_binary_to_multi(frames, target_objects=target_objects, min_area=args.min_component_area)
        mode = f"binary->multi(target={target_objects})"
    else:
        out_frames = remap_existing_labels(frames, max_eval_labels=max_eval_labels)
        mode = "remap-existing"

    os.makedirs(dst_dir, exist_ok=True)
    for name, out in zip(png_names, out_frames):
        Image.fromarray(out, mode="L").save(os.path.join(dst_dir, name))

    first_frame_path = os.path.join(dst_dir, "00000.png")
    if not os.path.isfile(first_frame_path):
        if png_names:
            Image.open(os.path.join(dst_dir, png_names[0])).save(first_frame_path)

    label_set = set()
    for arr in out_frames:
        label_set.update(int(v) for v in np.unique(arr) if int(v) > 0)

    print(f"Prepared DAVIS masks in: {dst_dir}; mode={mode}; mapped_labels={len(label_set)}; labels={sorted(label_set)}")


if __name__ == "__main__":
    main()
