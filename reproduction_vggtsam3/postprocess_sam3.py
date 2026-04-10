#!/usr/bin/env python3
import argparse
import os
import shutil

import numpy as np
from PIL import Image


def convert_masks(raw_mask_dir: str, out_mask_dir: str) -> None:
    os.makedirs(out_mask_dir, exist_ok=True)
    for name in sorted(os.listdir(raw_mask_dir)):
        if not name.lower().endswith(".png"):
            continue
        arr = np.array(Image.open(os.path.join(raw_mask_dir, name)))
        if arr.ndim > 2:
            arr = arr[..., 0]
        out = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(out, mode="L").save(os.path.join(out_mask_dir, name))


def _pick_indices(total: int, num: int):
    if total <= 0:
        return []
    if total <= num:
        return list(range(total))
    return np.linspace(0, total - 1, num, dtype=int).tolist()


def render_seg_demo(frame_dir: str, mask_dir: str, out_dir: str, num: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))])
    idxs = _pick_indices(len(frames), num)

    for k, idx in enumerate(idxs, 1):
        frame_name = frames[idx]
        mask_name = os.path.splitext(frame_name)[0] + ".png"

        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert("RGB"))
        m = np.array(Image.open(os.path.join(mask_dir, mask_name)).convert("L")) > 0

        color = np.zeros_like(img)
        color[..., 0] = 255
        alpha = 0.45
        overlay = img.copy()
        overlay[m] = ((1 - alpha) * img[m] + alpha * color[m]).astype(np.uint8)

        out_name = f"seg_{k:02d}_{idx:05d}.jpg"
        Image.fromarray(overlay).save(os.path.join(out_dir, out_name))


def render_mask_compare(old_dir: str, new_dir: str, out_dir: str, num: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([x for x in os.listdir(new_dir) if x.lower().endswith(".png")])
    idxs = _pick_indices(len(names), num)

    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        new_mask = np.array(Image.open(os.path.join(new_dir, name)).convert("L")) > 0

        old_path = os.path.join(old_dir, name)
        if os.path.isfile(old_path):
            old_mask = np.array(Image.open(old_path).convert("L")) > 0
            if old_mask.shape != new_mask.shape:
                old_mask = np.array(
                    Image.fromarray((old_mask.astype(np.uint8) * 255), mode="L").resize(
                        (new_mask.shape[1], new_mask.shape[0]), resample=Image.NEAREST
                    )
                ) > 0
        else:
            old_mask = np.zeros_like(new_mask, dtype=bool)

        both = old_mask & new_mask
        add = new_mask & (~old_mask)
        rem = old_mask & (~new_mask)

        old_u8 = old_mask.astype(np.uint8) * 255
        new_u8 = new_mask.astype(np.uint8) * 255
        old_rgb = np.stack([old_u8] * 3, axis=-1)
        new_rgb = np.stack([new_u8] * 3, axis=-1)

        diff = np.zeros((old_u8.shape[0], old_u8.shape[1], 3), dtype=np.uint8)
        diff[both] = [255, 255, 255]
        diff[add] = [0, 255, 0]
        diff[rem] = [255, 0, 0]

        panel = np.concatenate([old_rgb, new_rgb, diff], axis=1)
        out_name = f"compare_{k:02d}_{idx:05d}.png"
        Image.fromarray(panel).save(os.path.join(out_dir, out_name))


def export_inpaint_5(frame_dir: str, out_dir: str, num: int = 5) -> None:
    if not os.path.isdir(frame_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith((".png", ".jpg", ".jpeg"))])
    idxs = _pick_indices(len(names), num)
    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        out_name = f"inpaint_{k:02d}_{idx:05d}{os.path.splitext(name)[1]}"
        shutil.copy2(os.path.join(frame_dir, name), os.path.join(out_dir, out_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess SAM3 masks and render demos.")
    parser.add_argument("--raw_mask_dir", required=True)
    parser.add_argument("--new_mask_dir", required=True)
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--old_mask_dir", required=True)
    parser.add_argument("--seg_demo_dir", required=True)
    parser.add_argument("--mask_compare_dir", required=True)
    parser.add_argument("--inpaint_frames_dir", required=True)
    parser.add_argument("--inpaint_5_dir", required=True)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    convert_masks(args.raw_mask_dir, args.new_mask_dir)
    render_seg_demo(args.frame_dir, args.new_mask_dir, args.seg_demo_dir, num=args.num)
    render_mask_compare(args.old_mask_dir, args.new_mask_dir, args.mask_compare_dir, num=args.num)
    export_inpaint_5(args.inpaint_frames_dir, args.inpaint_5_dir, num=args.num)

    print("new_mask_dir:", args.new_mask_dir)
    print("seg_demo_dir:", args.seg_demo_dir)
    print("mask_compare_dir:", args.mask_compare_dir)
    if os.path.isdir(args.inpaint_frames_dir):
        print("inpaint_5_dir:", args.inpaint_5_dir)


if __name__ == "__main__":
    main()
