#!/usr/bin/env python3
import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image


def _safe_pick_indices(n: int, num: int):
    if n <= 0:
        return []
    if n <= num:
        return list(range(n))
    return np.linspace(0, n - 1, num, dtype=int).tolist()


def _load_mask_bool(mask_path: str, target_hw=None) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert('L')) > 0
    if target_hw is not None:
        th, tw = target_hw
        if mask.shape != (th, tw):
            mask = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def render_seg_demo(frame_dir: str, mask_dir: str, out_dir: str, num: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.jpg') or x.lower().endswith('.png')])
    idxs = _safe_pick_indices(len(frames), num)

    for k, idx in enumerate(idxs, 1):
        frame_name = frames[idx]
        mask_name = os.path.splitext(frame_name)[0] + '.png'

        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert('RGB'))
        ih, iw = img.shape[:2]
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.isfile(mask_path):
            m = _load_mask_bool(mask_path, target_hw=(ih, iw))
        else:
            m = np.zeros((ih, iw), dtype=bool)

        color = np.zeros_like(img)
        color[..., 0] = 255
        alpha = 0.45
        overlay = img.copy()
        overlay[m] = ((1 - alpha) * img[m] + alpha * color[m]).astype(np.uint8)

        out_name = f'seg_{k:02d}_{idx:05d}.jpg'
        Image.fromarray(overlay).save(os.path.join(out_dir, out_name))


def render_mask_compare(old_dir: str, new_dir: str, out_dir: str, num: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([x for x in os.listdir(new_dir) if x.lower().endswith('.png')])
    idxs = _safe_pick_indices(len(names), num)

    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        new_mask = _load_mask_bool(os.path.join(new_dir, name))

        old_path = os.path.join(old_dir, name)
        if os.path.isfile(old_path):
            old_mask = _load_mask_bool(old_path, target_hw=new_mask.shape)
        else:
            old_mask = np.zeros_like(new_mask, dtype=bool)

        both = old_mask & new_mask
        add = new_mask & (~old_mask)
        rem = old_mask & (~new_mask)

        old_u8 = (old_mask.astype(np.uint8) * 255)
        new_u8 = (new_mask.astype(np.uint8) * 255)
        old_rgb = np.stack([old_u8] * 3, axis=-1)
        new_rgb = np.stack([new_u8] * 3, axis=-1)

        diff = np.zeros((old_u8.shape[0], old_u8.shape[1], 3), dtype=np.uint8)
        diff[both] = [255, 255, 255]
        diff[add] = [0, 255, 0]
        diff[rem] = [255, 0, 0]

        panel = np.concatenate([old_rgb, new_rgb, diff], axis=1)
        out_name = f'compare_{k:02d}_{idx:05d}.png'
        Image.fromarray(panel).save(os.path.join(out_dir, out_name))


def export_inpaint_5(frame_dir: str, out_dir: str, num: int = 5) -> None:
    if not os.path.isdir(frame_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.png') or x.lower().endswith('.jpg')])
    idxs = _safe_pick_indices(len(names), num)
    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        out_name = f'inpaint_{k:02d}_{idx:05d}{os.path.splitext(name)[1]}'
        shutil.copy2(os.path.join(frame_dir, name), os.path.join(out_dir, out_name))


def main() -> None:
    parser = argparse.ArgumentParser(description='Render visualization outputs for VGGT masks and inpainting.')
    parser.add_argument('--frame_dir', required=True)
    parser.add_argument('--new_mask_dir', required=True)
    parser.add_argument('--old_mask_dir', required=True)
    parser.add_argument('--seg_demo_dir', required=True)
    parser.add_argument('--mask_compare_dir', required=True)
    parser.add_argument('--inpaint_frames_dir', required=True)
    parser.add_argument('--inpaint_5_dir', required=True)
    parser.add_argument('--num', type=int, default=5)
    args = parser.parse_args()

    render_seg_demo(args.frame_dir, args.new_mask_dir, args.seg_demo_dir, num=args.num)
    render_mask_compare(args.old_mask_dir, args.new_mask_dir, args.mask_compare_dir, num=args.num)
    export_inpaint_5(args.inpaint_frames_dir, args.inpaint_5_dir, num=args.num)

    print('seg_demo_dir:', args.seg_demo_dir)
    print('mask_compare_dir:', args.mask_compare_dir)
    if os.path.isdir(args.inpaint_frames_dir):
        print('inpaint_5_dir:', args.inpaint_5_dir)


if __name__ == '__main__':
    main()
