#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image


def convert_masks(raw_mask_dir: str, out_mask_dir: str) -> None:
    os.makedirs(out_mask_dir, exist_ok=True)
    for name in sorted(os.listdir(raw_mask_dir)):
        if not name.lower().endswith('.png'):
            continue
        arr = np.array(Image.open(os.path.join(raw_mask_dir, name)))
        out = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(out, mode='L').save(os.path.join(out_mask_dir, name))


def render_seg_demo(frame_dir: str, mask_dir: str, out_dir: str, num: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.jpg')])
    idxs = np.linspace(0, len(frames) - 1, num, dtype=int)

    for k, idx in enumerate(idxs, 1):
        frame_name = frames[idx]
        mask_name = frame_name.rsplit('.', 1)[0] + '.png'

        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert('RGB'))
        m = np.array(Image.open(os.path.join(mask_dir, mask_name)).convert('L')) > 0

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
    idxs = np.linspace(0, len(names) - 1, num, dtype=int)

    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        old_path = os.path.join(old_dir, name)
        if os.path.isfile(old_path):
            old_mask = np.array(Image.open(old_path).convert('L')) > 0
        else:
            # If old mask is unavailable for this frame, treat it as empty.
            new_shape = np.array(Image.open(os.path.join(new_dir, name)).convert('L')).shape
            old_mask = np.zeros(new_shape, dtype=bool)
        new_mask = np.array(Image.open(os.path.join(new_dir, name)).convert('L')) > 0

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
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.png')])
    idxs = np.linspace(0, len(names) - 1, num, dtype=int)
    for k, idx in enumerate(idxs, 1):
        name = names[idx]
        out_name = f'inpaint_{k:02d}_{idx:05d}.png'
        shutil.copy2(os.path.join(frame_dir, name), os.path.join(out_dir, out_name))


def export_mask_video(frame_dir: str, mask_dir: str, out_path: str, fps: float = 10.0, alpha: float = 0.5) -> None:
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.jpg')])
    if not frames:
        print(f'[export_mask_video] No frames found in {frame_dir}')
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mask_names = set([x for x in os.listdir(mask_dir) if x.lower().endswith('.png')])

    video_frames = []
    for frame_name in frames:
        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert('RGB'))

        base_name = frame_name.rsplit('.', 1)[0] + '.png'
        if base_name in mask_names:
            m = np.array(Image.open(os.path.join(mask_dir, base_name)).convert('L')) > 0
        else:
            m = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

        color_mask = np.zeros_like(img)
        color_mask[m] = [255, 0, 0]

        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
        video_frames.append(overlay)

    if not video_frames:
        print(f'[export_mask_video] No frames to write')
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(video_frames):
            cv2.imwrite(os.path.join(tmpdir, f'{i:05d}.jpg'), frame)

        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(tmpdir, '%05d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f'[export_mask_video] Saved to {out_path} ({len(video_frames)} frames @ {fps} fps)')
        except subprocess.CalledProcessError as e:
            print(f'[export_mask_video] ffmpeg failed: {e.stderr.decode() if e.stderr else e}')
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()
            print(f'[export_mask_video] Saved to {out_path} (fallback to mp4v)')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output_root', default='')
    parser.add_argument('--video_name', default='bmx-trees')
    parser.add_argument('--frame_dir', default='')
    parser.add_argument('--old_mask_dir', default='')
    parser.add_argument('--mask_video_path', type=str, default=None, help='Path to export mask overlay video')
    parser.add_argument('--mask_video_fps', type=float, default=10.0, help='FPS for mask video export')
    parser.add_argument('--mask_video_alpha', type=float, default=0.5, help='Alpha for mask overlay')
    args = parser.parse_args()

    root = args.root_dir
    video = args.video_name
    output_root = args.output_root if args.output_root else os.path.join(root, 'outputs')

    raw_mask_dir = os.path.join(output_root, 'tmp_sam2_masks_raw', video)
    new_mask_dir = os.path.join(output_root, f'{video}_mask_sam2')
    frame_dir = args.frame_dir if args.frame_dir else os.path.join(root, 'inputs', video)
    old_mask_dir = args.old_mask_dir if args.old_mask_dir else os.path.join(root, 'inputs', f'{video}_mask')

    vis_root = os.path.join(output_root, f'{video}_sam2_vis')
    seg_demo_dir = os.path.join(vis_root, 'seg_demo')
    compare_dir = os.path.join(vis_root, 'mask_compare')
    inpaint_5_dir = os.path.join(vis_root, 'inpaint_5frames')

    convert_masks(raw_mask_dir, new_mask_dir)
    render_seg_demo(frame_dir, new_mask_dir, seg_demo_dir, num=5)
    render_mask_compare(old_mask_dir, new_mask_dir, compare_dir, num=5)

    inpaint_frames_dir = os.path.join(output_root, f'{video}_propainter', video, 'frames')
    if os.path.isdir(inpaint_frames_dir):
        export_inpaint_5(inpaint_frames_dir, inpaint_5_dir, num=5)

    print('new_mask_dir:', new_mask_dir)
    print('seg_demo_dir:', seg_demo_dir)
    print('mask_compare_dir:', compare_dir)
    if os.path.isdir(inpaint_frames_dir):
        print('inpaint_5_dir:', inpaint_5_dir)

    if args.mask_video_path:
        export_mask_video(frame_dir, new_mask_dir, args.mask_video_path,
                         fps=args.mask_video_fps, alpha=args.mask_video_alpha)


if __name__ == '__main__':
    main()
