#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import tempfile

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


def export_mask_video(frame_dir: str, mask_dir: str, out_path: str, fps: float = 10.0, alpha: float = 0.5) -> None:
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith('.jpg') or x.lower().endswith('.png')])
    if not frames:
        print(f'[export_mask_video] No frames found in {frame_dir}')
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mask_names = set([x for x in os.listdir(mask_dir) if x.lower().endswith('.png')])

    video_frames = []
    for frame_name in frames:
        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert('RGB'))
        ih, iw = img.shape[:2]

        mask_name = os.path.splitext(frame_name)[0] + '.png'
        if mask_name in mask_names:
            m = _load_mask_bool(os.path.join(mask_dir, mask_name), target_hw=(ih, iw))
        else:
            m = np.zeros((ih, iw), dtype=bool)

        color_mask = np.zeros_like(img)
        color_mask[m] = [255, 0, 0]

        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
        video_frames.append(overlay)

    if not video_frames:
        print(f'[export_mask_video] No frames to write')
        return

    h, w = video_frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, '%05d.jpg')
        for i, frame in enumerate(video_frames):
            cv2.imwrite(os.path.join(tmpdir, f'{i:05d}.jpg'), frame)

        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f'[export_mask_video] Saved to {out_path} ({len(video_frames)} frames @ {fps} fps)')
        except subprocess.CalledProcessError as e:
            print(f'[export_mask_video] ffmpeg failed: {e.stderr.decode() if e.stderr else e}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()
            print(f'[export_mask_video] Saved to {out_path} (fallback to mp4v)')


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
    parser.add_argument('--mask_video_path', type=str, default=None, help='Path to export mask overlay video')
    parser.add_argument('--mask_video_fps', type=float, default=10.0, help='FPS for mask video export')
    parser.add_argument('--mask_video_alpha', type=float, default=0.5, help='Alpha for mask overlay')
    args = parser.parse_args()

    render_seg_demo(args.frame_dir, args.new_mask_dir, args.seg_demo_dir, num=args.num)
    render_mask_compare(args.old_mask_dir, args.new_mask_dir, args.mask_compare_dir, num=args.num)
    export_inpaint_5(args.inpaint_frames_dir, args.inpaint_5_dir, num=args.num)

    print('seg_demo_dir:', args.seg_demo_dir)
    print('mask_compare_dir:', args.mask_compare_dir)
    if os.path.isdir(args.inpaint_frames_dir):
        print('inpaint_5_dir:', args.inpaint_5_dir)

    if args.mask_video_path:
        export_mask_video(args.frame_dir, args.new_mask_dir, args.mask_video_path,
                         fps=args.mask_video_fps, alpha=args.mask_video_alpha)


if __name__ == '__main__':
    main()
