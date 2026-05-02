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


def export_mask_video(frame_dir: str, mask_dir: str, out_path: str, fps: float = 10.0, alpha: float = 0.5) -> None:
    frames = sorted([x for x in os.listdir(frame_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not frames:
        print(f"[export_mask_video] No frames found in {frame_dir}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mask_names = set([x for x in os.listdir(mask_dir) if x.lower().endswith(".png")])

    video_frames = []
    for frame_name in frames:
        img = np.array(Image.open(os.path.join(frame_dir, frame_name)).convert("RGB"))

        base_name = os.path.splitext(frame_name)[0] + ".png"
        if base_name in mask_names:
            m = np.array(Image.open(os.path.join(mask_dir, base_name)).convert("L")) > 0
        else:
            m = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

        color_mask = np.zeros_like(img)
        color_mask[m] = [255, 0, 0]

        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
        video_frames.append(overlay)

    if not video_frames:
        print(f"[export_mask_video] No frames to write")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(video_frames):
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), frame)

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "%05d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[export_mask_video] Saved to {out_path} ({len(video_frames)} frames @ {fps} fps)")
        except subprocess.CalledProcessError as e:
            print(f"[export_mask_video] ffmpeg failed: {e.stderr.decode() if e.stderr else e}")
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for frame in video_frames:
                out.write(frame)
            out.release()
            print(f"[export_mask_video] Saved to {out_path} (fallback to mp4v)")


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
    parser.add_argument("--mask_video_path", type=str, default=None, help="Path to export mask overlay video")
    parser.add_argument("--mask_video_fps", type=float, default=10.0, help="FPS for mask video export")
    parser.add_argument("--mask_video_alpha", type=float, default=0.5, help="Alpha for mask overlay")
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

    if args.mask_video_path:
        export_mask_video(args.frame_dir, args.new_mask_dir, args.mask_video_path,
                         fps=args.mask_video_fps, alpha=args.mask_video_alpha)


if __name__ == "__main__":
    main()
