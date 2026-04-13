import os
import shutil
import subprocess

import cv2
import numpy as np


def load_video(path, max_frames=None):
    if os.path.isdir(path):
        return load_image_sequence(path, max_frames)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    print(f"[Step 1] Loaded {len(frames)}/{total} frames  ({width}x{height} @ {fps:.1f} fps)")
    return frames, fps, width, height


def load_image_sequence(dir_path, max_frames=None):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = sorted([f for f in os.listdir(dir_path) if os.path.splitext(f.lower())[1] in exts])
    if not files:
        raise FileNotFoundError(f"No image files found in: {dir_path}")

    frames = []
    for fname in files[:max_frames]:
        img = cv2.imread(os.path.join(dir_path, fname))
        if img is None:
            continue
        frames.append(img)

    if not frames:
        raise ValueError(f"Failed to load any images from: {dir_path}")

    height, width = frames[0].shape[:2]
    fps = 25.0
    total = len(files)
    print(f"[Step 1] Loaded {len(frames)}/{total} frames from directory  ({width}x{height} @ {fps:.1f} fps)")
    return frames, fps, width, height


def write_video(frame_list, path, fps, width, height):
    def _open_writer(dst_path, codec_list):
        for codec in codec_list:
            out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if out.isOpened():
                return out, codec
        return None, None

    def _write_frames(out):
        for frame in frame_list:
            out.write(frame)
        out.release()

    def _resolve_ffmpeg_binary():
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    ext = os.path.splitext(path)[1].lower()

    if ext == ".mp4":
        tmp_path = f"{path}.tmp.mp4"
        out, codec = _open_writer(tmp_path, ["mp4v"])
        if out is None:
            raise RuntimeError(f"Cannot open MP4 writer for {path}")
        _write_frames(out)

        ffmpeg_bin = _resolve_ffmpeg_binary()
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-i",
                tmp_path,
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                path,
            ]
            try:
                subprocess.run(cmd, check=True)
                os.remove(tmp_path)
                print(f"[VideoIO] Wrote MP4 using ffmpeg/libx264 (source codec: {codec})")
                return
            except Exception as exc:
                print(f"[VideoIO] ffmpeg re-encode failed, fallback to OpenCV output: {exc}")
        else:
            print("[VideoIO] ffmpeg not found. Install ffmpeg for SSH preview compatibility.")

        os.replace(tmp_path, path)
        print(f"[VideoIO] Wrote MP4 using OpenCV codec: {codec}")
        return

    out, codec = _open_writer(path, ["XVID", "MJPG", "mp4v"])
    if out is None:
        raise RuntimeError(f"Cannot open video writer for {path}")
    _write_frames(out)
    print(f"[VideoIO] Wrote video using OpenCV codec: {codec}")


def build_side_by_side(original_frames, result_frames, height):
    return [
        np.hstack([orig, np.full((height, 4, 3), 60, np.uint8), result])
        for orig, result in zip(original_frames, result_frames)
    ]
