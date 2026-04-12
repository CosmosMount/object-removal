import os

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
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frame_list:
        out.write(frame)
    out.release()


def build_side_by_side(original_frames, result_frames, height):
    return [
        np.hstack([orig, np.full((height, 4, 3), 60, np.uint8), result])
        for orig, result in zip(original_frames, result_frames)
    ]
