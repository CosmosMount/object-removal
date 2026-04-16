#!/usr/bin/env python3
import argparse
import os
import glob

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + Optical Flow mask propagation")
    parser.add_argument("--root_dir", required=True, help="Project root directory.")
    parser.add_argument("--video_dir", required=True, help="Directory containing video frames.")
    parser.add_argument("--first_mask_dir", required=True, help="Directory with first-frame mask (indexed).")
    parser.add_argument("--old_mask_dir", required=True, help="Directory with old/provided masks for video mode.")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated masks.")
    parser.add_argument("--video_name", required=True, help="Video sequence name.")
    return parser.parse_args()


def load_frames(video_dir: str):
    frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    frames = []
    for f in frame_files:
        img = cv2.imread(f)
        if img is not None:
            frames.append(img)
    return frames


def load_first_mask(first_mask_dir: str, video_name: str):
    mask_path = os.path.join(first_mask_dir, video_name, "00000.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask
    return None


def load_old_mask(old_mask_dir: str, frame_idx: int):
    mask_path = os.path.join(old_mask_dir, f"{frame_idx:05d}.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask
    return None


def compute_optical_flow_on_mask(prev_gray, curr_gray, mask):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(
        maxCorners=60,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

    if pts is None or len(pts) < 3:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk_params)

    good_prev = pts[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) < 3:
        return None

    return good_next - good_prev


def filter_dynamic(mask, flow, motion_threshold=1.5):
    if flow is None or len(flow) < 3:
        return np.zeros_like(mask)

    magnitude = np.linalg.norm(flow, axis=1)
    avg_mag = np.mean(magnitude)

    if avg_mag >= motion_threshold:
        return mask
    else:
        return np.zeros_like(mask)


def dilate_mask(mask, kernel_size=15, adaptive=True):
    if mask.max() == 0:
        return mask

    k_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    k_ext = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2, kernel_size * 2))

    if not adaptive:
        return cv2.dilate(mask, k_body, iterations=2)

    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return mask

    y_min, y_max = int(ys.min()), int(ys.max())
    margin = max(int((y_max - y_min) * 0.22), 20)

    h, w = mask.shape[:2]
    top = np.zeros_like(mask)
    top[: y_min + margin, :] = mask[: y_min + margin, :]

    bottom = np.zeros_like(mask)
    bottom[y_max - margin :, :] = mask[y_max - margin :, :]

    mid = mask.copy()
    mid[: y_min + margin, :] = 0
    mid[y_max - margin :, :] = 0

    dilated = cv2.dilate(mid, k_body, iterations=2)
    dilated = np.maximum(dilated, cv2.dilate(top, k_ext, iterations=2))
    dilated = np.maximum(dilated, cv2.dilate(bottom, k_ext, iterations=2))
    return dilated


def save_binary_masks(masks, output_dir, video_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        out_path = os.path.join(output_dir, f"{i:05d}.png")
        binary = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(out_path, binary)


def create_visuals(frames, old_masks, new_masks, video_name, vis_root):
    os.makedirs(vis_root, exist_ok=True)

    seg_demo_dir = os.path.join(vis_root, "seg_demo")
    mask_compare_dir = os.path.join(vis_root, "mask_compare")
    os.makedirs(seg_demo_dir, exist_ok=True)
    os.makedirs(mask_compare_dir, exist_ok=True)

    key_frames = [i for i, mask in enumerate(new_masks) if mask.max() > 0]
    if not key_frames:
        key_frames = [0]

    samples = key_frames[:: max(1, len(key_frames) // 5)][:5]

    for idx in samples:
        if idx >= len(frames):
            continue

        frame = frames[idx]
        old_mask = old_masks[idx] if idx < len(old_masks) else np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        new_mask = new_masks[idx]

        frame_h, frame_w = frame.shape[:2]
        old_mask_resized = cv2.resize(old_mask, (frame_w, frame_h))
        new_mask_resized = cv2.resize(new_mask, (frame_w, frame_h))

        overlay = frame.copy()
        mask_color = np.array([0, 255, 0], dtype=np.uint8)
        mask_overlay = (new_mask_resized > 0).astype(np.uint8)[:, :, np.newaxis]
        overlay = overlay * (1 - mask_overlay) + mask_color * mask_overlay

        seg_path = os.path.join(seg_demo_dir, f"{idx:05d}.jpg")
        cv2.imwrite(seg_path, overlay)

        compare = np.hstack([
            frame,
            cv2.cvtColor(old_mask_resized, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(new_mask_resized, cv2.COLOR_GRAY2BGR)
        ])
        compare_path = os.path.join(mask_compare_dir, f"{idx:05d}.jpg")
        cv2.imwrite(compare_path, compare)

    print(f"   Segmentation demos: {seg_demo_dir}")
    print(f"   Mask comparisons:   {mask_compare_dir}")


def main():
    args = parse_args()

    print("[YOLO+OptFlow] Loading frames...")
    frames = load_frames(args.video_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {args.video_dir}")

    h, w = frames[0].shape[:2]
    print(f"   Frame size: {w}x{h}, frame count: {len(frames)}")

    print("[YOLO+OptFlow] Loading first-frame mask...")
    first_mask = load_first_mask(args.first_mask_dir, args.video_name)
    if first_mask is None:
        first_mask_path = os.path.join(args.first_mask_dir, args.video_name, "00000.png")
        if not os.path.exists(first_mask_path):
            print(f"   First mask not found at {first_mask_path}, using old mask for first frame")
            first_mask = load_old_mask(args.old_mask_dir, 0)

    if first_mask is None:
        first_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        first_mask = cv2.resize(first_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    old_masks = []
    for i in range(len(frames)):
        mask = load_old_mask(args.old_mask_dir, i)
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        old_masks.append(mask)

    print("[YOLO+OptFlow] Running YOLO on all frames + motion filter...")

    from ultralytics import YOLO
    yolo_model = YOLO(os.path.join(args.root_dir, "baseline", "yolov8n-seg.pt"))
    dynamic_classes = [0, 1, 2, 3, 5, 7]

    raw_masks = []
    for i, frame in enumerate(frames):
        result = yolo_model(frame, classes=dynamic_classes, conf=0.25, verbose=False)[0]
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None:
            for seg in result.masks.data:
                seg_map = seg.cpu().numpy()
                seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, (seg_map > 0.5).astype(np.uint8) * 255)
        raw_masks.append(mask)
        if (i + 1) % 30 == 0:
            print(f"   YOLO: {i + 1}/{len(frames)}")

    print(f"   YOLO detections: {sum(m.max() > 0 for m in raw_masks)}/{len(frames)} frames")

    propagated_masks = []
    flow_info = {}
    for i in range(len(frames)):
        mask = raw_masks[i]
        
        if mask.max() == 0 or i == 0:
            dilated_mask = dilate_mask(mask, kernel_size=15)
            propagated_masks.append(dilated_mask)
            continue

        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        flow = compute_optical_flow_on_mask(prev_gray, curr_gray, mask)

        if flow is None or len(flow) < 3:
            dilated_mask = dilate_mask(mask, kernel_size=15)
            propagated_masks.append(dilated_mask)
            continue

        magnitude = np.linalg.norm(flow, axis=1)
        avg_mag = float(np.mean(magnitude))
        flow_info[i] = avg_mag

        if avg_mag >= 1.5:
            dynamic_mask = mask
        else:
            dynamic_mask = np.zeros_like(mask)

        dilated_mask = dilate_mask(dynamic_mask, kernel_size=15)
        propagated_masks.append(dilated_mask)

        if (i + 1) % 30 == 0:
            print(f"   Motion filter: {i + 1}/{len(frames)}")

    dynamic_frames = sum(m.max() > 0 for m in propagated_masks)
    print(f"   Dynamic frames: {dynamic_frames}/{len(frames)} (thr=1.5px)")

    final_masks = propagated_masks

    output_mask_dir = os.path.join(args.output_dir, f"{args.video_name}_mask_optflow")
    print(f"[YOLO+OptFlow] Saving masks to {output_mask_dir}")
    save_binary_masks(final_masks, output_mask_dir, args.video_name)

    vis_root = os.path.join(args.output_dir, f"{args.video_name}_optflow_vis")
    print("[YOLO+OptFlow] Generating visualization...")
    create_visuals(frames, old_masks, final_masks, args.video_name, vis_root)

    print(f"[YOLO+OptFlow] Done. Masks saved to {output_mask_dir}")


if __name__ == "__main__":
    main()