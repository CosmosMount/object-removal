#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first-frame indexed mask for SAM2 initialization.")
    parser.add_argument("--first_frame", required=True, help="Path to first RGB frame.")
    parser.add_argument("--model_path", required=True, help="Path to YOLO segmentation model file.")
    parser.add_argument("--output_mask", required=True, help="Output PNG path for indexed mask.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--max_init_objects", type=int, default=4, help="Maximum number of initialized objects.")
    parser.add_argument(
        "--classes",
        default="0,1,2,3,5,7",
        help="Comma-separated class IDs to keep.",
    )
    return parser.parse_args()


def _safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _align_mask_to_frame(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    # Ultralytics may output masks at inference resolution; restore to frame size.
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def main() -> None:
    args = parse_args()

    frame = cv2.imread(args.first_frame)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {args.first_frame}")

    classes = [int(x) for x in args.classes.split(",") if x.strip() != ""]
    h, w = frame.shape[:2]
    model = YOLO(args.model_path)
    res = model(frame, classes=classes, conf=args.conf, verbose=False)[0]

    indexed_mask = np.zeros((h, w), dtype=np.uint8)
    max_init_objects = max(1, min(int(args.max_init_objects), 255))

    if res.masks is not None and len(res.masks.data) > 0:
        masks = res.masks.data.cpu().numpy() > 0.5
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        order = np.argsort(-areas)
        kept = 0
        for idx in order:
            if kept >= max_init_objects:
                break
            m = _align_mask_to_frame(masks[idx], h, w)
            if m.sum() == 0:
                continue
            write_region = np.logical_and(m, indexed_mask == 0)
            if write_region.sum() == 0:
                continue
            indexed_mask[write_region] = kept + 1
            kept += 1
    elif res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order = np.argsort(-areas)
        kept = 0
        for idx in order:
            if kept >= max_init_objects:
                break
            x1, y1, x2, y2 = _safe_box(*boxes_xyxy[idx], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            region = indexed_mask[y1:y2, x1:x2]
            region[region == 0] = kept + 1
            indexed_mask[y1:y2, x1:x2] = region
            kept += 1

    os.makedirs(os.path.dirname(args.output_mask), exist_ok=True)
    ok = cv2.imwrite(args.output_mask, indexed_mask)
    if not ok:
        raise RuntimeError(f"Failed to write first-frame mask to: {args.output_mask}")

    print(
        f"Saved first-frame mask: {args.output_mask}; "
        f"objects={int(indexed_mask.max())}; nonzero={int((indexed_mask > 0).sum())}"
    )


if __name__ == "__main__":
    main()
