import cv2
import numpy as np


def _try_yolo(frames, dynamic_classes):
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n-seg.pt")
        height, width = frames[0].shape[:2]
        masks = []

        for i, frame in enumerate(frames):
            result = model(frame, classes=dynamic_classes, verbose=False)[0]
            mask = np.zeros((height, width), dtype=np.uint8)
            if result.masks is not None:
                for seg in result.masks.data:
                    seg_map = seg.cpu().numpy()
                    seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, (seg_map > 0.5).astype(np.uint8) * 255)
            masks.append(mask)
            if (i + 1) % 30 == 0:
                print(f"   YOLO: {i + 1}/{len(frames)}")

        return masks, "YOLOv8-Seg"
    except Exception as exc:
        print(f"   YOLO unavailable ({exc}) - using MOG2 fallback")
        return None, None


def _mog2(frames, history, threshold, min_blob_area):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold,
        detectShadows=True,
    )

    for frame in frames:
        fgbg.apply(frame)

    fgbg2 = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold,
        detectShadows=True,
    )

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    masks = []

    for frame in frames:
        fg = fgbg2.apply(frame)
        fg[fg == 127] = 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(fg)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_blob_area:
                cv2.drawContours(clean, [cnt], -1, 255, -1)
        masks.append(clean)

    return masks, "MOG2 background subtraction"


def extract_masks(frames, cfg):
    print("[Step 2] Extracting masks ...")
    masks, method = _try_yolo(frames, cfg.dynamic_classes)
    if masks is None:
        masks, method = _mog2(frames, cfg.mog2_history, cfg.mog2_threshold, cfg.min_blob_area)

    detected = sum(1 for mask in masks if mask.max() > 0)
    print(f"[Step 2] {method} - detections in {detected}/{len(frames)} frames")
    return masks, method
