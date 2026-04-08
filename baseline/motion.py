import cv2
import numpy as np


def filter_dynamic(frames, masks, cfg):
    print("[Step 3] Optical flow dynamic filter ...")
    lk = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    fp = dict(
        maxCorners=cfg.lk_max_corners,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    out_masks = []
    flow_info = {}

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        if mask.max() == 0 or i == 0:
            out_masks.append(mask)
            continue

        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **fp)

        if pts is None or len(pts) < 3:
            out_masks.append(mask)
            continue

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **lk)
        good_o = pts[st == 1]
        good_n = nxt[st == 1]
        mag = float(np.mean(np.linalg.norm(good_n - good_o, axis=1))) if len(good_o) else 0.0
        flow_info[i] = (pts, nxt, st, mag)

        if mag >= cfg.motion_threshold:
            out_masks.append(mask)
        else:
            out_masks.append(np.zeros_like(mask))

    kept = sum(1 for mask in out_masks if mask.max() > 0)
    print(f"[Step 3] Dynamic frames: {kept}/{len(frames)}  (thr={cfg.motion_threshold}px)")
    return out_masks, flow_info
