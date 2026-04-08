import cv2
import numpy as np


def _temporal_fill(frames, masks, cfg, mode):
    n = len(frames)
    results = [frame.copy() for frame in frames]
    px_temporal = 0

    for i in range(n):
        mask = masks[i]
        if mask.max() == 0:
            continue

        filled = np.zeros(mask.shape, dtype=bool)
        neighbors = sorted(
            [
                j
                for j in range(max(0, i - cfg.temp_bg_window), min(n, i + cfg.temp_bg_window + 1))
                if j != i
            ],
            key=lambda j: abs(j - i),
        )

        target_px = int((mask > 0).sum())
        for j in neighbors:
            if filled.sum() == target_px:
                break
            borrow = (~filled) & (mask > 0) & (masks[j] == 0)
            if borrow.any():
                results[i][borrow] = frames[j][borrow]
                filled[borrow] = True

        px_temporal += int(filled.sum())

        if mode == "both":
            residual = ((mask > 0) & ~filled).astype(np.uint8) * 255
            if residual.max() > 0:
                results[i] = cv2.inpaint(results[i], residual, 5, cv2.INPAINT_TELEA)

        if (i + 1) % 30 == 0:
            print(f"   {i + 1}/{n}")

    return results, px_temporal


def _spatial_only(frames, masks):
    results = [frame.copy() for frame in frames]
    px_spatial = 0

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        if mask.max() == 0:
            continue
        results[i] = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
        px_spatial += int(mask.sum() // 255)
        if (i + 1) % 30 == 0:
            print(f"   {i + 1}/{len(frames)}")

    return results, px_spatial


def inpaint_video(frames, masks, cfg, mode=None):
    selected_mode = mode or cfg.inpaint_mode

    if selected_mode == "spatial":
        print("[Step 5+6] Spatial-only inpainting (cv2.inpaint Telea) ...")
        results, px_spatial = _spatial_only(frames, masks)
        print(f"[Step 5+6] Spatial: {px_spatial:,} px filled")
        return results, 0, px_spatial

    if selected_mode == "temporal":
        print(f"[Step 5+6] Temporal-only propagation (window={cfg.temp_bg_window}) ...")
        results, px_temporal = _temporal_fill(frames, masks, cfg, "temporal")
        print(f"[Step 5+6] Temporal: {px_temporal:,} px filled")
        return results, px_temporal, 0

    print(f"[Step 5+6] Temporal propagation + spatial fallback (window={cfg.temp_bg_window}) ...")
    results, px_temporal = _temporal_fill(frames, masks, cfg, "both")
    total_masked = sum(int(mask.sum() // 255) for mask in masks if mask.max() > 0)
    px_spatial = max(0, total_masked - px_temporal)
    total = px_temporal + px_spatial
    if total:
        print(
            f"[Step 5+6] Temporal {px_temporal:,}px ({100 * px_temporal / total:.0f}%)  "
            f"Spatial fallback {px_spatial:,}px ({100 * px_spatial / total:.0f}%)"
        )
    return results, px_temporal, px_spatial
