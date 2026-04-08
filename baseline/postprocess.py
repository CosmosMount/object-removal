import cv2
import numpy as np


def dilate_masks(masks, cfg):
    print(
        f"[Step 4] Dilation  kernel={cfg.dilation_kernel}px  adaptive={cfg.adaptive_dilation} ..."
    )

    k_body = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.dilation_kernel, cfg.dilation_kernel),
    )
    k_ext = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.dilation_kernel * 2, cfg.dilation_kernel * 2),
    )

    result = []
    for mask in masks:
        if mask.max() == 0:
            result.append(mask)
            continue

        if not cfg.adaptive_dilation:
            result.append(cv2.dilate(mask, k_body, iterations=2))
            continue

        ys = np.where(mask > 0)[0]
        if len(ys) == 0:
            result.append(mask)
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        margin = max(int((y_max - y_min) * 0.22), 20)

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
        result.append(dilated)

    print("[Step 4] Done")
    return result
