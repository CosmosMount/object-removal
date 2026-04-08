import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tint(frame, mask, color=(220, 60, 60), alpha=0.5):
    view = bgr_to_rgb(frame).astype(float)
    for channel, value in enumerate(color):
        view[:, :, channel][mask > 0] = view[:, :, channel][mask > 0] * (1 - alpha) + value * alpha
    return np.clip(view, 0, 255).astype(np.uint8)


def draw_flow(frame, pts_o, pts_n, status):
    view = bgr_to_rgb(frame).copy()
    if pts_o is None:
        return view

    for old, new, ok in zip(pts_o, pts_n, status):
        if ok[0]:
            cv2.arrowedLine(
                view,
                tuple(old.ravel().astype(int)),
                tuple(new.ravel().astype(int)),
                (40, 230, 80),
                2,
                tipLength=0.5,
            )
            cv2.circle(view, tuple(old.ravel().astype(int)), 3, (40, 230, 80), -1)
    return view


def _ax_img(ax, img, title):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, color="#dde0ff", fontsize=9.5, pad=5, fontweight="bold", multialignment="center")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4444aa")
        spine.set_linewidth(0.8)


def save_step_viz(out_dir, idx, frame, raw_mask, dyn_mask, dil_mask, result, flow, seg_name, cfg):
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(cfg.bg_color)
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.05)

    flow_data = flow.get(idx)
    panels = [
        (bgr_to_rgb(frame), f"1 Original  #{idx}"),
        (tint(frame, raw_mask, (220, 60, 60)), f"2 {seg_name}\n(raw detections - red tint)"),
        (
            draw_flow(frame, *(flow_data[:3])) if flow_data else tint(frame, dyn_mask, (60, 200, 80)),
            "3 Optical flow vectors\n(green arrows = motion)",
        ),
        (tint(frame, dyn_mask, (60, 200, 80)), f"4 Dynamic mask\n(threshold = {cfg.motion_threshold}px)"),
        (tint(frame, dil_mask, (255, 160, 0)), f"5 After dilation\n(kernel = {cfg.dilation_kernel}px)"),
        (bgr_to_rgb(result), "6 Inpainted result"),
    ]

    for k, (img, title) in enumerate(panels):
        _ax_img(fig.add_subplot(gs[k // 3, k % 3]), img, title)

    fig.suptitle(
        f"Part 1 Hand-crafted Pipeline  |  {seg_name}  |  Frame {idx:04d}",
        color="#aaaaff",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    path = os.path.join(out_dir, f"viz_frame_{idx:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=cfg.bg_color)
    plt.close(fig)
    return path


def save_strip(out_dir, frames, results, bg_color, n=8):
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.2, 6.5))
    fig.patch.set_facecolor(bg_color)

    for col, idx in enumerate(idxs):
        for row, (img, label) in enumerate(
            [
                (bgr_to_rgb(frames[idx]), f"Original\n#{idx}"),
                (bgr_to_rgb(results[idx]), f"Inpainted\n#{idx}"),
            ]
        ):
            axes[row][col].imshow(img)
            axes[row][col].set_title(label, color="#e0e0ff", fontsize=8)
            axes[row][col].axis("off")

    for row, label in enumerate(["ORIGINAL", "INPAINTED"]):
        axes[row][0].set_ylabel(label, color="#aaaaff", fontsize=10)

    fig.suptitle("Before / After - sampled frames", color="#aaaaff", fontsize=13, fontweight="bold")
    path = os.path.join(out_dir, "comparison_strip.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    return path


def save_coverage(out_dir, raw_masks, dyn_masks, dil_masks, bg_color):
    hw = raw_masks[0].shape[0] * raw_masks[0].shape[1] / 100

    def pct(ms):
        return [mask.sum() / 255 / hw for mask in ms]

    raw, dyn, dil = pct(raw_masks), pct(dyn_masks), pct(dil_masks)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor("#0e0e22")

    xs = range(len(raw))
    ax.fill_between(xs, raw, alpha=0.25, color="#ff6666", label="Raw detection")
    ax.fill_between(xs, dyn, alpha=0.35, color="#66cc66", label="After dynamic filter")
    ax.fill_between(xs, dil, alpha=0.2, color="#ffaa33", label="After dilation")
    ax.plot(xs, raw, color="#ff6666", lw=1)
    ax.plot(xs, dyn, color="#66cc66", lw=1.2)
    ax.plot(xs, dil, color="#ffaa33", lw=1, linestyle="--")

    ax.set_xlabel("Frame", color="#ccccff")
    ax.set_ylabel("Coverage (%)", color="#ccccff")
    ax.set_title("Mask coverage over time", color="#e0e0ff", fontweight="bold")
    ax.legend(facecolor="#22224a", edgecolor="#555588", labelcolor="#ddddff", fontsize=9)
    ax.tick_params(colors="#9999cc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    path = os.path.join(out_dir, "mask_coverage.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    return path


def save_intensity(out_dir, frames, masks, results, bg_color):
    diffs = []
    for frame, mask, result in zip(frames, masks, results):
        if mask.max() == 0:
            diffs.append(0.0)
            continue
        diffs.append(float(np.mean(np.abs(frame[mask > 0].astype(float) - result[mask > 0].astype(float)))))

    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor("#0e0e22")
    ax.fill_between(range(len(diffs)), diffs, alpha=0.4, color="#7799ff")
    ax.plot(diffs, color="#aabbff", lw=1.2)

    ax.set_xlabel("Frame", color="#ccccff")
    ax.set_ylabel("Mean |orig-result|", color="#ccccff")
    ax.set_title("Inpainting intensity (masked region)", color="#e0e0ff", fontweight="bold")
    ax.tick_params(colors="#9999cc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    path = os.path.join(out_dir, "inpaint_intensity.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    return path


def save_mode_comparison(out_dir, frames, res_spatial, res_temporal, res_both, bg_color, n=8):
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    fig, axes = plt.subplots(4, n, figsize=(n * 3.2, 10))
    fig.patch.set_facecolor(bg_color)

    rows = [
        (frames, "ORIGINAL"),
        (res_spatial, "SPATIAL ONLY\n(cv2.inpaint)"),
        (res_temporal, "TEMPORAL ONLY\n(bg propagation)"),
        (res_both, "TEMPORAL + SPATIAL\n(combined)"),
    ]
    colors = ["#ffffff", "#ff9999", "#99ffcc", "#ffdd88"]

    for row, (src, label) in enumerate(rows):
        for col, idx in enumerate(idxs):
            ax = axes[row][col]
            ax.imshow(bgr_to_rgb(src[idx]))
            ax.set_title(f"#{idx}", color="#aaaacc", fontsize=7)
            ax.axis("off")
        axes[row][0].set_ylabel(label, color=colors[row], fontsize=8, fontweight="bold", labelpad=6)

    fig.suptitle("Inpainting method comparison", color="#aaaaff", fontsize=13, fontweight="bold")
    path = os.path.join(out_dir, "mode_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    return path


def save_diff_heatmap(out_dir, frames, res_spatial, res_temporal, masks, sample_idx, bg_color):
    idx = sample_idx
    frame = frames[idx]
    mask = masks[idx]
    if mask.max() == 0:
        return None

    def diff_map(orig, res):
        diff = np.abs(orig.astype(float) - res.astype(float)).mean(axis=2)
        diff[mask == 0] = 0
        return diff

    d_spatial = diff_map(frame, res_spatial[idx])
    d_temporal = diff_map(frame, res_temporal[idx])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(bg_color)
    vmax = max(d_spatial.max(), d_temporal.max(), 1)

    for ax, img, title in [
        (axes[0], bgr_to_rgb(frame), f"Original #{idx}"),
        (axes[1], bgr_to_rgb(res_spatial[idx]), "Spatial result"),
        (axes[2], bgr_to_rgb(res_temporal[idx]), "Temporal result"),
    ]:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, color="#dde0ff", fontsize=10, fontweight="bold")

    axes[3].set_facecolor("#0e0e22")
    im = axes[3].imshow(d_spatial - d_temporal, cmap="RdYlGn", vmin=-vmax / 2, vmax=vmax / 2)
    axes[3].set_title(
        "Diff map\nGreen=temporal better\nRed=spatial better",
        color="#dde0ff",
        fontsize=9,
        fontweight="bold",
    )
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Spatial vs Temporal inpainting - frame {idx}", color="#aaaaff", fontsize=12, fontweight="bold")
    path = os.path.join(out_dir, f"diff_heatmap_frame{idx:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    return path
