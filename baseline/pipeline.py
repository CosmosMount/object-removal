import os
import time

import cv2

from inpainting import inpaint_video
from masking import extract_masks
from motion import filter_dynamic
from postprocess import dilate_masks
from video_io import build_side_by_side, load_video, write_video
from visualization import (
    save_coverage,
    save_diff_heatmap,
    save_intensity,
    save_mode_comparison,
    save_step_viz,
    save_strip,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_mask_sequence(mask_list, out_dir):
    ensure_dir(out_dir)
    for i, mask in enumerate(mask_list):
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.png"), mask)


def run_pipeline(video_path, output_dir, cfg, mode):
    t0 = time.time()
    ensure_dir(output_dir)
    viz_dir = ensure_dir(os.path.join(output_dir, "visualizations"))

    frames, fps, width, height = load_video(video_path, max_frames=cfg.max_frames)
    fps_out = cfg.output_fps or fps

    raw_masks, seg_method = extract_masks(frames, cfg)
    dynamic_masks, flow_info = filter_dynamic(frames, raw_masks, cfg)
    dilated_masks = dilate_masks(dynamic_masks, cfg)

    masks_root = ensure_dir(os.path.join(output_dir, "masks"))
    save_mask_sequence(raw_masks, os.path.join(masks_root, "raw"))
    save_mask_sequence(dynamic_masks, os.path.join(masks_root, "dynamic"))
    save_mask_sequence(dilated_masks, os.path.join(masks_root, "final"))

    if mode == "compare":
        print("\n[Mode: compare]  Running all three inpainting methods for comparison ...")
        res_spatial, _, _ = inpaint_video(frames, dilated_masks, cfg, "spatial")
        print()
        res_temporal, _, _ = inpaint_video(frames, dilated_masks, cfg, "temporal")
        print()
        res_both, px_temporal, px_spatial = inpaint_video(frames, dilated_masks, cfg, "both")
        result_frames = res_both
    else:
        res_spatial = None
        res_temporal = None
        res_both = None
        result_frames, px_temporal, px_spatial = inpaint_video(frames, dilated_masks, cfg, mode)

    print("[Viz] Generating visualisations ...")
    key_frames = [i for i, mask in enumerate(raw_masks) if mask.max() > 0]
    samples = key_frames[:: max(1, len(key_frames) // 3)][:3] if key_frames else [0]

    for idx in samples:
        path = save_step_viz(
            viz_dir,
            idx,
            frames[idx],
            raw_masks[idx],
            dynamic_masks[idx],
            dilated_masks[idx],
            result_frames[idx],
            flow_info,
            seg_method,
            cfg,
        )
        print(f"   Step-viz: {path}")

    print(f"   Strip     : {save_strip(viz_dir, frames, result_frames, cfg.bg_color)}")
    print(f"   Coverage  : {save_coverage(viz_dir, raw_masks, dynamic_masks, dilated_masks, cfg.bg_color)}")
    print(f"   Intensity : {save_intensity(viz_dir, frames, dilated_masks, result_frames, cfg.bg_color)}")

    if mode == "compare" and res_spatial is not None and res_temporal is not None and res_both is not None:
        path = save_mode_comparison(viz_dir, frames, res_spatial, res_temporal, res_both, cfg.bg_color)
        print(f"   Mode comparison  : {path}")
        if key_frames:
            heatmap_path = save_diff_heatmap(
                viz_dir,
                frames,
                res_spatial,
                res_temporal,
                dilated_masks,
                key_frames[len(key_frames) // 2],
                cfg.bg_color,
            )
            if heatmap_path:
                print(f"   Diff heatmap     : {heatmap_path}")

    output_video_path = os.path.join(output_dir, "inpainted_output.mp4")
    compare_video_path = os.path.join(output_dir, "compare_sidebyside.mp4")
    write_video(result_frames, output_video_path, fps_out, width, height)
    print(f"[Step 7] Output video   : {output_video_path}")

    side = build_side_by_side(frames, result_frames, height)
    write_video(side, compare_video_path, fps_out, width * 2 + 4, height)
    print(f"[Step 7] Side-by-side   : {compare_video_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Segmentation  : {seg_method}")
    print(f"  Inpaint mode  : {mode}")
    print(f"  Dilation      : kernel={cfg.dilation_kernel}px  adaptive={cfg.adaptive_dilation}")
    print(f"  Temporal fill : {px_temporal:,} px")
    print(f"  Spatial fill  : {px_spatial:,} px")
    print(f"  Visualisations: {viz_dir}/")
    print(f"{'=' * 60}\n")
