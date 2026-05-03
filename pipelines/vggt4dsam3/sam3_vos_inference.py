#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SAM3_DIR = os.path.join(ROOT_DIR, "external", "sam3")
if os.path.isdir(SAM3_DIR) and SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)

from sam3.model_builder import build_sam3_video_model


DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 VOS inference with init mask prompt.")
    parser.add_argument("--sam3_checkpoint", required=True, help="Local SAM3 checkpoint path.")
    parser.add_argument("--base_video_dir", required=True, help="Root dir containing frame folders for each video.")
    parser.add_argument("--input_mask_dir", required=True, help="Input init indexed masks dir.")
    parser.add_argument("--output_mask_dir", required=True, help="Output propagated indexed masks dir.")
    parser.add_argument("--video_list_file", default=None, help="Optional txt with one video name per line.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Threshold on SAM3 logits.")
    return parser.parse_args()


def load_mask(path: str) -> Tuple[np.ndarray, List[int], List[int]]:
    mask_img = Image.open(path)
    arr = np.array(mask_img)
    if arr.ndim > 2:
        arr = arr[..., 0]
    object_ids = [int(v) for v in np.unique(arr) if int(v) > 0]
    palette = mask_img.getpalette()
    return arr.astype(np.uint8), object_ids, palette


def save_indexed_mask(path: str, mask: np.ndarray, palette: List[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = Image.fromarray(mask.astype(np.uint8), mode="P")
    if palette is not None:
        out.putpalette(palette)
    else:
        out.putpalette(DAVIS_PALETTE)
    out.save(path)


def list_frame_names(video_dir: str) -> List[str]:
    names = []
    for name in os.listdir(video_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            names.append(os.path.splitext(name)[0])
    names.sort(key=lambda x: int(x) if x.isdigit() else x)
    return names


def resolve_init_mask_path(input_mask_dir: str, video_name: str) -> str:
    video_mask_dir = os.path.join(input_mask_dir, video_name)
    if not os.path.isdir(video_mask_dir):
        raise FileNotFoundError(f"Input mask dir not found: {video_mask_dir}")

    pngs = sorted(
        [p for p in os.listdir(video_mask_dir) if p.lower().endswith(".png")],
        key=lambda n: int(os.path.splitext(n)[0]) if os.path.splitext(n)[0].isdigit() else n,
    )
    if not pngs:
        raise FileNotFoundError(f"No init masks found in {video_mask_dir}")

    best_path = None
    best_area = -1
    for name in pngs:
        path = os.path.join(video_mask_dir, name)
        arr = np.array(Image.open(path))
        if arr.ndim > 2:
            arr = arr[..., 0]
        area = int((arr > 0).sum())
        if area > best_area:
            best_area = area
            best_path = path

    if best_path is None:
        raise RuntimeError(f"Failed to select an init mask from {video_mask_dir}")
    return best_path


def run_one_video(
    predictor,
    base_video_dir: str,
    input_mask_dir: str,
    output_mask_dir: str,
    video_name: str,
    score_thresh: float,
) -> None:
    video_dir = os.path.join(base_video_dir, video_name)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video frame dir not found: {video_dir}")

    frame_names = list_frame_names(video_dir)
    if not frame_names:
        raise RuntimeError(f"No frames found in {video_dir}")

    init_mask_path = resolve_init_mask_path(input_mask_dir, video_name)
    init_frame_name = os.path.splitext(os.path.basename(init_mask_path))[0]
    if init_frame_name not in frame_names:
        raise RuntimeError(
            f"Init mask frame {init_frame_name}.png not found in video frames for {video_name}"
        )
    init_frame_idx = frame_names.index(init_frame_name)

    first_mask, object_ids, palette = load_mask(init_mask_path)
    if len(object_ids) == 0:
        raise RuntimeError(f"No foreground object in init mask: {init_mask_path}")

    print(f"[sam3] init frame for {video_name}: {init_frame_name} (idx={init_frame_idx})")

    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    video_h = int(inference_state["video_height"])
    video_w = int(inference_state["video_width"])
    if first_mask.shape != (video_h, video_w):
        resized = Image.fromarray(first_mask, mode="L").resize((video_w, video_h), resample=Image.NEAREST)
        first_mask = np.array(resized).astype(np.uint8)

    for obj_id in object_ids:
        obj_mask = torch.from_numpy(first_mask == obj_id)
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=init_frame_idx,
            obj_id=int(obj_id),
            mask=obj_mask,
        )

    outputs: Dict[int, Dict[int, np.ndarray]] = {}
    for reverse in (False, True):
        for out_frame_idx, out_obj_ids, _, out_video_res_masks, _ in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=init_frame_idx,
            max_frame_num_to_track=len(frame_names),
            reverse=reverse,
            propagate_preflight=(not reverse),
        ):
            per_obj = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                per_obj[int(out_obj_id)] = (out_video_res_masks[i] > score_thresh).cpu().numpy()
            outputs[int(out_frame_idx)] = per_obj

    save_dir = os.path.join(output_mask_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx, per_obj in outputs.items():
        canvas = np.zeros((video_h, video_w), dtype=np.uint8)
        for obj_id in sorted(per_obj.keys(), reverse=True):
            canvas[per_obj[obj_id].reshape(video_h, video_w)] = np.uint8(obj_id)
        out_name = frame_names[frame_idx]
        save_indexed_mask(os.path.join(save_dir, f"{out_name}.png"), canvas, palette)


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.sam3_checkpoint):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")

    if args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_names = [line.strip() for line in f if line.strip()]
    else:
        video_names = sorted(os.listdir(args.base_video_dir))
        video_names = [n for n in video_names if os.path.isdir(os.path.join(args.base_video_dir, n))]

    if not video_names:
        raise RuntimeError("No videos to process")

    model = build_sam3_video_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    for video_name in video_names:
        print(f"[sam3] processing {video_name}")
        run_one_video(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=float(args.score_thresh),
        )

    print(f"Done. SAM3 masks saved to: {args.output_mask_dir}")


if __name__ == "__main__":
    main()
