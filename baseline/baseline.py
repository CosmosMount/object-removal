import argparse
import os
import subprocess
import warnings

from config import PipelineConfig
from pipeline import run_pipeline

warnings.filterwarnings("ignore")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Hand-crafted video object removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--output", default="output_part1")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Optical flow motion threshold (px)",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=15,
        help="Base dilation kernel size (px)",
    )
    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable adaptive dilation (same kernel everywhere)",
    )
    parser.add_argument(
        "--bg_window",
        type=int,
        default=40,
        help="Temporal background propagation window (frames)",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["temporal", "spatial", "both", "compare"],
        help="Inpainting mode. 'compare' runs all three and saves comparison images",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 7],
        help="YOLO class IDs to detect (e.g. --classes 0 1 2). 0=person",
    )
    parser.add_argument("--part", default="part1", help="Part label for metric summary")
    parser.add_argument("--gt_mask_dir", default="", help="GT mask directory for JM/JR evaluation")
    parser.add_argument("--gt_video", default="", help="GT video path for PSNR/SSIM evaluation")
    parser.add_argument("--gt_frames_dir", default="", help="GT frame directory for PSNR/SSIM evaluation")
    return parser


def main():
    args = build_arg_parser().parse_args()

    cfg = PipelineConfig(
        dynamic_classes=args.classes,
        motion_threshold=args.threshold,
        dilation_kernel=args.dilation,
        adaptive_dilation=not args.no_adaptive,
        temp_bg_window=args.bg_window,
        inpaint_mode=args.mode,
        max_frames=args.max_frames,
    )

    run_pipeline(args.video, args.output, cfg, args.mode)

    # Evaluation logic after pipeline execution
    print("[Evaluation] Starting evaluation of results...")

    # Define paths for evaluation
    eval_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluate_metrics.py")
    pred_mask_dir = os.path.join(args.output, "masks", "final")
    pred_video = os.path.join(args.output, "inpainted_output.mp4")

    # Set up evaluation command
    eval_cmd = [
        "conda",
        "run",
        "-n",
        "propainter",
        "python",
        eval_script,
        "--output_dir",
        os.path.join(args.output, "metrics"),
        "--part_label",
        args.part,
        "--experiment_name",
        "baseline",
        "--pred_mask_dir",
        pred_mask_dir,
        "--pred_video",
        pred_video,
    ]

    # Add optional ground truth parameters
    if args.gt_mask_dir:
        eval_cmd.extend(["--gt_mask_dir", args.gt_mask_dir])
    if args.gt_video:
        eval_cmd.extend(["--gt_video", args.gt_video])
    elif os.path.isfile(args.video):
        eval_cmd.extend(["--gt_video", args.video])

    # Always set gt_frames_dir for metrics
    gt_frames_dir = args.gt_frames_dir
    if not gt_frames_dir:
        # If davis mode, use DAVIS/JPEGImages/480p/<video_name>
        video_name = os.path.basename(args.video)
        video_name = os.path.splitext(video_name)[0]
        davis_frames = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DAVIS", "JPEGImages", "480p", video_name)
        if os.path.isdir(davis_frames):
            gt_frames_dir = davis_frames
        elif os.path.isdir(args.video):
            gt_frames_dir = args.video
    if gt_frames_dir:
        eval_cmd.extend(["--gt_frames_dir", gt_frames_dir])

    # Run evaluation command
    try:
        subprocess.run(eval_cmd, check=True)
        print("[Evaluation] Metrics evaluation completed successfully.")
    except Exception as exc:
        print(f"[WARN] Metric evaluation failed: {exc}")


if __name__ == "__main__":
    main()
