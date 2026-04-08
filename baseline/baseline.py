import argparse
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


if __name__ == "__main__":
    main()
