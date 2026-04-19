#!/bin/bash

# Comparison script for all pipelines

# Define common variables
DAVIS_INPUT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"
DAVIS_GT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--video)
			VIDEO_PATH="$2"
			shift 2
			;;
		--davis_seq)
			DAVIS_SEQ="$2"
			shift 2
			;;
		--davis_input_root)
			DAVIS_INPUT_ROOT="$2"
			shift 2
			;;
		--eval_davis)
			EVAL_DAVIS="$2"
			shift 2
			;;
		--davis_gt_root)
			DAVIS_GT_ROOT="$2"
			shift 2
			;;
		--davis_task)
			DAVIS_TASK="$2"
			shift 2
			;;
		--part_label)
			PART_LABEL="$2"
			shift 2
			;;
		--gt_mask_dir)
			GT_MASK_DIR="$2"
			shift 2
			;;
		--gt_video)
			GT_VIDEO="$2"
			shift 2
			;;
		--gt_frames_dir)
			GT_FRAMES_DIR="$2"
			shift 2
			;;
		--dyn_threshold_scale)
			VGGT_THRESHOLD_SCALE="$2"
			shift 2
			;;
		*)
			echo "Unknown argument: $1"
			echo "Usage:"
			echo "  Video mode: $0 --video /path/to/video.mp4"
			echo "  DAVIS mode: $0 --davis_seq bmx-trees [--davis_input_root ${ROOT_DIR}/DAVIS] [--eval_davis 1|0] [--davis_gt_root /path/to/DAVIS] [--davis_task semi-supervised|unsupervised]"
			exit 1
			;;
	esac
done

echo "Running baseline..."
conda run -n sam2 python baseline/baseline.py --video DAVIS/JPEGImages/480p/$DAVIS_SEQ --output outputs/baseline/$DAVIS_SEQ --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

echo "Running yoloopt pipeline..."
bash pipeline_yoloopt/yoloopt.sh --davis_seq $DAVIS_SEQ --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run yolosam2 pipeline
echo "Running yolosam2 pipeline..."
bash pipeline_yolosam2/yolosam2.sh --davis_seq $DAVIS_SEQ --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4d pipeline
echo "Running vggt4d pipeline..."
bash pipeline_vggt4d/vggt4d.sh --davis_seq $DAVIS_SEQ --dyn_threshold_scale "${VGGT_THRESHOLD_SCALE}" --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4dsam3 pipeline
echo "Running vggt4dsam3 pipeline..."
bash pipeline_vggt4dsam3/vggt4dsam3.sh --davis_seq $DAVIS_SEQ --dyn_threshold_scale "${VGGT_THRESHOLD_SCALE}" --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4dsam3sd pipeline
echo "Running vggt4dsam3sd pipeline..."
bash pipeline_vggt4dsam3sd/vggt4dsam3sd.sh --davis_seq $DAVIS_SEQ --dyn_threshold_scale "${VGGT_THRESHOLD_SCALE}" --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Summary
echo "All pipelines completed. Check outputs for metrics and visualizations."
