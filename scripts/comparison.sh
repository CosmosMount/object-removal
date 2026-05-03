#!/bin/bash

# Comparison script for all pipelines

# Define common variables
ROOT_DIR="/home/xyz/Desktop/yzhang/object-removal"
DAVIS_INPUT_ROOT="${ROOT_DIR}/data/DAVIS"
DAVIS_GT_ROOT="${ROOT_DIR}/data/DAVIS"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
PROPAINTER_ENV="propainter"
VGGT_THRESHOLD_SCALE="${VGGT_THRESHOLD_SCALE:-1.0}"

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
			echo "  DAVIS mode: $0 --davis_seq bmx-trees [--davis_input_root ${ROOT_DIR}/data/DAVIS] [--eval_davis 1|0] [--davis_gt_root /path/to/DAVIS] [--davis_task semi-supervised|unsupervised]"
			exit 1
			;;
	esac
done

echo "Running baseline..."
BASELINE_GT_MASK="${DAVIS_GT_ROOT}/Annotations_unsupervised/480p/${DAVIS_SEQ}"
if [[ ! -d "$BASELINE_GT_MASK" ]]; then
	BASELINE_GT_MASK="${DAVIS_GT_ROOT}/Annotations/480p/${DAVIS_SEQ}"
fi

conda run -n sam2 python baseline/baseline.py \
	--video "${DAVIS_INPUT_ROOT}/JPEGImages/480p/${DAVIS_SEQ}" \
	--output outputs/baseline/$DAVIS_SEQ \
	--gt_mask_dir "$BASELINE_GT_MASK" \
	--gt_frames_dir "${DAVIS_INPUT_ROOT}/JPEGImages/480p/${DAVIS_SEQ}"

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

# Metrics: collect from existing outputs (each pipeline already generates metrics internally)
GT_MASK_DIR="${DAVIS_GT_ROOT}/Annotations/480p/${DAVIS_SEQ}"
GT_FRAMES_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${DAVIS_SEQ}"

declare -A PIPELINE_OUTPUT_DIRS=(
	["baseline"]="${OUTPUTS_DIR}/baseline"
	["yoloopt"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}"
	["yolosam2"]="${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}"
	["vggt4d"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}"
	["vggt4dsam3"]="${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}"
	["vggt4dsam3sd"]="${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}"
)

declare -A PIPELINE_METRICS_PATHS=(
	["baseline"]="${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["yoloopt"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["yolosam2"]="${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4d"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4dsam3"]="${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4dsam3sd"]="${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
)

run_metrics() {
	local name=$1
	local output_dir=$2
	local mask_dir=$3
	local pred_video=$4
	local pred_frames=$5

	local metrics_path="${PIPELINE_METRICS_PATHS[$name]}"
	mkdir -p "${OUTPUTS_DIR}/metrics_${name}"

	if [[ -f "$metrics_path" ]]; then
		local existing_psnr existing_ssim
		existing_psnr=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(d.get('video_psnr'))" 2>/dev/null)
		existing_ssim=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(d.get('video_ssim'))" 2>/dev/null)

		if [[ "$existing_psnr" != "None" ]] && [[ -n "$existing_psnr" ]]; then
			echo "[Metrics] Found existing metrics for $name: $metrics_path"
			cp "$metrics_path" "${OUTPUTS_DIR}/metrics_${name}/metrics_summary.json"
			return 0
		else
			echo "[Metrics] Existing metrics found but missing video metrics, re-computing for $name..."
		fi
	fi

	if [[ -n "$pred_video" ]] && [[ ! -d "$pred_frames" ]]; then
		echo "[Metrics] Extracting frames from video for $name..."
		mkdir -p "$pred_frames"
		ffmpeg -i "$pred_video" -vf "fps=25,scale=ceil(iw/2)*2:ceil(ih/2)*2" "${pred_frames}/%04d.png" -y 2>/dev/null
	fi

	echo "[Metrics] Computing JM/JR and optional PSNR/SSIM for $name"
	local metrics_dir="${OUTPUTS_DIR}/metrics_${name}"

	local metrics_cmd=(
		conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/evaluate_metrics.py"
		--output_dir "${metrics_dir}"
		--part_label "${PART_LABEL}"
		--experiment_name "${name}"
		--pred_mask_dir "${mask_dir}"
		--pred_frames_dir "${pred_frames}"
	)

	if [[ -n "${GT_MASK_DIR}" ]] && [[ -d "${GT_MASK_DIR}" ]]; then
		metrics_cmd+=(--gt_mask_dir "${GT_MASK_DIR}")
	fi
	if [[ -n "${GT_FRAMES_DIR}" ]] && [[ -d "${GT_FRAMES_DIR}" ]]; then
		metrics_cmd+=(--gt_frames_dir "${GT_FRAMES_DIR}")
	fi

	"${metrics_cmd[@]}" || echo "WARN: metrics evaluation failed for $name"
}

# Run metrics for each pipeline (copies existing or computes if missing)
run_metrics "baseline" "${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}/masks/final" \
	"${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}/inpainted_output.mp4" \
	"${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}"

run_metrics "yoloopt" "${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/bmx-trees_mask_vggt" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/inpaint_out.mp4" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/bmx-trees_propainter/bmx-trees/frames"

run_metrics "vggt4d" "${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/bmx-trees_mask_vggt" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/inpaint_out.mp4" \
	"${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/bmx-trees_propainter/bmx-trees/frames"

run_metrics "yolosam2" "${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}/bmx-trees_mask_sam2" \
	"${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}/inpaint_out.mp4" \
	"${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}/bmx-trees_propainter/bmx-trees/frames"

run_metrics "vggt4dsam3" "${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}/bmx-trees_mask_sam3" \
	"${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}/inpaint_out.mp4" \
	"${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}/bmx-trees_propainter/bmx-trees/frames"

run_metrics "vggt4dsam3sd" "${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}" \
	"${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}/bmx-trees_mask_sam3" \
	"${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}/inpaint_out.mp4" \
	"${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}/bmx-trees_propainter/bmx-trees/frames"

# Summary
echo "All pipelines completed. Check outputs for metrics and visualizations."

generate_comparison_summary() {
	echo "[Summary] Generating comparison summary..."

	local summary_file="${OUTPUTS_DIR}/comparison_${DAVIS_SEQ}.md"
	local all_metrics=()

	for name in baseline yoloopt yolosam2 vggt4d vggt4dsam3 vggt4dsam3sd; do
		local metrics_json="${OUTPUTS_DIR}/metrics_${name}/metrics_summary.json"
		if [[ -f "$metrics_json" ]]; then
			all_metrics+=("$name:$metrics_json")
		fi
	done

	{
		echo "# Comparison Results: ${DAVIS_SEQ}"
		echo ""
		echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')"
		echo ""
		echo "## Metrics Summary"
		echo ""
		echo "| Method | Jaccard (JM) (↑) | Jaccard Recall (JR) (↑) | PSNR (↑) | SSIM (↑) |"
		echo "|--------|-----------------|--------------------------|----------|---------|"

		for item in "${all_metrics[@]}"; do
			local name="${item%%:*}"
			local json="${item##*:}"

			local jm jr psnr ssim
			jm=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('mask_jm', 'N/A'))" 2>/dev/null || echo "N/A")
			jr=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('mask_jr', 'N/A'))" 2>/dev/null || echo "N/A")
			psnr=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('video_psnr', 'N/A'))" 2>/dev/null || echo "N/A")
			ssim=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('video_ssim', 'N/A'))" 2>/dev/null || echo "N/A")

			if [[ "$jm" != "N/A" ]]; then jm=$(printf "%.4f" "$jm"); fi
			if [[ "$jr" != "N/A" ]]; then jr=$(printf "%.4f" "$jr"); fi
			if [[ "$psnr" != "N/A" ]]; then psnr=$(printf "%.2f" "$psnr"); fi
			if [[ "$ssim" != "N/A" ]]; then ssim=$(printf "%.4f" "$ssim"); fi

			echo "| $name | $jm | $jr | $psnr | $ssim |"
		done

		echo ""
		echo "## Notes"
		echo "- ↑ indicates higher is better"
		echo "- JM: Jaccard Mean (IoU)"
		echo "- JR: Jaccard Recall (threshold IoU)"
		echo "- PSNR: Peak Signal-to-Noise Ratio"
		echo "- SSIM: Structural Similarity Index"
		echo ""
		echo "## Output Locations"
		for item in "${all_metrics[@]}"; do
			local name="${item%%:*}"
			echo "- $name: ${OUTPUTS_DIR}/metrics_${name}/"
		done

	} > "$summary_file"

	echo "Summary saved to: $summary_file"
}

generate_comparison_summary
