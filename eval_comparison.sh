#!/bin/bash

# Evaluation and comparison only script
# Run this after pipelines complete to collect metrics and generate comparison summary

DAVIS_INPUT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"
DAVIS_GT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"
ROOT_DIR="/home/xyz/Desktop/yzhang/object-removal"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
PROPAINTER_ENV="propainter"

PART_LABEL="${PART_LABEL:-part1}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--davis_seq)
			DAVIS_SEQ="$2"
			shift 2
			;;
		--davis_input_root)
			DAVIS_INPUT_ROOT="$2"
			shift 2
			;;
		--davis_gt_root)
			DAVIS_GT_ROOT="$2"
			shift 2
			;;
		--part_label)
			PART_LABEL="$2"
			shift 2
			;;
		*)
			echo "Unknown argument: $1"
			echo "Usage: $0 --davis_seq SEQ_NAME [--davis_input_root PATH] [--davis_gt_root PATH] [--part_label LABEL]"
			exit 1
			;;
	esac
done

if [[ -z "$DAVIS_SEQ" ]]; then
	echo "Error: --davis_seq is required"
	echo "Usage: $0 --davis_seq SEQ_NAME"
	exit 1
fi

echo "Collecting metrics for: ${DAVIS_SEQ}"

GT_MASK_DIR="${DAVIS_GT_ROOT}/Annotations_unsupervised/480p/${DAVIS_SEQ}"
GT_FRAMES_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${DAVIS_SEQ}"

if [[ ! -d "$GT_MASK_DIR" ]]; then
	GT_MASK_DIR="${DAVIS_GT_ROOT}/Annotations/480p/${DAVIS_SEQ}"
fi

declare -A PIPELINE_METRICS_PATHS=(
	["baseline"]="${OUTPUTS_DIR}/baseline/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["yoloopt"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["yolosam2"]="${OUTPUTS_DIR}/yolosam2_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4d"]="${OUTPUTS_DIR}/vggt_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4dsam3"]="${OUTPUTS_DIR}/vggtsam3_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
	["vggt4dsam3sd"]="${OUTPUTS_DIR}/vggtsam3sd_davis/${DAVIS_SEQ}/metrics/metrics_summary.json"
)

collect_metrics() {
	local name=$1
	local src_path="${PIPELINE_METRICS_PATHS[$name]}"
	local dest_dir="${OUTPUTS_DIR}/metrics_${name}"
	local dest_path="${dest_dir}/metrics_summary.json"

	mkdir -p "${dest_dir}"

	if [[ -f "$src_path" ]]; then
		echo "[Metrics] Found: $name -> $src_path"
		cp "$src_path" "$dest_path"
	else
		echo "[Metrics] Not found: $name ($src_path)"
	fi
}

echo ""
echo "=== Collecting metrics ==="
for name in baseline yoloopt yolosam2 vggt4d vggt4dsam3 vggt4dsam3sd; do
	collect_metrics "$name"
done

echo ""
echo "=== Generating comparison summary ==="

generate_comparison_summary() {
	local summary_file="${OUTPUTS_DIR}/comparison_${DAVIS_SEQ}.md"

	{
		echo "# Comparison Results: ${DAVIS_SEQ}"
		echo ""
		echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')"
		echo ""
		echo "## Metrics Summary"
		echo ""
		echo "| Method | Jaccard Mean (JM) (↑) | Jaccard Recall (JR) (↑) | PSNR (↑) | SSIM (↑) |"
		echo "|--------|----------------------|-------------------------|---------|---------|"

		for name in baseline yoloopt yolosam2 vggt4d vggt4dsam3 vggt4dsam3sd; do
			local json="${OUTPUTS_DIR}/metrics_${name}/metrics_summary.json"
			if [[ ! -f "$json" ]]; then
				echo "| $name | - | - | - | - |"
				continue
			fi

			local jm jr psnr ssim
			jm=$(python3 -c "
import json
d = json.load(open('$json'))
v = d.get('mask_jm')
if v is None: print('N/A')
else: print('{:.4f}'.format(v))
" 2>/dev/null)
			jr=$(python3 -c "
import json
d = json.load(open('$json'))
v = d.get('mask_jr')
if v is None: print('N/A')
else: print('{:.4f}'.format(v))
" 2>/dev/null)
			psnr=$(python3 -c "
import json
d = json.load(open('$json'))
v = d.get('video_psnr')
if v is None: print('N/A')
else: print('{:.2f}'.format(v))
" 2>/dev/null)
			ssim=$(python3 -c "
import json
d = json.load(open('$json'))
v = d.get('video_ssim')
if v is None: print('N/A')
else: print('{:.4f}'.format(v))
" 2>/dev/null)

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
		for name in baseline yoloopt yolosam2 vggt4d vggt4dsam3 vggt4dsam3sd; do
			echo "- $name: ${OUTPUTS_DIR}/metrics_${name}/"
		done

	} > "$summary_file"

	echo "Summary saved to: $summary_file"
}

generate_comparison_summary

echo ""
echo "Done. Results:"
ls -la "${OUTPUTS_DIR}/comparison_${DAVIS_SEQ}.md"