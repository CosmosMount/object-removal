#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VGGT_ENV="${VGGT_ENV:-vggt}"
PREPROCESS_ENV="${PREPROCESS_ENV:-sam2}"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
DAVIS_ENV="${DAVIS_ENV:-davis}"

VGGT_DIR="${ROOT_DIR}/VGGT4D"
PROPAINTER_DIR="${ROOT_DIR}/ProPainter"
INPUTS_DIR="${ROOT_DIR}/inputs"

VIDEO_PATH=""
DAVIS_SEQ=""
DAVIS_INPUT_ROOT="${ROOT_DIR}/DAVIS"
EVAL_DAVIS=1
DAVIS_GT_ROOT="${ROOT_DIR}/DAVIS"
DAVIS_TASK="unsupervised"
PART_LABEL="part2"
GT_MASK_DIR=""
GT_VIDEO=""
GT_FRAMES_DIR=""

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

if [[ -z "${VIDEO_PATH}" && -z "${DAVIS_SEQ}" ]]; then
	echo "Usage:"
	echo "  Video mode: $0 --video /path/to/video.mp4"
	echo "  DAVIS mode: $0 --davis_seq bmx-trees [--davis_input_root ${ROOT_DIR}/DAVIS] [--eval_davis 1|0] [--davis_gt_root /path/to/DAVIS] [--davis_task semi-supervised|unsupervised]"
	exit 1
fi

if [[ -n "${VIDEO_PATH}" && -n "${DAVIS_SEQ}" ]]; then
	echo "ERROR: please provide either --video or --davis_seq, not both"
	exit 1
fi

if [[ "${DAVIS_TASK}" != "semi-supervised" && "${DAVIS_TASK}" != "unsupervised" ]]; then
	echo "ERROR: --davis_task must be either semi-supervised or unsupervised"
	exit 1
fi

if [[ -n "${VIDEO_PATH}" ]]; then
	MODE="video"
	if [[ ! -f "${VIDEO_PATH}" ]]; then
		echo "ERROR: video file not found: ${VIDEO_PATH}"
		exit 1
	fi
	VIDEO_NAME="$(basename "${VIDEO_PATH}")"
	VIDEO_NAME="${VIDEO_NAME%.*}"
	VIDEO_DIR="${INPUTS_DIR}/${VIDEO_NAME}"
	OLD_MASK_DIR="${INPUTS_DIR}/${VIDEO_NAME}_mask"
	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt/${VIDEO_NAME}"
	
	if [[ ! -d "${VIDEO_DIR}" ]]; then
		echo "   Extracting frames from video ${VIDEO_PATH}..."
		mkdir -p "${VIDEO_DIR}"
		ffmpeg -i "${VIDEO_PATH}" -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" "${VIDEO_DIR}/%05d.jpg" -y
		echo "   Frames extracted to ${VIDEO_DIR}"
	fi
else
	MODE="davis"
	VIDEO_NAME="${DAVIS_SEQ}"

	if [[ -d "${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}" ]]; then
		VIDEO_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
	elif [[ -d "${DAVIS_INPUT_ROOT}/${VIDEO_NAME}" ]]; then
		VIDEO_DIR="${DAVIS_INPUT_ROOT}/${VIDEO_NAME}"
	else
		echo "ERROR: DAVIS sequence not found in either:"
		echo "  ${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
		echo "  ${DAVIS_INPUT_ROOT}/${VIDEO_NAME}"
		exit 1
	fi

	if [[ -d "${DAVIS_INPUT_ROOT}/Annotations/480p/${VIDEO_NAME}" ]]; then
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/Annotations/480p/${VIDEO_NAME}"
	elif [[ -d "${DAVIS_INPUT_ROOT}/Annotations_unsupervised/480p/${VIDEO_NAME}" ]]; then
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/Annotations_unsupervised/480p/${VIDEO_NAME}"
	else
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/${VIDEO_NAME}_mask"
	fi

	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt_davis/${VIDEO_NAME}"
fi

VGGT_INPUT_ROOT="${OUTPUTS_DIR}/vggt_input"
VGGT_SCENE_INPUT="${VGGT_INPUT_ROOT}/${VIDEO_NAME}"
VGGT_OUTPUT_ROOT="${OUTPUTS_DIR}/vggt4d_outputs"
VGGT_SCENE_OUTPUT="${VGGT_OUTPUT_ROOT}/${VIDEO_NAME}"

NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_vggt"
VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_vggt_vis"
SEG_DEMO_DIR="${VIS_ROOT}/seg_demo"
MASK_COMPARE_DIR="${VIS_ROOT}/mask_compare"
INPAINT_5_DIR="${VIS_ROOT}/inpaint_5frames"
MASK_VIDEO_PATH="${VIS_ROOT}/mask_overlay.mp4"

PROPAINTER_OUT_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_propainter"
PROPAINTER_VIDEO_PATH="${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/inpaint_out.mp4"
FINAL_VIDEO_PATH="${OUTPUTS_DIR}/inpaint_out.mp4"

DAVIS_EVAL_RESULTS_ROOT="${OUTPUTS_DIR}/davis_eval_results"
DAVIS_EVAL_SEQ_DIR="${DAVIS_EVAL_RESULTS_ROOT}/${VIDEO_NAME}"
DAVIS_EVAL_SUBSET_ROOT="${OUTPUTS_DIR}/davis_eval_subset"
DAVIS_CSV_PATH="${DAVIS_EVAL_RESULTS_ROOT}/global_results-val.csv"
GT_MASK_DIR_FOR_METRICS="${GT_MASK_DIR}"
PRED_FRAMES_DIR="${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/frames"
GT_FRAMES_DIR_FOR_METRICS="${GT_FRAMES_DIR}"

mkdir -p "${OUTPUTS_DIR}"

# Ensure DAVIS_INPUT_ROOT and DAVIS_GT_ROOT are absolute paths
if [[ "${DAVIS_INPUT_ROOT}" != /* ]]; then
  DAVIS_INPUT_ROOT="${ROOT_DIR}/${DAVIS_INPUT_ROOT}"
fi
if [[ "${DAVIS_GT_ROOT}" != /* ]]; then
  DAVIS_GT_ROOT="${ROOT_DIR}/${DAVIS_GT_ROOT}"
fi

if [[ "${MODE}" == "video" ]]; then
	if [[ ! -d "${VIDEO_DIR}" ]]; then
		echo "[1/7] Splitting input video into frames in conda env: ${PREPROCESS_ENV}"
		cd "${ROOT_DIR}"
		conda run -n "${PREPROCESS_ENV}" python "${ROOT_DIR}/pipeline_vggt4d/sam2_preprocess.py" \
			--root_dir "${ROOT_DIR}" \
			--video "${VIDEO_PATH}"
	fi
fi

echo "[2/7] Preparing VGGT4D scene input"
rm -rf "${VGGT_INPUT_ROOT}"
mkdir -p "${VGGT_INPUT_ROOT}"
ln -s "${VIDEO_DIR}" "${VGGT_SCENE_INPUT}"

echo "[3/7] Running VGGT4D dynamic mask extraction in conda env: ${VGGT_ENV}"
cd "${VGGT_DIR}"
VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-20}"
if false \
	--input_dir "${VGGT_INPUT_ROOT}" \
	--output_dir "${VGGT_OUTPUT_ROOT}"; then
	echo "VGGT4D full-sequence inference finished."
else
	echo "WARN: Full-sequence VGGT4D run failed (likely OOM). Falling back to chunked inference."
	cd "${ROOT_DIR}"

	FRAME_LIST=( $(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | sort) )
	N_FRAMES="${#FRAME_LIST[@]}"
	if [[ "${N_FRAMES}" -eq 0 ]]; then
		echo "ERROR: no frames found in ${VIDEO_DIR}"
		exit 1
	fi

	rm -rf "${VGGT_OUTPUT_ROOT}" "${OUTPUTS_DIR}/vggt_chunks"
	mkdir -p "${VGGT_SCENE_OUTPUT}"

	START=0
	while [[ "${START}" -lt "${N_FRAMES}" ]]; do
		END=$((START + VGGT_CHUNK_SIZE))
		if [[ "${END}" -gt "${N_FRAMES}" ]]; then
			END="${N_FRAMES}"
		fi

		CHUNK_TAG="${START}_${END}"
		CHUNK_ROOT="${OUTPUTS_DIR}/vggt_chunks/chunk_${CHUNK_TAG}"
		CHUNK_INPUT_ROOT="${CHUNK_ROOT}/input"
		CHUNK_SCENE_INPUT="${CHUNK_INPUT_ROOT}/${VIDEO_NAME}"
		CHUNK_OUTPUT_ROOT="${CHUNK_ROOT}/output"
		CHUNK_SCENE_OUTPUT="${CHUNK_OUTPUT_ROOT}/${VIDEO_NAME}"

		mkdir -p "${CHUNK_SCENE_INPUT}"

		I="${START}"
		while [[ "${I}" -lt "${END}" ]]; do
			FRAME_PATH="${FRAME_LIST[${I}]}"
			FRAME_NAME="$(basename "${FRAME_PATH}")"
			ln -s "${FRAME_PATH}" "${CHUNK_SCENE_INPUT}/${FRAME_NAME}"
			I=$((I + 1))
		done

		echo "  Chunk ${START}:${END}"
		cd "${VGGT_DIR}"
		conda run -n "${VGGT_ENV}" python demo_vggt4d.py \
			--input_dir "${CHUNK_INPUT_ROOT}" \
			--output_dir "${CHUNK_OUTPUT_ROOT}" \
			--dyn_threshold_scale "${VGGT_THRESHOLD_SCALE}"

		if [[ ! -d "${CHUNK_SCENE_OUTPUT}" ]]; then
			echo "ERROR: missing chunk output dir: ${CHUNK_SCENE_OUTPUT}"
			exit 1
		fi

		LOCAL_MASKS=( $(find "${CHUNK_SCENE_OUTPUT}" -maxdepth 1 -type f -name 'dynamic_mask_*.png' | sort) )
		LOCAL_N="${#LOCAL_MASKS[@]}"
		EXPECTED_N=$((END - START))
		if [[ "${LOCAL_N}" -ne "${EXPECTED_N}" ]]; then
			echo "ERROR: chunk mask count mismatch for ${CHUNK_TAG}: got ${LOCAL_N}, expected ${EXPECTED_N}"
			exit 1
		fi

		J=0
		while [[ "${J}" -lt "${LOCAL_N}" ]]; do
			GLOBAL_IDX=$((START + J))
			cp -f "${LOCAL_MASKS[${J}]}" "${VGGT_SCENE_OUTPUT}/dynamic_mask_$(printf '%04d' "${GLOBAL_IDX}").png"
			J=$((J + 1))
		done

		START="${END}"
	done

	echo "VGGT4D chunked inference finished."
fi

if [[ ! -d "${VGGT_SCENE_OUTPUT}" ]]; then
	echo "ERROR: VGGT4D output scene not found: ${VGGT_SCENE_OUTPUT}"
	exit 1
fi

echo "[4/7] Converting VGGT dynamic masks to ProPainter/DAVIS mask format"
cd "${ROOT_DIR}"
conda run -n "${VGGT_ENV}" python "${ROOT_DIR}/pipeline_vggt4d/convert_vggt_masks.py" \
	--src_dir "${VGGT_SCENE_OUTPUT}" \
	--dst_dir "${NEW_MASK_DIR}" \
	--frame_dir "${VIDEO_DIR}" \
	--threshold 0

echo "[5/7] Rendering mask demos"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipeline_vggt4d/postprocess_vggt.py" \
	--frame_dir "${VIDEO_DIR}" \
	--new_mask_dir "${NEW_MASK_DIR}" \
	--old_mask_dir "${OLD_MASK_DIR}" \
	--seg_demo_dir "${SEG_DEMO_DIR}" \
	--mask_compare_dir "${MASK_COMPARE_DIR}" \
	--inpaint_frames_dir "${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/frames" \
	--inpaint_5_dir "${INPAINT_5_DIR}" \
	--num 5 \
	--mask_video_path "${MASK_VIDEO_PATH}"

if [[ ! -d "${NEW_MASK_DIR}" ]]; then
	echo "ERROR: missing mask directory ${NEW_MASK_DIR}"
	exit 1
fi

echo "[6/7] Running ProPainter inpainting in conda env: ${PROPAINTER_ENV}"
cd "${PROPAINTER_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
PROPAINTER_RESIZE_RATIO="${PROPAINTER_RESIZE_RATIO:-0.75}"
PROPAINTER_SUBVIDEO_LENGTH="${PROPAINTER_SUBVIDEO_LENGTH:-40}"
PROPAINTER_NEIGHBOR_LENGTH="${PROPAINTER_NEIGHBOR_LENGTH:-8}"
PROPAINTER_RAFT_ITER="${PROPAINTER_RAFT_ITER:-12}"
PROPAINTER_FP16="${PROPAINTER_FP16:-1}"

FP16_FLAG=""
if [[ "${PROPAINTER_FP16}" == "1" ]]; then
	FP16_FLAG="--fp16"
fi

echo "      resize_ratio=${PROPAINTER_RESIZE_RATIO}, subvideo_length=${PROPAINTER_SUBVIDEO_LENGTH}, neighbor_length=${PROPAINTER_NEIGHBOR_LENGTH}, raft_iter=${PROPAINTER_RAFT_ITER}, fp16=${PROPAINTER_FP16}, alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
conda run -n "${PROPAINTER_ENV}" python inference_propainter.py \
	--video "${VIDEO_DIR}" \
	--mask "${NEW_MASK_DIR}" \
	--output "${PROPAINTER_OUT_ROOT}" \
	--resize_ratio "${PROPAINTER_RESIZE_RATIO}" \
	--subvideo_length "${PROPAINTER_SUBVIDEO_LENGTH}" \
	--neighbor_length "${PROPAINTER_NEIGHBOR_LENGTH}" \
	--raft_iter "${PROPAINTER_RAFT_ITER}" \
	${FP16_FLAG} \
	--save_frames

echo "[7/7] Exporting inpaint samples and optional DAVIS eval"
cd "${ROOT_DIR}"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipeline_vggt4d/postprocess_vggt.py" \
	--frame_dir "${VIDEO_DIR}" \
	--new_mask_dir "${NEW_MASK_DIR}" \
	--old_mask_dir "${OLD_MASK_DIR}" \
	--seg_demo_dir "${SEG_DEMO_DIR}" \
	--mask_compare_dir "${MASK_COMPARE_DIR}" \
	--inpaint_frames_dir "${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/frames" \
	--inpaint_5_dir "${INPAINT_5_DIR}" \
	--num 5 \
	--mask_video_path "${MASK_VIDEO_PATH}"

if [[ -f "${PROPAINTER_VIDEO_PATH}" ]]; then
	cp -f "${PROPAINTER_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
fi

if [[ "${MODE}" == "davis" && "${EVAL_DAVIS}" == "1" ]]; then
	rm -rf "${DAVIS_EVAL_RESULTS_ROOT}"
	mkdir -p "${DAVIS_EVAL_SEQ_DIR}"

	VGGT_EVAL_TARGET_OBJECTS="${VGGT_EVAL_TARGET_OBJECTS:-2}"
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipeline_vggt4d/prepare_davis_eval_masks_multi.py" \
		--src_dir "${NEW_MASK_DIR}" \
		--dst_dir "${DAVIS_EVAL_SEQ_DIR}" \
		--max_eval_labels 20 \
		--target_objects "${VGGT_EVAL_TARGET_OBJECTS}"

	if [[ "${DAVIS_TASK}" == "semi-supervised" ]]; then
		ANN_FOLDER="Annotations"
		ALT_ANN_FOLDER="Annotations_unsupervised"
	else
		ANN_FOLDER="Annotations_unsupervised"
		ALT_ANN_FOLDER="Annotations"
	fi

	GT_SEQ_DIR="${DAVIS_GT_ROOT}/${ANN_FOLDER}/480p/${VIDEO_NAME}"
	if [[ ! -d "${GT_SEQ_DIR}" ]]; then
		ALT_GT_SEQ_DIR="${DAVIS_GT_ROOT}/${ALT_ANN_FOLDER}/480p/${VIDEO_NAME}"
		if [[ -d "${ALT_GT_SEQ_DIR}" ]]; then
			ANN_FOLDER="${ALT_ANN_FOLDER}"
			GT_SEQ_DIR="${ALT_GT_SEQ_DIR}"
		else
			echo "ERROR: ground-truth sequence not found for ${VIDEO_NAME}"
			exit 1
		fi
	fi

	rm -rf "${DAVIS_EVAL_SUBSET_ROOT}"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/JPEGImages/480p"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/${ANN_FOLDER}/480p"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/ImageSets/2017"
	ln -s "${VIDEO_DIR}" "${DAVIS_EVAL_SUBSET_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
	ln -s "${GT_SEQ_DIR}" "${DAVIS_EVAL_SUBSET_ROOT}/${ANN_FOLDER}/480p/${VIDEO_NAME}"
	printf "%s\n" "${VIDEO_NAME}" > "${DAVIS_EVAL_SUBSET_ROOT}/ImageSets/2017/val.txt"
	GT_MASK_DIR_FOR_METRICS="${GT_SEQ_DIR}"

	cd "${ROOT_DIR}/davis2017-evaluation"
	conda run -n "${DAVIS_ENV}" python evaluation_method.py \
		--task "${DAVIS_TASK}" \
		--set val \
		--davis_path "${DAVIS_EVAL_SUBSET_ROOT}" \
		--results_path "${DAVIS_EVAL_RESULTS_ROOT}"
fi

echo "[Metrics] Computing JM/JR and optional PSNR/SSIM"
METRICS_DIR="${OUTPUTS_DIR}/metrics"
METRICS_CMD=(
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/evaluate_metrics.py"
	--output_dir "${METRICS_DIR}"
	--part_label "${PART_LABEL}"
	--experiment_name "vggt4d"
	--pred_mask_dir "${NEW_MASK_DIR}"
	--pred_video "${FINAL_VIDEO_PATH}"
	--pred_frames_dir "${PRED_FRAMES_DIR}"
)

if [[ -f "${DAVIS_CSV_PATH}" ]]; then
	METRICS_CMD+=(--davis_csv "${DAVIS_CSV_PATH}")
fi
if [[ -n "${GT_MASK_DIR_FOR_METRICS}" ]]; then
	METRICS_CMD+=(--gt_mask_dir "${GT_MASK_DIR_FOR_METRICS}")
fi
if [[ -n "${GT_VIDEO}" ]]; then
	METRICS_CMD+=(--gt_video "${GT_VIDEO}")
fi

# Always set GT_FRAMES_DIR_FOR_METRICS
if [[ "${MODE}" == "davis" ]]; then
	GT_FRAMES_DIR_FOR_METRICS="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
else
	GT_FRAMES_DIR_FOR_METRICS="${VIDEO_DIR}"
fi
METRICS_CMD+=(--gt_frames_dir "${GT_FRAMES_DIR_FOR_METRICS}")

"${METRICS_CMD[@]}" || echo "WARN: metrics evaluation failed"

echo "Done. Outputs:"
echo "- VGGT output scene:  ${VGGT_SCENE_OUTPUT}"
echo "- VGGT mask dir:      ${NEW_MASK_DIR}"
echo "- Segmentation demos: ${SEG_DEMO_DIR}"
echo "- Mask comparisons:   ${MASK_COMPARE_DIR}"
echo "- Inpaint 5 frames:   ${INPAINT_5_DIR}"
echo "- Mask overlay video: ${MASK_VIDEO_PATH}"
echo "- Inpaint video:      ${FINAL_VIDEO_PATH}"
if [[ "${MODE}" == "davis" && "${EVAL_DAVIS}" == "1" ]]; then
	echo "- DAVIS CSV results:  ${DAVIS_EVAL_RESULTS_ROOT}/global_results-val.csv"
fi
echo "- Metric summary:    ${METRICS_DIR}/metrics_summary.json"
