#!/usr/bin/env bash
# VGGT4D first-frame init mask -> Track-Anything (XMem) full-video masks -> DiffuEraser inpainting.
# Mirrors pipelines/vggt4dsam3_diffueraser for VGGT + DiffuEraser wiring; replaces SAM3 with Track-Anything.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRACKANYTHING_ENV="${TRACKANYTHING_ENV:-trackanything}"
VGGT_ENV="${VGGT_ENV:-vggt}"
PREPROCESS_ENV="${PREPROCESS_ENV:-sam3}"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-diffueraser}"
DAVIS_ENV="${DAVIS_ENV:-davis}"

VGGT_DIR="${ROOT_DIR}/external/VGGT4D"
TRACKANYTHING_DIR="${ROOT_DIR}/external/Track-Anything"
TRACKANYTHING_MASKS_PY="${ROOT_DIR}/pipelines/trackanything_diffueraser/trackanything_masks.py"
DIFFUERASER_DIR="${ROOT_DIR}/external/DiffuEraser"
INPUTS_DIR="${ROOT_DIR}/data/inputs"

DIFFUERASER_WEIGHTS_ROOT="${DIFFUERASER_WEIGHTS_ROOT:-${DIFFUERASER_DIR}/weights}"
DIFFUERASER_BASE_MODEL="${DIFFUERASER_BASE_MODEL:-${DIFFUERASER_WEIGHTS_ROOT}/stable-diffusion-v1-5}"
DIFFUERASER_VAE="${DIFFUERASER_VAE:-${DIFFUERASER_WEIGHTS_ROOT}/sd-vae-ft-mse}"
DIFFUERASER_MODEL="${DIFFUERASER_MODEL:-${DIFFUERASER_WEIGHTS_ROOT}/diffuEraser}"
DIFFUERASER_PROPAINTER="${DIFFUERASER_PROPAINTER:-${DIFFUERASER_WEIGHTS_ROOT}/propainter}"

VIDEO_PATH=""
DAVIS_SEQ=""
DAVIS_INPUT_ROOT="${ROOT_DIR}/data/DAVIS"
EVAL_DAVIS=1
DAVIS_GT_ROOT="${ROOT_DIR}/data/DAVIS"
DAVIS_TASK="unsupervised"
PART_LABEL="part3"
GT_MASK_DIR=""
GT_VIDEO=""
GT_FRAMES_DIR=""

VGGT_THRESHOLD_SCALE="${VGGT_THRESHOLD_SCALE:-0.7}"
VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES:-20}"
VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-20}"

TRACKANYTHING_DEVICE="${TRACKANYTHING_DEVICE:-cuda:0}"
TRACKANYTHING_SAM_MODEL_TYPE="${TRACKANYTHING_SAM_MODEL_TYPE:-vit_h}"
TRACKANYTHING_INIT_SOURCE="${TRACKANYTHING_INIT_SOURCE:-mask}"
TRACKANYTHING_INIT_MASK_OVERRIDE=""
TRACKANYTHING_YOLO_MODEL="${TRACKANYTHING_YOLO_MODEL:-}"
TRACKANYTHING_YOLO_CONF="${TRACKANYTHING_YOLO_CONF:-0.25}"
TRACKANYTHING_CLASSES="${TRACKANYTHING_CLASSES:-all}"
TRACKANYTHING_MAX_OBJECTS="${TRACKANYTHING_MAX_OBJECTS:-4}"
TRACKANYTHING_MIN_AREA_RATIO="${TRACKANYTHING_MIN_AREA_RATIO:-0.0005}"
TRACKANYTHING_MAX_AREA_RATIO="${TRACKANYTHING_MAX_AREA_RATIO:-0.80}"
TRACKANYTHING_MASK_ONLY="${TRACKANYTHING_MASK_ONLY:-0}"

usage() {
	cat <<EOF
Usage:
  Video mode:
    $0 --video /path/to/video.mp4 [--dyn_threshold_scale 0.7] [--vggt_max_frames 20|100|all|0] [--vggt_chunk_size 20]

  DAVIS mode:
    $0 --davis_seq bmx-trees [--davis_input_root data/DAVIS] [--davis_gt_root data/DAVIS] [--davis_task unsupervised] [--dyn_threshold_scale 0.7] [--vggt_max_frames 20|100|all|0] [--vggt_chunk_size 20]

Pipeline: VGGT4D (init) -> gen_first_mask_from_vggt -> Track-Anything -> postprocess -> DiffuEraser.

Optional (override VGGT init; rare):
  --init_source yolo|sam_auto|mask   default mask (from VGGT)
  --init_mask /path/to.png          use this instead of VGGT output when set

Other Track-Anything flags (defaults match trackanything_diffueraser.sh):
  --device cuda:0  --sam_model_type vit_h  --mask_only 1
  --classes all  --yolo_conf 0.25  --max_objects 4  (only if init_source is yolo/sam_auto)

VGGT init coverage (only affects VGGT -> merge_all init mask; full clip still tracked after):
  --vggt_max_frames N        Run VGGT on first N frames only (default 20). Example: 100 for bike-packing-style static starts.
  --vggt_max_frames all|0    Run VGGT on every frame (slower, richer motion).
  --vggt_chunk_size 20       Chunk size for demo_vggt4d.py (default 20)
  Env: VGGT_MAX_FRAMES, VGGT_CHUNK_SIZE, VGGT_THRESHOLD_SCALE
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--video) VIDEO_PATH="$2"; shift 2 ;;
		--davis_seq) DAVIS_SEQ="$2"; shift 2 ;;
		--davis_input_root) DAVIS_INPUT_ROOT="$2"; shift 2 ;;
		--eval_davis) EVAL_DAVIS="$2"; shift 2 ;;
		--davis_gt_root) DAVIS_GT_ROOT="$2"; shift 2 ;;
		--davis_task) DAVIS_TASK="$2"; shift 2 ;;
		--dyn_threshold_scale) VGGT_THRESHOLD_SCALE="$2"; shift 2 ;;
		--part_label) PART_LABEL="$2"; shift 2 ;;
		--gt_mask_dir) GT_MASK_DIR="$2"; shift 2 ;;
		--gt_video) GT_VIDEO="$2"; shift 2 ;;
		--gt_frames_dir) GT_FRAMES_DIR="$2"; shift 2 ;;
		--init_source) TRACKANYTHING_INIT_SOURCE="$2"; shift 2 ;;
		--init_mask) TRACKANYTHING_INIT_MASK_OVERRIDE="$2"; shift 2 ;;
		--classes) TRACKANYTHING_CLASSES="$2"; shift 2 ;;
		--yolo_conf) TRACKANYTHING_YOLO_CONF="$2"; shift 2 ;;
		--yolo_model) TRACKANYTHING_YOLO_MODEL="$2"; shift 2 ;;
		--max_objects) TRACKANYTHING_MAX_OBJECTS="$2"; shift 2 ;;
		--min_area_ratio) TRACKANYTHING_MIN_AREA_RATIO="$2"; shift 2 ;;
		--max_area_ratio) TRACKANYTHING_MAX_AREA_RATIO="$2"; shift 2 ;;
		--device) TRACKANYTHING_DEVICE="$2"; shift 2 ;;
		--sam_model_type) TRACKANYTHING_SAM_MODEL_TYPE="$2"; shift 2 ;;
		--mask_only) TRACKANYTHING_MASK_ONLY="$2"; shift 2 ;;
		--vggt_max_frames) VGGT_MAX_FRAMES="$2"; shift 2 ;;
		--vggt_chunk_size) VGGT_CHUNK_SIZE="$2"; shift 2 ;;
		-h|--help) usage; exit 0 ;;
		*) echo "Unknown argument: $1"; usage; exit 1 ;;
	esac
done

if [[ "${DAVIS_INPUT_ROOT}" != /* ]]; then
	DAVIS_INPUT_ROOT="${ROOT_DIR}/${DAVIS_INPUT_ROOT}"
fi
if [[ "${DAVIS_GT_ROOT}" != /* ]]; then
	DAVIS_GT_ROOT="${ROOT_DIR}/${DAVIS_GT_ROOT}"
fi

if [[ -z "${VIDEO_PATH}" && -z "${DAVIS_SEQ}" ]]; then
	usage
	exit 1
fi
if [[ -n "${VIDEO_PATH}" && -n "${DAVIS_SEQ}" ]]; then
	echo "ERROR: provide either --video or --davis_seq, not both"
	exit 1
fi
if [[ "${DAVIS_TASK}" != "semi-supervised" && "${DAVIS_TASK}" != "unsupervised" ]]; then
	echo "ERROR: --davis_task must be semi-supervised or unsupervised"
	exit 1
fi
if [[ ! -d "${VGGT_DIR}" ]]; then
	echo "ERROR: VGGT4D dir not found: ${VGGT_DIR}"
	exit 1
fi
if [[ ! -d "${TRACKANYTHING_DIR}" ]]; then
	echo "ERROR: Track-Anything dir not found: ${TRACKANYTHING_DIR}"
	exit 1
fi
if [[ ! -f "${TRACKANYTHING_MASKS_PY}" ]]; then
	echo "ERROR: Track-Anything mask script not found: ${TRACKANYTHING_MASKS_PY}"
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
	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt_trackanything_diffueraser/${VIDEO_NAME}"
else
	MODE="davis"
	VIDEO_NAME="${DAVIS_SEQ}"
	if [[ -d "${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}" ]]; then
		VIDEO_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
	elif [[ -d "${DAVIS_INPUT_ROOT}/${VIDEO_NAME}" ]]; then
		VIDEO_DIR="${DAVIS_INPUT_ROOT}/${VIDEO_NAME}"
	else
		echo "ERROR: DAVIS sequence not found: ${VIDEO_NAME}"
		exit 1
	fi
	if [[ -d "${DAVIS_INPUT_ROOT}/Annotations/480p/${VIDEO_NAME}" ]]; then
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/Annotations/480p/${VIDEO_NAME}"
	elif [[ -d "${DAVIS_INPUT_ROOT}/Annotations_unsupervised/480p/${VIDEO_NAME}" ]]; then
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/Annotations_unsupervised/480p/${VIDEO_NAME}"
	else
		OLD_MASK_DIR="${DAVIS_INPUT_ROOT}/${VIDEO_NAME}_mask"
	fi
	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt_trackanything_diffueraser_davis/${VIDEO_NAME}"
fi

VGGT_INPUT_ROOT="${OUTPUTS_DIR}/vggt_input"
VGGT_SCENE_INPUT="${VGGT_INPUT_ROOT}/${VIDEO_NAME}"
VGGT_OUTPUT_ROOT="${OUTPUTS_DIR}/vggt4d_outputs"
VGGT_SCENE_OUTPUT="${VGGT_OUTPUT_ROOT}/${VIDEO_NAME}"
INIT_MASK_DIR="${OUTPUTS_DIR}/tmp_vggt_init_masks/${VIDEO_NAME}"

RAW_MASK_DIR="${OUTPUTS_DIR}/tmp_trackanything_masks_raw/${VIDEO_NAME}"
NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_trackanything"
VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_trackanything_vis"
TRACK_VIS_DIR="${VIS_ROOT}/track_vis"
SEG_DEMO_DIR="${VIS_ROOT}/seg_demo"
MASK_COMPARE_DIR="${VIS_ROOT}/mask_compare"
INPAINT_5_DIR="${VIS_ROOT}/inpaint_5frames"
MASK_VIDEO_PATH="${VIS_ROOT}/mask_overlay.mp4"

DIFFUERASER_OUT_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_diffueraser"
DIFFUERASER_FRAMES_DIR="${DIFFUERASER_OUT_ROOT}/frames"
DIFFUERASER_VIDEO_PATH="${DIFFUERASER_OUT_ROOT}/diffueraser_result.mp4"
DIFFUERASER_RAW_VIDEO_PATH="${DIFFUERASER_OUT_ROOT}/diffueraser_result_raw.mp4"
DIFFUERASER_INPUT_VIDEO_PATH="${DIFFUERASER_OUT_ROOT}/input_video.mp4"
DIFFUERASER_INPUT_MASK_PATH="${DIFFUERASER_OUT_ROOT}/input_mask.mp4"
FINAL_VIDEO_PATH="${OUTPUTS_DIR}/inpaint_out.mp4"

DAVIS_EVAL_RESULTS_ROOT="${OUTPUTS_DIR}/davis_eval_results"
DAVIS_EVAL_SEQ_DIR="${DAVIS_EVAL_RESULTS_ROOT}/${VIDEO_NAME}"
DAVIS_EVAL_SUBSET_ROOT="${OUTPUTS_DIR}/davis_eval_subset"
DAVIS_CSV_PATH="${DAVIS_EVAL_RESULTS_ROOT}/global_results-val.csv"
GT_MASK_DIR_FOR_METRICS="${GT_MASK_DIR}"
PRED_FRAMES_DIR="${DIFFUERASER_FRAMES_DIR}"
GT_FRAMES_DIR_FOR_METRICS="${GT_FRAMES_DIR}"

mkdir -p "${OUTPUTS_DIR}"

if [[ "${MODE}" == "video" ]]; then
	if [[ ! -d "${VIDEO_DIR}" ]]; then
		echo "[1/9] Splitting input video into frames in conda env: ${PREPROCESS_ENV}"
		cd "${ROOT_DIR}"
		conda run -n "${PREPROCESS_ENV}" python "${ROOT_DIR}/pipelines/yolosam2/sam2_preprocess.py" \
			--root_dir "${ROOT_DIR}" \
			--video "${VIDEO_PATH}"
	fi
fi

echo "[2/9] Preparing VGGT4D scene input"
rm -rf "${VGGT_INPUT_ROOT}"
mkdir -p "${VGGT_INPUT_ROOT}"
ln -s "${VIDEO_DIR}" "${VGGT_SCENE_INPUT}"

echo "[3/9] Running VGGT4D dynamic mask extraction in conda env: ${VGGT_ENV}"
cd "${ROOT_DIR}"

FRAME_LIST=( $(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | sort) )
TOTAL_FRAMES="${#FRAME_LIST[@]}"
if [[ "${TOTAL_FRAMES}" -eq 0 ]]; then
	echo "ERROR: no frames found in ${VIDEO_DIR}"
	exit 1
fi

VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-20}"
VGGT_THRESHOLD_SCALE="${VGGT_THRESHOLD_SCALE:-0.7}"
VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES:-20}"
VMF_LC="$(printf '%s' "${VGGT_MAX_FRAMES}" | tr '[:upper:]' '[:lower:]')"
if [[ "${VMF_LC}" == "all" ]] || [[ "${VGGT_MAX_FRAMES}" == "0" ]]; then
	N_FRAMES="${TOTAL_FRAMES}"
	echo "INFO: VGGT init mask will merge dynamic masks from all ${N_FRAMES} frames (vggt_max_frames=all)."
elif [[ "${TOTAL_FRAMES}" -gt "${VGGT_MAX_FRAMES}" ]]; then
	N_FRAMES="${VGGT_MAX_FRAMES}"
	FRAME_LIST=("${FRAME_LIST[@]:0:${N_FRAMES}}")
	echo "INFO: VGGT init uses first ${N_FRAMES} of ${TOTAL_FRAMES} frames (cap vggt_max_frames=${VGGT_MAX_FRAMES})."
else
	N_FRAMES="${TOTAL_FRAMES}"
	echo "INFO: VGGT init uses all ${N_FRAMES} frames (sequence shorter than vggt_max_frames=${VGGT_MAX_FRAMES})."
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

	echo "  Chunk ${START}:${END} (threshold_scale=${VGGT_THRESHOLD_SCALE})"
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

echo "VGGT4D chunked inference finished (${N_FRAMES} frames)."

if [[ ! -d "${VGGT_SCENE_OUTPUT}" ]]; then
	echo "ERROR: VGGT4D output scene not found: ${VGGT_SCENE_OUTPUT}"
	exit 1
fi

echo "[4/9] Building indexed init mask from VGGT4D output (merge_all, same as vggt4dsam3_diffueraser)"
rm -rf "${INIT_MASK_DIR}"
mkdir -p "${INIT_MASK_DIR}"
cd "${ROOT_DIR}"
conda run -n "${VGGT_ENV}" python "${ROOT_DIR}/pipelines/vggt4dsam3/gen_first_mask_from_vggt.py" \
	--vggt_scene_output "${VGGT_SCENE_OUTPUT}" \
	--output_dir "${INIT_MASK_DIR}" \
	--threshold 0 \
	--merge_all

INIT_MASK_COUNT="$(find "${INIT_MASK_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
if [[ "${INIT_MASK_COUNT}" == "0" ]]; then
	echo "ERROR: init mask was not created in ${INIT_MASK_DIR}"
	exit 1
fi

if [[ -n "${TRACKANYTHING_INIT_MASK_OVERRIDE}" ]]; then
	VGGT_INIT_MASK="${TRACKANYTHING_INIT_MASK_OVERRIDE}"
else
	VGGT_INIT_MASK="$(find "${INIT_MASK_DIR}" -maxdepth 1 -type f -name '*.png' | sort | head -n 1)"
fi
if [[ ! -f "${VGGT_INIT_MASK}" ]]; then
	echo "ERROR: VGGT init mask PNG not found: ${VGGT_INIT_MASK}"
	exit 1
fi
echo "INFO: Track-Anything init mask from VGGT: ${VGGT_INIT_MASK}"

if [[ "${TRACKANYTHING_INIT_SOURCE}" == "mask" ]]; then
	TRACK_INIT_MASK_ARG=(--init_mask "${VGGT_INIT_MASK}")
else
	TRACK_INIT_MASK_ARG=()
fi

echo "[5/9] Running Track-Anything mask tracking in conda env: ${TRACKANYTHING_ENV}"
rm -rf "${RAW_MASK_DIR}" "${NEW_MASK_DIR}" "${TRACK_VIS_DIR}"
mkdir -p "${RAW_MASK_DIR}" "${NEW_MASK_DIR}" "${TRACK_VIS_DIR}"

TRACK_CMD=(
	conda run -n "${TRACKANYTHING_ENV}" python "${TRACKANYTHING_MASKS_PY}"
	--frame_dir "${VIDEO_DIR}"
	--raw_mask_dir "${RAW_MASK_DIR}"
	--binary_mask_dir "${NEW_MASK_DIR}"
	--vis_dir "${TRACK_VIS_DIR}"
	--trackanything_dir "${TRACKANYTHING_DIR}"
	--device "${TRACKANYTHING_DEVICE}"
	--sam_model_type "${TRACKANYTHING_SAM_MODEL_TYPE}"
	--init_source "${TRACKANYTHING_INIT_SOURCE}"
	--classes "${TRACKANYTHING_CLASSES}"
	--yolo_conf "${TRACKANYTHING_YOLO_CONF}"
	--max_objects "${TRACKANYTHING_MAX_OBJECTS}"
	--min_area_ratio "${TRACKANYTHING_MIN_AREA_RATIO}"
	--max_area_ratio "${TRACKANYTHING_MAX_AREA_RATIO}"
)
if [[ "${#TRACK_INIT_MASK_ARG[@]}" -gt 0 ]]; then
	TRACK_CMD+=("${TRACK_INIT_MASK_ARG[@]}")
fi
if [[ -n "${TRACKANYTHING_YOLO_MODEL}" ]]; then
	TRACK_CMD+=(--yolo_model "${TRACKANYTHING_YOLO_MODEL}")
fi
"${TRACK_CMD[@]}"

MASK_COUNT="$(find "${NEW_MASK_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
if [[ "${MASK_COUNT}" == "0" ]]; then
	echo "ERROR: Track-Anything did not write masks in ${NEW_MASK_DIR}"
	exit 1
fi

echo "[6/9] Rendering mask demos"
mkdir -p "${SEG_DEMO_DIR}" "${MASK_COMPARE_DIR}" "${INPAINT_5_DIR}"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipelines/vggt4dsam3/postprocess_sam3.py" \
	--raw_mask_dir "${RAW_MASK_DIR}" \
	--new_mask_dir "${NEW_MASK_DIR}" \
	--frame_dir "${VIDEO_DIR}" \
	--old_mask_dir "${OLD_MASK_DIR}" \
	--seg_demo_dir "${SEG_DEMO_DIR}" \
	--mask_compare_dir "${MASK_COMPARE_DIR}" \
	--inpaint_frames_dir "${DIFFUERASER_FRAMES_DIR}" \
	--inpaint_5_dir "${INPAINT_5_DIR}" \
	--num 5 \
	--mask_video_path "${MASK_VIDEO_PATH}"

if [[ "${TRACKANYTHING_MASK_ONLY}" == "1" ]]; then
	echo "Done (mask_only=1)"
	echo "  vggt:  ${VGGT_SCENE_OUTPUT}"
	echo "  init:  ${VGGT_INIT_MASK}"
	echo "  masks: ${NEW_MASK_DIR}"
	echo "  vis:   ${VIS_ROOT}"
	exit 0
fi

echo "[7/9] Running DiffuEraser inpainting in conda env: ${DIFFUERASER_ENV}"

diffueraser_try_hf_hub_snapshot() {
	local varname="$1"
	local slug="$2"
	local marker="$3"
	local cur="${!varname:-}"
	if [[ -n "${cur}" && -d "${cur}" ]]; then
		return 0
	fi
	local hub_root="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
	local snaps="${hub_root}/${slug}/snapshots"
	if [[ ! -d "${snaps}" ]]; then
		return 0
	fi
	local d hit=""
	for d in "${snaps}"/*; do
		[[ -d "${d}" ]] || continue
		if [[ -f "${d}/${marker}" ]]; then
			hit="${d}"
			break
		fi
	done
	if [[ -n "${hit}" ]]; then
		echo "INFO: Using Hugging Face Hub cache for ${varname}: ${hit}"
		printf -v "${varname}" '%s' "${hit}"
	fi
}

diffueraser_try_hf_hub_snapshot DIFFUERASER_BASE_MODEL "models--stable-diffusion-v1-5--stable-diffusion-v1-5" "model_index.json"
diffueraser_try_hf_hub_snapshot DIFFUERASER_VAE "models--stabilityai--sd-vae-ft-mse" "config.json"

cd "${DIFFUERASER_DIR}"
PCM_CANON="${DIFFUERASER_DIR}/weights/PCM_Weights"
PCM_MISSPELLED="${DIFFUERASER_DIR}/weights/PCM_weights"
if [[ -L "${PCM_CANON}" && ! -e "${PCM_CANON}" ]]; then
	rm -f "${PCM_CANON}"
fi
if [[ ! -e "${PCM_CANON}" && -d "${PCM_MISSPELLED}" ]]; then
	ln -sfn "$(realpath "${PCM_MISSPELLED}")" "${PCM_CANON}"
fi
DIFFUERASER_PCM_LORA="${PCM_CANON}/sd15/pcm_sd15_smallcfg_2step_converted.safetensors"
if [[ ! -f "${DIFFUERASER_PCM_LORA}" ]]; then
	DIFFUERASER_PCM_FOUND="$(find "${DIFFUERASER_DIR}/weights" -type f -path '*/sd15/pcm_sd15_smallcfg_2step_converted.safetensors' 2>/dev/null | head -n 1)"
	if [[ -n "${DIFFUERASER_PCM_FOUND}" ]]; then
		DIFFUERASER_PCM_FOUND="$(realpath "${DIFFUERASER_PCM_FOUND}")"
		PCM_REPO_ROOT="$(realpath "$(dirname "$(dirname "${DIFFUERASER_PCM_FOUND}")")")"
		rm -f "${PCM_CANON}"
		ln -sfn "${PCM_REPO_ROOT}" "${PCM_CANON}"
	fi
fi

DIFFUERASER_WEIGHT_ERRORS=""
[[ -d "${DIFFUERASER_BASE_MODEL}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - SD1.5 base: ${DIFFUERASER_BASE_MODEL}"
[[ -d "${DIFFUERASER_VAE}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - VAE: ${DIFFUERASER_VAE}"
[[ -d "${DIFFUERASER_MODEL}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - DiffuEraser: ${DIFFUERASER_MODEL}"
[[ -d "${DIFFUERASER_PROPAINTER}" && -f "${DIFFUERASER_PROPAINTER}/ProPainter.pth" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - ProPainter prior: ${DIFFUERASER_PROPAINTER}"
[[ -f "${DIFFUERASER_PCM_LORA}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - PCM LoRA: ${DIFFUERASER_PCM_LORA}"
if [[ -n "${DIFFUERASER_WEIGHT_ERRORS}" ]]; then
	echo "ERROR: DiffuEraser pretrained assets are missing:${DIFFUERASER_WEIGHT_ERRORS}"
	exit 1
fi

DIFFUERASER_FPS="${DIFFUERASER_FPS:-24}"
DIFFUERASER_MASK_DILATION="${DIFFUERASER_MASK_DILATION:-8}"
DIFFUERASER_MAX_IMG_SIZE="${DIFFUERASER_MAX_IMG_SIZE:-960}"
DIFFUERASER_REF_STRIDE="${DIFFUERASER_REF_STRIDE:-10}"
DIFFUERASER_NEIGHBOR_LENGTH="${DIFFUERASER_NEIGHBOR_LENGTH:-10}"
DIFFUERASER_SUBVIDEO_LENGTH="${DIFFUERASER_SUBVIDEO_LENGTH:-50}"

DIFFUERASER_MEDIA_ROOT="${DIFFUERASER_OUT_ROOT}/media_input"
DIFFUERASER_VIDEO_FRAME_DIR="${DIFFUERASER_MEDIA_ROOT}/frames"
DIFFUERASER_MASK_FRAME_DIR="${DIFFUERASER_MEDIA_ROOT}/masks"
rm -rf "${DIFFUERASER_MEDIA_ROOT}"
mkdir -p "${DIFFUERASER_VIDEO_FRAME_DIR}" "${DIFFUERASER_MASK_FRAME_DIR}" "${DIFFUERASER_OUT_ROOT}"

DIFFUERASER_ALL_FRAMES=( $(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | sort) )
DIFFUERASER_ALL_MASKS=( $(find "${NEW_MASK_DIR}" -maxdepth 1 -type f -name '*.png' | sort) )
DIFFUERASER_FRAME_COUNT="${#DIFFUERASER_ALL_FRAMES[@]}"
DIFFUERASER_MASK_COUNT="${#DIFFUERASER_ALL_MASKS[@]}"
if [[ "${DIFFUERASER_FRAME_COUNT}" -eq 0 || "${DIFFUERASER_MASK_COUNT}" -eq 0 ]]; then
	echo "ERROR: DiffuEraser input frames or masks are empty"
	exit 1
fi
if [[ "${DIFFUERASER_FRAME_COUNT}" -ne "${DIFFUERASER_MASK_COUNT}" ]]; then
	echo "WARN: frame/mask count mismatch: frames=${DIFFUERASER_FRAME_COUNT}, masks=${DIFFUERASER_MASK_COUNT}; using shorter length"
	if [[ "${DIFFUERASER_MASK_COUNT}" -lt "${DIFFUERASER_FRAME_COUNT}" ]]; then
		DIFFUERASER_FRAME_COUNT="${DIFFUERASER_MASK_COUNT}"
	fi
fi

I=0
while [[ "${I}" -lt "${DIFFUERASER_FRAME_COUNT}" ]]; do
	ln -s "${DIFFUERASER_ALL_FRAMES[${I}]}" "${DIFFUERASER_VIDEO_FRAME_DIR}/$(printf '%05d' "${I}").jpg"
	ln -s "${DIFFUERASER_ALL_MASKS[${I}]}" "${DIFFUERASER_MASK_FRAME_DIR}/$(printf '%05d' "${I}").png"
	I=$((I + 1))
done

DIFFUERASER_VIDEO_LENGTH="${DIFFUERASER_VIDEO_LENGTH:-$(awk -v n="${DIFFUERASER_FRAME_COUNT}" -v fps="${DIFFUERASER_FPS}" 'BEGIN { q = n / fps; qi = int(q); print (q > qi ? qi + 1 : qi) }')}"
[[ "${DIFFUERASER_VIDEO_LENGTH}" -lt 1 ]] && DIFFUERASER_VIDEO_LENGTH=1

ffmpeg -y -loglevel error -framerate "${DIFFUERASER_FPS}" -start_number 0 \
	-i "${DIFFUERASER_VIDEO_FRAME_DIR}/%05d.jpg" \
	-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" \
	-c:v libx264 -pix_fmt yuv420p "${DIFFUERASER_INPUT_VIDEO_PATH}"
ffmpeg -y -loglevel error -framerate "${DIFFUERASER_FPS}" -start_number 0 \
	-i "${DIFFUERASER_MASK_FRAME_DIR}/%05d.png" \
	-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" \
	-c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p "${DIFFUERASER_INPUT_MASK_PATH}"

conda run -n "${DIFFUERASER_ENV}" python run_diffueraser.py \
	--input_video "${DIFFUERASER_INPUT_VIDEO_PATH}" \
	--input_mask "${DIFFUERASER_INPUT_MASK_PATH}" \
	--video_length "${DIFFUERASER_VIDEO_LENGTH}" \
	--mask_dilation_iter "${DIFFUERASER_MASK_DILATION}" \
	--max_img_size "${DIFFUERASER_MAX_IMG_SIZE}" \
	--save_path "${DIFFUERASER_OUT_ROOT}" \
	--ref_stride "${DIFFUERASER_REF_STRIDE}" \
	--neighbor_length "${DIFFUERASER_NEIGHBOR_LENGTH}" \
	--subvideo_length "${DIFFUERASER_SUBVIDEO_LENGTH}" \
	--base_model_path "${DIFFUERASER_BASE_MODEL}" \
	--vae_path "${DIFFUERASER_VAE}" \
	--diffueraser_path "${DIFFUERASER_MODEL}" \
	--propainter_model_dir "${DIFFUERASER_PROPAINTER}"

if [[ ! -f "${DIFFUERASER_VIDEO_PATH}" ]]; then
	echo "ERROR: DiffuEraser result video not found: ${DIFFUERASER_VIDEO_PATH}"
	exit 1
fi

mv -f "${DIFFUERASER_VIDEO_PATH}" "${DIFFUERASER_RAW_VIDEO_PATH}"
ffmpeg -y -loglevel error -i "${DIFFUERASER_RAW_VIDEO_PATH}" \
	-c:v libx264 -pix_fmt yuv420p -movflags +faststart "${DIFFUERASER_VIDEO_PATH}"
cp -f "${DIFFUERASER_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
rm -rf "${DIFFUERASER_FRAMES_DIR}"
mkdir -p "${DIFFUERASER_FRAMES_DIR}"
ffmpeg -y -loglevel error -i "${DIFFUERASER_VIDEO_PATH}" -start_number 0 "${DIFFUERASER_FRAMES_DIR}/%05d.png"

echo "[8/9] Exporting inpaint samples"
cd "${ROOT_DIR}"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipelines/vggt4dsam3/postprocess_sam3.py" \
	--raw_mask_dir "${RAW_MASK_DIR}" \
	--new_mask_dir "${NEW_MASK_DIR}" \
	--frame_dir "${VIDEO_DIR}" \
	--old_mask_dir "${OLD_MASK_DIR}" \
	--seg_demo_dir "${SEG_DEMO_DIR}" \
	--mask_compare_dir "${MASK_COMPARE_DIR}" \
	--inpaint_frames_dir "${DIFFUERASER_FRAMES_DIR}" \
	--inpaint_5_dir "${INPAINT_5_DIR}" \
	--num 5 \
	--mask_video_path "${MASK_VIDEO_PATH}"

if [[ "${MODE}" == "davis" && "${EVAL_DAVIS}" == "1" ]]; then
	echo "[9/9] Preparing DAVIS-evaluable masks and running DAVIS evaluation"
	rm -rf "${DAVIS_EVAL_RESULTS_ROOT}"
	mkdir -p "${DAVIS_EVAL_SEQ_DIR}"
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipelines/vggt4dsam3/prepare_davis_eval_masks.py" \
		--src_dir "${RAW_MASK_DIR}" \
		--dst_dir "${DAVIS_EVAL_SEQ_DIR}" \
		--max_eval_labels 20 \
		--binary

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

	cd "${ROOT_DIR}/external/davis2017-evaluation"
	conda run -n "${DAVIS_ENV}" python evaluation_method.py \
		--task "${DAVIS_TASK}" \
		--set val \
		--davis_path "${DAVIS_EVAL_SUBSET_ROOT}" \
		--results_path "${DAVIS_EVAL_RESULTS_ROOT}"
else
	echo "[9/9] Skipping DAVIS evaluation"
fi

echo "[Metrics] Computing JM/JR and optional PSNR/SSIM"
METRICS_DIR="${OUTPUTS_DIR}/metrics"
METRICS_CMD=(
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/evaluate_metrics.py"
	--output_dir "${METRICS_DIR}"
	--part_label "${PART_LABEL}"
	--experiment_name "vggt_trackanything_diffueraser"
	--pred_mask_dir "${NEW_MASK_DIR}"
	--pred_video "${FINAL_VIDEO_PATH}"
	--pred_frames_dir "${PRED_FRAMES_DIR}"
)
[[ -f "${DAVIS_CSV_PATH}" ]] && METRICS_CMD+=(--davis_csv "${DAVIS_CSV_PATH}")
[[ -n "${GT_MASK_DIR_FOR_METRICS}" ]] && METRICS_CMD+=(--gt_mask_dir "${GT_MASK_DIR_FOR_METRICS}")
[[ -n "${GT_VIDEO}" ]] && METRICS_CMD+=(--gt_video "${GT_VIDEO}")
[[ -n "${GT_FRAMES_DIR_FOR_METRICS}" ]] && METRICS_CMD+=(--gt_frames_dir "${GT_FRAMES_DIR_FOR_METRICS}")
"${METRICS_CMD[@]}"

echo "Done."
echo "  vggt:  ${VGGT_SCENE_OUTPUT}"
echo "  init:  ${VGGT_INIT_MASK}"
echo "  masks: ${NEW_MASK_DIR}"
echo "  video: ${FINAL_VIDEO_PATH}"
echo "  metrics: ${METRICS_DIR}"
