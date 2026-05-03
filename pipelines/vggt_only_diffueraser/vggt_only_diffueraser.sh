#!/usr/bin/env bash
# VGGT4D per-frame dynamic masks (no SAM3 / Track-Anything) -> postprocess -> DiffuEraser.
# For baseline comparison when motion-based VGGT signal is weak (e.g. nearly static targets).
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VGGT_ENV="${VGGT_ENV:-vggt}"
PREPROCESS_ENV="${PREPROCESS_ENV:-sam3}"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-diffueraser}"
DAVIS_ENV="${DAVIS_ENV:-davis}"

VGGT_DIR="${ROOT_DIR}/external/VGGT4D"
DIFFUERASER_DIR="${ROOT_DIR}/external/DiffuEraser"
INPUTS_DIR="${ROOT_DIR}/data/inputs"
VGGT_MASKS_PY="${ROOT_DIR}/pipelines/vggt_only_diffueraser/vggt_dynamic_masks_to_framewise.py"

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
VGGT_ALIGN_TAIL="${VGGT_ALIGN_TAIL:-hold_last}"

usage() {
	cat <<EOF
Usage:
  Video: $0 --video /path/to/video.mp4 [--scale 0.5] [--vggt_max_frames 20|100|all|0] [--vggt_chunk_size 20]
  DAVIS: $0 --davis_seq bike-packing [--davis_input_root data/DAVIS] [--eval_davis 1|0] [--scale 0.5] [--vggt_max_frames all] ...

Pipeline: VGGT4D -> align dynamic_mask_* to every frame -> postprocess (viz) -> DiffuEraser.

  --scale FLOAT              Same as --dyn_threshold_scale: passed to demo_vggt4d.py as --dyn_threshold_scale (default 0.7).
  --dyn_threshold_scale FLOAT   Alias of --scale (matches other vggt_* pipelines).
  --vggt_align_tail hold_last|zeros   When VGGT ran on fewer frames than the clip: repeat last VGGT mask or zeros (default ${VGGT_ALIGN_TAIL})
  Env: VGGT_THRESHOLD_SCALE, VGGT_MAX_FRAMES, VGGT_CHUNK_SIZE, VGGT_ALIGN_TAIL
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
		--scale|--dyn_threshold_scale) VGGT_THRESHOLD_SCALE="$2"; shift 2 ;;
		--vggt_max_frames) VGGT_MAX_FRAMES="$2"; shift 2 ;;
		--vggt_chunk_size) VGGT_CHUNK_SIZE="$2"; shift 2 ;;
		--vggt_align_tail) VGGT_ALIGN_TAIL="$2"; shift 2 ;;
		--part_label) PART_LABEL="$2"; shift 2 ;;
		--gt_mask_dir) GT_MASK_DIR="$2"; shift 2 ;;
		--gt_video) GT_VIDEO="$2"; shift 2 ;;
		--gt_frames_dir) GT_FRAMES_DIR="$2"; shift 2 ;;
		-h|--help) usage; exit 0 ;;
		*) echo "Unknown: $1"; usage; exit 1 ;;
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
	echo "ERROR: provide either --video or --davis_seq"
	exit 1
fi
if [[ "${DAVIS_TASK}" != "semi-supervised" && "${DAVIS_TASK}" != "unsupervised" ]]; then
	echo "ERROR: --davis_task must be semi-supervised or unsupervised"
	exit 1
fi
if [[ ! -d "${VGGT_DIR}" ]]; then
	echo "ERROR: VGGT4D not found: ${VGGT_DIR}"
	exit 1
fi
if [[ ! -f "${VGGT_MASKS_PY}" ]]; then
	echo "ERROR: missing ${VGGT_MASKS_PY}"
	exit 1
fi

if [[ -n "${VIDEO_PATH}" ]]; then
	MODE="video"
	if [[ ! -f "${VIDEO_PATH}" ]]; then
		echo "ERROR: video not found: ${VIDEO_PATH}"
		exit 1
	fi
	VIDEO_NAME="$(basename "${VIDEO_PATH}")"
	VIDEO_NAME="${VIDEO_NAME%.*}"
	VIDEO_DIR="${INPUTS_DIR}/${VIDEO_NAME}"
	OLD_MASK_DIR="${INPUTS_DIR}/${VIDEO_NAME}_mask"
	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt_only_diffueraser/${VIDEO_NAME}"
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
	OUTPUTS_DIR="${ROOT_DIR}/outputs/vggt_only_diffueraser_davis/${VIDEO_NAME}"
fi

VGGT_INPUT_ROOT="${OUTPUTS_DIR}/vggt_input"
VGGT_SCENE_INPUT="${VGGT_INPUT_ROOT}/${VIDEO_NAME}"
VGGT_OUTPUT_ROOT="${OUTPUTS_DIR}/vggt4d_outputs"
VGGT_SCENE_OUTPUT="${VGGT_OUTPUT_ROOT}/${VIDEO_NAME}"

RAW_MASK_DIR="${OUTPUTS_DIR}/tmp_vggt_only_raw/${VIDEO_NAME}"
NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_vggt_only"
VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_vggt_only_vis"
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
		echo "[1/8] Splitting input video in conda env: ${PREPROCESS_ENV}"
		cd "${ROOT_DIR}"
		conda run -n "${PREPROCESS_ENV}" python "${ROOT_DIR}/pipelines/yolosam2/sam2_preprocess.py" \
			--root_dir "${ROOT_DIR}" \
			--video "${VIDEO_PATH}"
	fi
fi

echo "[2/8] Preparing VGGT4D scene input"
rm -rf "${VGGT_INPUT_ROOT}"
mkdir -p "${VGGT_INPUT_ROOT}"
ln -sfn "${VIDEO_DIR}" "${VGGT_SCENE_INPUT}"

echo "[3/8] Running VGGT4D in conda env: ${VGGT_ENV}"
cd "${ROOT_DIR}"

FRAME_LIST=( $(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | sort) )
TOTAL_FRAMES="${#FRAME_LIST[@]}"
if [[ "${TOTAL_FRAMES}" -eq 0 ]]; then
	echo "ERROR: no frames in ${VIDEO_DIR}"
	exit 1
fi

VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-20}"
VGGT_THRESHOLD_SCALE="${VGGT_THRESHOLD_SCALE:-0.7}"
VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES:-20}"
VMF_LC="$(printf '%s' "${VGGT_MAX_FRAMES}" | tr '[:upper:]' '[:lower:]')"
if [[ "${VMF_LC}" == "all" ]] || [[ "${VGGT_MAX_FRAMES}" == "0" ]]; then
	N_FRAMES="${TOTAL_FRAMES}"
	echo "INFO: VGGT on all ${N_FRAMES} frames (vggt_max_frames=all)."
elif [[ "${TOTAL_FRAMES}" -gt "${VGGT_MAX_FRAMES}" ]]; then
	N_FRAMES="${VGGT_MAX_FRAMES}"
	FRAME_LIST=("${FRAME_LIST[@]:0:${N_FRAMES}}")
	echo "INFO: VGGT on first ${N_FRAMES} of ${TOTAL_FRAMES} frames (cap=${VGGT_MAX_FRAMES})."
else
	N_FRAMES="${TOTAL_FRAMES}"
	echo "INFO: VGGT on all ${N_FRAMES} frames (clip shorter than cap=${VGGT_MAX_FRAMES})."
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
		ln -sfn "${FRAME_PATH}" "${CHUNK_SCENE_INPUT}/${FRAME_NAME}"
		I=$((I + 1))
	done
	echo "  Chunk ${START}:${END} (threshold_scale=${VGGT_THRESHOLD_SCALE})"
	cd "${VGGT_DIR}"
	conda run -n "${VGGT_ENV}" python demo_vggt4d.py \
		--input_dir "${CHUNK_INPUT_ROOT}" \
		--output_dir "${CHUNK_OUTPUT_ROOT}" \
		--dyn_threshold_scale "${VGGT_THRESHOLD_SCALE}"
	if [[ ! -d "${CHUNK_SCENE_OUTPUT}" ]]; then
		echo "ERROR: missing chunk output: ${CHUNK_SCENE_OUTPUT}"
		exit 1
	fi
	LOCAL_MASKS=( $(find "${CHUNK_SCENE_OUTPUT}" -maxdepth 1 -type f -name 'dynamic_mask_*.png' | sort) )
	LOCAL_N="${#LOCAL_MASKS[@]}"
	EXPECTED_N=$((END - START))
	if [[ "${LOCAL_N}" -ne "${EXPECTED_N}" ]]; then
		echo "ERROR: mask count mismatch ${CHUNK_TAG}: got ${LOCAL_N}, expected ${EXPECTED_N}"
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
echo "VGGT4D finished (${N_FRAMES} frames)."

echo "[4/8] Aligning VGGT dynamic masks to full frame list (tail_policy=${VGGT_ALIGN_TAIL})"
rm -rf "$(dirname "${RAW_MASK_DIR}")"
mkdir -p "${RAW_MASK_DIR}"
cd "${ROOT_DIR}"
conda run -n "${VGGT_ENV}" python "${VGGT_MASKS_PY}" \
	--vggt_scene_output "${VGGT_SCENE_OUTPUT}" \
	--frame_dir "${VIDEO_DIR}" \
	--output_raw_dir "${RAW_MASK_DIR}" \
	--threshold 0 \
	--tail_policy "${VGGT_ALIGN_TAIL}"

rm -rf "${NEW_MASK_DIR}"
mkdir -p "${SEG_DEMO_DIR}" "${MASK_COMPARE_DIR}" "${INPAINT_5_DIR}" "${DIFFUERASER_FRAMES_DIR}"

echo "[5/8] Mask demos (postprocess)"
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

if [[ ! -d "${NEW_MASK_DIR}" ]]; then
	echo "ERROR: missing ${NEW_MASK_DIR}"
	exit 1
fi

echo "[6/8] DiffuEraser in conda env: ${DIFFUERASER_ENV}"

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
		echo "INFO: HF Hub cache for ${varname}: ${hit}"
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
[[ -d "${DIFFUERASER_BASE_MODEL}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - SD1.5: ${DIFFUERASER_BASE_MODEL}"
[[ -d "${DIFFUERASER_VAE}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - VAE: ${DIFFUERASER_VAE}"
[[ -d "${DIFFUERASER_MODEL}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - DiffuEraser: ${DIFFUERASER_MODEL}"
[[ -d "${DIFFUERASER_PROPAINTER}" && -f "${DIFFUERASER_PROPAINTER}/ProPainter.pth" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - ProPainter: ${DIFFUERASER_PROPAINTER}"
[[ -f "${DIFFUERASER_PCM_LORA}" ]] || DIFFUERASER_WEIGHT_ERRORS+=$'\n'"  - PCM LoRA: ${DIFFUERASER_PCM_LORA}"
if [[ -n "${DIFFUERASER_WEIGHT_ERRORS}" ]]; then
	echo "ERROR: DiffuEraser weights missing:${DIFFUERASER_WEIGHT_ERRORS}"
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
	echo "ERROR: empty frames or masks"
	exit 1
fi
if [[ "${DIFFUERASER_FRAME_COUNT}" -ne "${DIFFUERASER_MASK_COUNT}" ]]; then
	echo "WARN: frame/mask mismatch: ${DIFFUERASER_FRAME_COUNT} vs ${DIFFUERASER_MASK_COUNT}; using shorter"
	if [[ "${DIFFUERASER_MASK_COUNT}" -lt "${DIFFUERASER_FRAME_COUNT}" ]]; then
		DIFFUERASER_FRAME_COUNT="${DIFFUERASER_MASK_COUNT}"
	fi
fi

I=0
while [[ "${I}" -lt "${DIFFUERASER_FRAME_COUNT}" ]]; do
	ln -sfn "${DIFFUERASER_ALL_FRAMES[${I}]}" "${DIFFUERASER_VIDEO_FRAME_DIR}/$(printf '%05d' "${I}").jpg"
	ln -sfn "${DIFFUERASER_ALL_MASKS[${I}]}" "${DIFFUERASER_MASK_FRAME_DIR}/$(printf '%05d' "${I}").png"
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
	echo "ERROR: DiffuEraser output missing: ${DIFFUERASER_VIDEO_PATH}"
	exit 1
fi

mv -f "${DIFFUERASER_VIDEO_PATH}" "${DIFFUERASER_RAW_VIDEO_PATH}"
ffmpeg -y -loglevel error -i "${DIFFUERASER_RAW_VIDEO_PATH}" \
	-c:v libx264 -pix_fmt yuv420p -movflags +faststart "${DIFFUERASER_VIDEO_PATH}"
cp -f "${DIFFUERASER_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
rm -rf "${DIFFUERASER_FRAMES_DIR}"
mkdir -p "${DIFFUERASER_FRAMES_DIR}"
ffmpeg -y -loglevel error -i "${DIFFUERASER_VIDEO_PATH}" -start_number 0 "${DIFFUERASER_FRAMES_DIR}/%05d.png"

echo "[7/8] Inpaint sample exports"
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
	echo "[8/8] DAVIS eval"
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
			echo "ERROR: GT sequence missing for ${VIDEO_NAME}"
			exit 1
		fi
	fi
	rm -rf "${DAVIS_EVAL_SUBSET_ROOT}"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/JPEGImages/480p"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/${ANN_FOLDER}/480p"
	mkdir -p "${DAVIS_EVAL_SUBSET_ROOT}/ImageSets/2017"
	ln -sfn "${VIDEO_DIR}" "${DAVIS_EVAL_SUBSET_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
	ln -sfn "${GT_SEQ_DIR}" "${DAVIS_EVAL_SUBSET_ROOT}/${ANN_FOLDER}/480p/${VIDEO_NAME}"
	printf "%s\n" "${VIDEO_NAME}" > "${DAVIS_EVAL_SUBSET_ROOT}/ImageSets/2017/val.txt"
	GT_MASK_DIR_FOR_METRICS="${GT_SEQ_DIR}"
	cd "${ROOT_DIR}/external/davis2017-evaluation"
	conda run -n "${DAVIS_ENV}" python evaluation_method.py \
		--task "${DAVIS_TASK}" \
		--set val \
		--davis_path "${DAVIS_EVAL_SUBSET_ROOT}" \
		--results_path "${DAVIS_EVAL_RESULTS_ROOT}"
else
	echo "[8/8] Skipping DAVIS eval"
fi

echo "[Metrics]"
METRICS_DIR="${OUTPUTS_DIR}/metrics"
METRICS_CMD=(
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/evaluate_metrics.py"
	--output_dir "${METRICS_DIR}"
	--part_label "${PART_LABEL}"
	--experiment_name "vggt_only_diffueraser"
	--pred_mask_dir "${NEW_MASK_DIR}"
	--pred_video "${FINAL_VIDEO_PATH}"
	--pred_frames_dir "${PRED_FRAMES_DIR}"
)
[[ -f "${DAVIS_CSV_PATH}" ]] && METRICS_CMD+=(--davis_csv "${DAVIS_CSV_PATH}")
[[ -n "${GT_MASK_DIR_FOR_METRICS}" ]] && METRICS_CMD+=(--gt_mask_dir "${GT_MASK_DIR_FOR_METRICS}")
[[ -n "${GT_VIDEO}" ]] && METRICS_CMD+=(--gt_video "${GT_VIDEO}")
[[ -n "${GT_FRAMES_DIR_FOR_METRICS}" ]] && METRICS_CMD+=(--gt_frames_dir "${GT_FRAMES_DIR_FOR_METRICS}")
"${METRICS_CMD[@]}"

echo "Done (VGGT-only baseline)."
echo "  vggt:   ${VGGT_SCENE_OUTPUT}"
echo "  masks:  ${NEW_MASK_DIR}"
echo "  video:  ${FINAL_VIDEO_PATH}"
echo "  vis:    ${VIS_ROOT}"
echo "  metrics:${METRICS_DIR}"
