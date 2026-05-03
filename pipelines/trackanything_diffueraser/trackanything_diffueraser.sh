#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRACKANYTHING_ENV="${TRACKANYTHING_ENV:-trackanything}"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
DIFFUERASER_ENV="${DIFFUERASER_ENV:-diffueraser}"
DAVIS_ENV="${DAVIS_ENV:-davis}"

TRACKANYTHING_DIR="${ROOT_DIR}/external/Track-Anything"
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

TRACKANYTHING_DEVICE="${TRACKANYTHING_DEVICE:-cuda:0}"
TRACKANYTHING_SAM_MODEL_TYPE="${TRACKANYTHING_SAM_MODEL_TYPE:-vit_h}"
TRACKANYTHING_INIT_SOURCE="${TRACKANYTHING_INIT_SOURCE:-yolo}"
TRACKANYTHING_INIT_MASK=""
TRACKANYTHING_YOLO_MODEL="${TRACKANYTHING_YOLO_MODEL:-}"
TRACKANYTHING_YOLO_CONF="${TRACKANYTHING_YOLO_CONF:-0.25}"
TRACKANYTHING_CLASSES="${TRACKANYTHING_CLASSES:-all}"
TRACKANYTHING_MAX_OBJECTS="${TRACKANYTHING_MAX_OBJECTS:-4}"
TRACKANYTHING_MIN_AREA_RATIO="${TRACKANYTHING_MIN_AREA_RATIO:-0.0005}"
TRACKANYTHING_MAX_AREA_RATIO="${TRACKANYTHING_MAX_AREA_RATIO:-0.80}"
TRACKANYTHING_MASK_ONLY="${TRACKANYTHING_MASK_ONLY:-0}"
TRACKANYTHING_DYN_THRESHOLD_SCALE="${TRACKANYTHING_DYN_THRESHOLD_SCALE:-}"

usage() {
	cat <<EOF
Usage:
  Video mode:
    $0 --video /path/to/video.mp4 [options]

  DAVIS mode:
    $0 --davis_seq bike-packing [--davis_input_root ${ROOT_DIR}/data/DAVIS] [options]

Options:
  --init_source yolo|sam_auto|mask       Automatic first mask source (default: ${TRACKANYTHING_INIT_SOURCE})
  --init_mask /path/to/00000.png         Required for --init_source mask
  --classes all                          YOLO COCO classes to keep; default all classes
  --yolo_conf 0.25
  --yolo_model /path/to/yolov8n-seg.pt   Optional; otherwise ultralytics downloads yolov8n-seg.pt
  --max_objects 4
  --device cuda:0
  --mask_only 1                          Stop after Track-Anything masks and visualizations
  --eval_davis 1|0
  --dyn_threshold_scale 0.7              Accepted for command compatibility; ignored by Track-Anything
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
		--part_label) PART_LABEL="$2"; shift 2 ;;
		--gt_mask_dir) GT_MASK_DIR="$2"; shift 2 ;;
		--gt_video) GT_VIDEO="$2"; shift 2 ;;
		--gt_frames_dir) GT_FRAMES_DIR="$2"; shift 2 ;;
		--dyn_threshold_scale) TRACKANYTHING_DYN_THRESHOLD_SCALE="$2"; shift 2 ;;
		--init_source) TRACKANYTHING_INIT_SOURCE="$2"; shift 2 ;;
		--init_mask) TRACKANYTHING_INIT_MASK="$2"; shift 2 ;;
		--classes) TRACKANYTHING_CLASSES="$2"; shift 2 ;;
		--yolo_conf) TRACKANYTHING_YOLO_CONF="$2"; shift 2 ;;
		--yolo_model) TRACKANYTHING_YOLO_MODEL="$2"; shift 2 ;;
		--max_objects) TRACKANYTHING_MAX_OBJECTS="$2"; shift 2 ;;
		--min_area_ratio) TRACKANYTHING_MIN_AREA_RATIO="$2"; shift 2 ;;
		--max_area_ratio) TRACKANYTHING_MAX_AREA_RATIO="$2"; shift 2 ;;
		--device) TRACKANYTHING_DEVICE="$2"; shift 2 ;;
		--sam_model_type) TRACKANYTHING_SAM_MODEL_TYPE="$2"; shift 2 ;;
		--mask_only) TRACKANYTHING_MASK_ONLY="$2"; shift 2 ;;
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
if [[ ! -d "${TRACKANYTHING_DIR}" ]]; then
	echo "ERROR: Track-Anything dir not found: ${TRACKANYTHING_DIR}"
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
	OUTPUTS_DIR="${ROOT_DIR}/outputs/trackanything_diffueraser/${VIDEO_NAME}"

	if [[ ! -d "${VIDEO_DIR}" ]]; then
		echo "[1/6] Splitting input video into frames"
		mkdir -p "${VIDEO_DIR}"
		ffmpeg -y -loglevel error -i "${VIDEO_PATH}" -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" "${VIDEO_DIR}/%05d.jpg"
	fi
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
	OUTPUTS_DIR="${ROOT_DIR}/outputs/trackanything_diffueraser_davis/${VIDEO_NAME}"
fi

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

FRAME_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l | tr -d ' ')"
if [[ "${FRAME_COUNT}" == "0" ]]; then
	echo "ERROR: no frames found in ${VIDEO_DIR}"
	exit 1
fi

echo "[2/6] Running Track-Anything mask tracking in conda env: ${TRACKANYTHING_ENV}"
rm -rf "${RAW_MASK_DIR}" "${NEW_MASK_DIR}" "${TRACK_VIS_DIR}"
mkdir -p "${RAW_MASK_DIR}" "${NEW_MASK_DIR}" "${TRACK_VIS_DIR}"

TRACK_CMD=(
	conda run -n "${TRACKANYTHING_ENV}" python "${SCRIPT_DIR}/trackanything_masks.py"
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
if [[ -n "${TRACKANYTHING_INIT_MASK}" ]]; then
	TRACK_CMD+=(--init_mask "${TRACKANYTHING_INIT_MASK}")
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

echo "[3/6] Rendering mask demos"
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
	echo "  masks: ${NEW_MASK_DIR}"
	echo "  vis:   ${VIS_ROOT}"
	exit 0
fi

echo "[4/6] Running DiffuEraser inpainting in conda env: ${DIFFUERASER_ENV}"

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

echo "[5/6] Exporting inpaint samples"
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
	echo "[6/6] Preparing DAVIS-evaluable masks and running DAVIS evaluation"
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
	echo "[6/6] Skipping DAVIS evaluation"
fi

echo "[Metrics] Computing JM/JR and optional PSNR/SSIM"
METRICS_DIR="${OUTPUTS_DIR}/metrics"
METRICS_CMD=(
	conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/evaluate_metrics.py"
	--output_dir "${METRICS_DIR}"
	--part_label "${PART_LABEL}"
	--experiment_name "trackanything_diffueraser"
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
echo "  masks:   ${NEW_MASK_DIR}"
echo "  video:   ${FINAL_VIDEO_PATH}"
echo "  metrics: ${METRICS_DIR}"
