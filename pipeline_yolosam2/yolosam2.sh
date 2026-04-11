#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) SAM2 VOS segmentation on bmx-trees
# 2) Convert SAM2 masks to ProPainter binary masks
# 3) Save 5 segmentation demos and 5 old-vs-new mask comparisons
# 4) Run ProPainter inpainting
# 5) Export 5 final inpainted frames

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAM2_ENV="${SAM2_ENV:-sam2}"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
YOLO_ENV="${YOLO_ENV:-sam2}"
DAVIS_ENV="${DAVIS_ENV:-davis}"
export ROOT_DIR

PROPAINTER_DIR="${ROOT_DIR}/ProPainter"
SAM2_DIR="${ROOT_DIR}/sam2"
INPUTS_DIR="${ROOT_DIR}/inputs"

VIDEO_PATH=""
DAVIS_SEQ=""
DAVIS_INPUT_ROOT="${ROOT_DIR}/DAVIS"
EVAL_DAVIS=1
DAVIS_GT_ROOT="${ROOT_DIR}/DAVIS"
DAVIS_TASK="unsupervised"
SAM2_BASE_VIDEO_DIR=""
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
  SAM2_BASE_VIDEO_DIR="${INPUTS_DIR}"
  OUTPUTS_DIR="${ROOT_DIR}/outputs/yolosam2/${VIDEO_NAME}"
else
  MODE="davis"
  VIDEO_NAME="${DAVIS_SEQ}"

  # Support both a DAVIS root (with JPEGImages/480p) and a direct frame root.
  if [[ -d "${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}" ]]; then
    VIDEO_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
    SAM2_BASE_VIDEO_DIR="${DAVIS_INPUT_ROOT}/JPEGImages/480p"
  elif [[ -d "${DAVIS_INPUT_ROOT}/${VIDEO_NAME}" ]]; then
    VIDEO_DIR="${DAVIS_INPUT_ROOT}/${VIDEO_NAME}"
    SAM2_BASE_VIDEO_DIR="${DAVIS_INPUT_ROOT}"
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

  OUTPUTS_DIR="${ROOT_DIR}/outputs/davis/${VIDEO_NAME}"
  if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: DAVIS sequence folder not found: ${VIDEO_DIR}"
    exit 1
  fi
fi

TMP_INPUT_MASK_DIR="${OUTPUTS_DIR}/tmp_sam2_input_masks"
TMP_RAW_MASK_DIR="${OUTPUTS_DIR}/tmp_sam2_masks_raw"
NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_sam2"

VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_sam2_vis"
SEG_DEMO_DIR="${VIS_ROOT}/seg_demo"
MASK_COMPARE_DIR="${VIS_ROOT}/mask_compare"
INPAINT_5_DIR="${VIS_ROOT}/inpaint_5frames"

PROPAINTER_OUT_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_propainter"
PROPAINTER_VIDEO_PATH="${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/inpaint_out.mp4"
FINAL_VIDEO_PATH="${OUTPUTS_DIR}/inpaint_out.mp4"

VIDEO_LIST_FILE="${OUTPUTS_DIR}/video_list.txt"

DAVIS_EVAL_RESULTS_ROOT="${OUTPUTS_DIR}/davis_eval_results"
DAVIS_EVAL_SEQ_DIR="${DAVIS_EVAL_RESULTS_ROOT}/${VIDEO_NAME}"
DAVIS_EVAL_SUBSET_ROOT="${OUTPUTS_DIR}/davis_eval_subset"

if [[ "${DAVIS_TASK}" != "semi-supervised" && "${DAVIS_TASK}" != "unsupervised" ]]; then
  echo "ERROR: --davis_task must be either semi-supervised or unsupervised"
  exit 1
fi

mkdir -p "${OUTPUTS_DIR}"

if [[ "${MODE}" == "video" ]]; then
  echo "[1/8] Preprocessing video (split frames + first-frame YOLO mask) in conda env: ${YOLO_ENV}"
  cd "${ROOT_DIR}"
  conda run -n "${YOLO_ENV}" python "${ROOT_DIR}/pipeline_yolosam2/sam2_preprocess.py" \
    --root_dir "${ROOT_DIR}" \
    --video "${VIDEO_PATH}"
else
  echo "[1/8] DAVIS mode: generate first-frame YOLO mask for ${VIDEO_DIR} in conda env: ${YOLO_ENV}"
  rm -rf "${TMP_INPUT_MASK_DIR}"
  mkdir -p "${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}"
  FIRST_MASK_PATH="${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}/00000.png"
  FIRST_FRAME="$(find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.jpg' | sort | head -n 1)"
  if [[ -z "${FIRST_FRAME}" ]]; then
    echo "ERROR: no .jpg frames found in ${VIDEO_DIR}"
    exit 1
  fi

  FIRST_MASK_SCRIPT="${ROOT_DIR}/pipeline_yolosam2/gen_first_mask.py"
  conda run -n "${YOLO_ENV}" python "${FIRST_MASK_SCRIPT}" \
    --first_frame "${FIRST_FRAME}" \
    --model_path "${ROOT_DIR}/baseline/yolov8n-seg.pt" \
    --output_mask "${FIRST_MASK_PATH}" \
    --conf 0.25 \
    --max_init_objects 4 \
    --classes "0,1,2,3,5,7"

  if [[ ! -f "${FIRST_MASK_PATH}" ]]; then
    echo "ERROR: first-frame mask was not created: ${FIRST_MASK_PATH}"
    exit 1
  fi
fi

echo "[2/8] Preparing SAM2 input files..."
printf "%s\n" "${VIDEO_NAME}" > "${VIDEO_LIST_FILE}"

if [[ "${MODE}" == "video" ]]; then
  rm -rf "${TMP_INPUT_MASK_DIR}"
  mkdir -p "${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}"
  cp "${OLD_MASK_DIR}/00000.png" "${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}/"
fi

echo "[3/8] Running SAM2 VOS propagation in conda env: ${SAM2_ENV}"
rm -rf "${TMP_RAW_MASK_DIR}"
mkdir -p "${TMP_RAW_MASK_DIR}"
cd "${SAM2_DIR}"
conda run -n "${SAM2_ENV}" python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --base_video_dir "${SAM2_BASE_VIDEO_DIR}" \
  --input_mask_dir "${TMP_INPUT_MASK_DIR}" \
  --video_list_file "${VIDEO_LIST_FILE}" \
  --output_mask_dir "${TMP_RAW_MASK_DIR}" \
  --score_thresh 0.0

echo "[4/8] Converting SAM2 masks to ProPainter binary mask format (0/255, L mode)..."
cd "${ROOT_DIR}"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipeline_yolosam2/sam2_postprocess.py" \
  --root_dir "${ROOT_DIR}" \
  --output_root "${OUTPUTS_DIR}" \
  --video_name "${VIDEO_NAME}" \
  --frame_dir "${VIDEO_DIR}" \
  --old_mask_dir "${OLD_MASK_DIR}"

echo "[5/8] Rendering 5 SAM2 segmentation demo images..."
echo "[5/8] Segmentation demos generated in step 4."

echo "[6/8] Rendering 5 old-vs-new mask comparison images..."
echo "[6/8] Mask comparisons generated in step 4."

if [[ ! -d "${NEW_MASK_DIR}" ]]; then
  echo "ERROR: missing mask directory ${NEW_MASK_DIR}"
  exit 1
fi

echo "[7/8] Running ProPainter inpainting in conda env: ${PROPAINTER_ENV}"
cd "${PROPAINTER_DIR}"
# Use a conservative allocator config for torch 2.4.x to avoid
# expandable_segments internal asserts on some CUDA drivers.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

# Low-memory defaults (can be overridden by exporting these vars before running script).
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

echo "[8/8] Exporting 5 final inpainted frames..."
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/pipeline_yolosam2/sam2_postprocess.py" \
  --root_dir "${ROOT_DIR}" \
  --output_root "${OUTPUTS_DIR}" \
  --video_name "${VIDEO_NAME}" \
  --frame_dir "${VIDEO_DIR}" \
  --old_mask_dir "${OLD_MASK_DIR}"

if [[ -f "${PROPAINTER_VIDEO_PATH}" ]]; then
  cp -f "${PROPAINTER_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
fi

if [[ "${MODE}" == "davis" && "${EVAL_DAVIS}" == "1" ]]; then
  echo "[9/9] Preparing DAVIS-evaluable masks and running DAVIS evaluation in conda env: ${DAVIS_ENV}"
  rm -rf "${DAVIS_EVAL_RESULTS_ROOT}"
  mkdir -p "${DAVIS_EVAL_SEQ_DIR}"

  EVAL_MASK_SRC_DIR="${TMP_RAW_MASK_DIR}/${VIDEO_NAME}"
  if [[ ! -d "${EVAL_MASK_SRC_DIR}" ]]; then
    echo "WARN: multi-object raw mask dir not found, fallback to binary masks: ${NEW_MASK_DIR}"
    EVAL_MASK_SRC_DIR="${NEW_MASK_DIR}"
  fi
  if [[ ! -d "${EVAL_MASK_SRC_DIR}" ]]; then
    echo "ERROR: no available eval mask source directory"
    exit 1
  fi

    DAVIS_CONVERT_SCRIPT="${ROOT_DIR}/pipeline_yolosam2/prepare_davis_eval_masks.py"
    conda run -n "${PROPAINTER_ENV}" python "${DAVIS_CONVERT_SCRIPT}" \
    --src_dir "${EVAL_MASK_SRC_DIR}" \
    --dst_dir "${DAVIS_EVAL_SEQ_DIR}" \
    --max_eval_labels 20

  if [[ ! -f "${DAVIS_EVAL_SEQ_DIR}/00000.png" ]]; then
    echo "ERROR: failed to prepare DAVIS eval masks; missing ${DAVIS_EVAL_SEQ_DIR}/00000.png"
    exit 1
  fi

  EVAL_MASK_COUNT="$(find "${DAVIS_EVAL_SEQ_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
  if [[ "${EVAL_MASK_COUNT}" == "0" ]]; then
    echo "ERROR: failed to prepare DAVIS eval masks; no PNG files in ${DAVIS_EVAL_SEQ_DIR}"
    exit 1
  fi

  if [[ -z "${DAVIS_GT_ROOT}" ]]; then
    echo "ERROR: --davis_gt_root is required when --eval_davis 1"
    exit 1
  fi
  if [[ ! -d "${DAVIS_GT_ROOT}" ]]; then
    echo "ERROR: DAVIS GT root not found: ${DAVIS_GT_ROOT}"
    exit 1
  fi

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
      echo "WARN: ${ANN_FOLDER} missing for ${VIDEO_NAME}; fallback to ${ALT_ANN_FOLDER}"
      ANN_FOLDER="${ALT_ANN_FOLDER}"
      GT_SEQ_DIR="${ALT_GT_SEQ_DIR}"
    else
      echo "ERROR: ground-truth sequence not found in either:"
      echo "  ${DAVIS_GT_ROOT}/${ANN_FOLDER}/480p/${VIDEO_NAME}"
      echo "  ${DAVIS_GT_ROOT}/${ALT_ANN_FOLDER}/480p/${VIDEO_NAME}"
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

  cd "${ROOT_DIR}/davis2017-evaluation"
  conda run -n "${DAVIS_ENV}" python evaluation_method.py \
    --task "${DAVIS_TASK}" \
    --set val \
    --davis_path "${DAVIS_EVAL_SUBSET_ROOT}" \
    --results_path "${DAVIS_EVAL_RESULTS_ROOT}"
fi

echo "Done. Outputs:"
echo "- Segmentation demos: ${SEG_DEMO_DIR}"
echo "- Mask comparisons:   ${MASK_COMPARE_DIR}"
echo "- Inpaint 5 frames:   ${INPAINT_5_DIR}"
echo "- Inpaint video:      ${FINAL_VIDEO_PATH}"
if [[ "${MODE}" == "davis" ]]; then
  echo "- DAVIS eval masks:   ${DAVIS_EVAL_SEQ_DIR}"
  if [[ "${EVAL_DAVIS}" == "1" ]]; then
    echo "- DAVIS CSV results:  ${DAVIS_EVAL_RESULTS_ROOT}/global_results-val.csv"
  fi
fi
