#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROPAINTER_ENV="${PROPAINTER_ENV:-propainter}"
YOLO_ENV="${YOLO_ENV:-sam2}"
DAVIS_ENV="${DAVIS_ENV:-davis}"

PROPAINTER_DIR="${ROOT_DIR}/external/ProPainter"
INPUTS_DIR="${ROOT_DIR}/data/inputs"

VIDEO_PATH=""
DAVIS_SEQ=""
DAVIS_INPUT_ROOT="${ROOT_DIR}/data/DAVIS"
EVAL_DAVIS=1
DAVIS_GT_ROOT="${ROOT_DIR}/data/DAVIS"
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
    *)
      echo "Unknown argument: $1"
      echo "Usage:"
      echo "  Video mode: $0 --video /path/to/video.mp4"
      echo "  DAVIS mode: $0 --davis_seq bmx-trees [--davis_input_root ${ROOT_DIR}/data/DAVIS] [--eval_davis 1|0] [--davis_gt_root /path/to/DAVIS] [--davis_task semi-supervised|unsupervised]"
      exit 1
      ;;
  esac
done

# Ensure DAVIS_INPUT_ROOT and DAVIS_GT_ROOT are absolute paths
if [[ "${DAVIS_INPUT_ROOT}" != /* ]]; then
  DAVIS_INPUT_ROOT="${ROOT_DIR}/${DAVIS_INPUT_ROOT}"
fi
if [[ "${DAVIS_GT_ROOT}" != /* ]]; then
  DAVIS_GT_ROOT="${ROOT_DIR}/${DAVIS_GT_ROOT}"
fi

if [[ -z "${VIDEO_PATH}" && -z "${DAVIS_SEQ}" ]]; then
  echo "Usage:"
  echo "  Video mode: $0 --video /path/to/video.mp4"
  echo "  DAVIS mode: $0 --davis_seq bmx-trees [--davis_input_root ${ROOT_DIR}/data/DAVIS] [--eval_davis 1|0] [--davis_gt_root /path/to/DAVIS] [--davis_task semi-supervised|unsupervised]"
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
  OUTPUTS_DIR="${ROOT_DIR}/outputs/yoloopt/${VIDEO_NAME}"
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

  OUTPUTS_DIR="${ROOT_DIR}/outputs/yoloopt_davis/${VIDEO_NAME}"
  if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: DAVIS sequence folder not found: ${VIDEO_DIR}"
    exit 1
  fi
fi

TMP_FIRST_MASK_DIR="${OUTPUTS_DIR}/tmp_first_mask"
NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_optflow"

VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_optflow_vis"
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

if [[ "${DAVIS_TASK}" != "semi-supervised" && "${DAVIS_TASK}" != "unsupervised" ]]; then
  echo "ERROR: --davis_task must be either semi-supervised or unsupervised"
  exit 1
fi

mkdir -p "${OUTPUTS_DIR}"

if [[ "${MODE}" == "video" ]]; then
  echo "[1/5] Checking video frames in ${VIDEO_DIR}"
  if [[ ! -d "${VIDEO_DIR}" ]]; then
    VIDEO_FILE="${ROOT_DIR}/${VIDEO_NAME}.mp4"
    if [[ -f "${VIDEO_FILE}" ]]; then
      echo "   Extracting frames from video ${VIDEO_FILE}..."
      mkdir -p "${VIDEO_DIR}"
      ffmpeg -i "${VIDEO_FILE}" -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" "${VIDEO_DIR}/%05d.jpg" -y
      echo "   Frames extracted to ${VIDEO_DIR}"
    else
      echo "ERROR: video frames directory not found: ${VIDEO_DIR}"
      echo "Hint: Place frames in inputs/walking4/ or put walking4.mp4 in project root"
      exit 1
    fi
  fi

  rm -rf "${TMP_FIRST_MASK_DIR}"
  mkdir -p "${TMP_FIRST_MASK_DIR}/${VIDEO_NAME}"
  if [[ -f "${OLD_MASK_DIR}/00000.png" ]]; then
    cp "${OLD_MASK_DIR}/00000.png" "${TMP_FIRST_MASK_DIR}/${VIDEO_NAME}/"
    echo "   Using provided first-frame mask from ${OLD_MASK_DIR}/00000.png"
  else
    echo "ERROR: first-frame mask not found in ${OLD_MASK_DIR}"
    exit 1
  fi
else
  echo "[1/5] DAVIS mode: generate first-frame YOLO mask for ${VIDEO_DIR} in conda env: ${YOLO_ENV}"
  rm -rf "${TMP_FIRST_MASK_DIR}"
  mkdir -p "${TMP_FIRST_MASK_DIR}/${VIDEO_NAME}"
  FIRST_MASK_PATH="${TMP_FIRST_MASK_DIR}/${VIDEO_NAME}/00000.png"
  FIRST_FRAME="$(find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.jpg' | sort | head -n 1)"
  if [[ -z "${FIRST_FRAME}" ]]; then
    echo "ERROR: no .jpg frames found in ${VIDEO_DIR}"
    exit 1
  fi

  FIRST_MASK_SCRIPT="${ROOT_DIR}/pipelines/yoloopt/gen_first_mask.py"
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

echo "[2/5] Running YOLO + Optical Flow mask propagation in conda env: ${YOLO_ENV}"
cd "${ROOT_DIR}"
conda run -n "${YOLO_ENV}" python "${ROOT_DIR}/pipelines/yoloopt/optflow_postprocess.py" \
  --root_dir "${ROOT_DIR}" \
  --video_dir "${VIDEO_DIR}" \
  --first_mask_dir "${TMP_FIRST_MASK_DIR}" \
  --old_mask_dir "${OLD_MASK_DIR}" \
  --output_dir "${OUTPUTS_DIR}" \
  --video_name "${VIDEO_NAME}" \
  --mask_video_path "${MASK_VIDEO_PATH}"

echo "[3/5] Rendering 5 segmentation demo images..."
echo "[3/5] Segmentation demos generated in step 2."

echo "[4/5] Rendering 5 old-vs-new mask comparison images..."
echo "[4/5] Mask comparisons generated in step 2."

if [[ ! -d "${NEW_MASK_DIR}" ]]; then
  echo "ERROR: missing mask directory ${NEW_MASK_DIR}"
  exit 1
fi

echo "[5/5] Running ProPainter inpainting in conda env: ${PROPAINTER_ENV}"
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

if [[ -f "${PROPAINTER_VIDEO_PATH}" ]]; then
  cp -f "${PROPAINTER_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
fi

if [[ "${MODE}" == "davis" && "${EVAL_DAVIS}" == "1" ]]; then
  echo "[DAVIS Evaluation] Preparing DAVIS-evaluable masks and running evaluation in conda env: ${DAVIS_ENV}"
  rm -rf "${DAVIS_EVAL_RESULTS_ROOT}"
  mkdir -p "${DAVIS_EVAL_SEQ_DIR}"

  EVAL_MASK_SRC_DIR="${NEW_MASK_DIR}"
  if [[ ! -d "${EVAL_MASK_SRC_DIR}" ]]; then
    echo "ERROR: no available eval mask source directory"
    exit 1
  fi

  DAVIS_CONVERT_SCRIPT="${ROOT_DIR}/pipelines/yoloopt/prepare_davis_eval_masks.py"
  conda run -n "${PROPAINTER_ENV}" python "${DAVIS_CONVERT_SCRIPT}" \
    --src_dir "${EVAL_MASK_SRC_DIR}" \
    --dst_dir "${DAVIS_EVAL_SEQ_DIR}" \
    --max_eval_labels 20 \
    --binary

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
  GT_MASK_DIR_FOR_METRICS="${GT_SEQ_DIR}"

  cd "${ROOT_DIR}/external/davis2017-evaluation"
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
  --experiment_name "yoloopt"
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

if [[ "${MODE}" == "davis" ]]; then
  GT_FRAMES_DIR_FOR_METRICS="${DAVIS_INPUT_ROOT}/JPEGImages/480p/${VIDEO_NAME}"
else
  GT_FRAMES_DIR_FOR_METRICS="${VIDEO_DIR}"
fi
METRICS_CMD+=(--gt_frames_dir "${GT_FRAMES_DIR_FOR_METRICS}")

"${METRICS_CMD[@]}" || echo "WARN: metrics evaluation failed"

echo "Done. Outputs:"
echo "- Segmentation demos: ${SEG_DEMO_DIR}"
echo "- Mask comparisons:   ${MASK_COMPARE_DIR}"
echo "- Mask overlay video: ${MASK_VIDEO_PATH}"
echo "- Inpaint video:     ${FINAL_VIDEO_PATH}"
if [[ "${MODE}" == "davis" ]]; then
  echo "- DAVIS eval masks:   ${DAVIS_EVAL_SEQ_DIR}"
  if [[ "${EVAL_DAVIS}" == "1" ]]; then
    echo "- DAVIS CSV results:  ${DAVIS_EVAL_RESULTS_ROOT}/global_results-val.csv"
  fi
fi
echo "- Metric summary:    ${METRICS_DIR}/metrics_summary.json"