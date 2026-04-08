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
PROPAINTER_ENV="${PROPAINTER_ENV:-sam3d-objects}"
YOLO_ENV="${YOLO_ENV:-sam2}"
export ROOT_DIR

PROPAINTER_DIR="${ROOT_DIR}/ProPainter"
SAM2_DIR="${ROOT_DIR}/sam2"
INPUTS_DIR="${ROOT_DIR}/inputs"

VIDEO_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --video /path/to/video.mp4"
      exit 1
      ;;
  esac
done

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "Usage: $0 --video /path/to/video.mp4"
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "ERROR: video file not found: ${VIDEO_PATH}"
  exit 1
fi

VIDEO_NAME="$(basename "${VIDEO_PATH}")"
VIDEO_NAME="${VIDEO_NAME%.*}"
VIDEO_DIR="${INPUTS_DIR}/${VIDEO_NAME}"
OLD_MASK_DIR="${INPUTS_DIR}/${VIDEO_NAME}_mask"

OUTPUTS_DIR="${ROOT_DIR}/outputs/reproduction/${VIDEO_NAME}"
TMP_INPUT_MASK_DIR="${OUTPUTS_DIR}/tmp_sam2_input_masks"
TMP_RAW_MASK_DIR="${OUTPUTS_DIR}/tmp_sam2_masks_raw"
NEW_MASK_DIR="${OUTPUTS_DIR}/${VIDEO_NAME}_mask_sam2"

VIS_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_sam2_vis"
SEG_DEMO_DIR="${VIS_ROOT}/seg_demo"
MASK_COMPARE_DIR="${VIS_ROOT}/mask_compare"
INPAINT_5_DIR="${VIS_ROOT}/inpaint_5frames"

PROPAINTER_OUT_ROOT="${OUTPUTS_DIR}/${VIDEO_NAME}_propainter"

VIDEO_LIST_FILE="${INPUTS_DIR}/video_completion/bmx_video_list.txt"

mkdir -p "${OUTPUTS_DIR}"

echo "[1/7] Preprocessing video (split frames + first-frame YOLO mask) in conda env: ${YOLO_ENV}"
cd "${ROOT_DIR}"
conda run -n "${YOLO_ENV}" python "${ROOT_DIR}/reproduction/sam2_preprocess.py" \
  --root_dir "${ROOT_DIR}" \
  --video "${VIDEO_PATH}"

echo "[2/7] Preparing SAM2 input files..."
mkdir -p "${INPUTS_DIR}/video_completion"
printf "%s\n" "${VIDEO_NAME}" > "${VIDEO_LIST_FILE}"

rm -rf "${TMP_INPUT_MASK_DIR}"
mkdir -p "${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}"
cp "${OLD_MASK_DIR}/00000.png" "${TMP_INPUT_MASK_DIR}/${VIDEO_NAME}/"

echo "[3/7] Running SAM2 VOS propagation in conda env: ${SAM2_ENV}"
rm -rf "${TMP_RAW_MASK_DIR}"
mkdir -p "${TMP_RAW_MASK_DIR}"
cd "${SAM2_DIR}"
conda run -n "${SAM2_ENV}" python tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --base_video_dir "${INPUTS_DIR}" \
  --input_mask_dir "${TMP_INPUT_MASK_DIR}" \
  --video_list_file "${VIDEO_LIST_FILE}" \
  --output_mask_dir "${TMP_RAW_MASK_DIR}" \
  --score_thresh 0.0

echo "[4/7] Converting SAM2 masks to ProPainter binary mask format (0/255, L mode)..."
cd "${ROOT_DIR}"
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/reproduction/sam2_postprocess.py" \
    --root_dir "${ROOT_DIR}" \
  --output_root "${OUTPUTS_DIR}" \
    --video_name "${VIDEO_NAME}"

echo "[5/7] Rendering 5 SAM2 segmentation demo images..."
echo "[5/7] Segmentation demos generated in step 4."

echo "[6/7] Rendering 5 old-vs-new mask comparison images..."
echo "[6/7] Mask comparisons generated in step 4."

if [[ ! -d "${NEW_MASK_DIR}" ]]; then
    echo "ERROR: missing mask directory ${NEW_MASK_DIR}"
    exit 1
fi

echo "[7/8] Running ProPainter inpainting in conda env: ${PROPAINTER_ENV}"
cd "${PROPAINTER_DIR}"
# Reduce CUDA fragmentation and peak memory during long-video inference.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Low-memory defaults (can be overridden by exporting these vars before running script).
PROPAINTER_RESIZE_RATIO="${PROPAINTER_RESIZE_RATIO:-0.75}"
PROPAINTER_SUBVIDEO_LENGTH="${PROPAINTER_SUBVIDEO_LENGTH:-40}"
PROPAINTER_NEIGHBOR_LENGTH="${PROPAINTER_NEIGHBOR_LENGTH:-8}"
PROPAINTER_RAFT_ITER="${PROPAINTER_RAFT_ITER:-12}"

echo "      resize_ratio=${PROPAINTER_RESIZE_RATIO}, subvideo_length=${PROPAINTER_SUBVIDEO_LENGTH}, neighbor_length=${PROPAINTER_NEIGHBOR_LENGTH}, raft_iter=${PROPAINTER_RAFT_ITER}, fp16=on"
conda run -n "${PROPAINTER_ENV}" python inference_propainter.py \
  --video "../inputs/${VIDEO_NAME}" \
  --mask "../outputs/reproduction/${VIDEO_NAME}/${VIDEO_NAME}_mask_sam2" \
  --output "../outputs/reproduction/${VIDEO_NAME}/${VIDEO_NAME}_propainter" \
  --resize_ratio "${PROPAINTER_RESIZE_RATIO}" \
  --subvideo_length "${PROPAINTER_SUBVIDEO_LENGTH}" \
  --neighbor_length "${PROPAINTER_NEIGHBOR_LENGTH}" \
  --raft_iter "${PROPAINTER_RAFT_ITER}" \
  --fp16 \
  --save_frames

echo "[8/8] Exporting 5 final inpainted frames..."
conda run -n "${PROPAINTER_ENV}" python "${ROOT_DIR}/reproduction/sam2_postprocess.py" \
    --root_dir "${ROOT_DIR}" \
  --output_root "${OUTPUTS_DIR}" \
    --video_name "${VIDEO_NAME}"

echo "Done. Outputs:"
echo "- Segmentation demos: ${SEG_DEMO_DIR}"
echo "- Mask comparisons:   ${MASK_COMPARE_DIR}"
echo "- Inpaint 5 frames:   ${INPAINT_5_DIR}"
echo "- Inpaint video:      ${PROPAINTER_OUT_ROOT}/${VIDEO_NAME}/inpaint_out.mp4"
