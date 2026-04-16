#!/bin/bash

# Comparison script for all pipelines

# Define common variables
DAVIS_INPUT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"
DAVIS_GT_ROOT="/home/xyz/Desktop/yzhang/object-removal/DAVIS"
DAVIS_SEQ="bmx-trees"

# Run yolosam2 pipeline
echo "Running yolosam2 pipeline..."
bash pipeline_yolosam2/yolosam2.sh --davis_seq $DAVIS_SEQ --eval_davis 1 --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4d pipeline
echo "Running vggt4d pipeline..."
bash pipeline_vggt4d/vggt4d.sh --davis_seq $DAVIS_SEQ --eval_davis 1 --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4dsam3 pipeline
echo "Running vggt4dsam3 pipeline..."
bash pipeline_vggt4dsam3/vggt4dsam3.sh --davis_seq $DAVIS_SEQ --eval_davis 1 --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Run vggt4dsam3sd pipeline
echo "Running vggt4dsam3sd pipeline..."
bash pipeline_vggt4dsam3sd/vggt4dsam3sd.sh --davis_seq $DAVIS_SEQ --eval_davis 1 --davis_input_root $DAVIS_INPUT_ROOT --davis_gt_root $DAVIS_GT_ROOT --davis_task unsupervised

# Summary
echo "All pipelines completed. Check outputs for metrics and visualizations."
