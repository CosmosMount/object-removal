#!/bin/bash

# This script combines images from a specified folder in the DAVIS dataset into a video.
# Usage: ./video.sh <folder_name>

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

# Get the folder name from the argument
FOLDER_NAME=$1

# Define the input and output paths
INPUT_PATH="data/DAVIS/JPEGImages/480p/$FOLDER_NAME"
OUTPUT_VIDEO="outputs/origins/$FOLDER_NAME.mp4"

# Check if the input folder exists
if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Folder $INPUT_PATH does not exist."
    exit 1
fi

# Ensure the output directory exists
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

# Ensure the subdirectory for the video exists
OUTPUT_SUBDIR="outputs/origins"
mkdir -p "$OUTPUT_SUBDIR"

# Combine images into a video
ffmpeg -framerate 30 -i "$INPUT_PATH/%05d.jpg" -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_VIDEO"

# Check if the video was created successfully
if [ $? -eq 0 ]; then
    echo "Video created successfully: $OUTPUT_VIDEO"
else
    echo "Error: Failed to create video."
    exit 1
fi