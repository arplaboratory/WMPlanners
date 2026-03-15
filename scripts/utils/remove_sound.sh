#!/bin/bash

# Define the output directory
OUT_DIR="./processed_videos"
mkdir -p "$OUT_DIR"

# Loop through common video extensions (case-insensitive)
# This handles .mp4, .MP4, .mov, .MOV, etc.
shopt -s nocaseglob

for file in *.{mp4,mov,avi,mkv}; do
    # Check if any files actually match the pattern
    [ -e "$file" ] || continue

    # Get the filename without the extension
    filename=$(basename "$file")
    extension="${filename##*.}"
    name_only="${filename%.*}"

    echo "Processing: $file"

    # ffmpeg flags:
    # -an: Removes audio
    # -vcodec libx264: Encodes to H.264 (standard for vision tasks)
    # -crf 23: Good balance of quality/file size
    # -pix_fmt yuv420p: Ensures compatibility with most players/libraries
    ffmpeg -i "$file" -an -vcodec libx264 -crf 23 -pix_fmt yuv420p "$OUT_DIR/${name_only}.mp4" -y
done

shopt -u nocaseglob
echo "Done! Processed videos are in $OUT_DIR"
