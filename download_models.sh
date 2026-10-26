#!/usr/bin/env bash
# Downloads the YOLOv3 weights, which are too large to store on GitHub.
set -euo pipefail

cd "$(dirname "$0")"
DEST="models/yolo/yolov3.weights"
SHA256="523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"

if [ -f "$DEST" ]; then
    echo "$DEST already exists, skipping download."
else
    echo "Downloading yolov3.weights (~237 MB)..."
    curl -L --fail -o "$DEST" https://pjreddie.com/media/files/yolov3.weights
fi

echo "Verifying checksum..."
echo "$SHA256  $DEST" | sha256sum -c -
