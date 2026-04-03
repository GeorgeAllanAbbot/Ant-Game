#!/bin/bash
set -e

# Change working directory to the repo root
cd "$(dirname "$0")/.."

# Destination zip file name
OUTPUT_ZIP="ai_pytorch.zip"

echo "Creating ${OUTPUT_ZIP} ..."
rm -f "${OUTPUT_ZIP}"

# Copy ai_pytorch.py to AI/ai.py temporarily so it's in the correct directory inside the zip
cp AI/ai_pytorch.py AI/ai.py

# Zip the required structure
zip -r "${OUTPUT_ZIP}" \
  AI/main.py \
  AI/common.py \
  AI/protocol.py \
  AI/ai.py \
  SDK/ \
  tools/ \
  checkpoints/best_model.pt

# Clean up the temporary file
rm AI/ai.py

echo "Done. The package is at $(pwd)/${OUTPUT_ZIP}"
