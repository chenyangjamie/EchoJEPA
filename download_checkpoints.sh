#!/bin/bash
# Download EchoJEPA checkpoints from Google Drive, one file at a time

DEST_DIR="/raid2/nextgen/new_projects/EchoJEPA/checkpoints"
LOG="$DEST_DIR/download.log"
FOLDER_URL="https://drive.google.com/drive/folders/1RFEXMe8TTcABMBz4H_qtiLB43K9jD_lf"

mkdir -p "$DEST_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "[$(date)] EchoJEPA checkpoint download start"
echo "Destination: $DEST_DIR"
echo "============================================"

# Ensure gdown is usable
pip install -q -U gdown

echo ""
echo "[$(date)] Downloading folder (sequential, one file at a time)..."
echo ""

gdown --folder "$FOLDER_URL" \
    -O "$DEST_DIR" \
    --remaining-ok \
    --continue

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] All downloads complete!"
    echo "Files in $DEST_DIR:"
    ls -lh "$DEST_DIR"/*.pt 2>/dev/null || ls -lh "$DEST_DIR"/
else
    echo "[$(date)] Download finished with exit code $EXIT_CODE — check log above."
fi
echo "============================================"
