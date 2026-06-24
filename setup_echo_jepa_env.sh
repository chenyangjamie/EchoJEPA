#!/bin/bash
# Create conda env echo_jepa (Python 3.12) and install EchoJEPA

LOG="/raid2/nextgen/new_projects/EchoJEPA/setup_env.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "[$(date)] echo_jepa env setup start"
echo "============================================"

# Remove existing env if broken
if conda env list | grep -q "^echo_jepa "; then
    echo "[$(date)] Removing existing echo_jepa env..."
    conda env remove -n echo_jepa -y
fi

echo ""
echo "[$(date)] Creating echo_jepa env (Python 3.12)..."
conda create -n echo_jepa python=3.12 -y

echo ""
echo "[$(date)] Installing EchoJEPA and dependencies..."
conda run -n echo_jepa pip install -e /raid2/nextgen/new_projects/EchoJEPA

echo ""
echo "[$(date)] Installing gdown for checkpoint downloads..."
conda run -n echo_jepa pip install -U gdown

echo ""
echo "============================================"
echo "[$(date)] Setup complete! Verifying..."
conda run -n echo_jepa python -c "
import torch
import timm
import einops
import decord
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('timm, einops, decord: OK')
"
echo "============================================"
