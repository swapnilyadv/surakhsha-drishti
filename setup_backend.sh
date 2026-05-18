#!/bin/bash
# ============================================================
# Suraksha Drishti — Backend Setup Script
# Run once to install all dependencies
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   SURAKSHA DRISHTI — Backend Setup           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

PYTHON=/opt/homebrew/bin/python3.11
PIP="$PYTHON -m pip"

echo "[1/4] Checking Python 3.11..."
$PYTHON --version

echo ""
echo "[2/4] Upgrading pip..."
$PIP install --upgrade pip --quiet

echo ""
echo "[3/4] Installing backend dependencies..."
echo "      (This may take 3-5 minutes first time — PyTorch ~200MB)"
echo ""
$PIP install \
  "fastapi==0.115.5" \
  "uvicorn[standard]==0.32.1" \
  "python-multipart==0.0.12" \
  "websockets==14.1" \
  "onnxruntime==1.20.1" \
  "transformers==4.47.0" \
  "torch==2.5.1" \
  "python-dotenv==1.0.1"

echo ""
echo "[4/4] Verifying installations..."
$PYTHON -c "import fastapi; print('  ✓ fastapi', fastapi.__version__)"
$PYTHON -c "import uvicorn; print('  ✓ uvicorn', uvicorn.__version__)"
$PYTHON -c "import onnxruntime; print('  ✓ onnxruntime', onnxruntime.__version__)"
$PYTHON -c "import transformers; print('  ✓ transformers', transformers.__version__)"
$PYTHON -c "import torch; print('  ✓ torch', torch.__version__)"
$PYTHON -c "import cv2; print('  ✓ opencv', cv2.__version__)"
$PYTHON -c "import numpy; print('  ✓ numpy', numpy.__version__)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Setup Complete!                            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║   Start backend:                             ║"
echo "║   cd backend && python3.11 main.py           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
