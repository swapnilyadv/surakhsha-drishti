#!/bin/bash
# ============================================================
# Suraksha Drishti — Start AI Backend
# Uses pyenv Python 3.11.9 (clean, stable environment)
# ============================================================

PYTHON=/Users/swapnil/.pyenv/versions/3.11.9/bin/python3.11
BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   SURAKSHA DRISHTI — AI Backend Starting     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Backend: http://localhost:8765"
echo "  Stream:  http://localhost:8765/api/stream/mjpeg"
echo "  WS:      ws://localhost:8765/ws/detections"
echo ""

cd "$BACKEND_DIR"
$PYTHON main.py
