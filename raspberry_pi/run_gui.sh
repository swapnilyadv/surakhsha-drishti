#!/bin/bash
# Simple script to run the Violence Detection GUI on Raspberry Pi

echo "=================================================="
echo "   🛡️  Violence Detection System - Starting..."
echo "=================================================="
echo ""

# Check if required packages are installed
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "❌ Tkinter not found. Please run install_gui.sh first!"
    exit 1
fi

if ! python3 -c "import cv2" 2>/dev/null; then
    echo "❌ OpenCV not found. Please run install_gui.sh first!"
    exit 1
fi

echo "✅ Dependencies checked"
echo "🚀 Starting GUI application..."
echo ""

# Run the GUI application
python3 gui_violence_detector.py

echo ""
echo "👋 Application closed"
