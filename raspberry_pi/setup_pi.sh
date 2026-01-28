#!/bin/bash
# Suraksha Drishti - Raspberry Pi Quick Setup Script
# Run this script on your Raspberry Pi to set up everything

echo "=============================================="
echo "  Suraksha Drishti - Violence Detection"
echo "  Raspberry Pi Setup Script"
echo "=============================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    echo "Continuing anyway..."
fi

# Create project directory
echo "1. Creating project directory..."
mkdir -p ~/violence_detection
cd ~/violence_detection

# Update system
echo ""
echo "2. Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev

# Create virtual environment
echo ""
echo "3. Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "5. Installing Python packages..."
echo "   This may take 10-30 minutes on Raspberry Pi..."

# Install numpy first (required by others)
pip install numpy

# Install OpenCV
pip install opencv-python-headless

# Install TensorFlow (this takes a while on Pi)
echo ""
echo "   Installing TensorFlow (this will take a while)..."
pip install tensorflow

# Check installations
echo ""
echo "6. Verifying installations..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy these files to ~/violence_detection/:"
echo "   - violence_detector_pi.py"
echo "   - best_violence_model.keras (or violence_model.tflite)"
echo ""
echo "2. Connect your Zebronics USB camera"
echo ""
echo "3. Run the detector:"
echo "   cd ~/violence_detection"
echo "   source venv/bin/activate"
echo "   python violence_detector_pi.py"
echo ""
echo "To test camera:"
echo "   ls /dev/video*"
echo ""
