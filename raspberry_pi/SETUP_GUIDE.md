# Suraksha Drishti - Raspberry Pi Setup Guide
## Violence Detection System for Raspberry Pi 3 with Zebronics Camera

---

## 📦 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Raspberry Pi** | Raspberry Pi 3 Model B/B+ (or Pi 4 for better performance) |
| **Camera** | Zebronics USB Camera (or any USB webcam) |
| **Storage** | MicroSD Card 16GB+ (Class 10 recommended) |
| **Power** | 5V 2.5A Power Supply |
| **Optional** | Monitor/Display for viewing (or run headless) |

---

## 🔧 Step 1: Prepare Raspberry Pi OS

### 1.1 Install Raspberry Pi OS
1. Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Flash **Raspberry Pi OS (64-bit)** to your MicroSD card
3. Enable SSH during setup for remote access
4. Boot your Raspberry Pi

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.3 Enable Camera Support
```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

---

## 📦 Step 2: Install Dependencies on Raspberry Pi

### 2.1 Install System Packages
```bash
# Install required system libraries
sudo apt install -y python3-pip python3-venv
sudo apt install -y libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
sudo apt install -y libhdf5-dev libhdf5-serial-dev
sudo apt install -y libharfbuzz0b libwebp6 libtiff5 libjasper1 libilmbase23
sudo apt install -y libopenjp2-7 libavcodec-extra libavformat58 libswscale5
```

### 2.2 Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/violence_detection
cd ~/violence_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Python Packages
```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow (required for LSTM model)
# For Raspberry Pi 3, use the lightweight version
pip install tensorflow

# If above fails, try:
pip install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl

# Install OpenCV
pip install opencv-python-headless  # Use headless for no display
# OR
pip install opencv-python  # Use this if you have display

# Install other dependencies
pip install numpy
```

**Note:** TensorFlow on Raspberry Pi 3 can be slow. For better performance, consider:
- Using Raspberry Pi 4 (4GB+ RAM)
- Running in headless mode (no display)
- Reducing frame rate and resolution

---

## 📁 Step 3: Copy Files to Raspberry Pi

### Option A: Using SCP (from your Mac)
```bash
# On your Mac, run:
cd "/Users/swapnil/Desktop/AI Model"

# First, convert the model to TFLite
python raspberry_pi/convert_model.py

# Copy files to Pi (replace <PI_IP> with your Pi's IP address)
scp violence_model.tflite pi@<PI_IP>:~/violence_detection/
scp raspberry_pi/violence_detector_pi.py pi@<PI_IP>:~/violence_detection/
```

### Option B: Using USB Drive
1. Copy these files to USB drive:
   - `violence_model.tflite`
   - `violence_detector_pi.py`
2. Insert USB into Pi and copy to `~/violence_detection/`

---

## 📹 Step 4: Connect Zebronics Camera

### 4.1 Connect the Camera
1. Plug Zebronics USB camera into any USB port on Raspberry Pi
2. Wait a few seconds for it to be recognized

### 4.2 Verify Camera Detection
```bash
# Check if camera is detected
ls /dev/video*

# Should show something like: /dev/video0

# Test camera (if you have display)
raspivid -o test.h264 -t 5000
# OR for USB camera
ffmpeg -f v4l2 -video_size 640x480 -i /dev/video0 -frames 1 test.jpg
```

### 4.3 Check Camera ID
```bash
# List all video devices
v4l2-ctl --list-devices
```

If your camera shows as `/dev/video1`, update the config in `violence_detector_pi.py`:
```python
CONFIG = {
    'camera_id': 1,  # Change to 1 if camera is on video1
    ...
}
```

---

## ▶️ Step 5: Run Violence Detection

### 5.1 Activate Environment and Run
```bash
cd ~/violence_detection
source venv/bin/activate
python violence_detector_pi.py
```

### 5.2 Run on Boot (Auto-start)
Create a systemd service:

```bash
sudo nano /etc/systemd/system/violence-detector.service
```

Add this content:
```ini
[Unit]
Description=Suraksha Drishti Violence Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/violence_detection
ExecStart=/home/pi/violence_detection/venv/bin/python violence_detector_pi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable violence-detector
sudo systemctl start violence-detector

# Check status
sudo systemctl status violence-detector

# View logs
journalctl -u violence-detector -f
```

---

## 📧 Step 6: Configure Email Alerts

The email is already configured in `violence_detector_pi.py`. To change settings:

```python
CONFIG = {
    ...
    'email_enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'surakshadrishti.vesit@gmail.com',
    'sender_password': 'kaqq zozs gthm zpla',  # Gmail App Password
    'recipient_email': '2024.swapnil.yadav@ves.ac.in',
    ...
}
```

---

## 🖥️ Step 7: Headless Mode (No Monitor)

If running Pi without a display:

```python
CONFIG = {
    ...
    'show_display': False,  # Disable display window
    ...
}
```

---

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# Check USB devices
lsusb

# Check video devices
ls -la /dev/video*

# Try different camera ID (0, 1, 2...)
```

### Out of Memory
```bash
# Increase swap space
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo /etc/init.d/dphys-swapfile restart
```

### TFLite Model Not Loading
```bash
# Ensure model file exists
ls -la ~/violence_detection/violence_model.tflite

# Check tflite-runtime version
pip show tflite-runtime
```

### Email Not Sending
- Check internet connection: `ping google.com`
- Verify Gmail App Password is correct
- Check if "Less secure app access" is needed

---

## 📊 Performance Tips

1. **Lower resolution**: Reduce `frame_width` and `frame_height` in CONFIG
2. **Skip more frames**: Increase `skip_frames` value (e.g., 5 instead of 3)
3. **Disable display**: Set `show_display: False` for headless mode
4. **Use TFLite**: Always use the `.tflite` model instead of `.keras`
5. **Overclock Pi**: Use `raspi-config` to overclock (with proper cooling)

---

## 📞 Support

If you encounter issues:
1. Check the logs: `journalctl -u violence-detector -f`
2. Test camera separately: `raspistill -o test.jpg`
3. Verify model file: `ls -la ~/violence_detection/`

---

## 🎉 You're Done!

Your Raspberry Pi violence detection system is now set up. It will:
- ✅ Monitor video feed from Zebronics camera
- ✅ Detect violence in real-time
- ✅ Send email alerts with video recordings
- ✅ Start automatically on boot

**Suraksha Drishti - Keeping You Safe**
