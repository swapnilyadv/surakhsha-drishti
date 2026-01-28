# 🛡️ Suraksha Drishti - Violence Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-2.20-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-3.1-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/OpenCV-4.13-red.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>AI-powered real-time violence detection system with gender recognition and email alerts</b>
</p>

---

## 📋 Overview

**Suraksha Drishti** (सुरक्षा दृष्टि - "Safety Vision") is an AI-based violence detection system that uses deep learning to identify violent activities in real-time video streams. The system supports multiple input sources including CCTV cameras, webcams, and video uploads.

### 🎯 Key Features

- **🎥 Multi-Source Input**: Support for CCTV/IP cameras, USB webcams, and video file uploads
- **🤖 Deep Learning Model**: MobileNetV2 + Bidirectional LSTM architecture (89.22% accuracy)
- **👤 Gender Detection**: Real-time face detection with gender classification using DeepFace
- **📧 Email Alerts**: Automatic email notifications with video recordings when violence is detected
- **📱 Raspberry Pi Support**: Optimized deployment for IoT edge devices
- **🌐 Web Interface**: Clean, responsive Flask-based web application

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Video Input   │────▶│  Preprocessing  │────▶│   CNN + LSTM    │
│  (CCTV/Webcam)  │     │   (24 frames)   │     │     Model       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Email Alert    │◀────│  Violence?      │◀────│   Prediction    │
│  + Recording    │     │  (threshold)    │     │   Confidence    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Model Architecture
- **Feature Extractor**: MobileNetV2 (pretrained on ImageNet)
- **Temporal Analysis**: Bidirectional LSTM (128 + 64 units)
- **Input**: 24 frames × 112×112×3 (RGB)
- **Output**: Binary classification (Violence / Non-Violence)

---

## 📁 Project Structure

```
AI Model/
├── app.py                    # Main Flask web application
├── best_violence_model.keras # Trained model (89.22% accuracy)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── templates/
│   └── index.html           # Web interface
│
├── uploads/                 # Uploaded video files
├── alerts/                  # Violence alert recordings
│   └── alert_log.json      # Alert history
│
└── raspberry_pi/            # Raspberry Pi deployment
    ├── violence_detector_pi.py
    ├── violence_model.tflite
    ├── setup_pi.sh
    └── SETUP_GUIDE.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip (Python package manager)
- Webcam or video files for testing

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/suraksha-drishti.git
cd suraksha-drishti

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Web Interface
Open your browser and navigate to: `http://localhost:8080`

---

## 📊 Model Training

The model was trained on the **Real Life Violence Dataset** containing:
- 1000 Violence videos
- 1000 Non-Violence videos

### Training Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 89.22% |
| Training Samples | 600 videos |
| Validation Split | 20% |
| Epochs | 20 |

---

## 🔧 Configuration

### Email Alerts
Configure email alerts in `app.py`:

```python
ALERT_CONFIG = {
    'enabled': True,
    'email_enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',  # Use Gmail App Password
    'recipient_email': 'alert-recipient@email.com',
    'violence_threshold': 70  # Minimum confidence to trigger alert
}
```

### CCTV/IP Camera
Add your RTSP stream URL in the web interface:
```
rtsp://username:password@camera-ip:554/stream
```

---

## 📱 Raspberry Pi Deployment

For IoT edge deployment on Raspberry Pi 3/4:

```bash
# Copy files to Pi
scp -r raspberry_pi/ pi@<PI_IP>:~/violence_detection/

# On Pi, run setup
cd ~/violence_detection
chmod +x setup_pi.sh
./setup_pi.sh

# Run detector
python violence_detector_pi.py
```

See `raspberry_pi/SETUP_GUIDE.md` for detailed instructions.

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Backend** | Python 3.11, Flask 3.1 |
| **Deep Learning** | TensorFlow 2.20, Keras |
| **Computer Vision** | OpenCV 4.13 |
| **Face Detection** | DeepFace, RetinaFace, MTCNN |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | TensorFlow Lite (Raspberry Pi) |

---

## 📸 Screenshots

### Web Interface
- **Dashboard**: Real-time video monitoring with violence detection
- **Gender Detection**: Face detection with male/female ratio display
- **Alert System**: Email notifications with video recordings

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Swapnil Yadav**
- Email: 2024.swapnil.yadav@ves.ac.in
- Institution: VESIT (Vivekanand Education Society's Institute of Technology)

---

## 🙏 Acknowledgments

- Real Life Violence Dataset for training data
- TensorFlow team for the deep learning framework
- DeepFace library for face detection and gender recognition
- OpenCV community for computer vision tools

---

<p align="center">
  <b>🛡️ Suraksha Drishti - Keeping You Safe 🛡️</b>
</p>
