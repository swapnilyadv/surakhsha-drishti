#!/usr/bin/env python3
"""
Violence Detection System for Raspberry Pi 3 - Simplified Version
=================================================================
Works WITHOUT TensorFlow - uses OpenCV DNN module instead

Hardware Requirements:
- Raspberry Pi 3 (Model B or B+)
- Zebronics USB Camera (or any USB webcam)
- MicroSD Card (16GB+ recommended)

Author: Suraksha Drishti Team
"""

import cv2
import numpy as np
import time
import os
import threading
import smtplib
import json
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from collections import deque

# ======================= CONFIGURATION =======================

CONFIG = {
    # Camera settings
    'camera_id': 0,  # USB camera index (usually 0)
    'frame_width': 640,
    'frame_height': 480,
    'fps': 15,
    
    # Model settings
    'model_path': 'violence_model.tflite',
    'input_size': (112, 112),
    'sequence_length': 24,
    'skip_frames': 5,  # Process every 5th frame (faster on Pi 3)
    
    # Detection settings
    'violence_threshold': 60,
    'cooldown_seconds': 60,
    
    # Email alert settings
    'email_enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'surakshadrishti.vesit@gmail.com',
    'sender_password': 'kaqq zozs gthm zpla',
    'recipient_email': '2024.swapnil.yadav@ves.ac.in',
    
    # Recording settings
    'record_on_violence': True,
    'recording_duration': 10,
    'alerts_folder': 'alerts',
    
    # Display settings
    'show_display': True,
}

# ======================= GLOBAL VARIABLES =======================

frame_buffer = deque(maxlen=CONFIG['sequence_length'])
last_alert_time = 0
is_recording = False
alert_lock = threading.Lock()
interpreter = None

# ======================= MODEL LOADING =======================

def load_model():
    """Load TFLite model using TensorFlow Lite."""
    global interpreter
    
    model_path = CONFIG['model_path']
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please copy violence_model.tflite to ~/violence_detection/")
        return False
    
    try:
        # Try tflite_runtime first (if available)
        try:
            import tflite_runtime.interpreter as tflite
            print("Using tflite_runtime...")
        except ImportError:
            # Fall back to TensorFlow Lite
            import tensorflow as tf
            tflite = tf.lite
            print("Using TensorFlow Lite...")
        
        print(f"Loading model from {model_path}...")
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✓ Model loaded successfully!")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        return True
        
    except ImportError:
        print("ERROR: Neither tflite_runtime nor tensorflow is installed!")
        print("Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


def predict_violence(frame_sequence):
    """Predict violence probability from frame sequence."""
    global interpreter
    
    if interpreter is None:
        return 0.0
    
    try:
        # Prepare input
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Convert to expected shape [1, sequence_length, height, width, channels]
        input_data = np.array([frame_sequence], dtype=np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        violence_prob = float(output_data[0][1]) * 100  # Violence probability
        
        return violence_prob
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0


# ======================= VIDEO PROCESSING =======================

def preprocess_frame(frame):
    """Preprocess frame for model input."""
    resized = cv2.resize(frame, CONFIG['input_size'])
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def send_email_alert(video_path=None):
    """Send email alert with optional video attachment."""
    if not CONFIG['email_enabled']:
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = CONFIG['sender_email']
        msg['To'] = CONFIG['recipient_email']
        msg['Subject'] = f"⚠️ VIOLENCE ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        body = f"""
        VIOLENCE DETECTED!
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Location: Raspberry Pi Camera
        System: Suraksha Drishti Violence Detection
        
        {"Video recording attached." if video_path else ""}
        
        Please take immediate action.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach video if available
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(video_path)}')
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(CONFIG['smtp_server'], CONFIG['smtp_port'])
        server.starttls()
        server.login(CONFIG['sender_email'], CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        
        print("✓ Email alert sent!")
        
    except Exception as e:
        print(f"Email send error: {e}")


def record_video(frames, output_path):
    """Save frames to video file."""
    if not frames:
        return
    
    try:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"✓ Video saved: {output_path}")
        
    except Exception as e:
        print(f"Video save error: {e}")


def handle_violence_detection(recording_buffer):
    """Handle violence detection - send alerts and save video."""
    global last_alert_time
    
    current_time = time.time()
    
    with alert_lock:
        if current_time - last_alert_time < CONFIG['cooldown_seconds']:
            return
        
        last_alert_time = current_time
    
    print("\n" + "="*50)
    print("⚠️  VIOLENCE DETECTED!")
    print("="*50 + "\n")
    
    # Save video
    video_path = None
    if CONFIG['record_on_violence'] and recording_buffer:
        os.makedirs(CONFIG['alerts_folder'], exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(CONFIG['alerts_folder'], f'violence_{timestamp}.avi')
        
        # Record in separate thread
        thread = threading.Thread(target=record_video, args=(list(recording_buffer), video_path))
        thread.start()
    
    # Send email in separate thread
    if CONFIG['email_enabled']:
        email_thread = threading.Thread(target=send_email_alert, args=(video_path,))
        email_thread.start()


# ======================= MAIN LOOP =======================

def main():
    """Main detection loop."""
    global frame_buffer
    
    print("\n" + "="*60)
    print("SURAKSHA DRISHTI - Violence Detection System")
    print("Raspberry Pi Edition")
    print("="*60 + "\n")
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Initialize camera
    print(f"Initializing camera {CONFIG['camera_id']}...")
    cap = cv2.VideoCapture(CONFIG['camera_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['fps'])
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        print("Check connections and camera ID")
        return
    
    print("✓ Camera initialized!")
    print("\nStarting detection...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    recording_buffer = deque(maxlen=CONFIG['fps'] * CONFIG['recording_duration'])
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break
            
            frame_count += 1
            
            # Add to recording buffer
            recording_buffer.append(frame.copy())
            
            # Process every Nth frame
            if frame_count % CONFIG['skip_frames'] != 0:
                if CONFIG['show_display']:
                    cv2.imshow('Suraksha Drishti', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Preprocess and add to buffer
            processed = preprocess_frame(frame)
            frame_buffer.append(processed)
            
            # Predict when buffer is full
            violence_prob = 0
            if len(frame_buffer) == CONFIG['sequence_length']:
                violence_prob = predict_violence(list(frame_buffer))
            
            # Display
            status_text = f"Violence: {violence_prob:.1f}%"
            color = (0, 0, 255) if violence_prob > CONFIG['violence_threshold'] else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if violence_prob > CONFIG['violence_threshold']:
                cv2.putText(frame, "ALERT!", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Handle detection
                handle_violence_detection(recording_buffer)
            
            if CONFIG['show_display']:
                cv2.imshow('Suraksha Drishti', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System stopped")


if __name__ == '__main__':
    main()
