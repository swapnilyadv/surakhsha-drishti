#!/usr/bin/env python3
"""
Violence Detection System for Raspberry Pi 3
=============================================
Optimized for Raspberry Pi 3 with Zebronics USB Camera

Hardware Requirements:
- Raspberry Pi 3 (Model B or B+)
- Zebronics USB Camera (or any USB webcam)
- MicroSD Card (16GB+ recommended)
- Power supply (5V 2.5A)

Software Requirements:
- Raspberry Pi OS (Bullseye or later)
- Python 3.9+
- TensorFlow Lite (for optimized inference)

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

# Try to import TensorFlow Lite for Pi (more efficient)
try:
    import tensorflow as tf
    # For TFLite with Flex ops, we need full TensorFlow
    USE_TFLITE = True
    print("Using TensorFlow with Flex delegate for LSTM support")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        USE_TFLITE = True
        print("Using TensorFlow Lite Runtime")
        print("WARNING: Model may not work due to LSTM Flex ops")
    except ImportError:
        print("ERROR: TensorFlow not found!")
        print("Install with: pip install tensorflow")
        exit(1)

# ======================= CONFIGURATION =======================

CONFIG = {
    # Camera settings
    'camera_id': 0,  # USB camera index (usually 0)
    'frame_width': 640,
    'frame_height': 480,
    'fps': 15,  # Lower FPS for Pi performance
    
    # Model settings
    'model_path': 'best_violence_model.keras',  # Will be converted to TFLite
    'tflite_model_path': 'violence_model.tflite',
    'input_size': (112, 112),
    'sequence_length': 24,
    'skip_frames': 3,  # Process every Nth frame to save CPU
    
    # Detection settings
    'violence_threshold': 60,  # Confidence threshold (%)
    'cooldown_seconds': 60,  # Time between alerts
    
    # Email alert settings
    'email_enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'surakshadrishti.vesit@gmail.com',
    'sender_password': 'kaqq zozs gthm zpla',  # Gmail App Password
    'recipient_email': '2024.swapnil.yadav@ves.ac.in',
    
    # Recording settings
    'record_on_violence': True,
    'recording_duration': 10,  # seconds
    'alerts_folder': 'alerts',
    
    # Display settings (set False for headless Pi)
    'show_display': True,
    'display_width': 640,
    'display_height': 480
}

# ======================= GLOBAL VARIABLES =======================

frame_buffer = deque(maxlen=CONFIG['sequence_length'])
last_alert_time = 0
is_recording = False
alert_lock = threading.Lock()

# ======================= MODEL LOADING =======================

def convert_to_tflite(keras_model_path, tflite_path):
    """Convert Keras model to TFLite for faster inference on Pi."""
    print(f"Converting {keras_model_path} to TFLite format...")
    
    if not USE_TFLITE:
        # Use full TensorFlow to convert
        model = tf.keras.models.load_model(keras_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Quantize for speed
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Saved TFLite model to {tflite_path}")
        return True
    else:
        print("Cannot convert - need full TensorFlow. Using Keras model.")
        return False


def load_model():
    """Load the violence detection model."""
    global model, interpreter
    
    tflite_path = CONFIG['tflite_model_path']
    keras_path = CONFIG['model_path']
    
    # Try TFLite model first (smaller, faster)
    if os.path.exists(tflite_path):
        try:
            print(f"Loading TFLite model from {tflite_path}...")
            # Use TensorFlow's interpreter with Flex delegate for LSTM support
            interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_flex.so', {})] if os.path.exists('/usr/lib/libtensorflowlite_flex.so') else None
            )
            interpreter.allocate_tensors()
            model = None
            print("TFLite model loaded!")
            return True
        except Exception as e:
            print(f"TFLite load failed: {e}")
            print("Falling back to Keras model...")
    
    # Fall back to Keras model
    if os.path.exists(keras_path):
        print(f"Loading Keras model from {keras_path}...")
        model = tf.keras.models.load_model(keras_path)
        interpreter = None
        print("Keras model loaded!")
        return True
    
    print(f"ERROR: Model not found!")
    print(f"Looked for: {tflite_path} and {keras_path}")
    return False


def predict_violence(frames):
    """Run violence prediction on a sequence of frames."""
    global model, interpreter
    
    if len(frames) < CONFIG['sequence_length']:
        return None, 0
    
    # Preprocess frames
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, CONFIG['input_size'])
        normalized = resized.astype(np.float32) / 255.0
        processed.append(normalized)
    
    input_data = np.array([processed])
    
    if USE_TFLITE and interpreter is not None:
        # TFLite inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
    else:
        # Keras inference
        prediction = model.predict(input_data, verbose=0)
    
    confidence = float(prediction[0][0]) * 100
    is_violence = confidence >= 50
    
    if is_violence:
        return 'Violence', confidence
    else:
        return 'Non-Violence', 100 - confidence


# ======================= EMAIL ALERTS =======================

def send_email_alert(video_path, confidence):
    """Send email alert with video attachment."""
    global last_alert_time
    
    if not CONFIG['email_enabled']:
        print("Email alerts disabled")
        return False
    
    # Check cooldown
    current_time = time.time()
    if current_time - last_alert_time < CONFIG['cooldown_seconds']:
        remaining = int(CONFIG['cooldown_seconds'] - (current_time - last_alert_time))
        print(f"Alert cooldown: {remaining}s remaining")
        return False
    
    try:
        print("Sending email alert...")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = CONFIG['sender_email']
        msg['To'] = CONFIG['recipient_email']
        msg['Subject'] = f"🚨 VIOLENCE ALERT - Raspberry Pi Security System"
        
        # Email body
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="background: #dc3545; color: white; padding: 20px; border-radius: 10px;">
                <h1>⚠️ Violence Detected!</h1>
            </div>
            <div style="padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 10px;">
                <h2>Alert Details:</h2>
                <ul>
                    <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><strong>Confidence:</strong> {confidence:.1f}%</li>
                    <li><strong>Source:</strong> Raspberry Pi Camera</li>
                    <li><strong>Device:</strong> {os.uname().nodename}</li>
                </ul>
                <p>A video recording of the incident is attached.</p>
            </div>
            <div style="margin-top: 20px; color: #666;">
                <p>- Suraksha Drishti Security System</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Attach video if exists
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename=violence_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
                )
                msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(CONFIG['smtp_server'], CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(CONFIG['sender_email'], CONFIG['sender_password'])
            server.send_message(msg)
        
        last_alert_time = current_time
        print(f"✓ Alert email sent to {CONFIG['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        return False


# ======================= VIDEO RECORDING =======================

def record_violence_clip(camera, duration=10):
    """Record a short video clip when violence is detected."""
    global is_recording
    
    if is_recording:
        return None
    
    is_recording = True
    
    # Create alerts folder
    os.makedirs(CONFIG['alerts_folder'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(CONFIG['alerts_folder'], f'violence_{timestamp}.avi')
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        filename, 
        fourcc, 
        CONFIG['fps'], 
        (CONFIG['frame_width'], CONFIG['frame_height'])
    )
    
    print(f"Recording {duration}s clip to {filename}...")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = camera.read()
        if ret:
            out.write(frame)
        time.sleep(1 / CONFIG['fps'])
    
    out.release()
    is_recording = False
    
    print(f"✓ Saved recording: {filename}")
    return filename


# ======================= MAIN DETECTION LOOP =======================

def run_detection():
    """Main violence detection loop."""
    global frame_buffer
    
    print("\n" + "="*60)
    print("   SURAKSHA DRISHTI - Raspberry Pi Violence Detection")
    print("="*60)
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Initialize camera
    print(f"\nInitializing camera (ID: {CONFIG['camera_id']})...")
    camera = cv2.VideoCapture(CONFIG['camera_id'])
    
    if not camera.isOpened():
        print("ERROR: Could not open camera!")
        print("Check if Zebronics camera is connected.")
        print("Try: ls /dev/video*")
        return
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    camera.set(cv2.CAP_PROP_FPS, CONFIG['fps'])
    
    print(f"Camera initialized: {CONFIG['frame_width']}x{CONFIG['frame_height']} @ {CONFIG['fps']}fps")
    print(f"Violence threshold: {CONFIG['violence_threshold']}%")
    print(f"Email alerts: {'Enabled' if CONFIG['email_enabled'] else 'Disabled'}")
    print("\nPress 'q' to quit, 's' to take screenshot")
    print("-"*60)
    
    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    
    current_prediction = "Waiting..."
    current_confidence = 0
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            # Skip frames to reduce CPU load
            if frame_count % CONFIG['skip_frames'] == 0:
                frame_buffer.append(frame)
                
                # Run prediction when buffer is full
                if len(frame_buffer) >= CONFIG['sequence_length']:
                    prediction, confidence = predict_violence(list(frame_buffer))
                    
                    if prediction:
                        current_prediction = prediction
                        current_confidence = confidence
                        
                        # Check for violence
                        if prediction == 'Violence' and confidence >= CONFIG['violence_threshold']:
                            print(f"\n🚨 VIOLENCE DETECTED! Confidence: {confidence:.1f}%")
                            
                            # Record clip in background
                            def alert_thread():
                                video_path = record_violence_clip(camera, CONFIG['recording_duration'])
                                if video_path:
                                    send_email_alert(video_path, confidence)
                            
                            with alert_lock:
                                if not is_recording:
                                    threading.Thread(target=alert_thread, daemon=True).start()
            
            # Display (if enabled)
            if CONFIG['show_display']:
                # Draw status on frame
                status_color = (0, 0, 255) if current_prediction == 'Violence' else (0, 255, 0)
                
                # Background rectangle for text
                cv2.rectangle(frame, (5, 5), (350, 90), (0, 0, 0), -1)
                cv2.rectangle(frame, (5, 5), (350, 90), status_color, 2)
                
                # Status text
                cv2.putText(frame, f"Status: {current_prediction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Confidence: {current_confidence:.1f}%", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {current_fps} | Recording: {is_recording}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Suraksha Drishti - Violence Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    screenshot = f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                    cv2.imwrite(screenshot, frame)
                    print(f"Screenshot saved: {screenshot}")
            else:
                # Headless mode - just print status periodically
                if frame_count % 30 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {current_prediction}: {current_confidence:.1f}% | FPS: {current_fps}")
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


# ======================= MAIN ENTRY POINT =======================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         SURAKSHA DRISHTI - Violence Detection             ║
    ║              Raspberry Pi Security System                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create alerts folder
    os.makedirs(CONFIG['alerts_folder'], exist_ok=True)
    
    # Check for model file
    if not os.path.exists(CONFIG['model_path']) and not os.path.exists(CONFIG['tflite_model_path']):
        print(f"WARNING: Model file not found!")
        print(f"Please copy '{CONFIG['model_path']}' to the Raspberry Pi")
        print("Or convert to TFLite format for better performance.")
    
    run_detection()
