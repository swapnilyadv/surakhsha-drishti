"""
Violence Detection Web Application with Gender Detection
Supports: CCTV Camera, Webcam, and Video Upload
Features: Violence detection + Face detection with Gender classification
         + Alert system with video recording when violence is detected
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import base64
import threading
import time
from datetime import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'violence_detection_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['ALERTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'alerts')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALERTS_FOLDER'], exist_ok=True)

# Model configuration - MUST MATCH TRAINED MODEL
MODEL_PATH = "/Users/swapnil/Desktop/AI Model/best_violence_model.keras"
IMG_HEIGHT, IMG_WIDTH = 112, 112  # Match trained model
SEQUENCE_LENGTH = 24  # Match trained model (24 frames)

# Face detection configuration
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Alert configuration - Suraksha Drishti Violence Detection System
ALERT_CONFIG = {
    'enabled': True,
    'email_enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'surakshadrishti.vesit@gmail.com',  # Sender email
    'sender_password': 'kaqq zozs gthm zpla',  # Replace with Gmail App Password (16 chars)
    'recipient_email': '2024.swapnil.yadav@ves.ac.in',  # Where to send alerts (same email or different)
    'cooldown_seconds': 60,  # Minimum time between alerts
    'last_alert_time': 0,
    'violence_threshold': 70  # Minimum confidence to trigger alert
}

# Global variables
model = None
face_cascade = None
gender_model = None
webcam_stream = None
cctv_stream = None
alert_lock = threading.Lock()


# ======================= ALERT SYSTEM =======================

def send_violence_alert(video_path, confidence, source='Unknown'):
    """Send an alert when violence is detected."""
    global ALERT_CONFIG
    
    with alert_lock:
        current_time = time.time()
        
        # Check cooldown
        if current_time - ALERT_CONFIG['last_alert_time'] < ALERT_CONFIG['cooldown_seconds']:
            print(f"Alert skipped - cooldown active ({ALERT_CONFIG['cooldown_seconds']}s)")
            return False
        
        if not ALERT_CONFIG['enabled']:
            print("Alerts are disabled")
            return False
        
        ALERT_CONFIG['last_alert_time'] = current_time
        
        # Create alert record
        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_info = {
            'time': alert_time,
            'confidence': confidence,
            'source': source,
            'video_path': video_path
        }
        
        # Save alert log
        log_path = os.path.join(app.config['ALERTS_FOLDER'], 'alert_log.json')
        alerts = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    alerts = json.load(f)
            except:
                alerts = []
        alerts.append(alert_info)
        with open(log_path, 'w') as f:
            json.dump(alerts[-100:], f, indent=2)  # Keep last 100 alerts
        
        print(f"🚨 VIOLENCE ALERT! Confidence: {confidence:.1f}% | Source: {source}")
        print(f"   Video saved: {video_path}")
        
        # Send email if configured
        if ALERT_CONFIG['email_enabled'] and ALERT_CONFIG['sender_email'] and ALERT_CONFIG['recipient_email']:
            threading.Thread(target=send_email_alert, args=(video_path, confidence, source, alert_time)).start()
        
        return True


def send_email_alert(video_path, confidence, source, alert_time):
    """Send email alert with video attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = ALERT_CONFIG['sender_email']
        msg['To'] = ALERT_CONFIG['recipient_email']
        msg['Subject'] = f"🚨 VIOLENCE DETECTED - {confidence:.1f}% Confidence"
        
        body = f"""
        ⚠️ VIOLENCE DETECTION ALERT ⚠️
        
        Time: {alert_time}
        Source: {source}
        Confidence: {confidence:.1f}%
        
        A potential violent activity has been detected.
        Please review the attached video recording.
        
        --
        Violence Detection System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach video if exists and not too large
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            if file_size < 25 * 1024 * 1024:  # Less than 25MB
                with open(video_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename=violence_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
                    msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(ALERT_CONFIG['smtp_server'], ALERT_CONFIG['smtp_port'])
        server.starttls()
        server.login(ALERT_CONFIG['sender_email'], ALERT_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email alert sent to {ALERT_CONFIG['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")
        return False


def record_violence_clip(frames, source='webcam'):
    """Record a video clip when violence is detected."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(app.config['ALERTS_FOLDER'], f"violence_{source}_{timestamp}.mp4")
        
        if not frames or len(frames) == 0:
            return None
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
        
        for frame in frames:
            # Add timestamp overlay
            cv2.putText(frame, f"VIOLENCE DETECTED - {datetime.now().strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
        
        out.release()
        print(f"📹 Violence clip recorded: {video_path}")
        return video_path
        
    except Exception as e:
        print(f"Error recording violence clip: {e}")
        return None


# ======================= MODEL LOADING =======================

def load_models():
    """Load violence detection model and face cascade."""
    global model, face_cascade, gender_model
    
    print("Loading violence detection model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Violence detection model loaded!")
    except Exception as e:
        print(f"Error loading violence model: {e}")
        model = None
    
    print("Loading face detection cascade...")
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        print("Face cascade loaded!")
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        face_cascade = None
    
    # Load a simple gender classifier using DeepFace if available
    try:
        from deepface import DeepFace
        gender_model = "deepface"
        print("DeepFace loaded for gender detection!")
    except:
        print("DeepFace not available, using OpenCV for face detection only")
        gender_model = None
    
    return model is not None


# ======================= VIDEO PROCESSING =======================

def extract_frames_from_video(video_path, num_frames=SEQUENCE_LENGTH):
    """Extract evenly spaced frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    original_frames = []  # Store original frames for face detection
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            original_frames.append(frame.copy())
            
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            normalized_frame = rgb_frame / 255.0
            frames.append(normalized_frame)
    
    cap.release()
    
    while len(frames) < num_frames:
        if len(frames) > 0:
            frames.append(frames[-1])
            original_frames.append(original_frames[-1])
        else:
            return None, None
    
    return np.array(frames[:num_frames], dtype=np.float32), original_frames[:num_frames]


def detect_faces_and_gender(frame):
    """Detect faces in a frame and classify gender with high accuracy.
    Uses DeepFace with RetinaFace backend for better detection of rotated faces."""
    global face_cascade, gender_model
    
    results = {
        'faces': [],
        'male_count': 0,
        'female_count': 0
    }
    
    if frame is None or frame.size == 0:
        return results
    
    all_faces_detected = []
    
    # Try DeepFace first - it handles rotations better than Haar
    if gender_model == "deepface":
        try:
            from deepface import DeepFace
            
            # Try with RetinaFace first (best accuracy), fallback to others
            backends_to_try = ['retinaface', 'mtcnn', 'opencv']
            
            for backend in backends_to_try:
                try:
                    analyses = DeepFace.analyze(
                        frame, 
                        actions=['gender'], 
                        enforce_detection=False, 
                        silent=True,
                        detector_backend=backend
                    )
                    
                    if analyses:
                        if not isinstance(analyses, list):
                            analyses = [analyses]
                        
                        for analysis in analyses:
                            if 'region' in analysis:
                                region = analysis['region']
                                x = int(region.get('x', 0))
                                y = int(region.get('y', 0))
                                w = int(region.get('w', 0))
                                h = int(region.get('h', 0))
                                
                                # Skip invalid regions
                                if w <= 10 or h <= 10:
                                    continue
                                
                                # Get gender with confidence
                                gender_data = analysis.get('gender', {})
                                if isinstance(gender_data, dict):
                                    man_conf = float(gender_data.get('Man', 0))
                                    woman_conf = float(gender_data.get('Woman', 0))
                                    
                                    # Apply female boost (15%) to counter model bias
                                    woman_conf_adjusted = woman_conf * 1.15
                                    
                                    if man_conf > woman_conf_adjusted:
                                        gender = 'Male'
                                        confidence = man_conf
                                    else:
                                        gender = 'Female'
                                        confidence = woman_conf
                                else:
                                    dominant = analysis.get('dominant_gender', 'Unknown')
                                    gender = 'Male' if dominant == 'Man' else 'Female'
                                    confidence = 85.0
                                
                                face_info = {
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h,
                                    'gender': gender,
                                    'confidence': round(float(confidence), 1)
                                }
                                
                                # Check for duplicate faces
                                is_duplicate = False
                                for existing in all_faces_detected:
                                    if is_same_face(face_info, existing):
                                        if face_info['confidence'] > existing['confidence']:
                                            all_faces_detected.remove(existing)
                                        else:
                                            is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    all_faces_detected.append(face_info)
                        
                        # If found faces, break out of backend loop
                        if all_faces_detected:
                            break
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"DeepFace error: {e}")
    
    # If no faces found with DeepFace, try rotations with Haar Cascade
    if not all_faces_detected and face_cascade is not None:
        # Try original and 180-degree rotation (most common)
        rotations = [
            (0, None),
            (180, cv2.ROTATE_180)
        ]
        
        for angle, rotation_code in rotations:
            if rotation_code is not None:
                rotated_frame = cv2.rotate(frame, rotation_code)
            else:
                rotated_frame = frame
            
            try:
                gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Transform back if rotated
                    if angle == 180:
                        x = rotated_frame.shape[1] - x - w
                        y = rotated_frame.shape[0] - y - h
                    
                    face_info = {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'gender': 'Unknown',
                        'confidence': 0.0
                    }
                    all_faces_detected.append(face_info)
                    
            except Exception as e:
                pass
            
            if all_faces_detected:
                break
        
        # Analyze gender for Haar-detected faces
        for face in all_faces_detected:
            if face['gender'] == 'Unknown' and gender_model == "deepface":
                try:
                    from deepface import DeepFace
                    
                    x, y, w, h = face['x'], face['y'], face['width'], face['height']
                    pad = int(max(w, h) * 0.4)
                    
                    y1 = max(0, y - pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0 and face_img.shape[0] > 30 and face_img.shape[1] > 30:
                        face_img_resized = cv2.resize(face_img, (224, 224))
                        
                        analysis = DeepFace.analyze(
                            face_img_resized, 
                            actions=['gender'], 
                            enforce_detection=False, 
                            silent=True
                        )
                        
                        if analysis:
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            
                            gender_data = analysis.get('gender', {})
                            if isinstance(gender_data, dict):
                                man_conf = float(gender_data.get('Man', 0))
                                woman_conf = float(gender_data.get('Woman', 0))
                                
                                woman_conf_adjusted = woman_conf * 1.15
                                
                                if man_conf > woman_conf_adjusted:
                                    face['gender'] = 'Male'
                                    face['confidence'] = round(man_conf, 1)
                                else:
                                    face['gender'] = 'Female'
                                    face['confidence'] = round(woman_conf, 1)
                except Exception:
                    pass
    
    # Build final results
    for face in all_faces_detected:
        results['faces'].append(face)
        if face['gender'] == 'Male':
            results['male_count'] += 1
        elif face['gender'] == 'Female':
            results['female_count'] += 1
    
    return results


def is_same_face(face1, face2, iou_threshold=0.3):
    """Check if two face detections are the same face using IoU."""
    x1, y1, w1, h1 = face1['x'], face1['y'], face1['width'], face1['height']
    x2, y2, w2, h2 = face2['x'], face2['y'], face2['width'], face2['height']
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return False
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou > iou_threshold


def analyze_video_for_gender(original_frames, sample_every=2):
    """Analyze frames for gender detection with improved accuracy using voting."""
    all_gender_results = {
        'male_count': 0,
        'female_count': 0,
        'total_faces': 0,
        'frames_with_faces': 0,
        'annotated_frames': [],
        'unique_males': 0,
        'unique_females': 0
    }
    
    # Collect gender votes from all frames to determine unique individuals
    # Using spatial clustering to track same faces across frames
    all_detections = []
    
    for i, frame in enumerate(original_frames):
        if i % sample_every == 0:  # Analyze more frequently (every 2nd frame)
            results = detect_faces_and_gender(frame)
            
            if results['faces']:
                all_gender_results['frames_with_faces'] += 1
                
                # Store detections with frame index
                for face in results['faces']:
                    all_detections.append({
                        'frame': i,
                        'x': face['x'],
                        'y': face['y'],
                        'width': face['width'],
                        'height': face['height'],
                        'gender': face['gender'],
                        'confidence': face.get('confidence', 50)
                    })
                
                # Draw bounding boxes on frame
                annotated_frame = frame.copy()
                for face in results['faces']:
                    x, y, w, h = face['x'], face['y'], face['width'], face['height']
                    gender = face['gender']
                    conf = face.get('confidence', 0)
                    
                    # Color: Blue for Male, Pink for Female, Gray for Unknown
                    if gender == 'Male':
                        color = (255, 150, 0)  # Blue
                    elif gender == 'Female':
                        color = (255, 0, 255)  # Pink/Magenta
                    else:
                        color = (128, 128, 128)  # Gray
                    
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Add label with confidence
                    label = f"{gender}"
                    if conf > 0:
                        label += f" {conf:.0f}%"
                    
                    # Draw background for text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
                    cv2.putText(annotated_frame, label, (x + 2, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                all_gender_results['annotated_frames'].append(annotated_frame)
    
    # Estimate unique individuals by clustering detections
    # Group faces by position (assuming people don't move too fast)
    if all_detections:
        # Simple clustering: group faces that are close together across frames
        person_tracks = []
        
        for detection in all_detections:
            matched = False
            center_x = detection['x'] + detection['width'] // 2
            center_y = detection['y'] + detection['height'] // 2
            
            # Try to match with existing tracks
            for track in person_tracks:
                # Check if this detection is near any detection in the track
                last_det = track[-1]
                last_center_x = last_det['x'] + last_det['width'] // 2
                last_center_y = last_det['y'] + last_det['height'] // 2
                
                # Distance threshold based on face size
                threshold = max(detection['width'], detection['height']) * 1.5
                distance = ((center_x - last_center_x)**2 + (center_y - last_center_y)**2)**0.5
                
                # Frame gap check (should be within 10 frames)
                frame_gap = abs(detection['frame'] - last_det['frame'])
                
                if distance < threshold and frame_gap < 10:
                    track.append(detection)
                    matched = True
                    break
            
            if not matched:
                person_tracks.append([detection])
        
        # For each track, use voting to determine final gender
        for track in person_tracks:
            male_votes = 0
            female_votes = 0
            male_confidence = 0
            female_confidence = 0
            
            for det in track:
                conf = det.get('confidence', 50)
                if det['gender'] == 'Male':
                    male_votes += 1
                    male_confidence += conf
                elif det['gender'] == 'Female':
                    female_votes += 1
                    female_confidence += conf
            
            # Weighted voting: consider both count and confidence
            male_score = male_votes * (male_confidence / max(male_votes, 1))
            female_score = female_votes * (female_confidence / max(female_votes, 1))
            
            if male_score > female_score:
                all_gender_results['unique_males'] += 1
            elif female_score > male_score:
                all_gender_results['unique_females'] += 1
        
        all_gender_results['total_faces'] = len(person_tracks)
    
    # Use unique counts for final result
    all_gender_results['male_count'] = all_gender_results['unique_males']
    all_gender_results['female_count'] = all_gender_results['unique_females']
    
    return all_gender_results
    
    return all_gender_results


def predict_violence(frames):
    """Make prediction on processed frames."""
    global model
    
    if model is None:
        return None, 0
    
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class]) * 100
    
    # Apply threshold adjustment - be more sensitive to violence
    # If violence probability is above 40%, lean towards violence
    violence_prob = float(prediction[0][1]) * 100
    if violence_prob > 40:
        return 'Violence', violence_prob
    
    class_names = ['Non-Violence', 'Violence']
    return class_names[predicted_class], confidence


def extract_frames_from_buffer(frame_buffer):
    """Process frames from a buffer for real-time detection."""
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None
    
    frames = []
    indices = np.linspace(0, len(frame_buffer) - 1, SEQUENCE_LENGTH, dtype=int)
    
    for idx in indices:
        frame = frame_buffer[idx]
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames.append(frame)
    
    return np.array(frames, dtype=np.float32)


# ======================= CAMERA STREAMING =======================

class CameraStream:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.frame_buffer = []
        self.current_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_prediction = {"prediction": None, "confidence": 0}
        self.gender_info = {"male_count": 0, "female_count": 0, "faces": []}
        self.violence_detected = False
        self.violence_frames = []  # Buffer for recording violence clips
        self.alert_sent = False
        
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.is_running = True
        self.alert_sent = False
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detection_loop, daemon=True).start()
        threading.Thread(target=self._gender_loop, daemon=True).start()
        return True
    
    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.frame_buffer = []
        self.violence_frames = []
        
    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
                    self.frame_buffer.append(frame.copy())
                    if len(self.frame_buffer) > 60:
                        self.frame_buffer.pop(0)
                    
                    # Keep violence frames for recording
                    if self.violence_detected:
                        self.violence_frames.append(frame.copy())
                        if len(self.violence_frames) > 150:  # ~5 seconds at 30fps
                            self.violence_frames.pop(0)
            time.sleep(0.033)
            
    def _detection_loop(self):
        while self.is_running:
            with self.lock:
                buffer_copy = self.frame_buffer.copy()
            
            if len(buffer_copy) >= SEQUENCE_LENGTH:
                frames = extract_frames_from_buffer(buffer_copy)
                if frames is not None:
                    prediction, confidence = predict_violence(frames)
                    self.last_prediction = {
                        "prediction": prediction,
                        "confidence": confidence
                    }
                    
                    # Check for violence and trigger alert
                    if prediction == "Violence" and confidence >= ALERT_CONFIG['violence_threshold']:
                        self.violence_detected = True
                        
                        # Trigger alert if not recently sent
                        if not self.alert_sent or (time.time() - ALERT_CONFIG['last_alert_time'] > ALERT_CONFIG['cooldown_seconds']):
                            # Record clip from buffer
                            with self.lock:
                                frames_to_record = self.violence_frames.copy() if self.violence_frames else buffer_copy
                            
                            source_name = 'webcam' if self.source == 0 else 'cctv'
                            video_path = record_violence_clip(frames_to_record, source=source_name)
                            
                            if video_path:
                                send_violence_alert(video_path, confidence, source=source_name.upper())
                                self.alert_sent = True
                    else:
                        self.violence_detected = False
                        
            time.sleep(1)
    
    def _gender_loop(self):
        while self.is_running:
            with self.lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None
            
            if frame is not None:
                results = detect_faces_and_gender(frame)
                self.gender_info = results
            
            time.sleep(0.5)
            
    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            
            frame = self.current_frame.copy()
            
            # Draw face boxes with gender
            for face in self.gender_info.get('faces', []):
                x, y, w, h = face['x'], face['y'], face['width'], face['height']
                gender = face['gender']
                
                color = (255, 0, 0) if gender == 'Male' else (255, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add prediction overlay
            pred = self.last_prediction
            if pred["prediction"]:
                is_violence = pred["prediction"] == "Violence"
                color = (0, 0, 255) if is_violence else (0, 255, 0)
                text = f"{pred['prediction']}: {pred['confidence']:.1f}%"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                if is_violence and pred["confidence"] > 50:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            
            # Add gender count
            male = self.gender_info.get('male_count', 0)
            female = self.gender_info.get('female_count', 0)
            gender_text = f"Male: {male} | Female: {female}"
            cv2.putText(frame, gender_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame


# ======================= ROUTES =======================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload with violence and gender detection."""
    global model
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing: {filepath}")
        
        # Extract frames for violence detection
        frames, original_frames = extract_frames_from_video(filepath)
        if frames is None:
            return jsonify({'error': 'Could not process video'}), 400
        
        # Violence prediction
        prediction, confidence = predict_violence(frames)
        print(f"Violence Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        # Gender detection
        gender_results = analyze_video_for_gender(original_frames, sample_every=2)
        print(f"Gender: Male={gender_results['male_count']}, Female={gender_results['female_count']}")
        
        # Calculate gender ratio
        total_gender = gender_results['male_count'] + gender_results['female_count']
        if total_gender > 0:
            male_ratio = (gender_results['male_count'] / total_gender) * 100
            female_ratio = (gender_results['female_count'] / total_gender) * 100
        else:
            male_ratio = 0
            female_ratio = 0
        
        # Trigger alert if violence detected
        if prediction == 'Violence' and confidence >= ALERT_CONFIG['violence_threshold']:
            send_violence_alert(filepath, confidence, source='Upload')
        
        # Save annotated frame if available
        annotated_frame_path = None
        if gender_results['annotated_frames']:
            annotated_filename = f"annotated_{filename}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            cv2.imwrite(annotated_path, gender_results['annotated_frames'][0])
            annotated_frame_path = f"/uploads/{annotated_filename}"
        
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'is_violence': prediction == 'Violence',
            'filename': filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gender': {
                'male_count': gender_results['male_count'],
                'female_count': gender_results['female_count'],
                'total_faces': gender_results['total_faces'],
                'male_ratio': round(male_ratio, 1),
                'female_ratio': round(female_ratio, 1)
            },
            'annotated_frame': annotated_frame_path
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/webcam/start', methods=['POST'])
def start_webcam():
    global webcam_stream
    
    if webcam_stream and webcam_stream.is_running:
        return jsonify({'status': 'already_running'})
    
    webcam_stream = CameraStream(source=0)
    if webcam_stream.start():
        return jsonify({'status': 'started'})
    else:
        return jsonify({'error': 'Could not open webcam'}), 500


@app.route('/webcam/stop', methods=['POST'])
def stop_webcam():
    global webcam_stream
    
    if webcam_stream:
        webcam_stream.stop()
        webcam_stream = None
    
    return jsonify({'status': 'stopped'})


@app.route('/webcam/feed')
def webcam_feed():
    def generate():
        global webcam_stream
        while webcam_stream and webcam_stream.is_running:
            frame = webcam_stream.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam/status')
def webcam_status():
    global webcam_stream
    
    if webcam_stream and webcam_stream.is_running:
        # Convert all values to Python native types for JSON serialization
        prediction = webcam_stream.last_prediction.get('prediction', 'Unknown')
        confidence = webcam_stream.last_prediction.get('confidence', 0)
        gender_info = webcam_stream.gender_info.copy() if webcam_stream.gender_info else {}
        
        # Convert numpy types to Python types
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        confidence = float(confidence)
        
        for key in gender_info:
            if hasattr(gender_info[key], 'item'):
                gender_info[key] = gender_info[key].item()
            elif isinstance(gender_info[key], (int, float)):
                gender_info[key] = int(gender_info[key]) if isinstance(gender_info[key], int) else float(gender_info[key])
        
        return jsonify({
            'running': True,
            'prediction': str(prediction),
            'confidence': confidence,
            'gender': gender_info
        })
    
    return jsonify({'running': False})


@app.route('/cctv/start', methods=['POST'])
def start_cctv():
    global cctv_stream
    
    data = request.get_json()
    rtsp_url = data.get('url', '')
    
    if not rtsp_url:
        return jsonify({'error': 'RTSP URL is required'}), 400
    
    if cctv_stream and cctv_stream.is_running:
        cctv_stream.stop()
    
    cctv_stream = CameraStream(source=rtsp_url)
    if cctv_stream.start():
        return jsonify({'status': 'started'})
    else:
        return jsonify({'error': 'Could not connect to CCTV'}), 500


@app.route('/cctv/stop', methods=['POST'])
def stop_cctv():
    global cctv_stream
    
    if cctv_stream:
        cctv_stream.stop()
        cctv_stream = None
    
    return jsonify({'status': 'stopped'})


@app.route('/cctv/feed')
def cctv_feed():
    def generate():
        global cctv_stream
        while cctv_stream and cctv_stream.is_running:
            frame = cctv_stream.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cctv/status')
def cctv_status():
    global cctv_stream
    
    if cctv_stream and cctv_stream.is_running:
        # Convert numpy types for JSON
        prediction = cctv_stream.last_prediction.get('prediction', 'Unknown')
        confidence = cctv_stream.last_prediction.get('confidence', 0)
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        
        return jsonify({
            'running': True,
            'prediction': str(prediction),
            'confidence': float(confidence),
            'gender': cctv_stream.gender_info
        })
    
    return jsonify({'running': False})


# ======================= ALERT API ENDPOINTS =======================

@app.route('/alerts/config', methods=['GET', 'POST'])
def alerts_config():
    """Get or update alert configuration."""
    global ALERT_CONFIG
    
    if request.method == 'GET':
        # Return current config (without password)
        safe_config = {
            'enabled': ALERT_CONFIG['enabled'],
            'email_enabled': ALERT_CONFIG['email_enabled'],
            'recipient_email': ALERT_CONFIG['recipient_email'],
            'sender_email': ALERT_CONFIG['sender_email'],
            'cooldown_seconds': ALERT_CONFIG['cooldown_seconds'],
            'violence_threshold': ALERT_CONFIG['violence_threshold'],
            'has_credentials': bool(ALERT_CONFIG['sender_password'])
        }
        return jsonify(safe_config)
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'enabled' in data:
            ALERT_CONFIG['enabled'] = data['enabled']
        if 'email_enabled' in data:
            ALERT_CONFIG['email_enabled'] = data['email_enabled']
        if 'recipient_email' in data:
            ALERT_CONFIG['recipient_email'] = data['recipient_email']
        if 'sender_email' in data:
            ALERT_CONFIG['sender_email'] = data['sender_email']
        if 'sender_password' in data:
            ALERT_CONFIG['sender_password'] = data['sender_password']
        if 'cooldown_seconds' in data:
            ALERT_CONFIG['cooldown_seconds'] = int(data['cooldown_seconds'])
        if 'violence_threshold' in data:
            ALERT_CONFIG['violence_threshold'] = float(data['violence_threshold'])
        
        return jsonify({'status': 'updated', 'config': {
            'enabled': ALERT_CONFIG['enabled'],
            'email_enabled': ALERT_CONFIG['email_enabled'],
            'cooldown_seconds': ALERT_CONFIG['cooldown_seconds'],
            'violence_threshold': ALERT_CONFIG['violence_threshold']
        }})


@app.route('/alerts/history')
def alerts_history():
    """Get alert history."""
    log_path = os.path.join(app.config['ALERTS_FOLDER'], 'alert_log.json')
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                alerts = json.load(f)
            return jsonify({'alerts': alerts[-50:]})  # Last 50 alerts
        except:
            return jsonify({'alerts': []})
    
    return jsonify({'alerts': []})


@app.route('/alerts/test', methods=['POST'])
def test_alert():
    """Send a test alert."""
    if not ALERT_CONFIG['sender_email'] or not ALERT_CONFIG['recipient_email']:
        return jsonify({'error': 'Email not configured'}), 400
    
    # Create a test video path
    test_path = None
    success = send_email_alert(test_path, 100.0, 'TEST', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if success:
        return jsonify({'status': 'Test alert sent successfully'})
    else:
        return jsonify({'error': 'Failed to send test alert'}), 500


@app.route('/alerts/videos')
def alert_videos():
    """List recorded violence videos."""
    videos = []
    alerts_folder = app.config['ALERTS_FOLDER']
    
    if os.path.exists(alerts_folder):
        for f in os.listdir(alerts_folder):
            if f.endswith('.mp4'):
                filepath = os.path.join(alerts_folder, f)
                videos.append({
                    'filename': f,
                    'path': f'/alerts/{f}',
                    'size': os.path.getsize(filepath),
                    'created': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    videos.sort(key=lambda x: x['created'], reverse=True)
    return jsonify({'videos': videos[:20]})  # Last 20 videos


@app.route('/alerts/<filename>')
def serve_alert_video(filename):
    """Serve recorded alert videos."""
    return send_from_directory(app.config['ALERTS_FOLDER'], filename)


@app.route('/model/status')
def model_status():
    global model
    return jsonify({'loaded': model is not None})


if __name__ == '__main__':
    load_models()
    
    print("\n" + "="*60)
    print("Violence Detection Web Application (with Gender Detection)")
    print("="*60)
    print("Open your browser: http://127.0.0.1:8080")
    print("="*60)
    print("\n📧 To enable email alerts, configure:")
    print("   POST /alerts/config with sender_email, sender_password, recipient_email")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
