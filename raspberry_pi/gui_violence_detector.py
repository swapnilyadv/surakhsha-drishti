#!/usr/bin/env python3
"""
Suraksha Drishti - Violence Detection GUI for Raspberry Pi
==========================================================
Modern UI matching the Web Application
Optimized for Raspberry Pi with Keras/TFLite support
"""

import os
import cv2
import numpy as np
import threading
import time
import json
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from collections import deque
from datetime import datetime

# --- SETTINGS ---
THEME_BG = "#1a1a2e"
THEME_ACCENT = "#e94560"
THEME_GREEN = "#4caf50"
THEME_WHITE = "#ffffff"
THEME_GRAY = "#a0a0a0"

CONFIG = {
    "model_keras": "best_violence_model.keras",
    "model_tflite": "violence_model.tflite",
    "input_size": (112, 112),
    "sequence_length": 24,
    "camera_id": 0,
    "violence_threshold": 65,
    "email_enabled": True,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "surakshadrishti.vesit@gmail.com",
    "sender_password": "kaqq zozs gthm zpla",
    "recipient_email": "2024.swapnil.yadav@ves.ac.in",
    "police_email": "police.alert.system@gmail.com",
    "cooldown_seconds": 60,
    "record_evidence": True,
    "pre_recording_buffer": 100, # Capture 100 frames before detection
    "post_recording_buffer": 200 # Capture 200 frames after detection
}

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class ViolenceDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Suraksha Drishti - Violence Detection System")
        self.window.geometry("1100x750")
        self.window.configure(bg=THEME_BG)
        
        # Initialize Variables
        self.cap = None
        self.model = None
        self.interpreter = None
        self.frame_buffer = deque(maxlen=CONFIG["sequence_length"])
        self.is_running = False
        self.prediction_text = "INITIALIZING..."
        self.prediction_prob = 0
        self.people_count = 0
        self.gender_info = "N/A"
        
        # NEW: Add a frame counter for skipping
        self.frame_counter = 0
        
        # New Video/Alert Variables
        self.full_frame_buffer = deque(maxlen=CONFIG["pre_recording_buffer"])
        self.post_violence_buffer = []
        self.violence_was_detected = False
        self.last_alert_time = 0
        
        # Fixed Face Cascade path to use your local file
        xml_path = "haarcascade_frontalface_default.xml"
        if os.path.exists(xml_path):
            self.face_cascade = cv2.CascadeClassifier(xml_path)
            print("✓ Local Face Cascade Loaded")
        else:
            # Fallback to system default if available
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                print("⚠️ Warning: No face detector found!")

        self.setup_ui()
        self.load_model_safe()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.window, bg=THEME_BG, pady=20)
        header_frame.pack(fill="x")
        
        title_label = tk.Label(header_frame, text="SURAKSHA DRISHTI", font=("Segoe UI", 28, "bold"), 
                              fg=THEME_ACCENT, bg=THEME_BG)
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Advanced AI Violence Detection System", 
                                 font=("Segoe UI", 12), fg=THEME_GRAY, bg=THEME_BG)
        subtitle_label.pack()

        # Main Layout
        main_frame = tk.Frame(self.window, bg=THEME_BG)
        main_frame.pack(expand=True, fill="both", padx=30, pady=10)
        
        # Left Side: Video
        self.video_frame = tk.Frame(main_frame, bg="#0f3460", bd=2, relief="flat")
        self.video_frame.pack(side="left", expand=True, fill="both")
        
        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Right Side: Controls & Stats
        sidebar = tk.Frame(main_frame, bg=THEME_BG, width=300)
        sidebar.pack(side="right", fill="y", padx=(20, 0))
        
        # Stats Box
        stats_frame = tk.LabelFrame(sidebar, text="REAL-TIME MONITOR", font=("Segoe UI", 10, "bold"),
                                   bg=THEME_BG, fg=THEME_WHITE, padx=15, pady=15, bd=1, relief="solid")
        stats_frame.pack(fill="x", pady=(0, 20))
        
        # Violence Status Box
        self.status_box = tk.Label(stats_frame, text="WAITING", font=("Segoe UI", 20, "bold"),
                                  bg="#222", fg=THEME_WHITE, pady=15)
        self.status_box.pack(fill="x", pady=10)
        
        self.prob_label = tk.Label(stats_frame, text="Confidence: 0%", font=("Segoe UI", 11),
                                  bg=THEME_BG, fg=THEME_GRAY)
        self.prob_label.pack(anchor="w")
        
        # Analytics box
        analytics_frame = tk.Frame(stats_frame, bg=THEME_BG, pady=10)
        analytics_frame.pack(fill="x")
        
        self.people_label = tk.Label(analytics_frame, text="People in Frame: 0", font=("Segoe UI", 11),
                                    bg=THEME_BG, fg=THEME_WHITE)
        self.people_label.pack(anchor="w")
        
        self.gender_label = tk.Label(analytics_frame, text="Gender Context: N/A", font=("Segoe UI", 11),
                                    bg=THEME_BG, fg=THEME_WHITE)
        self.gender_label.pack(anchor="w")

        # Controls box
        ctrl_frame = tk.Frame(sidebar, bg=THEME_BG)
        ctrl_frame.pack(fill="x", pady=20)
        
        self.start_btn = tk.Button(ctrl_frame, text="START CAMERA", command=self.toggle_camera,
                                  font=("Segoe UI", 12, "bold"), bg=THEME_GREEN, fg="white", 
                                  relief="flat", pady=10, cursor="hand2")
        self.start_btn.pack(fill="x", pady=5)
        
        exit_btn = tk.Button(ctrl_frame, text="EXIT SYSTEM", command=self.window.quit,
                            font=("Segoe UI", 12, "bold"), bg="#444", fg="white", 
                            relief="flat", pady=10, cursor="hand2")
        exit_btn.pack(fill="x", pady=5)
        
        # Status Bar
        self.footer_status = tk.Label(self.window, text="System Ready", bd=1, relief="flat", 
                                    anchor="w", bg="#16213e", fg=THEME_GRAY, font=("Segoe UI", 9))
        self.footer_status.pack(side="bottom", fill="x")

    def load_model_safe(self):
        """Loads TFLite or sets up Motion Fallback if AI fails."""
        self.update_status("Loading Model...")
        try:
            import tflite_runtime.interpreter as tflite
            # Try to load the TFLite model
            self.interpreter = tflite.Interpreter(model_path=CONFIG["model_tflite"])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_loaded = True
            print("✓ TFLite Model Loaded Successfully")
            self.update_status("System Online: TFLite Ready")
        except Exception as e:
            print(f"AI Model Error (likely Flex ops): {e}")
            self.model_loaded = False
            self.update_status("Running: Motion Analysis Mode (AI Offline)")

    def update_status(self, text):
        self.footer_status.config(text=text)

    def toggle_camera(self):
        if not self.is_running:
            # Try V4L2 backend on Linux to avoid GStreamer error
            self.cap = cv2.VideoCapture(CONFIG["camera_id"], cv2.CAP_V4L2)
            
            # NEW: Set a lower resolution to reduce lag
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.cap.isOpened():
                # Fallback to default if V4L2 isn't working
                self.cap = cv2.VideoCapture(CONFIG["camera_id"])
                
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access webcam.")
                return
            self.is_running = True
            self.start_btn.config(text="STOP CAMERA", bg="#f44336")
            self.window.after(10, self.update_frame)
        else:
            self.is_running = False
            self.start_btn.config(text="STOP CAMERA", bg=THEME_GREEN)
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.window.after(10, self.update_frame)
            return
            
        self.frame_counter += 1

        # Store full quality frame for evidence
        self.full_frame_buffer.append(frame.copy())
        
        # Handle ongoing evidence recording
        if self.violence_was_detected:
            self.post_violence_buffer.append(frame.copy())
            if len(self.post_violence_buffer) >= CONFIG["post_recording_buffer"]:
                self.save_and_send_evidence()

        # --- OPTIMIZATION: Process only every 3rd frame ---
        if self.frame_counter % 3 == 0:
            # Add frame to buffer for sequence prediction
            resized_for_model = cv2.resize(frame, CONFIG["input_size"])
            normalized = resized_for_model.astype(np.float32) / 255.0
            self.frame_buffer.append(normalized)
            
            # Run prediction if buffer is full
            if len(self.frame_buffer) == CONFIG["sequence_length"]:
                self.run_inference()

            # Simple Face detection for "People Count" and UI feedback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if self.face_cascade is not None else []
            self.people_count = len(faces)
        else:
            # On skipped frames, just grab the last known face data
            faces = [] # Or you could store the last known faces to make it look smoother

        # UI Updates (run every frame for smoothness)
        # Update Status Box Color
        if "VIOLENCE" in self.prediction_text:
            self.status_box.config(text=self.prediction_text, bg="#f44336", fg="white")
        else:
            self.status_box.config(text=self.prediction_text, bg="#4caf50", fg="white")
            
        self.prob_label.config(text=f"Confidence: {self.prediction_prob:.1f}%")
        self.people_label.config(text=f"People in Frame: {self.people_count}")
        
        # Draw bounding boxes and detect "Gender" (Simplified for Pi)
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            # Improved logic: Male/Female based on face dimensions (placeholder for Pi speed)
            is_male = (w * h) > 3500 
            gender = "Male" if is_male else "Female"
            color = (76, 175, 80) if is_male else (233, 69, 96) # Green vs Pink
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, f"{gender}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update the side label
            self.gender_info = f"Last Detected: {gender}"

        self.gender_label.config(text=f"Gender Context: {self.gender_info}")

        # Convert to TK Image
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        # NEW: Use a faster resizing algorithm
        img = img.resize((700, 525), Image.Resampling.BILINEAR)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_frame)

    def run_inference(self):
        """Unified analysis: Uses AI if loaded, else uses Motion Energy."""
        if not self.model_loaded:
            # --- MOTION ANALYSIS FALLBACK ---
            # If the AI model failed (no Flex support), we use Pixel Difference
            # to detect 'Violence-like' motion.
            if len(self.frame_buffer) < 2: return
            
            f1 = self.frame_buffer[-1]
            f2 = self.frame_buffer[-2]
            
            # Simple motion math: Pixel difference between frames
            diff = cv2.absdiff((f1 * 255).astype(np.uint8), (f2 * 255).astype(np.uint8))
            motion_score = np.mean(diff)
            
            # Threshold for "Aggressive" movement (calibrated for Pi 3)
            if motion_score > 35: # High value = sudden/violent movement
                self.prediction_text = "🚨 VIOLENCE DETECTED"
                self.prediction_prob = min(motion_score * 2.5, 99.0)
            else:
                self.prediction_text = "NORMAL"
                self.prediction_prob = 100 - (motion_score * 2)
            return

        # AI-based inference for machines with proper runtime
        try:
            input_data = np.array([list(self.frame_buffer)], dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            prob = float(prediction[0][0]) * 100
            if prob >= CONFIG["violence_threshold"]:
                self.prediction_text = "🚨 VIOLENCE DETECTED"
                self.prediction_prob = prob
            else:
                self.prediction_text = "NORMAL"
                self.prediction_prob = 100 - prob
        except Exception as e:
            # Fallback to motion if inference crashes mid-run
            self.model_loaded = False
            print(f"Warning: Inference crashed, switching to Motion Mode: {e}")

        if "VIOLENCE" in self.prediction_text:
            self.trigger_evidence_capture()

    def trigger_evidence_capture(self):
        """Start full evidence capture when violence is flagged."""
        if not self.violence_was_detected:
            current_time = time.time()
            if current_time - self.last_alert_time > CONFIG["cooldown_seconds"]:
                print("🚨 Violence Detected! Capturing start-to-end evidence...")
                self.violence_was_detected = True
                self.post_violence_buffer = [] # Reset for new capture

    def save_and_send_evidence(self):
        """Stitches start-to-end video and sends to police and user."""
        self.violence_was_detected = False
        self.last_alert_time = time.time()
        
        # Combine buffers
        full_evidence = list(self.full_frame_buffer) + self.post_violence_buffer
        self.post_violence_buffer = []
        
        # Save Video
        filename = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(os.getcwd(), filename)
        
        h, w = full_evidence[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 15.0, (w, h))
        
        for f in full_evidence:
            cv2.putText(f, "EVIDENCE LOG - " + datetime.now().strftime('%H:%M:%S'), 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(f)
        out.release()
        
        print(f"📹 Evidence video saved: {out_path}")
        
        # Send Email in thread
        threading.Thread(target=self.send_alerts, args=(out_path,)).start()

    def send_alerts(self, video_path):
        try:
            for email in [CONFIG["recipient_email"], CONFIG["police_email"]]:
                if not email: continue
                msg = MIMEMultipart()
                msg['From'] = CONFIG["sender_email"]
                msg['To'] = email
                msg['Subject'] = "🚨 CRITICAL: VIOLENCE EVIDENCE DETECTED"
                
                body = f"VIOLENCE DETECTED\nTime: {datetime.now()}\n\nPlease find attached the 'start-to-end' video evidence as proof."
                msg.attach(MIMEText(body, 'plain'))
                
                with open(video_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(video_path)}')
                    msg.attach(part)
                
                server = smtplib.SMTP(CONFIG['smtp_server'], CONFIG['smtp_port'])
                server.starttls()
                server.login(CONFIG['sender_email'], CONFIG['sender_password'])
                server.send_message(msg)
                server.quit()
                print(f"📧 Alert sent to {email}")
        except Exception as e:
            print(f"❌ Alert failed: {e}")
