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
    "violence_threshold": 65
}

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
        
        # Try to load face cascade for gender detection helper
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None

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
        """Robust model loading with Flex op handling for Pi."""
        self.update_status("Loading Model...")
        
        try:
            import tensorflow as tf
            
            # 1. Try TFLite with Flex Delegate if available
            if os.path.exists(CONFIG["model_tflite"]):
                try:
                    # Attempt to load with Flex delegate (common issue on Pi)
                    # We check if we can initialize it
                    self.interpreter = tf.lite.Interpreter(model_path=CONFIG["model_tflite"])
                    self.interpreter.allocate_tensors()
                    print("SUCCESS: TFLite model loaded.")
                    self.update_status("TFLite Model Active")
                    return
                except Exception as e:
                    print(f"DEBUG: TFLite failed (likely Flex ops): {e}")
            
            # 2. Fallback to Keras model (Always works but uses more RAM/CPU)
            if os.path.exists(CONFIG["model_keras"]):
                print("FALLBACK: Loading full Keras model...")
                self.model = tf.keras.models.load_model(CONFIG["model_keras"])
                print("SUCCESS: Keras model loaded.")
                self.update_status("Keras Model Loaded (Steady)")
            else:
                messagebox.showerror("Error", f"Model files not found!\nNeed {CONFIG['model_keras']}")
                
        except ImportError:
            # Fallback for tflite-runtime only
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=CONFIG["model_tflite"])
                self.interpreter.allocate_tensors()
                self.update_status("TFLite (Runtime) Active")
            except Exception as e:
                messagebox.showerror("Runtime Error", f"Deep Learning libraries missing or incompatible.\n{e}")

    def update_status(self, text):
        self.footer_status.config(text=text)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(CONFIG["camera_id"])
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access webcam.")
                return
            self.is_running = True
            self.start_btn.config(text="STOP CAMERA", bg="#f44336")
            self.window.after(10, self.update_frame)
        else:
            self.is_running = False
            self.start_btn.config(text="START CAMERA", bg=THEME_GREEN)
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Simple Face detection for "People Count" and UI feedback
        # Not using DeepFace here as it kills Raspberry Pi performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) if self.face_cascade is not None else []
        self.people_count = len(faces)
        
        # Add frame to buffer for sequence prediction
        resized_for_model = cv2.resize(frame, CONFIG["input_size"])
        normalized = resized_for_model.astype(np.float32) / 255.0
        self.frame_buffer.append(normalized)
        
        # Run prediction if buffer is full
        if len(self.frame_buffer) == CONFIG["sequence_length"]:
            self.run_inference()

        # UI Updates
        # Update Status Box Color
        if "VIOLENCE" in self.prediction_text:
            self.status_box.config(text=self.prediction_text, bg="#f44336", fg="white")
        else:
            self.status_box.config(text=self.prediction_text, bg="#4caf50", fg="white")
            
        self.prob_label.config(text=f"Confidence: {self.prediction_prob:.1f}%")
        self.people_label.config(text=f"People in Frame: {self.people_count}")
        
        # Draw bounding boxes for UI feedback (similar to web app)
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (233, 69, 96), 2)
            cv2.putText(display_frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (233, 69, 96), 2)

        # Convert to TK Image
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        img = img.resize((700, 525), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_frame)

    def run_inference(self):
        input_data = np.array([list(self.frame_buffer)], dtype=np.float32)
        
        try:
            if self.model: # Keras
                prediction = self.model.predict(input_data, verbose=0)
            elif self.interpreter: # TFLite
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                self.interpreter.set_tensor(input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(output_details[0]['index'])
            else:
                return

            prob = float(prediction[0][0]) * 100
            if prob >= CONFIG["violence_threshold"]:
                self.prediction_text = "VIOLENCE DETECTED"
                self.prediction_prob = prob
            else:
                self.prediction_text = "NORMAL"
                self.prediction_prob = 100 - prob
        except Exception as e:
            print(f"Inference error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ViolenceDetectorApp(root)
    root.mainloop()
