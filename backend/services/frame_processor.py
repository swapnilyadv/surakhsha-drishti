"""
Frame Processor — Central AI Surveillance Pipeline
====================================================
Orchestrates the full detection pipeline:

  Camera Feed
    → YOLO Human Detection        (every 5th webcam frame)
    → MediaPipe Pose Estimation   (every 5th webcam frame, on detected persons)
    → ViT Violence Detection      (every 20th frame, rate-limited 2/sec)
    → ONNX Weapon Detection       (every 5th webcam frame)
    → Overlay Rendering           (every frame for smooth MJPEG video)
    → JSON Result                 (shared with FastAPI async layer)
    → WebSocket Broadcast         (triggered by FastAPI broadcaster)

Threading model:
  Thread 1 (webcam-capture):
    - Reads frames from camera at ~15-20 FPS
    - Encodes JPEG for MJPEG stream (with overlays)
    - Pushes every Nth frame to AI queue

  Thread 2 (ai-process):
    - Dequeues frames
    - Runs Human → Pose → Violence → Weapon inference
    - Updates shared DetectionResult (lock-protected)
    - Computes rolling FPS counter

Main/Async thread:
  - get_detection_result()  → dict  (safe to call from async)
  - get_latest_jpeg()       → bytes (MJPEG stream consumer)

Performance design:
  - Weapon + Human + Pose: every 5th frame  (~3 FPS at 15 FPS capture)
  - Violence (ViT):         every 20th frame + rate-limited
  - MJPEG encoding:         every frame (smooth 15-20 FPS video)
  - Queue max size 3:       drops AI frames when busy (non-blocking)
  - JPEG quality 65:        good quality vs. encoding CPU tradeoff
"""

import cv2
import time
import queue
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from services.human_detector  import HumanDetector
from services.weapon_detector import WeaponDetector
from services.violence_detector import ViolenceDetector
from services.pose_estimator  import PoseEstimator

logger = logging.getLogger(__name__)

# ── Frame skip config ─────────────────────────────────────────────────────────
WEBCAM_FPS_TARGET  = 20          # target webcam capture rate (FPS)
AI_EVERY_N_FRAMES  = 5           # run human/pose/weapon detection every Nth frame
VIOLENCE_EVERY_N   = 4           # run violence every 4th AI frame (= every 20th webcam)
JPEG_QUALITY       = 65          # MJPEG JPEG quality 0–100
MAX_QUEUE_SIZE     = 3           # AI queue depth (drop frames when full)
FPS_WINDOW         = 30          # rolling window for FPS calculation (frames)


# ── Shared detection result ───────────────────────────────────────────────────
@dataclass
class DetectionResult:
    """Latest AI inference state — written by AI thread, read by async layer."""

    # Person detection
    person_count:  int  = 0
    persons:       list = field(default_factory=list)   # bboxes + aggression scores

    # Violence
    violence_detected:    bool  = False
    violence_label:       str   = "Non Violence"
    violence_confidence:  float = 0.0
    action:               str   = "Normal"
    aggression_score:     float = 0.0

    # Weapon
    weapon_detected:    bool  = False
    weapon_label:       str   = ""
    weapon_confidence:  float = 0.0
    weapon_detections:  list  = field(default_factory=list)

    # Gender counts (from frontend face-api — not updated here, kept for WS payload compat)
    male_count:   int = 0
    female_count: int = 0

    # Timing
    fps:         float = 0.0
    timestamp:   float = field(default_factory=time.time)
    frame_count: int   = 0


class FrameProcessor:
    """
    Central pipeline manager.

    Exposes:
      start()                → starts background threads
      stop()                 → graceful shutdown
      get_detection_result() → latest detection dict (async-safe)
      get_latest_jpeg()      → latest JPEG bytes for MJPEG stream
      update_gender_counts() → called by WebSocket handler when frontend sends gender data
    """

    def __init__(
        self,
        weapon_detector:   WeaponDetector,
        violence_detector: ViolenceDetector,
        human_detector:    HumanDetector,
        pose_estimator:    PoseEstimator,
        camera_index:      int = 0,
        evidence_manager = None,
        alert_manager = None,
    ):
        self.weapon_detector   = weapon_detector
        self.violence_detector = violence_detector
        self.human_detector    = human_detector
        self.pose_estimator    = pose_estimator
        self.camera_index      = camera_index
        self.evidence_manager  = evidence_manager
        self.alert_manager     = alert_manager

        # Shared state (lock-protected)
        self._result       = DetectionResult()
        self._result_lock  = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock    = threading.Lock()

        # AI inference queue (capture thread → AI thread)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        # Control
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # FPS rolling window
        self._frame_times: list[float] = []

        # Temporal gender smoothing buffers
        self._male_history = []
        self._female_history = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start webcam capture + AI processing threads."""
        if self._running:
            return
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="webcam-capture"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="ai-process"
        )
        self._capture_thread.start()
        self._process_thread.start()
        logger.info("[FrameProcessor] Started capture + processing threads")

    def stop(self):
        """Gracefully stop all background threads."""
        self._running = False
        # Unblock AI thread if waiting on empty queue
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
        if self._process_thread:
            self._process_thread.join(timeout=3)

        # Release pose estimator resources
        if self.pose_estimator.loaded:
            self.pose_estimator.close()

        logger.info("[FrameProcessor] Stopped")

    def change_source(self, new_source):
        """Safely stops active streams, changes source, and restarts the processor."""
        logger.info(f"[FrameProcessor] Changing input source from {self.camera_index} to {new_source}")
        
        # Stop background loops
        self._running = False
        
        # Unblock process thread
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
            
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
        if self._process_thread:
            self._process_thread.join(timeout=3)
            
        # Re-initialize state
        self.camera_index = new_source
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._frame_times = []
        self._running = True
        
        # Restart threads
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture-thread"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="ai-process"
        )
        self._capture_thread.start()
        self._process_thread.start()
        logger.info(f"[FrameProcessor] Restarted capture + processing on new source: {new_source}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_detection_result(self) -> dict:
        """
        Return the latest detection result as a JSON-serialisable dict.
        Safe to call from the async FastAPI layer.

        Fully compatible with both the Next.js frontend WS contract
        AND the user's professional CCTV JSON specification.
        """
        with self._result_lock:
            r = self._result
            # Build persons list for JSON (strip numpy types, add standard 'label' key)
            persons_json = [
                {
                    "label":            "Person",
                    "bbox":             p.get("bbox", []),
                    "aggression_score": p.get("aggression_score", 0.0),
                    "pose_flags":       p.get("pose_flags", []),
                    "keypoints":        [],    # heavy — only expose on demand
                    "confidence":       p.get("confidence", 0.0),
                }
                for p in r.persons
            ]

            # Limit FPS representation to round integer for the user's spec
            user_fps = int(round(r.fps)) if r.fps > 0 else 30

            return {
                # ── Core WS fields (frontend contract) ────────────────────────
                "weapon_detected":    r.weapon_detected,
                "weapon_label":       r.weapon_label,
                "weapon_confidence":  round(r.weapon_confidence, 3),
                "violence_detected":  r.violence_detected,
                "violence_label":     r.violence_label,
                "violence_confidence": round(r.violence_confidence, 3),
                "male_count":         r.male_count,
                "female_count":       r.female_count,
                "timestamp":          r.timestamp,
                "frame_count":        r.frame_count,

                # ── Extended detection fields ──────────────────────────────────
                "action":             r.action,
                "aggression_score":   round(r.aggression_score, 3),
                "fps":                user_fps,
                "person_count":       r.person_count,
                "persons":            persons_json,

                # ── User Specific Schema Mapping ──────────────────────────────
                "violence":           r.violence_detected,
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return latest JPEG frame bytes for MJPEG stream consumer."""
        with self._jpeg_lock:
            return self._latest_jpeg

    def update_gender_counts(self, male: int, female: int):
        """
        Update gender counts sent from frontend (face-api).
        Applies a temporal moving average to smooth flickers and return stabilized counts.
        If confidence was low, the count naturally stabilizes or drops.
        """
        with self._result_lock:
            self._male_history.append(male)
            self._female_history.append(female)
            
            # Keep last 5 updates
            if len(self._male_history) > 5:
                self._male_history.pop(0)
            if len(self._female_history) > 5:
                self._female_history.pop(0)
                
            # Average and round to stabilize predictions
            smoothed_male = int(round(sum(self._male_history) / len(self._male_history)))
            smoothed_female = int(round(sum(self._female_history) / len(self._female_history)))
            
            self._result.male_count   = smoothed_male
            self._result.female_count = smoothed_female

    # ── Capture Thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Thread 1: Reads frames from webcam or video file at target FPS.
        - Always encodes MJPEG (smooth video regardless of AI speed)
        - Pushes every AI_EVERY_N_FRAMES-th frame to AI queue
        - Auto-loops if source is a video file
        """
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            logger.error(f"[Capture] Cannot open source {self.camera_index}")
            self._run_placeholder_loop()
            return

        is_video_file = isinstance(self.camera_index, str)
        fps_target = WEBCAM_FPS_TARGET

        if is_video_file:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps_target = video_fps
        else:
            # Configure webcam for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          WEBCAM_FPS_TARGET)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # single-frame buffer = min latency

        frame_interval = 1.0 / fps_target
        last_time      = time.time()
        frame_num      = 0

        logger.info(f"[Capture] Source opened successfully (Target FPS: {fps_target:.2f})")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    # Loop video file continuously
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("[Capture] Frame read failed — retrying")
                time.sleep(0.1)
                continue

            frame_num += 1

            # Always encode MJPEG for smooth streaming
            self._encode_jpeg(frame, frame_num)

            # Push every Nth frame to AI processing queue (non-blocking drop)
            if frame_num % AI_EVERY_N_FRAMES == 0:
                try:
                    self._frame_queue.put_nowait((frame.copy(), frame_num))
                except queue.Full:
                    pass    # AI busy — drop frame (video stays smooth)

            # Throttle to target FPS
            elapsed    = time.time() - last_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

        cap.release()
        logger.info("[Capture] Camera released")

    def _run_placeholder_loop(self):
        """Show 'Camera Offline' placeholder when camera unavailable."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        # Dark background with camera offline text
        cv2.rectangle(placeholder, (0, 0), (640, 480), (10, 10, 20), -1)
        cv2.putText(
            placeholder, "CAMERA OFFLINE",
            (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 200), 2, cv2.LINE_AA
        )
        cv2.putText(
            placeholder, "Check camera connection and restart",
            (80, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1, cv2.LINE_AA
        )

        while self._running:
            # Animate timestamp on placeholder
            ts_frame = placeholder.copy()
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                ts_frame, ts,
                (270, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 80), 1, cv2.LINE_AA
            )
            self._encode_jpeg(ts_frame, 0)
            time.sleep(0.5)

    # ── AI Processing Thread ──────────────────────────────────────────────────

    def _process_loop(self):
        """
        Thread 2: Dequeues frames and runs AI inference pipeline.

        Pipeline order (CPU budget rationale):
          1. Human detection  — fast ONNX (every dequeued frame)
          2. Pose estimation  — MediaPipe lite (every dequeued frame, on cropped persons)
          3. Weapon detection — ONNX (every dequeued frame)
          4. Violence detect  — ViT (every 4th dequeued frame = every 20th webcam)
        """
        process_count = 0

        while self._running:
            try:
                item = self._frame_queue.get(timeout=1.0)

                # Shutdown sentinel
                if item is None:
                    break

                frame, frame_num = item
                process_count += 1

                # ── 1. Human Detection ─────────────────────────────────────────
                person_detections = self.human_detector.detect(frame)

                # ── 2. Pose Estimation ─────────────────────────────────────────
                # Run on full frame (MediaPipe handles multi-person internally)
                pose_persons = self.pose_estimator.detect(frame) if self.pose_estimator.loaded else []

                # ── 3. Weapon Detection ────────────────────────────────────────
                weapon_result = self.weapon_detector.detect(frame)

                # ── 4. Violence Detection (every frame to feed temporal sequence analysis) ──
                # We always run the detector, which tracks joint trajectories over time.
                # The detector's rate-limiter prevents heavy deep-learning model calls,
                # ensuring ultra-smooth performance and low CPU usage.
                violence_result = self.violence_detector.detect(frame, pose_persons)

                # ── 5. Merge person data: YOLO bbox + pose aggression ──────────
                merged_persons = self._merge_person_data(
                    person_detections, pose_persons
                )

                # ── 6. Update FPS counter ──────────────────────────────────────
                fps = self._update_fps()

                # ── 7. Write shared result (lock held briefly) ─────────────────
                with self._result_lock:
                    self._result.person_count       = len(person_detections)
                    self._result.persons            = merged_persons

                    self._result.violence_detected  = violence_result.get("violence_detected", False)
                    self._result.violence_label     = violence_result.get("label", "Non Violence")
                    self._result.violence_confidence = violence_result.get("confidence", 0.0)
                    self._result.action             = violence_result.get("action", "Normal")
                    self._result.aggression_score   = violence_result.get("aggression_score", 0.0)

                    self._result.weapon_detected    = weapon_result.get("weapon_detected", False)
                    self._result.weapon_label       = weapon_result.get("label", "")
                    self._result.weapon_confidence  = weapon_result.get("confidence", 0.0)
                    self._result.weapon_detections  = weapon_result.get("detections", [])

                    self._result.fps         = fps
                    self._result.timestamp   = time.time()
                    self._result.frame_count = frame_num

                if self.alert_manager:
                    is_threat = self._result.violence_detected or self._result.weapon_detected
                    threat_label = self._result.violence_label if self._result.violence_detected else (self._result.weapon_label if self._result.weapon_detected else "NOMINAL")
                    self.alert_manager.update_threat_state(
                        violence_detected=is_threat,
                        threat_type=threat_label,
                        confidence=self._result.violence_confidence if self._result.violence_detected else self._result.weapon_confidence
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ProcessThread] Unexpected error: {e}", exc_info=True)

    # ── MJPEG Encoding + Overlay Rendering ───────────────────────────────────

    def _encode_jpeg(self, frame: np.ndarray, frame_num: int):
        """Render detection overlays on frame and encode as JPEG for MJPEG stream."""
        try:
            display = self._draw_overlays(frame.copy(), frame_num)
            
            # Feed annotated display frames to the EvidenceManager
            if self.evidence_manager:
                is_threat = self._result.violence_detected or self._result.weapon_detected
                threat_label = self._result.violence_label if self._result.violence_detected else (self._result.weapon_label if self._result.weapon_detected else "NOMINAL")
                cam_label = "Webcam Unit"
                if isinstance(self.camera_index, str):
                    from pathlib import Path
                    cam_label = Path(self.camera_index).stem
                elif self.camera_index == 0:
                    cam_label = "Live Webcam"
                    
                self.evidence_manager.add_frame(
                    display, 
                    is_threat=is_threat, 
                    threat_type=threat_label, 
                    camera_id=f"CAM-{self.camera_index}", 
                    camera_label=cam_label,
                    male_count=self._result.male_count,
                    female_count=self._result.female_count,
                    weapon_detected=self._result.weapon_detected,
                    weapon_type=self._result.weapon_label,
                    confidence=self._result.violence_confidence if self._result.violence_detected else self._result.weapon_confidence
                )
                
            ok, buf = cv2.imencode(
                ".jpg", display,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
        except Exception as e:
            logger.debug(f"[JPEG] Encode error: {e}")

    def _draw_overlays(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Draw AI detection overlays on the display frame:
          - Weapon bounding boxes (red)
          - Person bounding boxes (green/orange/red based on aggression)
          - Pose skeleton lines + joints
          - FPS counter + timestamp HUD
          - Alert banner when violence/weapon detected
        """
        with self._result_lock:
            r = self._result

        h, w = frame.shape[:2]
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # ── Person bounding boxes + aggression colour coding ─────────────────
        for p in r.persons:
            bbox = p.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                agg = p.get("aggression_score", 0.0)
                # Green → orange → red
                col_r = int(min(255, agg * 2 * 255))
                col_g = int(min(255, (1 - agg) * 2 * 255))
                colour = (0, col_g, col_r)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

                # Clean simple label: display ONLY "Person"
                label = "Person"
                lw = len(label) * 9
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + lw, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        # ── Pose skeleton overlay ────────────────────────────────────────────
        if self.pose_estimator.loaded and r.persons:
            # Only draw skeletons for persons that have keypoints
            pose_only = [p for p in r.persons if p.get("keypoints")]
            if pose_only:
                self.pose_estimator.draw_skeleton(frame, pose_only, h, w)

        # ── Weapon bounding boxes (red overlay) ──────────────────────────────
        for det in r.weapon_detections:
            box = det.get("box", [])
            if len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                lbl = f"{det['label']} {det['confidence']:.0%}"
                lw = len(lbl) * 9
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + lw, y1), (0, 0, 200), -1)
                cv2.putText(frame, lbl, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Alert banner (top strip) when violence/weapon active ─────────────
        if r.violence_detected or r.weapon_detected:
            banner_colour = (0, 0, 180) if r.weapon_detected else (0, 0, 220)
            # Blink effect: show every other second
            if int(time.time()) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, 28), banner_colour, -1)
                alert_text = (
                    f"⚠ WEAPON: {r.weapon_label.upper()}" if r.weapon_detected
                    else f"⚠ {r.action.upper()} DETECTED ({r.violence_confidence:.0%})"
                )
                cv2.putText(frame, alert_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Bottom-left HUD: FPS + timestamp ─────────────────────────────────
        hud_lines = [
            f"FPS: {r.fps:.1f}",
            f"PERSONS: {r.person_count}",
            f"{now_str}",
        ]
        hud_y_base = h - 10 - (len(hud_lines) - 1) * 18
        cv2.rectangle(frame, (0, hud_y_base - 14), (150, h), (0, 0, 0), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (5, hud_y_base + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 100), 1, cv2.LINE_AA)

        # ── Top-right: action label ───────────────────────────────────────────
        action_col = (0, 100, 255) if r.violence_detected else (0, 220, 100)
        action_text = r.action.upper()
        (tw, th), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, action_text, (w - tw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_col, 1, cv2.LINE_AA)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_avg_aggression(self, pose_persons: list) -> float:
        """Compute the maximum aggression score across all detected persons."""
        if not pose_persons:
            return 0.0
        scores = [p.get("aggression_score", 0.0) for p in pose_persons]
        # Use max rather than average — one very aggressive person = threat
        return float(max(scores))

    def _merge_person_data(
        self,
        yolo_persons: list[dict],
        pose_persons:  list[dict],
    ) -> list[dict]:
        """
        Merge YOLO person bboxes with MediaPipe pose data.

        Strategy:
          - YOLO gives reliable bounding boxes (person_count × bbox)
          - MediaPipe gives pose keypoints + aggression score
          - We match them by index (assumes same ordering works well enough
            for 1-4 persons; full IoU matching would be overkill here)

        Returns merged list:
          [{"bbox": [...], "confidence": float, "aggression_score": float,
            "pose_flags": [...], "keypoints": [...]}, ...]
        """
        merged = []

        # Start from YOLO detections (ground truth for bboxes)
        for i, yp in enumerate(yolo_persons):
            entry = {
                "bbox":             yp.get("bbox", []),
                "confidence":       yp.get("confidence", 0.0),
                "aggression_score": 0.0,
                "pose_flags":       [],
                "keypoints":        [],
            }
            # Match with pose data (index-based for simplicity)
            if i < len(pose_persons):
                pp = pose_persons[i]
                entry["aggression_score"] = pp.get("aggression_score", 0.0)
                entry["pose_flags"]       = pp.get("pose_flags", [])
                entry["keypoints"]        = pp.get("keypoints", [])
            merged.append(entry)

        # If pose detected more persons than YOLO (e.g. partial occlusion),
        # include extra pose entries without a bbox
        for i in range(len(yolo_persons), len(pose_persons)):
            pp = pose_persons[i]
            merged.append({
                "bbox":             [],
                "confidence":       0.0,
                "aggression_score": pp.get("aggression_score", 0.0),
                "pose_flags":       pp.get("pose_flags", []),
                "keypoints":        pp.get("keypoints", []),
            })

        return merged

    def _update_fps(self) -> float:
        """Compute rolling-window FPS for the AI processing thread."""
        now = time.time()
        self._frame_times.append(now)
        # Keep only the last N frame timestamps
        if len(self._frame_times) > FPS_WINDOW:
            self._frame_times = self._frame_times[-FPS_WINDOW:]
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed
