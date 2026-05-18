"""
Frame Processor
================
Orchestrates webcam capture + AI inference with frame skipping.

Architecture:
  - Thread 1 (capture_thread): reads webcam frames → puts in queue
  - Thread 2 (process_thread): pulls frames → runs AI → updates result
  - Main thread: reads latest result + MJPEG frame via async API

Frame skipping strategy (CPU optimization):
  - Weapon detection:  every 5th frame  (~3 FPS at 15 FPS webcam)
  - Violence detection: every 20th frame (~0.75 FPS, also rate-limited to 1/sec)
  - MJPEG stream:      every frame (smooth video, no AI)
"""

import cv2
import time
import queue
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from services.weapon_detector import WeaponDetector
from services.violence_detector import ViolenceDetector

logger = logging.getLogger(__name__)

# ─── Frame Skip Configuration ─────────────────────────────────────────────────
WEBCAM_FPS_TARGET = 15       # target webcam capture rate
WEAPON_EVERY_N = 5           # run weapon detection every 5th frame
VIOLENCE_EVERY_N = 20        # run violence detection every 20th frame
JPEG_QUALITY = 60            # MJPEG quality (lower = less CPU encoding)
MAX_QUEUE_SIZE = 3           # cap queue to prevent memory buildup


@dataclass
class DetectionResult:
    """Latest AI detection state — shared between threads."""
    weapon_detected: bool = False
    weapon_label: str = ""
    weapon_confidence: float = 0.0
    weapon_detections: list = field(default_factory=list)

    violence_detected: bool = False
    violence_label: str = "Non Violence"
    violence_confidence: float = 0.0

    male_count: int = 0
    female_count: int = 0

    timestamp: float = field(default_factory=time.time)
    frame_count: int = 0


class FrameProcessor:
    """
    Manages webcam capture + AI inference in background threads.
    Frontend accesses:
      - get_detection_result() → lightweight dict
      - get_latest_jpeg()      → raw JPEG bytes for MJPEG stream
    """

    def __init__(self, weapon_detector: WeaponDetector, violence_detector: ViolenceDetector, camera_index: int = 0):
        self.weapon_detector = weapon_detector
        self.violence_detector = violence_detector
        self.camera_index = camera_index

        # Shared state (protected by locks)
        self._result = DetectionResult()
        self._result_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()

        # Frame queue between capture and processing threads
        self._frame_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        # Control flags
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

    def start(self):
        """Start capture + processing threads."""
        if self._running:
            return
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="webcam-capture")
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True, name="ai-process")
        self._capture_thread.start()
        self._process_thread.start()
        logger.info("[FrameProcessor] Started capture + processing threads")

    def stop(self):
        """Gracefully stop all threads."""
        self._running = False
        # Unblock queue if process thread is waiting
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
        if self._process_thread:
            self._process_thread.join(timeout=3)
        logger.info("[FrameProcessor] Stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_detection_result(self) -> dict:
        """Return latest detection as a lightweight dict (safe to call from async)."""
        with self._result_lock:
            r = self._result
            return {
                "weapon_detected": r.weapon_detected,
                "weapon_label": r.weapon_label,
                "weapon_confidence": r.weapon_confidence,
                "violence_detected": r.violence_detected,
                "violence_label": r.violence_label,
                "violence_confidence": r.violence_confidence,
                "male_count": r.male_count,
                "female_count": r.female_count,
                "timestamp": r.timestamp,
                "frame_count": r.frame_count,
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return latest JPEG frame bytes for MJPEG stream."""
        with self._jpeg_lock:
            return self._latest_jpeg

    # ── Capture Thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Thread 1: Reads frames from webcam at ~15 FPS.
        Puts frames into queue for AI processing.
        Also encodes JPEG for MJPEG stream.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"[Capture] Cannot open camera {self.camera_index}")
            # Generate a placeholder frame so MJPEG stream still works
            self._generate_placeholder_frames()
            return

        # Set webcam properties for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS_TARGET)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer lag

        frame_interval = 1.0 / WEBCAM_FPS_TARGET
        last_time = time.time()
        frame_num = 0

        logger.info("[Capture] Webcam opened successfully")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[Capture] Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_num += 1

            # Always update MJPEG stream (smooth video)
            self._encode_jpeg(frame, frame_num)

            # Only enqueue every Nth frame for AI (non-blocking put)
            if frame_num % WEAPON_EVERY_N == 0:
                try:
                    self._frame_queue.put_nowait((frame.copy(), frame_num))
                except queue.Full:
                    pass  # Drop frame — AI is busy, that's OK

            # Throttle to target FPS
            elapsed = time.time() - last_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

        cap.release()
        logger.info("[Capture] Camera released")

    def _generate_placeholder_frames(self):
        """Generate a 'Camera Offline' placeholder JPEG when camera unavailable."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "CAMERA OFFLINE", (160, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
        cv2.putText(placeholder, "Check camera connection", (140, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        while self._running:
            self._encode_jpeg(placeholder, 0)
            time.sleep(0.1)

    def _encode_jpeg(self, frame: np.ndarray, frame_num: int):
        """Encode frame as JPEG and update shared buffer."""
        # Optionally draw detection overlays
        display_frame = self._draw_overlays(frame.copy(), frame_num)

        ret, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()

    def _draw_overlays(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Draw ONLY weapon bounding boxes on frame. Status info is tracked in the frontend panel."""
        with self._result_lock:
            r = self._result

        # Weapon bounding boxes only — everything else shown in right-side panel
        for det in r.weapon_detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{det['label']} {det['confidence']:.0%}"
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 9, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    # ── Processing Thread ─────────────────────────────────────────────────────

    def _process_loop(self):
        """
        Thread 2: Consumes frames from queue, runs AI inference.
        Weapon: every dequeued frame (~every 5th webcam frame)
        Violence: every 4th dequeued frame (~every 20th webcam frame, rate-limited to 1/sec)
        """
        process_count = 0

        while self._running:
            try:
                item = self._frame_queue.get(timeout=1.0)
                if item is None:
                    break

                frame, frame_num = item
                process_count += 1

                weapon_result = {"weapon_detected": False, "label": "", "confidence": 0.0, "detections": []}
                violence_result = {"violence_detected": False, "label": "Non Violence", "confidence": 0.0}

                # Weapon detection — every dequeued frame
                weapon_result = self.weapon_detector.detect(frame)

                # Violence detection — every 4th dequeued frame (= every 20th webcam frame)
                # Also rate-limited to 1/sec inside ViolenceDetector
                if process_count % 4 == 0:
                    violence_result = self.violence_detector.detect(frame)

                # Update shared result (lock held briefly)
                with self._result_lock:
                    self._result = DetectionResult(
                        weapon_detected=weapon_result["weapon_detected"],
                        weapon_label=weapon_result.get("label", ""),
                        weapon_confidence=weapon_result.get("confidence", 0.0),
                        weapon_detections=weapon_result.get("detections", []),
                        violence_detected=violence_result["violence_detected"],
                        violence_label=violence_result.get("label", "Non Violence"),
                        violence_confidence=violence_result.get("confidence", 0.0),
                        timestamp=time.time(),
                        frame_count=frame_num,
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ProcessThread] Error: {e}")
