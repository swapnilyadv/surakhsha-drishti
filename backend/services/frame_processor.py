"""
Frame Processor — Central Upgraded AI Surveillance Pipeline
===========================================================
Orchestrates YOLO, MediaPipe, and ViT inference with temporal sequence trackers,
dynamic threat scoring, adaptive skipping, and smart coordinate mapping.

Threading Architecture:
  - Thread 1 (webcam-capture): Pulls frames, runs frame-difference motion estimators.
  - Thread 2 (ai-process): Async CPU queue reader running temporal and threat engines.
  - Event triggers feed direct REST/WebSocket outputs with zero blocking.
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
from services.pose_estimator  import PoseEstimator
from services.violence_detector import ViolenceDetector
from services.threat_engine import ThreatEngine
from services.event_manager import EventManager

logger = logging.getLogger("suraksha.frame_processor")

# ── Dynamic skipping bounds ───────────────────────────────────────────────────
WEBCAM_FPS_TARGET  = 20          # target camera speed
MIN_SKIP_FRAMES    = 4           # min skip rate
MAX_SKIP_FRAMES    = 10          # max skip rate to save CPU under heavy loads
JPEG_QUALITY       = 65
MAX_QUEUE_SIZE     = 3
FPS_WINDOW         = 30

@dataclass
class DetectionResult:
    """Latest AI states read by FastAPI WebSocket/REST pipelines."""
    person_count:  int  = 0
    persons:       list = field(default_factory=list)

    # Violence & Action
    violence_detected:    bool  = False
    violence_label:       str   = "Non Violence"
    violence_confidence:  float = 0.0
    violence_smoothed:    float = 0.0
    active_threshold:     float = 0.78
    action:               str   = "Normal"
    aggression_score:     float = 0.0

    # Weapons
    weapon_detected:    bool  = False
    weapon_label:       str   = ""
    weapon_confidence:  float = 0.0
    weapon_detections:  list  = field(default_factory=list)

    # Threat Score & Events
    threat_level:       str   = "LOW"
    threat_score:       float = 0.0
    fall_detected:      bool  = False
    repeated_strikes:   bool  = False
    chasing_detected:   bool  = False
    motion_intensity:   float = 0.0
    
    # Interaction Aware Fields
    punch_detected:     bool  = False
    attacker_bbox:      list  = field(default_factory=list)
    victim_bbox:        list  = field(default_factory=list)
    punch_arrow:        list  = field(default_factory=list)

    # Gender metrics
    male_count:   int = 0
    female_count: int = 0

    fps:         float = 0.0
    timestamp:   float = field(default_factory=time.time)
    frame_count: int   = 0


class FrameProcessor:
    """
    Central Upgraded Pipeline Manager.
    
    Coordinates inference workers, performance skip modules, and threat score calculations.
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

        # Stored metadata (Task 3)
        self.latitude: float = 19.0760
        self.longitude: float = 72.8777
        self.location_name: str = "Webcam Unit"

        # Modular Subsystems
        self.threat_engine = ThreatEngine()
        self.event_manager = EventManager()

        # Shared state
        self._result       = DetectionResult()
        self._result_lock  = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock    = threading.Lock()

        # Input Queue
        self._frame_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        # Performance tuning & thread states
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._frame_times: list[float] = []

        # Adaptive skipping tracker
        self.current_skip_rate = 5
        self.avg_inference_duration = 0.04  # Initial estimate (40ms)

        # Gender smooth buffers
        self._male_history = []
        self._female_history = []

    def start(self):
        """Webcam capture and AI thread initialization."""
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
        logger.info("[FrameProcessor] Upgraded engine threads launched successfully.")

    def stop(self):
        """Gracefully stop thread workers."""
        self._running = False
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._process_thread:
            self._process_thread.join(timeout=2)

        if self.pose_estimator.loaded:
            self.pose_estimator.close()

        # Upload cleanup guard: clean up temporary file on deactivation
        if isinstance(self.camera_index, str) and "temp_upload" in self.camera_index:
            try:
                p = Path(self.camera_index)
                if p.exists():
                    p.unlink()
                    logger.info(f"[FrameProcessor] Cleaned up temporary upload video on stop: {p.name}")
            except Exception as e:
                logger.error(f"[FrameProcessor] Failed to clean up temp file: {e}")

        logger.info("[FrameProcessor] Upgraded engine deactivated safely.")

    def change_source(self, new_source):
        """Changes video source index and resets worker queues."""
        logger.info(f"[FrameProcessor] Relaying capture stream -> {new_source}")
        self._running = False
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
            
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._process_thread:
            self._process_thread.join(timeout=2)
            
        # Upload cleanup guard: clean up temporary file before switching source
        if isinstance(self.camera_index, str) and "temp_upload" in self.camera_index:
            try:
                p = Path(self.camera_index)
                if p.exists():
                    p.unlink()
                    logger.info(f"[FrameProcessor] Cleaned up temporary upload video on switch: {p.name}")
            except Exception as e:
                logger.error(f"[FrameProcessor] Failed to clean up temp file: {e}")

        self.camera_index = new_source
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._frame_times = []
        self._running = True
        
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture-thread"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="ai-process"
        )
        self._capture_thread.start()
        self._process_thread.start()
        logger.info("[FrameProcessor] Upgraded streams re-attached successfully.")

    def get_detection_result(self) -> dict:
        """JSON compatible real-time coordinate dictionary containing active events."""
        with self._result_lock:
            r = self._result
            persons_json = [
                {
                    "label":            "Person",
                    "bbox":             p.get("bbox", []),
                    "aggression_score": p.get("aggression_score", 0.0),
                    "pose_flags":       p.get("pose_flags", []),
                    "keypoints":        [],
                    "confidence":       p.get("confidence", 0.0),
                }
                for p in r.persons
            ]
            user_fps = int(round(r.fps)) if r.fps > 0 else 30

            return {
                # WebSocket compatible channels
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

                # Advanced Threat Engine metadata
                "action":             r.action,
                "aggression_score":   round(r.aggression_score, 3),
                "fps":                user_fps,
                "person_count":       r.person_count,
                "persons":            persons_json,
                "violence":           r.violence_detected,

                # Phase 2 & 5 indicators
                "threat_level":       r.threat_level,
                "threat_score":       round(r.threat_score, 3),
                "fall_detected":      r.fall_detected,
                "repeated_strikes":   r.repeated_strikes,
                "chasing_detected":   r.chasing_detected,
                "motion_intensity":   round(r.motion_intensity, 3),

                # Interaction Aware fields (Task 1, 2, 6)
                "punch_detected":     r.punch_detected,
                "attacker_bbox":      r.attacker_bbox,
                "victim_bbox":        r.victim_bbox,
                "punch_arrow":        r.punch_arrow,
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def update_gender_counts(self, male: int, female: int):
        with self._result_lock:
            self._male_history.append(male)
            self._female_history.append(female)
            if len(self._male_history) > 5:
                self._male_history.pop(0)
            if len(self._female_history) > 5:
                self._female_history.pop(0)
            self._result.male_count = int(round(sum(self._male_history) / len(self._male_history)))
            self._result.female_count = int(round(sum(self._female_history) / len(self._female_history)))

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"[Capture] Offline camera feed index {self.camera_index}")
            self._run_placeholder_loop()
            return

        is_video_file = isinstance(self.camera_index, str)
        fps_target = WEBCAM_FPS_TARGET

        if is_video_file:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps_target = video_fps
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          WEBCAM_FPS_TARGET)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        frame_interval = 1.0 / fps_target
        last_time      = time.time()
        frame_num      = 0

        logger.info(f"[Capture] Stream connected successfully (Target FPS: {fps_target:.2f})")

        while self._running:
            try:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("[Capture] Frame loss — repeating block")
                    time.sleep(0.05)
                    continue

                if frame is None or frame.size == 0:
                    logger.warning("[Capture] Empty/Null frame read.")
                    time.sleep(0.05)
                    continue

                frame_num += 1

                # Compute CPU differencing index fast
                motion_idx = self.violence_detector.temporal_classifier.calculate_frame_motion(frame)

                # Smooth encoded HUD HUD stream
                self._encode_jpeg(frame, frame_num)

                # Adaptive skip mechanism: Uses active processing time to compute skip thresholds
                skip_step = self.current_skip_rate
                if frame_num % skip_step == 0:
                    try:
                        self._frame_queue.put_nowait((frame.copy(), frame_num, motion_idx))
                    except queue.Full:
                        pass

                elapsed    = time.time() - last_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
            except Exception as e:
                logger.error(f"[Capture] Capture thread error: {e}", exc_info=True)
                time.sleep(0.05)

        cap.release()

    def _run_placeholder_loop(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(placeholder, (0, 0), (640, 480), (12, 12, 22), -1)
        cv2.putText(
            placeholder, "SURVEY UNIT OFFLINE",
            (110, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 100, 240), 2, cv2.LINE_AA
        )
        cv2.putText(
            placeholder, "Re-connecting to surveillance streams...",
            (120, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 85, 105), 1, cv2.LINE_AA
        )

        while self._running:
            ts_frame = placeholder.copy()
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                ts_frame, ts,
                (270, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 80), 1, cv2.LINE_AA
            )
            self._encode_jpeg(ts_frame, 0)
            time.sleep(0.5)

    def _process_loop(self):
        process_count = 0
        while self._running:
            try:
                item = self._frame_queue.get(timeout=1.0)
                if item is None:
                    break

                t_start = time.perf_counter()

                frame, frame_num, motion_idx = item
                process_count += 1

                # 1. Human Bboxes
                person_detections = self.human_detector.detect(frame)

                # 2. Pose estimation on key frames
                pose_persons = self.pose_estimator.detect(frame) if self.pose_estimator.loaded else []

                # 3. Weapon scan
                weapon_result = self.weapon_detector.detect(frame)

                # Merge coordinates first so that both keypoints and bounding boxes are associated (Task 5)
                merged_persons = self._merge_person_data(person_detections, pose_persons)

                # 4. Temporal violence classification (blended internally)
                violence_result = self.violence_detector.detect(frame, merged_persons)

                # 5. Extract specialized dynamic events from upgraded temporal engine
                temp_cls = self.violence_detector.temporal_classifier
                fall_active = temp_cls.get_fall_active()
                repeated_strikes = temp_cls.get_repeated_strikes_active()
                chasing_active = temp_cls.get_chasing_active()
                
                # Fetch multi-person interaction fields (Task 1, 2, 6)
                punch_detected = temp_cls.punch_detected
                attacker_bbox = temp_cls.attacker_bbox
                victim_bbox = temp_cls.victim_bbox
                punch_arrow = temp_cls.punch_arrow

                # Calculate combined FPS
                fps = self._update_fps()

                # 6. Dynamic Threat Score Calculation
                w_conf = weapon_result.get("confidence", 0.0) if weapon_result.get("weapon_detected", False) else 0.0
                threat_level, threat_score = self.threat_engine.calculate_threat(
                    aggression_score=violence_result.get("aggression_score", 0.0),
                    motion_intensity=motion_idx,
                    person_count=len(person_detections),
                    weapon_detected=weapon_result.get("weapon_detected", False),
                    weapon_confidence=w_conf,
                    repeated_strikes=repeated_strikes,
                    fall_detected=fall_active,
                    chasing_detected=chasing_active,
                    punch_detected=punch_detected,
                    recoil_active=temp_cls.recoil_active
                )

                # 7. Timeline logs coordination
                cam_label = self.location_name if self.location_name else "Webcam Unit"
                if not self.location_name:
                    if isinstance(self.camera_index, str):
                        cam_label = Path(self.camera_index).stem
                    elif self.camera_index == 0:
                        cam_label = "Live Webcam"

                if fall_active and not self._result.fall_detected:
                    self.event_manager.log_event(
                        f"CAM-{self.camera_index}", cam_label, "IMPACT_COLLAPSE", "HIGH", "Person fall detected after sudden motion spike."
                    )
                if repeated_strikes and not self._result.repeated_strikes:
                    self.event_manager.log_event(
                        f"CAM-{self.camera_index}", cam_label, "FIGHTING_STRIKES", "CRITICAL", "Repeated aggressive arm strikes counted."
                    )
                if chasing_active and not self._result.chasing_detected:
                    self.event_manager.log_event(
                        f"CAM-{self.camera_index}", cam_label, "AGGRESSIVE_CHASE", "HIGH", "Converging bounding velocities indicating a chase."
                    )

                # Save metrics safely
                with self._result_lock:
                    self._result.person_count       = len(person_detections)
                    self._result.persons            = merged_persons

                    self._result.violence_detected  = violence_result.get("violence_detected", False)
                    self._result.violence_label     = violence_result.get("label", "Non Violence")
                    self._result.violence_confidence = violence_result.get("confidence", 0.0)
                    self._result.violence_smoothed   = violence_result.get("smoothed_confidence", 0.0)
                    self._result.active_threshold    = violence_result.get("active_threshold", 0.78)
                    self._result.action             = violence_result.get("action", "Normal")
                    self._result.aggression_score   = violence_result.get("aggression_score", 0.0)

                    self._result.weapon_detected    = weapon_result.get("weapon_detected", False)
                    self._result.weapon_label       = weapon_result.get("label", "")
                    self._result.weapon_confidence  = weapon_result.get("confidence", 0.0)
                    self._result.weapon_detections  = weapon_result.get("detections", [])

                    self._result.threat_level       = threat_level
                    self._result.threat_score       = threat_score
                    self._result.fall_detected      = fall_active
                    self._result.repeated_strikes   = repeated_strikes
                    self._result.chasing_detected   = chasing_active
                    self._result.motion_intensity   = motion_idx
                    
                    # Interaction metrics (Task 1, 2, 6)
                    self._result.punch_detected     = punch_detected
                    self._result.attacker_bbox      = attacker_bbox
                    self._result.victim_bbox        = victim_bbox
                    self._result.punch_arrow        = punch_arrow

                    self._result.fps         = fps
                    self._result.timestamp   = time.time()
                    self._result.frame_count = frame_num

                # Trigger Smart Auto-recording with merged cooldowns
                if self.evidence_manager:
                    is_threat = self._result.violence_detected or self._result.weapon_detected or threat_level in ["HIGH", "CRITICAL"]
                    threat_label = f"[{threat_level}] {self._result.action}"
                    
                    self.evidence_manager.add_frame(
                        frame,
                        is_threat=is_threat,
                        threat_type=threat_label,
                        camera_id=f"CAM-{self.camera_index}",
                        camera_label=cam_label,
                        male_count=self._result.male_count,
                        female_count=self._result.female_count,
                        weapon_detected=self._result.weapon_detected,
                        weapon_type=self._result.weapon_label,
                        confidence=max(self._result.violence_confidence, self._result.weapon_confidence, self._result.threat_score),
                        lat=self.latitude,
                        lng=self.longitude
                    )

                if self.alert_manager:
                    is_threat = self._result.violence_detected or self._result.weapon_detected
                    self.alert_manager.update_threat_state(
                        violence_detected=is_threat,
                        threat_type=f"[{threat_level}] {self._result.action}",
                        confidence=self._result.threat_score
                    )

                # Performance measure
                t_duration = time.perf_counter() - t_start
                self.avg_inference_duration = 0.90 * self.avg_inference_duration + 0.10 * t_duration

                # Adaptive Skipped frame coordinator based on execution timing (Mac CPU safeguard)
                # Task 5: For uploaded videos, enforce smoother skip_rate = 2 and bypass adaptive skipping
                if isinstance(self.camera_index, str) and ("temp_upload" in self.camera_index or self.camera_index.endswith((".mp4", ".mov", ".avi", ".mkv"))):
                    self.current_skip_rate = 2
                else:
                    if self.avg_inference_duration > 0.120:  # > 120ms: heavy CPU load -> skip more
                        self.current_skip_rate = min(MAX_SKIP_FRAMES, self.current_skip_rate + 1)
                    elif self.avg_inference_duration < 0.060: # < 60ms: CPU free -> skip less
                        self.current_skip_rate = max(MIN_SKIP_FRAMES, self.current_skip_rate - 1)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ProcessThread] AI pipeline failure: {e}", exc_info=True)

    def _encode_jpeg(self, frame: np.ndarray, frame_num: int):
        try:
            display = self._draw_overlays(frame.copy(), frame_num)
            ok, buf = cv2.imencode(
                ".jpg", display,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
        except Exception as e:
            logger.debug(f"[JPEG] Encode issue: {e}")

    def _draw_overlays(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        with self._result_lock:
            r = self._result

        h, w = frame.shape[:2]
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Draw bboxes
        for p in r.persons:
            bbox = p.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Check for Attacker / Victim override (Task 6)
                is_attacker = False
                is_victim = False
                if r.punch_detected:
                    # Match bounding boxes by proximity/overlap
                    if len(r.attacker_bbox) == 4:
                        ax1, ay1, ax2, ay2 = map(int, r.attacker_bbox)
                        if abs(x1 - ax1) < 15 and abs(y1 - ay1) < 15:
                            is_attacker = True
                    if len(r.victim_bbox) == 4:
                        vx1, vy1, vx2, vy2 = map(int, r.victim_bbox)
                        if abs(x1 - vx1) < 15 and abs(y1 - vy1) < 15:
                            is_victim = True

                if is_attacker:
                    colour = (0, 0, 255)  # RED in BGR
                    label = "ATTACKER"
                elif is_victim:
                    colour = (0, 255, 255)  # YELLOW in BGR
                    label = "VICTIM"
                else:
                    agg = p.get("aggression_score", 0.0)
                    col_r = int(min(255, agg * 2.5 * 255))
                    col_g = int(min(255, (1.0 - agg) * 2.0 * 255))
                    colour = (0, col_g, col_r)
                    label = "Person"

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
                lw = len(label) * 9
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + lw, y1), colour, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255) if (is_attacker or is_victim) else colour, 1, cv2.LINE_AA)

        # Draw attack direction arrow (Task 6)
        if r.punch_detected and len(r.punch_arrow) == 4:
            ax, ay, bx, by = r.punch_arrow
            # Draw thick red direction arrow
            cv2.arrowedLine(frame, (ax, ay), (bx, by), (0, 0, 255), 3, cv2.LINE_AA, 0, 0.15)
            
            # Center HUD Overlay banner: "PUNCH DETECTED"
            text = "PUNCH DETECTED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.80
            thickness = 2
            (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = (w - t_w) // 2
            ty = 60
            
            # Draw glassy red backdrop
            overlay = frame.copy()
            cv2.rectangle(overlay, (tx - 15, ty - t_h - 10), (tx + t_w + 15, ty + 10), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            
            # Neon-yellow border
            cv2.rectangle(frame, (tx - 15, ty - t_h - 10), (tx + t_w + 15, ty + 10), (0, 255, 255), 2, cv2.LINE_AA)
            
            # Shadow and front text
            cv2.putText(frame, text, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Pose skeletons
        if self.pose_estimator.loaded and r.persons:
            pose_only = [p for p in r.persons if p.get("keypoints")]
            if pose_only:
                self.pose_estimator.draw_skeleton(frame, pose_only, h, w)

        # Weapons
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

        # Red Alert Strip
        if r.violence_detected or r.weapon_detected:
            banner_colour = (0, 0, 200) if r.weapon_detected else (0, 60, 220)
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, 28), banner_colour, -1)
                alert_text = (
                    f"⚠ CRITICAL WEAPON: {r.weapon_label.upper()}" if r.weapon_detected
                    else f"⚠ Threat: {r.action.upper()} ({r.threat_level})"
                )
                cv2.putText(frame, alert_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom HUD
        source_label = "WEBCAM"
        if isinstance(self.camera_index, str):
            if "temp_upload" in self.camera_index:
                source_label = "UPLOAD VIDEO"
            else:
                source_label = f"CCTV: {Path(self.camera_index).stem.upper()}"
        elif self.camera_index != 0:
            source_label = f"CCTV CAM {self.camera_index}"

        hud_lines = [
            f"SOURCE: {source_label}",
            f"FPS: {r.fps:.1f}",
            f"Skip: every {self.current_skip_rate}f",
            f"Threat: {r.threat_level} ({r.threat_score:.0%})",
            f"{now_str}",
        ]
        hud_y_base = h - 10 - (len(hud_lines) - 1) * 18
        cv2.rectangle(frame, (0, hud_y_base - 14), (160, h), (0, 0, 0), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (5, hud_y_base + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 100), 1, cv2.LINE_AA)

        # Top-right display
        action_col = (0, 0, 255) if r.threat_level in ["HIGH", "CRITICAL"] else (0, 220, 100)
        action_text = f"{r.action.upper()} - {r.threat_level}"
        (tw, th), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, action_text, (w - tw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_col, 1, cv2.LINE_AA)

        # Real-time Production Debug HUD Overlay (Task 7 & 8)
        debug_lines = [
            f"CONF: {r.violence_confidence:.2f}",
            f"AVG: {r.violence_smoothed:.2f}",
            f"STATE: {'VIOLENCE' if r.violence_detected else 'VIOLENCE' if r.repeated_strikes else 'NORMAL'}",
            f"FPS: {r.fps:.0f}",
        ]
        hud_w = 140
        hud_h = len(debug_lines) * 16 + 10
        hud_x = w - hud_w - 10
        hud_y = 35

        # Glassmorphic transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Neon-green border
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 220, 100), 1, cv2.LINE_AA)

        # Render tactical telemetry text
        for idx, line in enumerate(debug_lines):
            cv2.putText(frame, line, (hud_x + 8, hud_y + 18 + idx * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 100), 1, cv2.LINE_AA)

        return frame

    def _merge_person_data(self, yolo_persons: list, pose_persons: list) -> list:
        merged = []
        for i, yp in enumerate(yolo_persons):
            entry = {
                "bbox":             yp.get("bbox", []),
                "confidence":       yp.get("confidence", 0.0),
                "aggression_score": 0.0,
                "pose_flags":       [],
                "keypoints":        [],
            }
            if i < len(pose_persons):
                pp = pose_persons[i]
                entry["aggression_score"] = pp.get("aggression_score", 0.0)
                entry["pose_flags"]       = pp.get("pose_flags", [])
                entry["keypoints"]        = pp.get("keypoints", [])
            merged.append(entry)

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
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) > FPS_WINDOW:
            self._frame_times = self._frame_times[-FPS_WINDOW:]
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed
