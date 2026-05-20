"""
Violence Classifier Service — Advanced Real-time Production Engine
===================================================================
Stateful temporal sequence analyzer featuring advanced hysteresis thresholding,
weighted confidence smoothing, momentum spike suppression, and false-positive guards.

Optimized for Edge CPU deployment (MacBook Air / Raspberry Pi).
"""

import os
import math
import time
import logging
import collections
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("suraksha.violence_classifier")

# ── Config Constants ──────────────────────────────────────────────────────────
ONNX_MODEL_PATH = "/Users/swapnil/Desktop/my project/surakhsha-drishti/backend/models/violence_model.onnx"
MAX_HISTORY_FRAMES = 30       
VELOCITY_WINDOW    = 3        
ACCEL_WINDOW       = 3        
VIOLENCE_BUFFER_SIZE = 12

# Target settings
FRAME_SEQUENCE = 16
ROLLING_AVERAGE = 3
TRIGGER_FRAMES = 2
COOLDOWN_FRAMES = 10

ACTIVATE_THRESHOLD = 0.78
DEACTIVATE_THRESHOLD = 0.55

# Kinematic thresholds
ARM_SWING_VELOCITY_THRESH = 0.16
COMBAT_STANCE_ANGLE       = 135.0
HIGH_KICK_HEIGHT_THRESH   = 0.10
RAISED_FIST_THRESH        = 0.04
FAST_PUNCH_VELOCITY       = 0.20
AGGRESSION_DECAY          = 0.96

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# ============================================================
# 1. ADVANCED THRESHOLD ENGINE & SMOOTHING UTILITIES
# ============================================================
class ThresholdManager:
    """Manages advanced hysteresis triggering, consecutive confirmations, and state cooldowns."""
    def __init__(self, activate_thresh=0.78, deactivate_thresh=0.55, trigger_frames=3, cooldown_frames=10):
        self.activate_thresh = activate_thresh
        self.deactivate_thresh = deactivate_thresh
        self.trigger_frames = trigger_frames
        self.cooldown_frames = cooldown_frames
        
        self.consecutive_positives = 0
        self.cooldown_left = 0
        self.is_active = False

    def update(self, raw_conf: float, smoothed_conf: float) -> bool:
        """Applies dynamic rules to transition alert states."""
        # 1. Check raw signal persistence
        if raw_conf >= self.activate_thresh:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = 0

        # 2. Hysteresis trigger rules
        if self.is_active:
            # Under alert state: respect minimal lock duration (cooldown protection)
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
            else:
                # Cooldown expired: check deactivation boundary
                if smoothed_conf < self.deactivate_thresh:
                    self.is_active = False
        else:
            # Under normal state: require concurrent smoothed & consecutive raw bounds
            if smoothed_conf >= self.activate_thresh and self.consecutive_positives >= self.trigger_frames:
                self.is_active = True
                self.cooldown_left = self.cooldown_frames

        return self.is_active

    def get_active_threshold(self) -> float:
        return self.deactivate_thresh if self.is_active else self.activate_thresh


class SmoothingUtility:
    """Performs sliding deques, weighted rolling averages, momentum tracking, and spike suppression."""
    def __init__(self, size=5, momentum=0.80):
        self.buffer = collections.deque(maxlen=size)
        self.momentum = momentum
        self.last_val = 0.0

    def add(self, val: float) -> float:
        """Pushes prediction value, computes weighted average, and dampens single-frame jumps."""
        self.buffer.append(val)
        n = len(self.buffer)
        if n == 0:
            return 0.0

        # Weighted Prediction Average (linear scaling giving latest frames more priority)
        weights = np.linspace(0.5, 1.0, n)
        weights /= weights.sum()
        weighted_avg = float(sum(self.buffer[i] * weights[i] for i in range(n)))

        # Confidence Momentum & Spike Suppression (prevents rapid glare/contrast jumps)
        if abs(weighted_avg - self.last_val) > 0.40:
            smoothed_val = self.momentum * self.last_val + (1.0 - self.momentum) * weighted_avg
        else:
            smoothed_val = 0.65 * self.last_val + 0.35 * weighted_avg

        self.last_val = float(smoothed_val)
        return self.last_val


# ============================================================
# 2. STATEFUL COORDINATOR & KEYPOINT TRACKERS
# ============================================================
class PersonTrajectory:
    """Stores a sliding history of tracked keypoints and detects temporal patterns."""

    def __init__(self, person_id: int):
        self.person_id = person_id
        self.history: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)
        self.timestamps: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)
        self.centroid_history: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)
        self.bbox_history: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)

        # Computed dynamics
        self.wrist_velocities: list[float] = [0.0, 0.0]  
        self.ankle_velocities: list[float] = [0.0, 0.0]  
        self.body_acceleration: float = 0.0
        self.is_falling: bool = False
        self.pose_flags: list[str] = []
        self.aggression_score: float = 0.0

        # Specialized temporal signals
        self.is_running: bool = False
        self.repeated_arm_strikes: bool = False
        self.wrist_velocity_history: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)

    def update(self, keypoints: list[list[float]], bbox: list[int] = None):
        """Add fresh keypoint / bounding box data and compute velocity physics."""
        self.history.append(keypoints)
        self.timestamps.append(time.time())
        if bbox:
            self.bbox_history.append(bbox)

        # Centroid calculation
        landmark_indices = [0, 11, 12, 23, 24]  
        valid_pts = []
        for idx in landmark_indices:
            if idx < len(keypoints):
                x, y, vis = keypoints[idx]
                if vis > 0.35:
                    valid_pts.append((x, y))

        if valid_pts:
            cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
            cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)
            self.centroid_history.append((cx, cy))
        elif bbox:
            cx = (bbox[0] + bbox[2]) / 2.0 / 640.0
            cy = (bbox[1] + bbox[3]) / 2.0 / 480.0
            self.centroid_history.append((cx, cy))
        else:
            self.centroid_history.append((0.5, 0.5))

        self._compute_dynamics()

    def _compute_dynamics(self):
        n = len(self.history)
        if n < 2:
            return

        self.wrist_velocities = [self._calc_joint_velocity(15), self._calc_joint_velocity(16)]
        self.ankle_velocities = [self._calc_joint_velocity(27), self._calc_joint_velocity(28)]
        self.body_acceleration = self._calc_body_acceleration()

        max_wrist_speed = max(self.wrist_velocities)
        self.wrist_velocity_history.append(max_wrist_speed)

        self.is_running = self._detect_suspicious_running()
        self.repeated_arm_strikes = self._detect_repeated_arm_strikes()
        self.is_falling = self._detect_fall()

        # Pose heuristics
        kps = self.history[-1]
        self.pose_flags = []
        raw_score = 0.0

        def get_kp(idx: int) -> Optional[tuple[float, float]]:
            if idx >= len(kps):
                return None
            x, y, vis = kps[idx]
            return (x, y) if vis > 0.35 else None

        l_shoulder = get_kp(11)
        r_shoulder = get_kp(12)
        l_elbow    = get_kp(13)
        r_elbow    = get_kp(14)
        l_wrist    = get_kp(15)
        r_wrist    = get_kp(16)
        l_hip      = get_kp(23)
        r_hip      = get_kp(24)
        l_ankle    = get_kp(27)
        r_ankle    = get_kp(28)

        # Raised fists
        raised_fist = False
        if l_wrist and l_shoulder and l_wrist[1] < l_shoulder[1] - RAISED_FIST_THRESH:
            raised_fist = True
        if r_wrist and r_shoulder and r_wrist[1] < r_shoulder[1] - RAISED_FIST_THRESH:
            raised_fist = True
        if raised_fist:
            self.pose_flags.append("raised_fist")
            raw_score += 0.35

        # Kicking
        kicking = False
        if l_ankle and l_hip and l_ankle[1] < l_hip[1] - HIGH_KICK_HEIGHT_THRESH:
            kicking = True
        if r_ankle and r_hip and r_ankle[1] < r_hip[1] - HIGH_KICK_HEIGHT_THRESH:
            kicking = True
        if kicking:
            self.pose_flags.append("high_kick")
            raw_score += 0.45

        # Punching
        punching = False
        if self.wrist_velocities[0] > FAST_PUNCH_VELOCITY and l_wrist and l_shoulder and abs(l_wrist[0] - l_shoulder[0]) > 0.22:
            punching = True
        if self.wrist_velocities[1] > FAST_PUNCH_VELOCITY and r_wrist and r_shoulder and abs(r_wrist[0] - r_shoulder[0]) > 0.22:
            punching = True
        if punching:
            self.pose_flags.append("punching_motion")
            raw_score += 0.50

        # Stance
        if l_shoulder and l_elbow and l_wrist and r_shoulder and r_elbow and r_wrist:
            l_angle = self._calc_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = self._calc_angle(r_shoulder, r_elbow, r_wrist)
            if l_angle < COMBAT_STANCE_ANGLE or r_angle < COMBAT_STANCE_ANGLE:
                self.pose_flags.append("combat_stance")
                raw_score += 0.15

        if self.repeated_arm_strikes:
            self.pose_flags.append("repeated_strikes")
            raw_score += 0.30
        if self.is_running:
            self.pose_flags.append("suspicious_running")
            raw_score += 0.10
        if self.is_falling:
            self.pose_flags.append("fall_detected")
            raw_score += 0.35

        # Check active motion requirement (Task 2 & 3)
        has_active_motion = (
            max_wrist_speed > 0.022 or 
            self.body_acceleration > 0.45 or 
            self.repeated_arm_strikes or
            punching or
            kicking
        )
        
        combat_pose_detected = ("combat_stance" in self.pose_flags or "raised_fist" in self.pose_flags)
        motion_low = not has_active_motion
        
        if combat_pose_detected and motion_low:
            # Suppress pose score heavily (Task 3)
            raw_score = 0.0
            if "raised_fist" in self.pose_flags:
                self.pose_flags.remove("raised_fist")
            if "combat_stance" in self.pose_flags:
                self.pose_flags.remove("combat_stance")

        # EMA decay blend or rapid post-action decay (Task 1)
        if not has_active_motion and raw_score == 0.0:
            # Rapid decay (decay rate of 0.65 instead of AGGRESSION_DECAY = 0.92)
            self.aggression_score = round(self.aggression_score * 0.65, 3)
        else:
            self.aggression_score = round(self.aggression_score * AGGRESSION_DECAY + raw_score * (1.0 - AGGRESSION_DECAY), 3)

    def _calc_joint_velocity(self, kp_idx: int) -> float:
        n = len(self.history)
        if n < VELOCITY_WINDOW:
            return 0.0
        curr_kp = self.history[-1]
        prev_kp = self.history[-VELOCITY_WINDOW]
        if kp_idx >= len(curr_kp) or kp_idx >= len(prev_kp):
            return 0.0
        c_pt, p_pt = curr_kp[kp_idx], prev_kp[kp_idx]
        if c_pt[2] < 0.35 or p_pt[2] < 0.35:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-VELOCITY_WINDOW]
        if dt <= 0:
            return 0.0
        return math.sqrt((c_pt[0] - p_pt[0])**2 + (c_pt[1] - p_pt[1])**2) / dt

    def _calc_body_acceleration(self) -> float:
        n = len(self.centroid_history)
        if n < ACCEL_WINDOW * 2:
            return 0.0
        c_end   = self.centroid_history[-1]
        c_mid   = self.centroid_history[-ACCEL_WINDOW]
        c_start = self.centroid_history[-ACCEL_WINDOW * 2]
        t_end   = self.timestamps[-1]
        t_mid   = self.timestamps[-ACCEL_WINDOW]
        t_start = self.timestamps[-ACCEL_WINDOW * 2]
        dt1 = t_end - t_mid
        dt2 = t_mid - t_start
        if dt1 <= 0 or dt2 <= 0:
            return 0.0
        v1 = math.sqrt((c_end[0] - c_mid[0])**2 + (c_end[1] - c_mid[1])**2) / dt1
        v2 = math.sqrt((c_mid[0] - c_start[0])**2 + (c_mid[1] - c_start[1])**2) / dt2
        return abs(v1 - v2) / (t_end - t_start)

    def _detect_suspicious_running(self) -> bool:
        n = len(self.centroid_history)
        if n < 5:
            return False
        cx_curr, cy_curr = self.centroid_history[-1]
        cx_prev, cy_prev = self.centroid_history[-5]
        dt = self.timestamps[-1] - self.timestamps[-5]
        if dt <= 0:
            return False
        speed = math.sqrt((cx_curr - cx_prev)**2 + (cy_curr - cy_prev)**2) / dt
        return speed > 0.28

    def _detect_repeated_arm_strikes(self) -> bool:
        n = len(self.wrist_velocity_history)
        if n < 12:
            return False
        speeds = list(self.wrist_velocity_history)
        peaks = 0
        for i in range(1, len(speeds) - 1):
            if speeds[i] > FAST_PUNCH_VELOCITY * 0.8:
                if speeds[i] > speeds[i-1] and speeds[i] > speeds[i+1]:
                    peaks += 1
        return peaks >= 3

    def _detect_fall(self) -> bool:
        n = len(self.centroid_history)
        if n < 6:
            return False
        cy_curr = self.centroid_history[-1][1]
        cy_prev = self.centroid_history[-6][1]
        dt = self.timestamps[-1] - self.timestamps[-6]
        if dt <= 0:
            return False
        y_velocity = (cy_curr - cy_prev) / dt
        return y_velocity > 0.25 and self.body_acceleration > 1.2

    @staticmethod
    def _calc_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


class ViolenceClassifier:
    """Production Spatial-Temporal classification engine loading unified LSTM ONNX models."""

    def __init__(self):
        self.trajectories: dict[int, PersonTrajectory] = {}
        self.violence_score_history: collections.deque = collections.deque(maxlen=VIOLENCE_BUFFER_SIZE)
        self.motion_intensity_history: collections.deque = collections.deque(maxlen=MAX_HISTORY_FRAMES)
        self.fast_violence_mode = False
        self.no_new_aggression_frames = 0
        self.recent_violence_counter = 0
        
        # ONNX variables
        self.onnx_session = None
        self.onnx_loaded = False
        self.frame_history = collections.deque(maxlen=FRAME_SEQUENCE)

        # Advanced Modules
        self.threshold_manager = ThresholdManager(
            activate_thresh=ACTIVATE_THRESHOLD,
            deactivate_thresh=DEACTIVATE_THRESHOLD,
            trigger_frames=TRIGGER_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES
        )
        self.smoothing_util = SmoothingUtility(size=ROLLING_AVERAGE, momentum=0.65)

        # Multi-Person Interaction & Impact telemetry
        self.aggression_boost = 0.0
        self.punch_detected = False
        self.attacker_id = None
        self.victim_id = None
        self.attacker_bbox = []
        self.victim_bbox = []
        self.punch_arrow = []
        self.recoil_active = False
        self.last_temporal_confidence = 0.0

        self._prev_gray = None
        self.load_onnx()

    def load_onnx(self):
        """Pre-warms the CPU-optimized ONNX model."""
        if os.path.exists(ONNX_MODEL_PATH):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 2
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
                self.onnx_loaded = True
                logger.info("[ViolenceClassifier] Production ONNX LSTM loaded successfully!")
            except Exception as e:
                logger.error(f"[ViolenceClassifier] Failed to load ONNX model: {e}")

    def predict_frame_sequence(self, frame: np.ndarray) -> tuple[float, float, float, bool]:
        """
        Runs LSTM ONNX sequence prediction with double prediction smoothing and false positive guards.
        
        Returns:
            (raw_confidence, smoothed_confidence, active_threshold, is_violent)
        """
        if not self.onnx_loaded or self.onnx_session is None:
            return 0.0, 0.0, ACTIVATE_THRESHOLD, False

        try:
            # Preprocess BGR frame to 160x160 RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (160, 160))
            
            # Normalize frame parameters
            img_normalized = small.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_normalized - mean) / std
            
            # Transpose to (C, H, W)
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            self.frame_history.append(img_transposed)

            # Pad sliding history window with copies if not filled yet
            while len(self.frame_history) < FRAME_SEQUENCE:
                self.frame_history.append(img_transposed)

            # Sequentialize history sequence: Shape (1, 16, 3, 160, 160)
            sequence_data = np.stack(list(self.frame_history), axis=0)
            sequence_data = np.expand_dims(sequence_data, axis=0)

            # Run ONNX inference
            outputs = self.onnx_session.run(["logits"], {"input_sequence": sequence_data})
            logits = outputs[0][0]
            probs = softmax(logits)
            
            # Index 1 = Positive (Harassment/Violence)
            raw_prob = float(probs[1])

            # Blend active multi-person interaction aggression boost (Task 3)
            raw_prob = min(1.0, raw_prob + self.aggression_boost)

            # ── TASK 3 — FALSE POSITIVE REDUCTION ──
            motion_intensity = self.calculate_frame_motion(frame)
            
            # Rule 1: Motion Consistency Guard (punish stationary glare and background camera casts)
            if motion_intensity < 0.08:
                # Extreme spike suppression during zero/very low movement
                raw_prob *= 0.10
            elif motion_intensity < 0.15:
                # Moderate damping during normal stretching or waving
                raw_prob *= 0.50

            # ── TASK 4 — THREAT CONFIDENCE STABILITY ──
            smoothed_prob = self.smoothing_util.add(raw_prob)
            self.last_temporal_confidence = raw_prob

            # ── TASK 1 & 5 — HYSTERESIS & DEACTIVATE DECAY ──
            is_violent = self.threshold_manager.update(raw_prob, smoothed_prob)
            active_thresh = self.threshold_manager.get_active_threshold()

            return raw_prob, smoothed_prob, active_thresh, is_violent

        except Exception as e:
            logger.error(f"[ViolenceClassifier] ONNX sequence predict failed: {e}")
            return 0.0, 0.0, ACTIVATE_THRESHOLD, False

    def process_pose_data(self, persons: list[dict], original_h: int, original_w: int) -> tuple[str, float]:
        """Track coordinates, assess proximity, compute dynamic states, and output action labels."""
        self._track_and_update_trajectories(persons)
        
        # 1. Decay the aggression boost over frames (Task 3)
        self.aggression_boost *= 0.92
        self.punch_detected = False
        self.recoil_active = False
        self.attacker_id = None
        self.victim_id = None
        self.attacker_bbox = []
        self.victim_bbox = []
        self.punch_arrow = []

        if not self.trajectories:
            self.violence_score_history.append(0.0)
            return "Normal", 0.0

        max_aggression = 0.0
        active_flags = set()
        for traj in self.trajectories.values():
            if traj.aggression_score > max_aggression:
                max_aggression = traj.aggression_score
            active_flags.update(traj.pose_flags)

        t_keys = list(self.trajectories.keys())
        is_interactive = len(t_keys) >= 2
        proximity_dist = 1.0
        aggressive_convergence = False
        chaotic_limb_overlap = False
        grappling_detected = False

        if is_interactive and len(t_keys) >= 2:
            ax, ay = self.trajectories[t_keys[0]].centroid_history[-1] if self.trajectories[t_keys[0]].centroid_history else (0.5, 0.5)
            bx, by = self.trajectories[t_keys[1]].centroid_history[-1] if self.trajectories[t_keys[1]].centroid_history else (0.5, 0.5)
            proximity_dist = math.sqrt((ax - bx)**2 + (ay - by)**2)

            # TASK 2/3: Body collision persistence / Grappling / Overlap detection
            t_a = self.trajectories[t_keys[0]]
            t_b = self.trajectories[t_keys[1]]
            if t_a.bbox_history and t_b.bbox_history:
                bbox_a = t_a.bbox_history[-1]
                bbox_b = t_b.bbox_history[-1]
                if len(bbox_a) == 4 and len(bbox_b) == 4:
                    # Calculate intersection box
                    ix1 = max(bbox_a[0], bbox_b[0])
                    iy1 = max(bbox_a[1], bbox_b[1])
                    ix2 = min(bbox_a[2], bbox_b[2])
                    iy2 = min(bbox_a[3], bbox_b[3])
                    
                    if ix2 > ix1 and iy2 > iy1:
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
                        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
                        min_area = min(area_a, area_b)
                        if min_area > 0 and (overlap_area / min_area) > 0.35:
                            chaotic_limb_overlap = True
                            if proximity_dist < 0.16:
                                grappling_detected = True

            # Centroid convergence (approaching fast)
            if len(t_a.centroid_history) >= 2 and len(t_b.centroid_history) >= 2:
                acx_now, acy_now = t_a.centroid_history[-1]
                acx_prev, acy_prev = t_a.centroid_history[-2]
                bcx_now, bcy_now = t_b.centroid_history[-1]
                bcx_prev, bcy_prev = t_b.centroid_history[-2]
                
                prev_dist = math.sqrt((acx_prev - bcx_prev)**2 + (acy_prev - bcy_prev)**2)
                now_dist = math.sqrt((acx_now - bcx_now)**2 + (acy_now - bcy_now)**2)
                
                # If they are converging fast
                if prev_dist - now_dist > 0.015:
                    aggressive_convergence = True

        # 2. Advanced Multi-Directional Impact Solver (Task 2, 3, 5)
        if len(t_keys) >= 2:
            for i in range(len(t_keys)):
                for j in range(len(t_keys)):
                    if i == j:
                        continue
                    t_a = self.trajectories[t_keys[i]]  # Attacker candidate
                    t_b = self.trajectories[t_keys[j]]  # Victim candidate

                    if len(t_a.history) < 2 or not t_b.bbox_history:
                        continue

                    # Keypoint extraction helper
                    def get_absolute_kp(keypoints, idx):
                        if idx >= len(keypoints):
                            return None
                        x, y, vis = keypoints[idx]
                        if vis < 0.30:  # More tolerant vis threshold for robust overlap
                            return None
                        return (int(x * original_w), int(y * original_h))

                    v_bbox = t_b.bbox_history[-1]
                    if len(v_bbox) != 4:
                        continue
                    vx1, vy1, vx2, vy2 = v_bbox
                    # Pad boundaries slightly (10%) to catch close-range punches/strikes (Task 3)
                    pad_x = int(0.10 * (vx2 - vx1))
                    pad_y = int(0.10 * (vy2 - vy1))
                    vx1_pad, vx2_pad = vx1 - pad_x, vx2 + pad_x
                    vy1_pad, vy2_pad = vy1 - pad_y, vy2 + pad_y

                    # Attacker wrist/elbow joints for direction-independent punch & elbow strikes
                    kps_now = t_a.history[-1]
                    kps_prev = t_a.history[-2]

                    # Track wrists (15, 16) and elbows (13, 14) for direction-independent assaults
                    for kp_idx, joint_name in [(15, "left_wrist"), (16, "right_wrist"), (13, "left_elbow"), (14, "right_elbow")]:
                        pt_now = get_absolute_kp(kps_now, kp_idx)
                        pt_prev = get_absolute_kp(kps_prev, kp_idx)

                        if pt_now and pt_prev:
                            wx, wy = pt_now
                            # Check collision with padded victim box (covers side, upward, downward punches & pushes)
                            if vx1_pad <= wx <= vx2_pad and vy1_pad <= wy <= vy2_pad:
                                w_curr_norm = kps_now[kp_idx]
                                w_prev_norm = kps_prev[kp_idx]
                                # Euclidean velocity magnitude (Direction-Independent)
                                vel_norm = math.sqrt(
                                    (w_curr_norm[0] - w_prev_norm[0])**2 +
                                    (w_curr_norm[1] - w_prev_norm[1])**2
                                )
                                
                                # Burst velocity threshold
                                if vel_norm > 0.038:
                                    self.punch_detected = True
                                    self.attacker_id = t_a.person_id
                                    self.victim_id = t_b.person_id
                                    self.attacker_bbox = t_a.bbox_history[-1] if t_a.bbox_history else []
                                    self.victim_bbox = t_b.bbox_history[-1] if t_b.bbox_history else []
                                    
                                    if t_a.centroid_history and t_b.centroid_history:
                                        acx, acy = t_a.centroid_history[-1]
                                        bcx, bcy = t_b.centroid_history[-1]
                                        self.punch_arrow = [
                                            int(acx * original_w), int(acy * original_h),
                                            int(bcx * original_w), int(bcy * original_h)
                                        ]

                                    # Boost 45% (Task 3 & 6 rebalancing)
                                    self.aggression_boost = min(1.0, self.aggression_boost + 0.45)

                                    # TASK 2: Recoil confirmation
                                    if len(t_b.centroid_history) >= 3:
                                        bcx_now, bcy_now = t_b.centroid_history[-1]
                                        bcx_prev, bcy_prev = t_b.centroid_history[-3]
                                        
                                        dcx = bcx_now - bcx_prev
                                        dcy = bcy_now - bcy_prev
                                        recoil_dist = math.sqrt(dcx**2 + dcy**2)

                                        # Vector direction alignment check
                                        px_vec = bcx_now - acx
                                        py_vec = bcy_now - acy
                                        p_mag = math.sqrt(px_vec**2 + py_vec**2)

                                        if recoil_dist > 0.012 and p_mag > 0:
                                            dot = (dcx * px_vec + dcy * py_vec) / (recoil_dist * p_mag)
                                            if dot > 0.2:  # Moving away in line of punch direction
                                                self.recoil_active = True
                                                # Boost 25% (Task 3)
                                                self.aggression_boost = min(1.0, self.aggression_boost + 0.25)
                                    break

        # Repeated strikes active check (Task 3 & 6 rebalancing)
        repeated_strikes = any(t.repeated_arm_strikes for t in self.trajectories.values())
        if repeated_strikes:
            self.aggression_boost = min(1.0, self.aggression_boost + 0.35)

        # ── TASK 2 — ADD CHAOTIC MOTION AGGRESSION Heuristics ──
        chaotic_movement_detected = False
        repeated_directional_changes = False
        violent_bbox_jitter = False
        arm_velocity_bursts = False
        rapid_centroid_acceleration = False

        if len(t_keys) >= 2:
            t_a = self.trajectories[t_keys[0]]
            t_b = self.trajectories[t_keys[1]]

            # Rapid centroid acceleration
            if t_a.body_acceleration > 1.0 or t_b.body_acceleration > 1.0:
                rapid_centroid_acceleration = True

            # Violent bounding box jitter (sudden aggressive size or location shaking)
            if len(t_a.bbox_history) >= 3:
                b1, b2, b3 = list(t_a.bbox_history)[-3], list(t_a.bbox_history)[-2], list(t_a.bbox_history)[-1]
                w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
                w2, h2 = b2[2]-b2[0], b2[3]-b2[1]
                w3, h3 = b3[2]-b3[0], b3[3]-b3[1]
                jitter = abs(w3 - w2) + abs(w2 - w1) + abs(h3 - h2) + abs(h2 - h1)
                if jitter > 22.0:
                    violent_bbox_jitter = True

            # Repeated directional changes in motion (aggressive flailing/struggling/shaking)
            if len(t_a.centroid_history) >= 4:
                cents = list(t_a.centroid_history)[-4:]
                diffs = [(cents[i][0] - cents[i-1][0], cents[i][1] - cents[i-1][1]) for i in range(1, 4)]
                x_dirs = [d[0] > 0 for d in diffs]
                y_dirs = [d[1] > 0 for d in diffs]
                if (x_dirs[0] != x_dirs[1] and x_dirs[1] != x_dirs[2]) or (y_dirs[0] != y_dirs[1] and y_dirs[1] != y_dirs[2]):
                    repeated_directional_changes = True

            # Repeated arm velocity bursts
            max_wrist_vel = max(max(t_a.wrist_velocities), max(t_b.wrist_velocities))
            if max_wrist_vel > 0.035:
                arm_velocity_bursts = True

            # Overlapping skeleton motion and high joint velocity sum
            total_joint_vel = sum(t_a.wrist_velocities) + sum(t_b.wrist_velocities)
            if total_joint_vel > 0.08:
                chaotic_movement_detected = True

        elif len(t_keys) == 1:
            # Merged body tracking optimization
            t_single = self.trajectories[t_keys[0]]
            if t_single.body_acceleration > 1.0:
                rapid_centroid_acceleration = True
            
            if len(t_single.bbox_history) >= 3:
                b1, b2, b3 = list(t_single.bbox_history)[-3], list(t_single.bbox_history)[-2], list(t_single.bbox_history)[-1]
                w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
                w2, h2 = b2[2]-b2[0], b2[3]-b2[1]
                w3, h3 = b3[2]-b3[0], b3[3]-b3[1]
                jitter = abs(w3 - w2) + abs(w2 - w1) + abs(h3 - h2) + abs(h2 - h1)
                if jitter > 22.0:
                    violent_bbox_jitter = True

            if len(t_single.centroid_history) >= 4:
                cents = list(t_single.centroid_history)[-4:]
                diffs = [(cents[i][0] - cents[i-1][0], cents[i][1] - cents[i-1][1]) for i in range(1, 4)]
                x_dirs = [d[0] > 0 for d in diffs]
                y_dirs = [d[1] > 0 for d in diffs]
                if (x_dirs[0] != x_dirs[1] and x_dirs[1] != x_dirs[2]) or (y_dirs[0] != y_dirs[1] and y_dirs[1] != y_dirs[2]):
                    repeated_directional_changes = True

            max_wrist_vel = max(t_single.wrist_velocities)
            if max_wrist_vel > 0.035:
                arm_velocity_bursts = True

            total_joint_vel = sum(t_single.wrist_velocities)
            if total_joint_vel > 0.08:
                chaotic_movement_detected = True

        # Apply chaotic motion boosts to internal stats
        if chaotic_movement_detected:
            self.aggression_boost = min(1.0, self.aggression_boost + 0.20)

        # ============================================================
        # TASKS 6 & 9: LIGHTWEIGHT CAMERA SHAKE FILTER (Centroid Vector Analysis)
        # ============================================================
        camera_motion_detected = False
        
        # If we have 2 or more people, check if their centroids are displaced in the same direction and magnitude
        if len(t_keys) >= 2:
            displacements = []
            for tk in t_keys:
                traj = self.trajectories[tk]
                if len(traj.centroid_history) >= 2:
                    c1 = traj.centroid_history[-2]
                    c2 = traj.centroid_history[-1]
                    dx = c2[0] - c1[0]
                    dy = c2[1] - c1[1]
                    displacements.append((dx, dy))
            
            if len(displacements) >= 2:
                # Check if all vectors are similar in direction and magnitude
                all_similar = True
                for i in range(len(displacements)):
                    for j in range(i + 1, len(displacements)):
                        dx1, dy1 = displacements[i]
                        dx2, dy2 = displacements[j]
                        mag1 = math.sqrt(dx1**2 + dy1**2)
                        mag2 = math.sqrt(dx2**2 + dy2**2)
                        
                        # We only evaluate camera shake if there's actual motion (magnitude > 0.005)
                        if mag1 > 0.005 and mag2 > 0.005:
                            dot_product = dx1 * dx2 + dy1 * dy2
                            cosine_sim = dot_product / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0
                            
                            # Same direction (cosine similarity > 0.85) and similar magnitude (ratio between 0.5 and 2.0)
                            mag_ratio = mag1 / mag2 if mag2 > 0 else 0.0
                            if cosine_sim < 0.85 or mag_ratio < 0.5 or mag_ratio > 2.0:
                                all_similar = False
                                break
                        else:
                            all_similar = False
                            break
                    if not all_similar:
                        break
                
                if all_similar:
                    camera_motion_detected = True

        # Even for 1 person, check if the single person centroid displacement matches global camera motion
        elif len(t_keys) == 1:
            traj = self.trajectories[t_keys[0]]
            if len(traj.centroid_history) >= 2:
                c1 = traj.centroid_history[-2]
                c2 = traj.centroid_history[-1]
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                mag = math.sqrt(dx**2 + dy**2)
                
                # Check if frame motion differencing is high and matches single centroid shift
                frame_motion = self.motion_intensity_history[-1] if self.motion_intensity_history else 0.0
                if mag > 0.015 and frame_motion > 0.40:
                    # Centroid motion is fully accounted for by global camera movement
                    camera_motion_detected = True

        # ============================================================
        # TASK 1: CALM INTERACTION FILTER (Hugging, Handshakes, Calm contact)
        # ============================================================
        self.calm_interaction_mode = False
        
        smooth_motion = False
        low_arm_velocity = True
        low_jitter = True
        synchronized_centroids = True
        
        # Require dynamic aggression signal to enable close_combat_mode (Task 5)
        has_aggression_signal = (
            repeated_strikes or
            arm_velocity_bursts or
            chaotic_movement_detected or
            repeated_directional_changes or
            violent_bbox_jitter or
            rapid_centroid_acceleration or
            self.punch_detected or
            self.recoil_active
        )

        if len(t_keys) >= 2:
            t_a = self.trajectories[t_keys[0]]
            t_b = self.trajectories[t_keys[1]]
            
            # Smooth movement: low centroid acceleration
            if t_a.body_acceleration < 0.45 and t_b.body_acceleration < 0.45:
                smooth_motion = True
                
            # Low arm velocity
            max_wrist_vel_a = max(t_a.wrist_velocities) if t_a.wrist_velocities else 0.0
            max_wrist_vel_b = max(t_b.wrist_velocities) if t_b.wrist_velocities else 0.0
            if max_wrist_vel_a > 0.022 or max_wrist_vel_b > 0.022:
                low_arm_velocity = False
                
            # Low bbox jitter
            if len(t_a.bbox_history) >= 2 and len(t_b.bbox_history) >= 2:
                b1_a, b2_a = list(t_a.bbox_history)[-2], list(t_a.bbox_history)[-1]
                b1_b, b2_b = list(t_b.bbox_history)[-2], list(t_b.bbox_history)[-1]
                jitter_a = abs((b2_a[2]-b2_a[0]) - (b1_a[2]-b1_a[0])) + abs((b2_a[3]-b2_a[1]) - (b1_a[3]-b1_a[1]))
                jitter_b = abs((b2_b[2]-b2_b[0]) - (b1_b[2]-b1_b[0])) + abs((b2_b[3]-b2_b[1]) - (b1_b[3]-b1_b[1]))
                if jitter_a > 10.0 or jitter_b > 10.0:
                    low_jitter = False
            
            # Synchronized centroids
            if len(t_a.centroid_history) >= 2 and len(t_b.centroid_history) >= 2:
                acx_now, acy_now = t_a.centroid_history[-1]
                acx_prev, acy_prev = t_a.centroid_history[-2]
                bcx_now, bcy_now = t_b.centroid_history[-1]
                bcx_prev, bcy_prev = t_b.centroid_history[-2]
                
                vel_a = math.sqrt((acx_now - acx_prev)**2 + (acy_now - acy_prev)**2)
                vel_b = math.sqrt((bcx_now - bcx_prev)**2 + (bcy_now - bcy_prev)**2)
                
                if abs(vel_a - vel_b) > 0.015:
                    synchronized_centroids = False
                    
            if smooth_motion and low_arm_velocity and low_jitter and synchronized_centroids and not has_aggression_signal:
                self.calm_interaction_mode = True

        elif len(t_keys) == 1:
            t_single = self.trajectories[t_keys[0]]
            if t_single.body_acceleration < 0.40:
                smooth_motion = True
            max_wrist_vel = max(t_single.wrist_velocities) if t_single.wrist_velocities else 0.0
            if max_wrist_vel > 0.022:
                low_arm_velocity = False
            if smooth_motion and low_arm_velocity and not has_aggression_signal:
                self.calm_interaction_mode = True

        # ============================================================
        # TASK 2: CLOSE-COMBAT DETECTION Heuristics & Dedicated Mode
        # ============================================================
        close_combat_aggression = False
        self.close_combat_mode = False
        
        # Geometry chaos and relative limb proximity
        if len(t_keys) >= 2:
            t_a = self.trajectories[t_keys[0]]
            t_b = self.trajectories[t_keys[1]]
            
            # Head-to-head proximity (keypoint 0 is Nose)
            head_a = t_a.history[-1][0] if t_a.history and len(t_a.history[-1]) > 0 else None
            head_b = t_b.history[-1][0] if t_b.history and len(t_b.history[-1]) > 0 else None
            head_dist = 1.0
            if head_a and head_b and head_a[2] > 0.30 and head_b[2] > 0.30:
                head_dist = math.sqrt((head_a[0] - head_b[0])**2 + (head_a[1] - head_b[1])**2)
                if head_dist < 0.12:
                    close_combat_aggression = True
                    
            # High Torso/Limbs overlap
            if chaotic_limb_overlap or proximity_dist < 0.22:
                close_combat_aggression = True

            # Chaotic relative body geometry
            if t_a.body_acceleration > 1.2 or t_b.body_acceleration > 1.2:
                close_combat_aggression = True

            # Trigger close_combat_mode only if we have high proximity/overlap COMBINED with aggression signals (Task 5)
            if has_aggression_signal:
                if (chaotic_limb_overlap and proximity_dist < 0.25) or (head_dist < 0.15) or close_combat_aggression:
                    self.close_combat_mode = True

        elif len(t_keys) == 1:
            # Merged person close-combat (YOLO body merge)
            t_single = self.trajectories[t_keys[0]]
            if has_aggression_signal and (t_single.body_acceleration > 1.1 or self.last_temporal_confidence > 0.40):
                close_combat_aggression = True
                self.close_combat_mode = True

        # ============================================================
        # TASK 5: GRAPPLING / CHOKING SUPPORT
        # ============================================================
        is_grappling_or_choking = (
            grappling_detected or 
            self.close_combat_mode or
            (is_interactive and proximity_dist < 0.18 and chaotic_limb_overlap) or
            (not is_interactive and close_combat_aggression and self.last_temporal_confidence > 0.45)
        )

        # ============================================================
        # TASK 7: HYBRID WEIGHTED AGGRESSION SCORING (Rebalanced Weights)
        # ============================================================
        violence_score = 0
        
        # 1. Chaotic Motion (15 points) - Task 2
        chaotic_motion = (chaotic_movement_detected or repeated_directional_changes 
                          or violent_bbox_jitter or arm_velocity_bursts or rapid_centroid_acceleration)
        if chaotic_motion:
            violence_score += 15

        # 2. Repeated Fast Arm Movement (20 points) - Task 2
        repeated_arm_speed = (repeated_strikes or arm_velocity_bursts)
        if repeated_arm_speed:
            violence_score += 20

        # 3. Skeleton Overlap (15 points) - Task 2
        skeleton_overlap = chaotic_limb_overlap
        if skeleton_overlap:
            violence_score += 15

        # 4. Human Proximity Aggression (10 points) - Task 2
        close_proximity = (is_interactive and proximity_dist < 0.35)
        if close_proximity:
            violence_score += 10

        # 5. Temporal Model Confidence (30 points) - Task 2
        if self.last_temporal_confidence > 0.60:
            violence_score += 30

        # 6. Wrist Collision (20 points) - Task 2
        if self.punch_detected:
            violence_score += 20

        # 7. Recoil (10 points) - Task 2
        if self.recoil_active:
            violence_score += 10

        # 8. Close Combat / Grappling Specific Boost (15 points) - Task 2
        if close_combat_aggression or self.close_combat_mode:
            violence_score += 15

        # ============================================================
        # TASK 4: REDUCE CLOSE-COMBAT BOOST (Task 4: scaled down from 1.30 to 1.10)
        # ============================================================
        if self.close_combat_mode or is_grappling_or_choking:
            # Dynamically boost/scale aggression score to ensure sensitivity
            violence_score = int(violence_score * 1.10)

        # ============================================================
        # TASK 3: HUG / HANDSHAKE SUPPRESSION
        # ============================================================
        if self.calm_interaction_mode:
            # Heavily suppress aggression score for calm interactions (clamp to LOW/MEDIUM)
            # If they are just standing close/hugging/shaking hands: clamp to LOW (max 25)
            # If moderate friendly movement: clamp to MEDIUM (max 42)
            if not has_aggression_signal:
                violence_score = min(25, violence_score)
            else:
                violence_score = min(42, violence_score)

        # ============================================================
        # TASK 4 & 5: AGGRESSION OVERRIDE MODE (Robust close-combat fallback)
        # ============================================================
        if not self.calm_interaction_mode:
            if (self.last_temporal_confidence > 0.55 or chaotic_motion) and repeated_arm_speed and (skeleton_overlap or close_combat_aggression or is_grappling_or_choking):
                # Force CRITICAL/HIGH threat (minimum score of 75)
                violence_score = max(75, violence_score)
            elif is_grappling_or_choking:
                # Force at least HIGH (minimum score of 60)
                violence_score = max(60, violence_score)

        # ============================================================
        # TASK 8: DEMO MODE OPTIMIZATION (Precise Target Boundaries)
        # ============================================================
        # Skip single-person clamp if close combat mode or close combat aggression is active
        if not is_interactive and not self.close_combat_mode and not close_combat_aggression:
            # Single person sitting / walking: LOW max (Task 8)
            # Single person exercise/dancing: MEDIUM max (Task 8)
            if "high_kick" in active_flags or "suspicious_running" in active_flags or chaotic_motion:
                violence_score = min(45, violence_score)  # Clamp to MEDIUM max
            else:
                violence_score = min(28, violence_score)  # Clamp to LOW max
        elif is_interactive and close_proximity and not (self.punch_detected or self.recoil_active or repeated_strikes or chaotic_motion or close_combat_aggression or is_grappling_or_choking):
            # Two close people, slow/nominal interaction: clamp to MEDIUM (max 48)
            violence_score = min(48, violence_score)

        # Ensure repeated aggression scales all the way up to CRITICAL
        if repeated_strikes and (self.punch_detected or self.close_combat_mode):
            violence_score = max(82, violence_score)

        # ============================================================
        # TASK 7: GLOBAL MOTION SUPPRESSION FOR CAMERA SHAKE
        # ============================================================
        if camera_motion_detected:
            # Reduce aggression heavily (Task 7)
            violence_score = int(violence_score * 0.2)

        # ============================================================
        # TASK 1: FAST VIOLENCE PATHWAY DETECTION
        # ============================================================
        extreme_wrist_speed = False
        explosive_accel = False
        
        # Check active motion requirement (Task 2 & 3)
        any_active_motion = False
        for tk in t_keys:
            traj = self.trajectories[tk]
            max_wrist_vel = max(traj.wrist_velocities) if traj.wrist_velocities else 0.0
            if max_wrist_vel > 0.045:
                extreme_wrist_speed = True
            if traj.body_acceleration > 1.35:
                explosive_accel = True
            
            # Trajectory active motion check
            if max_wrist_vel > 0.022 or traj.body_acceleration > 0.45 or traj.repeated_arm_strikes:
                any_active_motion = True

        self.fast_violence_mode = (
            (self.punch_detected or self.recoil_active or repeated_strikes or extreme_wrist_speed or explosive_accel or (self.last_temporal_confidence > 0.65))
            and not camera_motion_detected
        )

        # Require active motion and active aggression signals (Task 1 & 2)
        if not has_aggression_signal and not any_active_motion:
            self.no_new_aggression_frames += 1
        else:
            self.no_new_aggression_frames = 0

        # Update self.recent_violence_counter (Task 6)
        if self.fast_violence_mode:
            self.recent_violence_counter = 30  # About 2-3 seconds at 10-15 FPS
        elif self.recent_violence_counter > 0:
            self.recent_violence_counter -= 1

        # Scale to 0.0 - 1.0 representation
        combined_score = min(1.0, violence_score / 100.0)

        # Rapid post-action decay (Task 1 & 7: decay by 0.65 instead of 0.92)
        if self.no_new_aggression_frames >= 3:
            decayed_history = collections.deque(maxlen=VIOLENCE_BUFFER_SIZE)
            for val in self.violence_score_history:
                decayed_history.append(val * 0.65)
            self.violence_score_history = decayed_history
            combined_score *= 0.65

        # ============================================================
        # TASK 4: DYNAMIC SMOOTHING & DUAL PIPELINE
        # ============================================================
        if self.fast_violence_mode:
            # Bypass rolling average (buffer size = 1) (Task 4)
            self.violence_score_history.clear()
            self.violence_score_history.append(combined_score)
            smoothed_score = combined_score
        else:
            # Normal mode: use rolling average over 3 frames
            self.violence_score_history.append(combined_score)
            while len(self.violence_score_history) > 3:
                self.violence_score_history.popleft()
            smoothed_score = sum(self.violence_score_history) / len(self.violence_score_history)

        # Resolve exact tactical action label (Task 8 Final Targets & Task 6 Active vs. Recent)
        smoothed_score_pct = smoothed_score * 100
        action = "Normal"
        
        # ACTIVE VS RECENT VIOLENCE SEPARATION (Task 6)
        is_active_fight = self.fast_violence_mode or has_aggression_signal or any_active_motion
        violence_was_recent = self.recent_violence_counter > 0

        if smoothed_score_pct >= 55.0:
            if is_active_fight:
                if repeated_strikes or arm_velocity_bursts:
                    action = "Fighting (Repeated Strikes)"
                elif self.punch_detected:
                    action = "Punch Detected"
                elif self.recoil_active:
                    action = "Physical Assault (Recoil)"
                elif grappling_detected or is_grappling_or_choking:
                    action = "Physical Grappling"
                else:
                    action = "Fighting (Chaotic Motion)"
            elif violence_was_recent:
                # Active motion has stopped, but violence happened recently (Task 6)
                action = "Recent Violence (Cooldown)"
            else:
                action = "Suspicious Interaction"
        elif smoothed_score_pct >= 30.0:
            if is_active_fight:
                if "fall_detected" in active_flags:
                    action = "Sudden Fall"
                elif "suspicious_running" in active_flags:
                    action = "Suspicious Running"
                else:
                    action = "Suspicious Interaction"
            elif violence_was_recent:
                action = "Recent Violence (Cooldown)"
            else:
                action = "Suspicious Interaction"
        else:
            if violence_was_recent and smoothed_score_pct > 15.0:
                action = "Recent Violence (Cooldown)"
            else:
                action = "Normal"

        if len(active_flags) == 0 or (len(active_flags) == 1 and "combat_stance" in active_flags and max_aggression < 0.30):
            if not self.punch_detected and not chaotic_motion and not close_combat_aggression and not violence_was_recent:
                action = "Normal"
                smoothed_score *= 0.4

        final_score = round(smoothed_score, 3)
        for person in persons:
            kps = person.get("keypoints", [])
            bbox = person.get("bbox", [])
            if not kps and not bbox:
                continue
            
            # Find matching trajectory
            cx, cy = 0.5, 0.5
            if kps:
                valid_pts = [(kp[0], kp[1]) for kp in kps if kp[2] > 0.35]
                if valid_pts:
                    cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
                    cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)
            elif bbox:
                cx = (bbox[0] + bbox[2]) / 2.0 / 640.0
                cy = (bbox[1] + bbox[3]) / 2.0 / 480.0
                
            matched_traj = None
            min_dist = 0.25
            for traj in self.trajectories.values():
                if traj.centroid_history:
                    pcx, pcy = traj.centroid_history[-1]
                    dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_traj = traj
            
            if matched_traj is not None:
                # Blend trajectory score with the overall smoothed scene score
                person["aggression_score"] = max(matched_traj.aggression_score, final_score)
                person["pose_flags"] = list(set(person.get("pose_flags", []) + matched_traj.pose_flags))
            else:
                if final_score > 0.30:
                    person["aggression_score"] = final_score

        return action, final_score

    def calculate_frame_motion(self, frame) -> float:
        """Fast CPU-efficient frame differencing acting as motion magnitude index."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            if self._prev_gray is None:
                self._prev_gray = gray
                return 0.0

            diff = cv2.absdiff(self._prev_gray, gray)
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            motion_pixels = float(np.sum(thresh == 255))
            intensity = min(1.0, motion_pixels / (frame.shape[0] * frame.shape[1] * 0.35))
            self._prev_gray = gray
            self.motion_intensity_history.append(intensity)
            return round(intensity, 3)
        except Exception:
            return 0.0

    def get_repeated_strikes_active(self) -> bool:
        return any(t.repeated_arm_strikes for t in self.trajectories.values())

    def get_fall_active(self) -> bool:
        return any(t.is_falling for t in self.trajectories.values())

    def get_chasing_active(self) -> bool:
        return self._evaluate_chasing() > 0.05

    def _track_and_update_trajectories(self, detected_persons: list[dict]):
        fresh_trajectories = {}
        for idx, person in enumerate(detected_persons):
            kps = person.get("keypoints", [])
            bbox = person.get("bbox", [])
            if not kps and not bbox:
                continue

            cx, cy = 0.5, 0.5
            if kps:
                valid_pts = [(kp[0], kp[1]) for kp in kps if kp[2] > 0.35]
                if valid_pts:
                    cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
                    cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)
            elif bbox:
                cx = (bbox[0] + bbox[2]) / 2.0 / 640.0
                cy = (bbox[1] + bbox[3]) / 2.0 / 480.0

            best_id = None
            min_dist = 0.22
            for pid, traj in self.trajectories.items():
                if traj.centroid_history:
                    pcx, pcy = traj.centroid_history[-1]
                    dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = pid

            if best_id is not None and best_id not in fresh_trajectories:
                traj = self.trajectories[best_id]
                traj.update(kps, bbox)
                fresh_trajectories[best_id] = traj
            else:
                new_id = int(time.time() * 1000) + idx
                traj = PersonTrajectory(new_id)
                traj.update(kps, bbox)
                fresh_trajectories[new_id] = traj

        self.trajectories = fresh_trajectories

    def _evaluate_proximity_aggression(self) -> float:
        t_keys = list(self.trajectories.keys())
        if len(t_keys) < 2:
            return 0.0
        max_proximity_boost = 0.0
        for i in range(len(t_keys)):
            for j in range(i + 1, len(t_keys)):
                t_a = self.trajectories[t_keys[i]]
                t_b = self.trajectories[t_keys[j]]
                if t_a.centroid_history and t_b.centroid_history:
                    ax, ay = t_a.centroid_history[-1]
                    bx, by = t_b.centroid_history[-1]
                    dist = math.sqrt((ax - bx)**2 + (ay - by)**2)
                    if dist < 0.24:
                        accel = t_a.body_acceleration + t_b.body_acceleration
                        active_pose = len(t_a.pose_flags) + len(t_b.pose_flags)
                        if accel > 0.6 or active_pose > 0:
                            max_proximity_boost = max(max_proximity_boost, 0.30)
                        else:
                            max_proximity_boost = max(max_proximity_boost, 0.10)
        return max_proximity_boost

    def _evaluate_chasing(self) -> float:
        t_keys = list(self.trajectories.keys())
        if len(t_keys) < 2:
            return 0.0
        for i in range(len(t_keys)):
            for j in range(i + 1, len(t_keys)):
                t_a = self.trajectories[t_keys[i]]
                t_b = self.trajectories[t_keys[j]]
                if len(t_a.centroid_history) < 5 or len(t_b.centroid_history) < 5:
                    continue
                
                ax_now, ay_now = t_a.centroid_history[-1]
                bx_now, by_now = t_b.centroid_history[-1]
                dist_now = math.sqrt((ax_now - bx_now)**2 + (ay_now - by_now)**2)

                ax_prev, ay_prev = t_a.centroid_history[-5]
                bx_prev, by_prev = t_b.centroid_history[-5]
                dist_prev = math.sqrt((ax_prev - bx_prev)**2 + (ay_prev - by_prev)**2)

                if dist_prev - dist_now > 0.08 and (t_a.is_running or t_b.is_running):
                    return 0.18
        return 0.0
