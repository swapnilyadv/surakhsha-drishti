"""
Pose Estimator Service
=======================
Uses MediaPipe Pose Landmarker (Lite task model) for real-time skeleton tracking.

Capabilities:
  - 33-point body skeleton detection per person
  - Aggression heuristic analysis:
      • Raised fists / punching
      • High kick detection
      • Rapid limb extension (aggression score)
      • Falling / collapse detection
  - Returns per-person keypoints + aggression score
  - CPU-optimised: uses 'lite' model, runs every Nth frame

Architecture note:
  MediaPipe Tasks API (PoseLandmarker) is used instead of the legacy
  mediapipe.solutions.pose because it supports multi-person detection
  and is the current recommended approach.
"""

import logging
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe landmark indices (COCO-style 33 keypoints) ─────────────────────
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
KP_NOSE          = 0
KP_LEFT_SHOULDER = 11
KP_RIGHT_SHOULDER= 12
KP_LEFT_ELBOW    = 13
KP_RIGHT_ELBOW   = 14
KP_LEFT_WRIST    = 15
KP_RIGHT_WRIST   = 16
KP_LEFT_HIP      = 23
KP_RIGHT_HIP     = 24
KP_LEFT_KNEE     = 25
KP_RIGHT_KNEE    = 26
KP_LEFT_ANKLE    = 27
KP_RIGHT_ANKLE   = 28

# Skeleton connections for overlay drawing
POSE_CONNECTIONS = [
    (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW,     KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER,  KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP,       KP_RIGHT_HIP),
    (KP_LEFT_HIP,       KP_LEFT_KNEE),
    (KP_LEFT_KNEE,      KP_LEFT_ANKLE),
    (KP_RIGHT_HIP,      KP_RIGHT_KNEE),
    (KP_RIGHT_KNEE,     KP_RIGHT_ANKLE),
]

# Aggression scoring weights
AGGRESSION_RAISED_FIST   = 0.45   # wrist above shoulder
AGGRESSION_HIGH_KICK     = 0.55   # ankle above hip
AGGRESSION_ARM_EXTENSION = 0.25   # elbow/wrist far from body centre
AGGRESSION_FALL_DETECT   = 0.40   # nose below hip level (falling/collapse)

# Confidence threshold — landmarks below this are ignored
LANDMARK_VISIBILITY_THRESH = 0.45


class PoseEstimator:
    """
    MediaPipe Pose Landmarker wrapper.
    - Loads the 'lite' .task model for low-latency CPU inference
    - Detects skeleton + computes per-person aggression score
    - Returns structured JSON-serialisable result per frame
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.landmarker = None
        self.loaded = False
        self._options = None
        
        # Landmark smoothing state (maps track_id -> smoothed keypoints)
        self.smoothed_kps = {}
        self.last_centroids = {}
        self.alpha = 0.45  # EMA smoothing factor (0.45 = sweet spot for smoothness vs latency)

    def load(self) -> bool:
        """Load MediaPipe PoseLandmarker. Returns True on success."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core import base_options as mp_base

            if not Path(self.model_path).exists():
                logger.warning(
                    f"[PoseEstimator] Model not found at {self.model_path} — "
                    "pose estimation disabled"
                )
                return False

            base_opts = mp_base.BaseOptions(model_asset_path=self.model_path)

            # IMAGE mode: we pass frames one-by-one (not video/live-stream)
            # This is the most stable mode for a threaded pipeline.
            self._options = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                output_segmentation_masks=False,
                num_poses=4,                            # detect up to 4 people
                min_pose_detection_confidence=0.50,
                min_pose_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                running_mode=mp_vision.RunningMode.IMAGE,
            )

            self.landmarker = mp_vision.PoseLandmarker.create_from_options(self._options)
            self.loaded = True
            logger.info(f"[PoseEstimator] Model loaded: {self.model_path}")
            return True

        except ImportError as e:
            logger.warning(f"[PoseEstimator] mediapipe not installed: {e} — disabled")
            return False
        except Exception as e:
            logger.error(f"[PoseEstimator] Load failed: {e}")
            return False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run pose estimation on a BGR frame.

        Returns a list of person dicts:
        {
          "keypoints": [[x, y, visibility], ...],   # 33 points, normalised 0-1
          "aggression_score": float,                 # 0.0 – 1.0
          "pose_flags": [str],                       # e.g. ["raised_fist", "high_kick"]
        }
        Returns [] if model not loaded or no persons found.
        """
        if not self.loaded or self.landmarker is None:
            return []

        if frame is None or frame.size == 0 or len(frame.shape) < 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("[PoseEstimator] Null/Empty frame received.")
            return []

        try:
            import mediapipe as mp

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.landmarker.detect(mp_image)

            persons = []
            fresh_smoothed = {}
            fresh_centroids = {}

            for idx, pose_landmarks in enumerate(result.pose_landmarks):
                # Build flat keypoint list: [[x, y, visibility], ...]
                keypoints = [
                    [round(lm.x, 4), round(lm.y, 4), round(lm.visibility, 3)]
                    for lm in pose_landmarks
                ]

                # Centroid computation for tracking association
                valid_pts = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.45]
                if valid_pts:
                    cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
                    cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)
                else:
                    cx, cy = 0.5, 0.5

                # Find closest historical centroid
                matched_id = None
                min_dist = 0.15
                for pid, prev_c in self.last_centroids.items():
                    dist = math.sqrt((cx - prev_c[0])**2 + (cy - prev_c[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = pid

                if matched_id is None:
                    matched_id = int(time.time() * 1000) + idx

                # Apply Exponential Moving Average (EMA) smoothing
                if matched_id in self.smoothed_kps:
                    prev_kps = self.smoothed_kps[matched_id]
                    smoothed_keypoints = []
                    for k in range(len(keypoints)):
                        curr_x, curr_y, curr_v = keypoints[k]
                        prev_x, prev_y, prev_v = prev_kps[k]
                        
                        sx = self.alpha * curr_x + (1.0 - self.alpha) * prev_x
                        sy = self.alpha * curr_y + (1.0 - self.alpha) * prev_y
                        smoothed_keypoints.append([round(sx, 4), round(sy, 4), curr_v])
                    keypoints = smoothed_keypoints

                fresh_smoothed[matched_id] = keypoints
                fresh_centroids[matched_id] = (cx, cy)

                aggression, flags = self._compute_aggression(keypoints)

                persons.append({
                    "keypoints": keypoints,
                    "aggression_score": round(aggression, 3),
                    "pose_flags": flags,
                })

            self.smoothed_kps = fresh_smoothed
            self.last_centroids = fresh_centroids
            return persons

        except Exception as e:
            logger.debug(f"[PoseEstimator] Inference error: {e}")
            return []

    # ── Aggression heuristics ────────────────────────────────────────────────

    def _get_lm(self, keypoints: list, idx: int) -> Optional[tuple]:
        """Return (x, y) for landmark if visibility >= threshold, else None."""
        if idx >= len(keypoints):
            return None
        x, y, vis = keypoints[idx]
        if vis < LANDMARK_VISIBILITY_THRESH:
            return None
        return (x, y)

    def _compute_aggression(self, keypoints: list) -> tuple[float, list[str]]:
        """
        Heuristic aggression scoring.

        Analyses landmark geometry to detect:
          - Raised fists (wrist above shoulder)
          - High kicks (ankle above hip level)
          - Wide arm extension (arm span relative to torso)
          - Falling / collapse (nose below hip)

        Returns: (aggression_score: float, flags: list[str])
        """
        score = 0.0
        flags = []

        # Helper for safe landmark retrieval
        g = self._get_lm

        l_shoulder = g(keypoints, KP_LEFT_SHOULDER)
        r_shoulder = g(keypoints, KP_RIGHT_SHOULDER)
        l_wrist    = g(keypoints, KP_LEFT_WRIST)
        r_wrist    = g(keypoints, KP_RIGHT_WRIST)
        l_elbow    = g(keypoints, KP_LEFT_ELBOW)
        r_elbow    = g(keypoints, KP_RIGHT_ELBOW)
        l_hip      = g(keypoints, KP_LEFT_HIP)
        r_hip      = g(keypoints, KP_RIGHT_HIP)
        l_ankle    = g(keypoints, KP_LEFT_ANKLE)
        r_ankle    = g(keypoints, KP_RIGHT_ANKLE)
        nose       = g(keypoints, KP_NOSE)

        # ── 1. Raised fist detection (wrist Y above shoulder Y) ───────────────
        # In MediaPipe, Y=0 is top of image, Y=1 is bottom.
        raised = False
        if l_wrist and l_shoulder and l_wrist[1] < l_shoulder[1] - 0.05:
            raised = True
        if r_wrist and r_shoulder and r_wrist[1] < r_shoulder[1] - 0.05:
            raised = True
        if raised:
            score += AGGRESSION_RAISED_FIST
            flags.append("raised_fist")

        # ── 2. High kick detection (ankle Y above hip Y significantly) ────────
        kicked = False
        if l_ankle and l_hip and l_ankle[1] < l_hip[1] - 0.10:
            kicked = True
        if r_ankle and r_hip and r_ankle[1] < r_hip[1] - 0.10:
            kicked = True
        if kicked:
            score += AGGRESSION_HIGH_KICK
            flags.append("high_kick")

        # ── 3. Wide arm extension (elbows far from body midline) ──────────────
        if l_shoulder and r_shoulder and l_elbow and r_elbow:
            torso_w = abs(l_shoulder[0] - r_shoulder[0])
            arm_span = abs(l_elbow[0] - r_elbow[0])
            if torso_w > 0.02 and arm_span > torso_w * 2.2:
                score += AGGRESSION_ARM_EXTENSION
                flags.append("wide_arm_extension")

        # ── 4. Punching motion (wrist far extended past shoulder plane) ────────
        punching = False
        if l_wrist and l_shoulder and l_elbow:
            # Wrist X far beyond shoulder X on the opposite side
            if abs(l_wrist[0] - l_shoulder[0]) > 0.25:
                punching = True
        if r_wrist and r_shoulder and r_elbow:
            if abs(r_wrist[0] - r_shoulder[0]) > 0.25:
                punching = True
        if punching and "raised_fist" in flags:
            score += 0.15   # bonus if fist is raised AND extended
            flags.append("punching_motion")

        # ── 5. Falling / collapse (nose below hip level) ──────────────────────
        hip_y = None
        if l_hip and r_hip:
            hip_y = (l_hip[1] + r_hip[1]) / 2
        elif l_hip:
            hip_y = l_hip[1]
        elif r_hip:
            hip_y = r_hip[1]

        if nose and hip_y and nose[1] > hip_y + 0.05:
            score += AGGRESSION_FALL_DETECT
            flags.append("falling")

        # Clamp to [0, 1]
        return min(score, 1.0), flags

    def draw_skeleton(
        self,
        frame: np.ndarray,
        persons: list[dict],
        frame_h: int,
        frame_w: int,
        person_bboxes: Optional[list] = None,
    ) -> np.ndarray:
        """
        Draw skeleton overlay on frame.
        - Skeleton colour shifts from green → orange → red based on aggression_score
        - Joints drawn as filled circles
        - Bones drawn as lines
        """
        for i, person in enumerate(persons):
            kps = person["keypoints"]
            agg = person.get("aggression_score", 0.0)

            # Colour: green (safe) → yellow → red (aggressive)
            r = int(min(255, agg * 2 * 255))
            g = int(min(255, (1 - agg) * 2 * 255))
            colour = (0, g, r)   # BGR
            joint_colour = (255, 255, 255)

            # Convert normalised coords → pixel coords
            def px(kp_idx):
                if kp_idx >= len(kps):
                    return None
                x, y, vis = kps[kp_idx]
                if vis < LANDMARK_VISIBILITY_THRESH:
                    return None
                return (int(x * frame_w), int(y * frame_h))

            # Draw skeleton bones
            for a, b in POSE_CONNECTIONS:
                pt_a = px(a)
                pt_b = px(b)
                if pt_a and pt_b:
                    cv2.line(frame, pt_a, pt_b, colour, 2, cv2.LINE_AA)

            # Draw joints
            for idx in range(min(len(kps), 33)):
                pt = px(idx)
                if pt:
                    cv2.circle(frame, pt, 4, joint_colour, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 4, colour, 1, cv2.LINE_AA)

            # Aggression label near nose/shoulder
            label_pt = px(KP_NOSE) or px(KP_LEFT_SHOULDER)
            if label_pt:
                flags_str = " ".join(person.get("pose_flags", []))
                label = f"AGG:{agg:.0%}"
                if flags_str:
                    label += f" [{flags_str}]"
                lx, ly = label_pt
                cv2.rectangle(frame, (lx - 2, ly - 16), (lx + len(label) * 7, ly + 2),
                               (0, 0, 0), -1)
                cv2.putText(frame, label, (lx, ly - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        return frame

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
        self.loaded = False
        logger.info("[PoseEstimator] Released")
