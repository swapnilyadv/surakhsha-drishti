"""
Violence Classifier Service
============================
Temporal sequence analyser and classifier for body movement trajectories,
velocities, accelerations, postures, and multi-person interactions.

Features:
  - Tracks person centroid trajectories across frames
  - Calculates joint velocities (wrists, elbows, shoulders, ankles)
  - Calculates body acceleration (detects sudden falls or aggressive charges)
  - Analyzes body postures (combat stance, raised fists, high kicks, wide swings)
  - Evaluates close-proximity interactions between multiple tracked persons
  - Implements multi-frame temporal smoothing and threshold buffering to filter
    out casual gestures, stretching, walking, dancing, and hand waving.
"""

import math
import time
import logging
from collections import deque
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# ── Config Constants ──────────────────────────────────────────────────────────
MAX_HISTORY_FRAMES = 15       # ~1 second of movement history (at 15-20 FPS)
VELOCITY_WINDOW    = 3        # frames over which to compute velocity
ACCEL_WINDOW       = 3        # frames over which to compute acceleration

# Posture & Motion Thresholds
ARM_SWING_VELOCITY_THRESH = 0.15   # normal arm swing velocity threshold
COMBAT_STANCE_ANGLE       = 140.0   # bent elbows angle threshold (degrees)
HIGH_KICK_HEIGHT_THRESH   = 0.12   # ankle relative to hip vertical difference
RAISED_FIST_THRESH        = 0.05   # wrist relative to shoulder vertical difference
FAST_PUNCH_VELOCITY       = 0.22   # fast wrist extension velocity (punch)
AGGRESSION_DECAY          = 0.95   # decay factor for temporal persistence
VIOLENCE_BUFFER_SIZE      = 8      # sliding window for final decision smoothing


class PersonTrajectory:
    """Stores history of a tracked individual's keypoints and computes dynamics."""

    def __init__(self, person_id: int):
        self.person_id = person_id
        # Keypoints history: deque of lists of [x, y, visibility]
        self.history: deque[list[list[float]]] = deque(maxlen=MAX_HISTORY_FRAMES)
        self.timestamps: deque[float] = deque(maxlen=MAX_HISTORY_FRAMES)
        self.centroid_history: deque[tuple[float, float]] = deque(maxlen=MAX_HISTORY_FRAMES)

        # Computed dynamics
        self.wrist_velocities: list[float] = [0.0, 0.0]  # [left, right]
        self.ankle_velocities: list[float] = [0.0, 0.0]  # [left, right]
        self.body_acceleration: float = 0.0
        self.is_falling: bool = False
        self.pose_flags: list[str] = []
        self.aggression_score: float = 0.0

    def update(self, keypoints: list[list[float]]):
        """Append fresh keypoints, update timestamps, and compute motion physics."""
        self.history.append(keypoints)
        self.timestamps.append(time.time())
        
        # Compute centroid (nose, shoulders, hips average for stability)
        landmark_indices = [0, 11, 12, 23, 24]  # Nose, L/R shoulders, L/R hips
        valid_pts = []
        for idx in landmark_indices:
            if idx < len(keypoints):
                x, y, vis = keypoints[idx]
                if vis > 0.4:
                    valid_pts.append((x, y))
        
        if valid_pts:
            cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
            cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)
            self.centroid_history.append((cx, cy))
        else:
            self.centroid_history.append((0.5, 0.5))

        self._compute_dynamics()

    def _compute_dynamics(self):
        """Extract velocity, acceleration, posture flags, and aggression score."""
        n = len(self.history)
        if n < 2:
            return

        # ── 1. Calculate Velocities (wrists, ankles) ─────────────────────────
        self.wrist_velocities = [self._calc_joint_velocity(15), self._calc_joint_velocity(16)] # Left, Right wrist
        self.ankle_velocities = [self._calc_joint_velocity(27), self._calc_joint_velocity(28)] # Left, Right ankle

        # ── 2. Calculate Body Acceleration (using centroid shift) ─────────────
        self.body_acceleration = self._calc_body_acceleration()

        # ── 3. Posture Analysis ──────────────────────────────────────────────
        kps = self.history[-1]
        self.pose_flags = []
        raw_score = 0.0

        # Helper to get current coordinates of a joint
        def get_kp(idx: int) -> Optional[tuple[float, float]]:
            if idx >= len(kps):
                return None
            x, y, vis = kps[idx]
            return (x, y) if vis > 0.4 else None

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
        nose       = get_kp(0)

        # A. Raised fists detection (wrists above shoulders)
        raised_fist = False
        if l_wrist and l_shoulder and l_wrist[1] < l_shoulder[1] - RAISED_FIST_THRESH:
            raised_fist = True
        if r_wrist and r_shoulder and r_wrist[1] < r_shoulder[1] - RAISED_FIST_THRESH:
            raised_fist = True
        if raised_fist:
            self.pose_flags.append("raised_fist")
            raw_score += 0.35

        # B. High kick detection (ankles above hip level)
        kicking = False
        if l_ankle and l_hip and l_ankle[1] < l_hip[1] - HIGH_KICK_HEIGHT_THRESH:
            kicking = True
        if r_ankle and r_hip and r_ankle[1] < r_hip[1] - HIGH_KICK_HEIGHT_THRESH:
            kicking = True
        if kicking:
            self.pose_flags.append("high_kick")
            raw_score += 0.45

        # C. Punching motion / fast extension
        punching = False
        if (self.wrist_velocities[0] > FAST_PUNCH_VELOCITY and l_wrist and l_shoulder and abs(l_wrist[0] - l_shoulder[0]) > 0.25):
            punching = True
        if (self.wrist_velocities[1] > FAST_PUNCH_VELOCITY and r_wrist and r_shoulder and abs(r_wrist[0] - r_shoulder[0]) > 0.25):
            punching = True
        if punching:
            self.pose_flags.append("punching_motion")
            raw_score += 0.50

        # D. Falling / collapse detection
        hip_y = None
        if l_hip and r_hip:
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
        elif l_hip:
            hip_y = l_hip[1]
        elif r_hip:
            hip_y = r_hip[1]

        if nose and hip_y and nose[1] > hip_y + 0.08:
            self.is_falling = True
            self.pose_flags.append("falling")
            raw_score += 0.40
        else:
            self.is_falling = False

        # E. Combat Stance (bent arms close to torso + body velocity)
        if l_shoulder and l_elbow and l_wrist and r_shoulder and r_elbow and r_wrist:
            l_angle = self._calc_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = self._calc_angle(r_shoulder, r_elbow, r_wrist)
            if l_angle < COMBAT_STANCE_ANGLE or r_angle < COMBAT_STANCE_ANGLE:
                self.pose_flags.append("combat_stance")
                raw_score += 0.20

        # F. Rapid arm swing velocity detection (fighting/flailing)
        max_wrist_vel = max(self.wrist_velocities)
        if max_wrist_vel > ARM_SWING_VELOCITY_THRESH:
            self.pose_flags.append("fast_arm_movement")
            raw_score += min(0.3, max_wrist_vel * 1.2)

        # Decay previous aggression and blend with fresh raw indicators
        self.aggression_score = round(self.aggression_score * AGGRESSION_DECAY + raw_score * (1.0 - AGGRESSION_DECAY), 3)

    def _calc_joint_velocity(self, kp_idx: int) -> float:
        """Calculate joint Euclidean distance shift per second."""
        n = len(self.history)
        if n < VELOCITY_WINDOW:
            return 0.0
        
        curr_kp = self.history[-1]
        prev_kp = self.history[-VELOCITY_WINDOW]
        
        if kp_idx >= len(curr_kp) or kp_idx >= len(prev_kp):
            return 0.0
            
        c_pt, p_pt = curr_kp[kp_idx], prev_kp[kp_idx]
        if c_pt[2] < 0.4 or p_pt[2] < 0.4:
            return 0.0
            
        dt = self.timestamps[-1] - self.timestamps[-VELOCITY_WINDOW]
        if dt <= 0:
            return 0.0
            
        dist = math.sqrt((c_pt[0] - p_pt[0])**2 + (c_pt[1] - p_pt[1])**2)
        return dist / dt

    def _calc_body_acceleration(self) -> float:
        """Compute rate of change of body centroid velocity (acceleration)."""
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

    @staticmethod
    def _calc_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """Calculate degree angle between three 2D points with b as vertex."""
        ang = math.degrees(
            math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        )
        return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


class ViolenceClassifier:
    """
    Stateful temporal classifier that combines body pose dynamics,
    velocities, accelerations, and multi-person proximity metrics to determine
    if a fight or aggressive action is occurring.
    """

    def __init__(self):
        # Maps person_id -> PersonTrajectory object
        self.trajectories: dict[int, PersonTrajectory] = {}
        # Sliding buffer of combined violence scores for final decision smoothing
        self.violence_score_history: deque[float] = deque(maxlen=VIOLENCE_BUFFER_SIZE)

    def process_pose_data(self, persons: list[dict], original_h: int, original_w: int) -> tuple[str, float]:
        """
        Ingest frame pose keypoints, perform temporal analysis, check proximity,
        and calculate final action label + aggregate aggression confidence.
        """
        # 1. Coordinate tracking & maintenance
        self._track_and_update_trajectories(persons)
        
        if not self.trajectories:
            self.violence_score_history.append(0.0)
            return "Normal", 0.0

        # 2. Extract highest individual aggression indicators
        max_aggression = 0.0
        active_flags = set()
        for traj in self.trajectories.values():
            if traj.aggression_score > max_aggression:
                max_aggression = traj.aggression_score
            active_flags.update(traj.pose_flags)

        # 3. Multi-person interaction / proximity analysis (crucial for fighting)
        proximity_boost = self._evaluate_proximity_aggression(original_h, original_w)

        # Combine pose aggression + proximity boost
        combined_score = max_aggression + proximity_boost
        
        # Apply special posture boosts
        if "raised_fist" in active_flags and "punching_motion" in active_flags:
            combined_score += 0.15
        if "high_kick" in active_flags:
            combined_score += 0.10

        combined_score = min(combined_score, 1.0)
        self.violence_score_history.append(combined_score)

        # 4. Temporal sequence confidence smoothing (EMA + low-pass)
        smoothed_score = sum(self.violence_score_history) / len(self.violence_score_history)

        # 5. Resolve action label with smart filters to ignore casual walks/stretching
        action = "Normal"
        if smoothed_score > 0.65:
            if "high_kick" in active_flags:
                action = "Kicking"
            elif "punching_motion" in active_flags or "raised_fist" in active_flags:
                action = "Fighting"
            elif "falling" in active_flags:
                action = "Falling"
            else:
                action = "Aggressive Movement"
        elif smoothed_score > 0.40:
            if "falling" in active_flags:
                action = "Falling"
            else:
                action = "Suspicious"

        # Safe default if walk/casual stretch is likely
        if len(active_flags) == 0 or (len(active_flags) == 1 and "combat_stance" in active_flags and max_aggression < 0.35):
            action = "Normal"
            smoothed_score *= 0.5

        return action, round(smoothed_score, 3)

    def _track_and_update_trajectories(self, detected_persons: list[dict]):
        """Associate detected persons with existing trajectory histories via simple centroid distance."""
        fresh_trajectories = {}
        
        for idx, person in enumerate(detected_persons):
            kps = person.get("keypoints", [])
            if not kps:
                continue

            # Calculate centroid of new keypoints
            valid_pts = [(kp[0], kp[1]) for kp in kps if kp[2] > 0.4]
            if not valid_pts:
                continue
            cx = sum(pt[0] for pt in valid_pts) / len(valid_pts)
            cy = sum(pt[1] for pt in valid_pts) / len(valid_pts)

            # Match to nearest previous trajectory
            best_id = None
            min_dist = 0.18  # centroid distance tracking threshold
            
            for pid, traj in self.trajectories.items():
                if traj.centroid_history:
                    pcx, pcy = traj.centroid_history[-1]
                    dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = pid

            if best_id is not None and best_id not in fresh_trajectories:
                # Update existing trajectory
                traj = self.trajectories[best_id]
                traj.update(kps)
                fresh_trajectories[best_id] = traj
            else:
                # Create a new trajectory tracker
                new_id = int(time.time() * 1000) + idx
                traj = PersonTrajectory(new_id)
                traj.update(kps)
                fresh_trajectories[new_id] = traj

        # Prune dead trajectories older than current frame
        self.trajectories = fresh_trajectories

    def _evaluate_proximity_aggression(self, h: int, w: int) -> float:
        """Boost violence metric when multiple individuals are in highly interactive proximity."""
        t_keys = list(self.trajectories.keys())
        if len(t_keys) < 2:
            return 0.0

        proximity_boost = 0.0
        
        # Check pairwise distances between all active trajectories
        for i in range(len(t_keys)):
            for j in range(i + 1, len(t_keys)):
                traj_a = self.trajectories[t_keys[i]]
                traj_b = self.trajectories[t_keys[j]]
                
                if traj_a.centroid_history and traj_b.centroid_history:
                    cx_a, cy_a = traj_a.centroid_history[-1]
                    cx_b, cy_b = traj_b.centroid_history[-1]
                    
                    dist = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
                    
                    # Close physical proximity threshold (normalised unit space)
                    if dist < 0.22:
                        # Proximity + high body acceleration or active flags = fight confirmation!
                        combined_accel = traj_a.body_acceleration + traj_b.body_acceleration
                        active_pose_count = len(traj_a.pose_flags) + len(traj_b.pose_flags)
                        
                        if combined_accel > 0.8 or active_pose_count > 0:
                            proximity_boost = max(proximity_boost, 0.28)
                        else:
                            # Just close proximity with no aggressive signatures
                            proximity_boost = max(proximity_boost, 0.08)

        return proximity_boost
