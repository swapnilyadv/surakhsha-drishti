"""
Threat Engine — Suraksha Drishti AI
===================================
Calculates dynamic surveillance threat levels (LOW, MEDIUM, HIGH, CRITICAL)
by combining raw AI predictions, crowd parameters, and temporal event detections.
"""

import logging

logger = logging.getLogger("suraksha.threat_engine")

class ThreatEngine:
    """
    Stateful threat scoring coordinator that aggregates real-time
    intelligence metrics and flags emergency escalations.
    """
    def __init__(self):
        self.high_threat_consecutive_count = 0

    def calculate_threat(
        self,
        aggression_score: float,
        motion_intensity: float,
        person_count: int,
        weapon_detected: bool,
        weapon_confidence: float,
        repeated_strikes: bool = False,
        fall_detected: bool = False,
        chasing_detected: bool = False,
        punch_detected: bool = False,
        recoil_active: bool = False
    ) -> tuple[str, float]:
        """
        Combines parameters mathematically to generate a threat score (0.0 to 1.0)
        and map it to a classification level: LOW, MEDIUM, HIGH, CRITICAL.
        """
        # 1. Normalize crowd density component (max reference = 6 persons for high density)
        crowd_density = min(1.0, person_count / 6.0)

        # 2. Base weighted combination (Task 6 rebalanced weighting)
        # - Aggression score has 50% weight (proximity and collision persistence)
        # - Weapon presence has 30% weight
        # - Spatial motion intensity has 10% weight
        # - Crowd factor has 10% weight
        base_score = (aggression_score * 0.50) + \
                     (weapon_confidence * 0.30 if weapon_detected else 0.0) + \
                     (motion_intensity * 0.10) + \
                     (crowd_density * 0.10)

        # 3. Dynamic context-aware boosts (Task 6 rebalanced weightings)
        boost = 0.0
        if repeated_strikes:
            boost += 0.22  # Increased repeated strikes weight
        if fall_detected:
            boost += 0.10
        if chasing_detected:
            boost += 0.12
        if punch_detected:
            boost += 0.20  # Proximity combat/collision persistence boost
        if recoil_active:
            boost += 0.15  # Recoil confirmation displacement boost

        # Combined score capped between 0.0 and 1.0
        combined_score = min(1.0, max(0.0, base_score + boost))

        # 4. Resolve Threat Level (Task 4 Rebalanced Boundaries)
        tentative_level = "LOW"
        if combined_score >= 0.75 or (weapon_detected and aggression_score > 0.45):
            tentative_level = "CRITICAL"
        elif combined_score >= 0.55:
            tentative_level = "HIGH"
        elif combined_score >= 0.30:
            tentative_level = "MEDIUM"
        else:
            tentative_level = "LOW"

        # Apply persistence gating: require 3 consecutive frames of HIGH/CRITICAL before escalating
        # EXCEPT when rapid escalation conditions are met:
        # - repeated aggressive movement persists (repeated_strikes is True)
        # - motion intensity is extremely high (motion_intensity > 0.65)
        # - interactive punches/combat are active (punch_detected or aggression_score > 0.70)
        rapid_escalation = (
            repeated_strikes or 
            motion_intensity > 0.65 or 
            punch_detected or
            aggression_score > 0.70
        )

        if tentative_level in ["HIGH", "CRITICAL"]:
            self.high_threat_consecutive_count += 1
        else:
            self.high_threat_consecutive_count = 0

        if tentative_level in ["HIGH", "CRITICAL"] and self.high_threat_consecutive_count < 3 and not rapid_escalation:
            level = "MEDIUM"
        else:
            level = tentative_level

        return level, round(combined_score, 3)
