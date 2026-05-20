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
        recoil_active: bool = False,
        fast_violence_mode: bool = False
    ) -> tuple[str, float]:
        """
        Combines parameters mathematically to generate a threat score (0.0 to 1.0)
        and map it to a classification level: LOW, MEDIUM, HIGH, CRITICAL.
        """
        # Centralized shared threat score (Task 4)
        combined_score = aggression_score
        
        # Boost if weapon is verified
        if weapon_detected and weapon_confidence > 0.40:
            combined_score = max(combined_score, 0.85)

        # 4. Resolve Threat Level (Task 5 Rebalanced Boundaries)
        score_pct = combined_score * 100.0
        tentative_level = "LOW"
        if score_pct >= 80.0:
            tentative_level = "CRITICAL"
        elif score_pct >= 55.0:
            tentative_level = "HIGH"
        elif score_pct >= 30.0:
            tentative_level = "MEDIUM"
        else:
            tentative_level = "LOW"

        # Apply persistence gating: require 3 consecutive frames of HIGH/CRITICAL before escalating
        # EXCEPT when rapid escalation conditions are met:
        # - repeated aggressive movement persists (repeated_strikes is True)
        # - motion intensity is extremely high (motion_intensity > 0.65)
        # - interactive punches/combat are active (punch_detected or aggression_score > 0.70)
        # - fast_violence_mode is active (Task 2 & 5)
        rapid_escalation = (
            repeated_strikes or 
            motion_intensity > 0.65 or 
            punch_detected or
            aggression_score > 0.70 or
            fast_violence_mode
        )

        if tentative_level in ["HIGH", "CRITICAL"]:
            self.high_threat_consecutive_count += 1
        else:
            self.high_threat_consecutive_count = 0

        if tentative_level in ["HIGH", "CRITICAL"] and self.high_threat_consecutive_count < 3 and not rapid_escalation:
            level = "MEDIUM"
        else:
            level = tentative_level

        # Perfect consistency adjustment: scale threat score to match threat level's exact range (Task 4)
        if level == "LOW":
            combined_score = min(0.29, combined_score)
        elif level == "MEDIUM":
            if combined_score >= 0.55:
                combined_score = 0.54
            elif combined_score < 0.30:
                combined_score = 0.30
        elif level == "HIGH":
            if combined_score >= 0.80:
                combined_score = 0.79
            elif combined_score < 0.55:
                combined_score = 0.55
        elif level == "CRITICAL":
            if combined_score < 0.80:
                combined_score = 0.80

        return level, round(combined_score, 3)
