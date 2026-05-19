import time
import logging

logger = logging.getLogger("suraksha.alert")

class AlertManager:
    """
    Intelligent Alert & Alarm Manager.
    - Manages live threat states, alarm audio activations, and acknowledgment handshakes.
    - User Acknowledge stops the alarm sound ONLY, allowing the AI to keep recording safely.
    - Automatically resets acknowledgment state when a brand new violence event starts.
    """
    def __init__(self):
        self.violence_detected = False
        self.threat_type = "NOMINAL"
        self.confidence = 0.0
        
        # Alarm states
        self.alarm_active = False
        self.acknowledged = False
        self.last_violence_state = False

    def update_threat_state(self, violence_detected: bool, threat_type: str = "Fighting", confidence: float = 0.0):
        """
        Processes AI threat detection outputs and updates alarms.
        - Triggers active alarms immediately on new threat.
        - Resets the acknowledge flag on NEW distinct threat events.
        """
        self.violence_detected = violence_detected
        self.threat_type = threat_type if violence_detected else "NOMINAL"
        self.confidence = confidence if violence_detected else 0.0

        if violence_detected:
            # If threat was previously offline, this is a NEW threat event -> reset acknowledge!
            if not self.last_violence_state:
                logger.info(f"[AlertManager] New threat event triggered: {threat_type} ({confidence:.2f})")
                self.acknowledged = False
            
            # Alarm remains active unless the user explicitly muted it
            self.alarm_active = not self.acknowledged
        else:
            self.alarm_active = False

        self.last_violence_state = violence_detected

    def acknowledge(self):
        """
        Acknowledges active alarm.
        - Mutes the audio alert immediately (sets alarm_active to False).
        - Prevents alarm from re-triggering until a brand-new violence event is identified.
        """
        if self.violence_detected:
            logger.info("[AlertManager] Alarm acknowledged by user. Muting audio alert.")
            self.acknowledged = True
            self.alarm_active = False
            return True
        return False

    def get_state(self) -> dict:
        """Returns the serialized alert state for websocket/HTTP endpoints."""
        return {
            "violence":       self.violence_detected,
            "alarm_active":   self.alarm_active,
            "acknowledged":   self.acknowledged,
            "event":          self.threat_type,
            "confidence":     self.confidence,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S")
        }
