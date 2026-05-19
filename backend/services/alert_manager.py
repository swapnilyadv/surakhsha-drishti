import time
import logging

logger = logging.getLogger("suraksha.alert")

class AlertManager:
    """
    Intelligent Alert & Global Alarm Manager.
    - Manages live threat states, alarm audio activations, and acknowledgment handshakes.
    - User Acknowledge mutes the audio locally, but AI stays active.
    - Allows manual emergency override triggers to force alarm reactivations globally.
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
                logger.info(f"[Alert] New threat event triggered: {threat_type} ({confidence:.2f})")
                self.acknowledged = False
            
            # Alarm remains active unless the user explicitly muted it
            self.alarm_active = not self.acknowledged
        else:
            # Do not force reset if it was triggered via manual emergency escalation!
            if self.threat_type != "EMERGENCY_BACKUP":
                self.alarm_active = False

        self.last_violence_state = violence_detected

    def trigger_emergency(self, threat_label: str = "EMERGENCY_BACKUP"):
        """
        Force triggers the global alarm. Used for Emergency Escalation.
        Resets acknowledgement so that all stations hear the sound again.
        """
        logger.info(f"[Alert] Manual global alarm force-triggered: {threat_label}")
        self.acknowledged = False
        self.alarm_active = True
        self.threat_type = threat_label
        self.violence_detected = True

    def mute_alarm_globally(self):
        """Mutes the active alarm globally. Used when dispatch is accepted."""
        logger.info("[Alert] Alarm muted globally.")
        self.alarm_active = False
        self.acknowledged = True

    def acknowledge(self):
        """
        Acknowledges active alarm locally.
        - Mutes the audio alert immediately (sets alarm_active to False).
        - Prevents alarm from re-triggering until a brand-new violence event is identified.
        """
        if self.violence_detected or self.threat_type == "EMERGENCY_BACKUP":
            logger.info("[Alert] Alarm acknowledged locally. Muting audio alert.")
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
