"""
Event Manager — Suraksha Drishti AI
==================================
Logs dynamic event timelines (Falls, Chases, Strikes) and updates live lists
for WebSocket broadcasting and frontend activity mapping.
"""

import time
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger("suraksha.event_manager")

class EventManager:
    """
    Manages active event timeline queues, logs historical incidents,
    and formats visual alert streams for connected operators.
    """
    def __init__(self, max_history: int = 50):
        self.timeline: deque[dict] = deque(maxlen=max_history)
        self._lock = []
        
    def log_event(self, camera_id: str, camera_label: str, event_type: str, severity: str, details: str):
        """Creates a visual historical timeline entry."""
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        iso_time = datetime.now().toISOString() if hasattr(datetime.now(), "toISOString") else datetime.now().isoformat()
        
        event_entry = {
            "id": f"EVT-{int(time.time() * 1000)}",
            "cameraId": camera_id,
            "cameraLabel": camera_label,
            "timestamp": timestamp,
            "isoTime": iso_time,
            "type": event_type,
            "severity": severity,
            "details": details,
        }
        self.timeline.append(event_entry)
        logger.info(f"[EventManager] Visual event logged: [{severity}] {event_type} on {camera_label} — {details}")
        return event_entry

    def get_recent_events(self) -> list[dict]:
        """Returns visual event lists ordered by newest first."""
        return list(self.timeline)[::-1]
