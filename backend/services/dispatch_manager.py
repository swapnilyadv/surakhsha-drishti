import time
import logging
import threading

logger = logging.getLogger("suraksha.dispatch")

class DispatchManager:
    """
    Sub-system: Tracks police response status across connected terminals.
    - Synchronizes dispatch assignments and blocks double-dispatch collisions.
    - Captures emergency backup escalation states.
    - Interfaces with websocket channels to broadcast status changes.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.dispatches = {}    # maps evidence_id -> dispatch details
        self.escalations = {}   # maps evidence_id -> escalation details

    def accept_dispatch(self, evidence_id: str, station: str, officer: str) -> dict | None:
        """
        Locks in a police station's dispatch commitment.
        Prevents multiple stations from accepting the same emergency.
        """
        with self._lock:
            if evidence_id in self.dispatches:
                logger.warning(f"[Dispatch] Rejected double dispatch claim on {evidence_id} by {station}")
                return None
            
            dispatch_info = {
                "event": "dispatch_accepted",
                "evidence_id": evidence_id,
                "station": station,
                "officer": officer,
                "status": "POLICE_DISPATCHED",
                "timestamp": time.time()
            }
            self.dispatches[evidence_id] = dispatch_info
            logger.info(f"[Dispatch] Verified dispatch accepted by {station} (Officer {officer}) for {evidence_id}")
            return dispatch_info

    def escalate(self, evidence_id: str, station: str, lat: float, lng: float) -> dict:
        """Logs an emergency backup escalation request."""
        with self._lock:
            escalation_info = {
                "event": "need_more_help",
                "evidence_id": evidence_id,
                "station": station,
                "location": {
                    "lat": lat,
                    "lng": lng
                },
                "status": "More Help Requested",
                "timestamp": time.time()
            }
            self.escalations[evidence_id] = escalation_info
            logger.info(f"[Dispatch] Escalation request verified for {evidence_id} by {station}")
            return escalation_info

    def get_dispatch_state(self, evidence_id: str) -> dict | None:
        with self._lock:
            return self.dispatches.get(evidence_id)

    def get_escalation_state(self, evidence_id: str) -> dict | None:
        with self._lock:
            return self.escalations.get(evidence_id)

    def reset_incident(self, evidence_id: str):
        """Removes dispatch and escalation tracking for resolved incidents."""
        with self._lock:
            self.dispatches.pop(evidence_id, None)
            self.escalations.pop(evidence_id, None)
            logger.info(f"[Dispatch] Cleared dispatch and escalation status for resolved incident: {evidence_id}")
