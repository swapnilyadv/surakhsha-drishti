import os
import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from services.recording_pipeline import RecordingPipeline

logger = logging.getLogger("suraksha.evidence")

class EvidenceManager:
    """
    Evidence Vault & Persistent Database Manager.
    - Delegates core frame capture and asynchronous H264 conversion to RecordingPipeline.
    - Synchronizes incident metadata in evidence_db.json.
    - Tracks male/female counts, weapon presence, location coordinates, and alerts.
    """
    def __init__(self, output_dir: str = "recordings", fps: float = 20.0, pre_buffer_seconds: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.output_dir / "evidence_db.json"
        
        # Lower-level rolling queue and OpenCV writer sub-system
        self.recording_manager = RecordingPipeline(output_dir, fps, pre_buffer_seconds)
        
        self.history = []
        self._lock = threading.Lock()
        
        # Accumulators for active recording metrics
        self.active_camera_id = "CAM-01"
        self.active_camera_label = "Webcam Unit"
        self.active_threat_type = "VIOLENCE"
        
        self.active_max_male = 0
        self.active_max_female = 0
        self.active_weapon_detected = False
        self.active_weapon_type = "None"
        self.active_max_confidence = 0.0
        self.active_lat = 19.0760
        self.active_lng = 72.8777
        
        self.start_time = 0.0
        self.last_threat_time = 0.0
        
        self._load_database()

    @property
    def is_recording(self) -> bool:
        return self.recording_manager.is_recording

    def start(self):
        """Starts the underlying recording writer thread."""
        self.recording_manager.start()

    def stop(self):
        """Stops the underlying recording writer thread."""
        self.recording_manager.stop()

    def _load_database(self):
        """Loads evidence records from the persistent JSON file."""
        with self._lock:
            if self.db_path.exists():
                try:
                    with open(self.db_path, "r") as f:
                        self.history = json.load(f)
                    logger.info(f"[Evidence] Loaded {len(self.history)} records from persistent database.")
                except Exception as e:
                    logger.error(f"[Evidence] Database load error: {e}")
                    self.history = []
            else:
                self.history = []
                self._save_database_locked()

    def _save_database_locked(self):
        """Persists evidence records to the JSON file."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.debug("[Evidence] Database saved successfully.")
        except Exception as e:
            logger.error(f"[Evidence] Database save error: {e}")

    def add_frame(self, frame, is_threat: bool, threat_type: str = "VIOLENCE", camera_id: str = "CAM-01", camera_label: str = "Webcam Unit", male_count: int = 0, female_count: int = 0, weapon_detected: bool = False, weapon_type: str = "None", confidence: float = 0.0, lat: float = 19.0760, lng: float = 72.8777):
        """
        Accepts frame inputs from pipeline.
        - Pipelines frames to rolling buffer.
        - Triggers active recording session on threat start.
        - Accumulates peak AI metrics (genders, weapons, confidence) over the session.
        """
        # Pipe frame to VideoWriter or buffer
        self.recording_manager.add_frame(frame)
        
        with self._lock:
            if is_threat:
                self.last_threat_time = time.time()
                self.active_threat_type = threat_type
                
                # Accumulate high-water stats
                self.active_max_male = max(self.active_max_male, male_count)
                self.active_max_female = max(self.active_max_female, female_count)
                if weapon_detected:
                    self.active_weapon_detected = True
                    self.active_weapon_type = weapon_type
                self.active_max_confidence = max(self.active_max_confidence, confidence)

            # Handle automatic trigger start on brand-new threat
            if not self.is_recording and is_threat:
                self.active_camera_id = camera_id
                self.active_camera_label = camera_label
                self.active_lat = lat
                self.active_lng = lng
                self.start_time = time.time()
                
                # Reset peaks
                self.active_max_male = male_count
                self.active_max_female = female_count
                self.active_weapon_detected = weapon_detected
                self.active_weapon_type = weapon_type
                self.active_max_confidence = confidence
                
                h, w = frame.shape[:2]
                self.recording_manager.start_recording(w, h, camera_id)
                logger.info(f"[Evidence] Threat trigger registered. Active recording initiated for camera {camera_id} at ({lat}, {lng}).")

    def update_lifecycle(self) -> dict | None:
        """
        Manages the recording duration lifecycle.
        - Keeps recording active for at least 15 seconds.
        - Continues automatically if threat remains active.
        - Employs a 5-second post-event cooling buffer before stopping.
        """
        with self._lock:
            if not self.is_recording:
                return None
            
            now = time.time()
            elapsed = now - self.start_time
            time_since_threat = now - self.last_threat_time
            
            # Stop ONLY when min 15s duration met AND threat has stopped for at least 5s
            if elapsed >= 15.0 and time_since_threat >= 5.0:
                logger.info(f"[Evidence] Stopping recording session. Total duration: {elapsed:.1f}s")
                return self._stop_recording_locked()
        return None

    def stop_recording(self) -> dict | None:
        """Manually forces a stop of the active recording."""
        with self._lock:
            if self.is_recording:
                return self._stop_recording_locked()
        return None

    def _stop_recording_locked(self) -> dict | None:
        """Stops the VideoWriter and constructs the persistent database record."""
        if not self.is_recording:
            return None
        
        final_file, duration = self.recording_manager.stop_recording()
        if not final_file:
            return None
        
        ev_id = f"EVD-REC-{int(time.time())}"
        timestamp_label = datetime.now().strftime("%I:%M %p")
        iso_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        evidence_item = {
            "id": ev_id,
            "cameraId": self.active_camera_id,
            "cameraLabel": self.active_camera_label,
            "timestamp": timestamp_label,
            "isoTime": iso_str,
            "confidence": float(round(self.active_max_confidence if self.active_max_confidence > 0 else 0.88, 2)),
            "type": self.active_threat_type,
            "videoUrl": f"/recordings/{final_file.name}",
            "status": "Active",
            "duration": f"{duration} sec",
            "locationName": self.active_camera_label,
            "lat": float(self.active_lat),
            "lng": float(self.active_lng),
            "maleCount": int(self.active_max_male),
            "femaleCount": int(self.active_max_female),
            "weaponDetected": bool(self.active_weapon_detected),
            "weaponType": str(self.active_weapon_type) if self.active_weapon_detected else "None"
        }
        
        self.history.insert(0, evidence_item)
        self._save_database_locked()
        
        logger.info(f"[Evidence] Incident recorded and committed to database: {ev_id}")
        return evidence_item

    def update_evidence_status(self, evidence_id: str, updates: dict) -> dict | None:
        """Updates and persists fields on an existing evidence entry."""
        with self._lock:
            for item in self.history:
                if item["id"] == evidence_id:
                    item.update(updates)
                    self._save_database_locked()
                    logger.info(f"[Evidence] Record {evidence_id} updated: {updates}")
                    return item
            logger.warning(f"[Evidence] Record {evidence_id} not found in database.")
            return None

    def get_all_evidence(self) -> list:
        """Returns all evidence entries from the persistent database."""
        with self._lock:
            return list(self.history)

    def clear_all(self):
        """Clears the persistent evidence database."""
        with self._lock:
            self.history = []
            self._save_database_locked()
            logger.info("[Evidence] Database cleared.")
