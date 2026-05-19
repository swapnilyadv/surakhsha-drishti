import os
import cv2
import time
import queue
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("suraksha.evidence")

class EvidenceManager:
    """
    Thread-safe Evidence Recording System.
    - Maintains a rolling pre-buffer (5-10 seconds) to capture pre-event trigger history.
    - Automatically manages VideoWriter lifecycles.
    - Guarantees minimum 15-second recordings.
    - Implements temporal post-buffering to prevent rapid start/stop flickering.
    - Exposes recorded clips via static endpoints and logs history.
    """
    def __init__(self, output_dir: str = "recordings", fps: float = 20.0, pre_buffer_seconds: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        
        # Max frames in rolling pre-buffer
        self.pre_buffer_size = int(fps * pre_buffer_seconds)
        self.pre_buffer = deque(maxlen=self.pre_buffer_size)
        
        # Recording state
        self.is_recording = False
        self.writer = None
        self.active_file = None
        self.start_time = 0.0
        self.last_threat_time = 0.0
        
        # Event details
        self.current_event = None
        self.camera_label = "Webcam Unit"
        self.camera_id = "CAM-01"
        self.threat_type = "VIOLENCE"
        
        # History store
        self.history = []
        self._lock = threading.Lock()
        
        # Async writing queue to prevent FPS drops
        self.write_queue = queue.Queue()
        self._write_thread = None
        self._write_running = False

    def start(self):
        """Starts the async background writing thread."""
        with self._lock:
            if self._write_running:
                return
            self._write_running = True
            self._write_thread = threading.Thread(target=self._async_writer_loop, daemon=True, name="evidence-writer")
            self._write_thread.start()
            logger.info("[EvidenceManager] Background writer thread started.")

    def stop(self):
        """Stops the manager and flushes any active recordings."""
        self._write_running = False
        self.write_queue.put(None)
        if self._write_thread:
            self._write_thread.join(timeout=2)
        self.stop_recording()

    def add_frame(self, frame, is_threat: bool, threat_type: str = "VIOLENCE", camera_id: str = "CAM-01", camera_label: str = "Webcam Unit"):
        """
        Receives every video frame from the FrameProcessor.
        - Updates the pre-buffer.
        - Triggers or extends recording on active threat.
        - Throws frames into the async writer queue if active.
        """
        if frame is None:
            return

        with self._lock:
            self.camera_id = camera_id
            self.camera_label = camera_label
            
            if is_threat:
                self.last_threat_time = time.time()
                self.threat_type = threat_type

            # Add to pre-buffer if not currently recording
            if not self.is_recording:
                self.pre_buffer.append(frame.copy())
                
                # Threat detected — start recording!
                if is_threat:
                    self._start_recording_locked(frame.shape[1], frame.shape[0])
            else:
                # Active recording — queue the frame for writing
                try:
                    self.write_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass # prevent block if queue overflowing

    def _start_recording_locked(self, width: int, height: int):
        """Initializes VideoWriter and dumps the pre-buffer frames into the write queue."""
        self.is_recording = True
        self.start_time = time.time()
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"EVD_{self.camera_id}_{timestamp_str}.mp4"
        self.active_file = self.output_dir / filename
        
        # H264 fallback to standard MP4V
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.active_file), fourcc, self.fps, (width, height))
        
        logger.info(f"[EvidenceManager] Triggered auto-recording! Output: {self.active_file}")
        
        # Dump pre-buffer history into the queue to capture context BEFORE the fight
        while self.pre_buffer:
            pre_frame = self.pre_buffer.popleft()
            self.write_queue.put(pre_frame)

    def update_lifecycle(self):
        """
        Lifecycle manager: checks if recording should stop.
        - Must be called regularly from a timer or processor thread.
        - Ensures minimum 15-second clip duration.
        - Uses 5-second post-violence temporal buffer.
        """
        with self._lock:
            if not self.is_recording:
                return None
            
            now = time.time()
            elapsed = now - self.start_time
            time_since_threat = now - self.last_threat_time
            
            # Stop ONLY when minimum 15 seconds passed AND threat has ceased for at least 5 seconds
            if elapsed >= 15.0 and time_since_threat >= 5.0:
                return self._stop_recording_locked()
        return None

    def stop_recording(self):
        """Public stop override."""
        with self._lock:
            if self.is_recording:
                return self._stop_recording_locked()
        return None

    def _stop_recording_locked(self):
        """Internal worker to release resources and save evidence metadata."""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        duration = int(time.time() - self.start_time)
        
        # Signal queue to write any remaining frames
        # We don't close the writer immediately to let the background thread finish writing frames in queue
        self.write_queue.put("FLUSH_AND_CLOSE")
        
        timestamp_label = datetime.now().strftime("%I:%M %p")
        iso_str = datetime.now().toISOString() if hasattr(datetime.now(), 'toISOString') else datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        ev_id = f"EVD-REC-{int(time.time())}"
        
        evidence_item = {
            "id": ev_id,
            "cameraId": self.camera_id,
            "cameraLabel": self.camera_label,
            "timestamp": timestamp_label,
            "isoTime": iso_str,
            "confidence": 0.88,
            "type": self.threat_type,
            "videoUrl": f"/recordings/{self.active_file.name}",
            "status": "Active",
            "duration": f"{duration} sec",
            "locationName": "System CCTV Feed"
        }
        
        self.history.append(evidence_item)
        logger.info(f"[EvidenceManager] Auto-recording stopped! Saved {duration}s clip: {self.active_file.name}")
        
        # Reset state
        self.writer = None
        self.active_file = None
        self.pre_buffer.clear()
        
        return evidence_item

    def _async_writer_loop(self):
        """Worker loop reading frames from queue and writing to disk."""
        active_writer = None
        
        while self._write_running:
            try:
                item = self.write_queue.get(timeout=0.5)
                if item is None:
                    break
                    
                if isinstance(item, str) and item == "FLUSH_AND_CLOSE":
                    if active_writer:
                        active_writer.release()
                        active_writer = None
                    self.write_queue.task_done()
                    continue
                
                # Fetch writer dynamically
                with self._lock:
                    current_writer = self.writer
                
                if current_writer:
                    active_writer = current_writer
                    active_writer.write(item)
                    
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[EvidenceManager] Writer thread error: {e}")
                if active_writer:
                    active_writer.release()
                    active_writer = None

        if active_writer:
            active_writer.release()
