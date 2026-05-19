import os
import cv2
import time
import queue
import logging
import threading
import subprocess
from collections import deque
from pathlib import Path

logger = logging.getLogger("suraksha.recording")

class RecordingManager:
    """
    Sub-system: Handles frame-by-frame buffering and writing.
    - Maintains a rolling pre-buffer (5 seconds) to capture pre-incident frames.
    - Automates OpenCV VideoWriter lifecycle on a background thread to prevent GUI lag.
    - Translates raw output into web-playable H264 standard via ffmpeg post-processing.
    """
    def __init__(self, output_dir: str = "recordings", fps: float = 20.0, pre_buffer_seconds: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre_buffer_size = int(fps * pre_buffer_seconds)
        self.pre_buffer = deque(maxlen=self.pre_buffer_size)
        
        self.is_recording = False
        self.writer = None
        self.temp_file = None
        self.active_file = None
        self.start_time = 0.0
        
        self._lock = threading.Lock()
        self.write_queue = queue.Queue()
        self._write_thread = None
        self._write_running = False

    def start(self):
        """Starts the background frame writing loop."""
        with self._lock:
            if self._write_running:
                return
            self._write_running = True
            self._write_thread = threading.Thread(target=self._async_writer_loop, daemon=True, name="recording-writer")
            self._write_thread.start()
            logger.info("[RecordingManager] Async writer service initialized.")

    def stop(self):
        """Stops the background writer and terminates any ongoing recordings."""
        self._write_running = False
        self.write_queue.put(None)
        if self._write_thread:
            self._write_thread.join(timeout=2)
        self.stop_recording()

    def add_frame(self, frame):
        """Adds a frame to either the pre-event buffer or the active file writer queue."""
        if frame is None:
            return
        with self._lock:
            if not self.is_recording:
                self.pre_buffer.append(frame.copy())
            else:
                try:
                    self.write_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Prevent FPS degradation under load

    def start_recording(self, width: int, height: int, camera_id: str) -> Path:
        """Starts a recording session and flushes pre-buffer frames into the writer queue."""
        with self._lock:
            if self.is_recording:
                return self.active_file
            
            self.is_recording = True
            self.start_time = time.time()
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            # Raw recording uses standard mp4v (fast CPU encoding)
            temp_filename = f"raw_EVD_{camera_id}_{timestamp_str}.mp4"
            # Final browser-compatible recording target
            final_filename = f"EVD_{camera_id}_{timestamp_str}.mp4"
            
            self.temp_file = self.output_dir / temp_filename
            self.active_file = self.output_dir / final_filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.temp_file), fourcc, self.fps, (width, height))
            
            logger.info(f"[RecordingManager] Dynamic recording started -> Raw temp file: {self.temp_file}")
            
            # Dump pre-buffer history to capture seconds preceding the trigger
            while self.pre_buffer:
                self.write_queue.put(self.pre_buffer.popleft())
            
            return self.active_file

    def stop_recording(self) -> tuple[Path, int]:
        """Stops active recording session and initiates H264 conversion in a background thread."""
        with self._lock:
            if not self.is_recording:
                return None, 0
            
            self.is_recording = False
            duration = int(time.time() - self.start_time)
            
            # Signal background writer thread to flush and release VideoWriter
            self.write_queue.put("FLUSH_AND_CLOSE")
            
            temp_path = self.temp_file
            final_path = self.active_file
            
            self.writer = None
            self.temp_file = None
            self.active_file = None
            self.pre_buffer.clear()
            
            # Delegate transcoding to an asynchronous thread so frames processing isn't stalled
            threading.Thread(
                target=self._convert_to_h264,
                args=(temp_path, final_path),
                daemon=True,
                name="h264-converter"
            ).start()
            
            return final_path, duration

    def _convert_to_h264(self, temp_path: Path, final_path: Path):
        """Uses system FFmpeg to transcode the raw file into browser-playable H264 format."""
        logger.info(f"[RecordingManager] Converting {temp_path.name} to web-compatible H264...")
        try:
            # -y overwrites, libx264 is H264 encoder, pix_fmt yuv420p for browser support
            # Set preset to 'ultrafast' to ensure conversion finishes extremely quickly on Mac
            cmd = [
                "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(temp_path),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-profile:v", "baseline", "-level", "3.0",
                str(final_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                logger.info(f"[RecordingManager] Transcode finished! H264 saved at {final_path.name}")
                # Remove temporary raw file to clean up disk space
                if temp_path.exists():
                    os.remove(temp_path)
            else:
                logger.error(f"[RecordingManager] FFmpeg transcoding failed: {result.stderr}")
                # Fallback: rename the raw file to final filename so the UI still has a video
                if temp_path.exists():
                    import shutil
                    shutil.copy2(temp_path, final_path)
        except Exception as e:
            logger.error(f"[RecordingManager] Transcoding error: {e}")
            if temp_path.exists():
                import shutil
                shutil.copy2(temp_path, final_path)

    def _async_writer_loop(self):
        """Reads frames from queue and writes them to the current OpenCV VideoWriter."""
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
                
                with self._lock:
                    current_writer = self.writer
                
                if current_writer:
                    active_writer = current_writer
                    active_writer.write(item)
                
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[RecordingManager] Writer thread exception: {e}")
                if active_writer:
                    active_writer.release()
                    active_writer = None
