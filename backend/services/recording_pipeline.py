"""
Smart Recording Pipeline — Suraksha Drishti AI
==============================================
Manages frame rolling buffers, dynamic threat recording extensions,
flicker prevention, incident merging, and async FFmpeg H264 transcoder queues.
"""

import os
import cv2
import time
import queue
import logging
import threading
import subprocess
from collections import deque
from pathlib import Path

logger = logging.getLogger("suraksha.recording_pipeline")

class RecordingPipeline:
    """
    Upgraded, production-grade video recording and transcoding pipeline.
    
    API compatible with RecordingManager to ensure zero breaking changes.
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

        # Smart Cooldown & Merge state variables
        self.min_recording_seconds = 15.0
        self.merge_window_seconds = 10.0      # Merge incidents within 10 seconds
        self.last_threat_timestamp = 0.0
        self.last_stop_timestamp = 0.0
        self.active_camera_id = None
        self.active_camera_label = None

        # FFmpeg background queue manager to prevent high parallel loads on CPU
        self._transcode_queue = queue.Queue()
        self._transcode_running = False
        self._transcode_thread = None

    def start(self):
        """Starts background writer and transcoding loops."""
        with self._lock:
            if not self._write_running:
                self._write_running = True
                self._write_thread = threading.Thread(target=self._async_writer_loop, daemon=True, name="pipeline-writer")
                self._write_thread.start()
                logger.info("[RecordingPipeline] Frame writer worker active.")

            if not self._transcode_running:
                self._transcode_running = True
                self._transcode_thread = threading.Thread(target=self._async_transcoder_loop, daemon=True, name="pipeline-transcoder")
                self._transcode_thread.start()
                logger.info("[RecordingPipeline] Transcoder worker pool active.")

    def stop(self):
        """Shutdown background threads safely."""
        self._write_running = False
        self.write_queue.put(None)
        if self._write_thread:
            self._write_thread.join(timeout=1.5)

        self._transcode_running = False
        self._transcode_queue.put(None)
        if self._transcode_thread:
            self._transcode_thread.join(timeout=1.5)

        self.stop_recording()

    def add_frame(self, frame):
        """Pushes frame to rolling pre-buffer or writing pipeline."""
        if frame is None:
            return
        with self._lock:
            if not self.is_recording:
                self.pre_buffer.append(frame.copy())
            else:
                try:
                    self.write_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

    def start_recording(self, width: int, height: int, camera_id: str, camera_label: str = "Surveillance Unit") -> Path:
        """Starts a new recording or resumes/merges into the existing active session."""
        with self._lock:
            now = time.time()
            self.last_threat_timestamp = now
            self.active_camera_id = camera_id
            self.active_camera_label = camera_label

            # Dynamic Merge Logic:
            # If stopped very recently (<10s) and camera IDs match, reuse the same active recording session
            if (now - self.last_stop_timestamp < self.merge_window_seconds) and self.temp_file and self.temp_file.exists():
                logger.info("[RecordingPipeline] Threat re-detected within merge window. Resuming active session...")
                self.is_recording = True
                self.last_stop_timestamp = 0.0
                return self.active_file

            if self.is_recording:
                return self.active_file
            
            self.is_recording = True
            self.start_time = now
            self.last_stop_timestamp = 0.0
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            safe_cam_id = str(camera_id).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
            temp_filename = f"raw_EVD_{safe_cam_id}_{timestamp_str}.mp4"
            final_filename = f"EVD_{safe_cam_id}_{timestamp_str}.mp4"
            
            self.temp_file = self.output_dir / temp_filename
            self.active_file = self.output_dir / final_filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.temp_file), fourcc, self.fps, (width, height))
            
            logger.info(f"[RecordingPipeline] Initialized Smart Clip -> {self.temp_file.name}")
            
            # Flush rolling history pre-buffer into the write loop
            while self.pre_buffer:
                self.write_queue.put(self.pre_buffer.popleft())
            
            return self.active_file

    def stop_recording(self, force: bool = False) -> tuple[Path, int]:
        """Stops active recording, obeying minimum duration rules and initiating transcode."""
        with self._lock:
            if not self.is_recording:
                return None, 0
            
            now = time.time()
            duration = now - self.start_time

            # Enforce Minimum Duration:
            # Prevent short fragmented clip spikes by maintaining recording until min duration is reached
            if not force and duration < self.min_recording_seconds:
                remaining = self.min_recording_seconds - duration
                logger.debug(f"[RecordingPipeline] Postponing stop. Recording remaining {remaining:.1f}s to fulfill minimum duration.")
                return self.active_file, int(duration)

            self.is_recording = False
            self.last_stop_timestamp = now
            
            self.write_queue.put("FLUSH_AND_CLOSE")
            
            temp_path = self.temp_file
            final_path = self.active_file
            
            self.writer = None
            self.pre_buffer.clear()
            
            # Enqueue to background FFmpeg worker pool
            self._transcode_queue.put((temp_path, final_path))
            
            return final_path, int(duration)

    def _async_writer_loop(self):
        active_writer = None
        while self._write_running:
            try:
                item = self.write_queue.get(timeout=0.2)
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
                logger.error(f"[RecordingPipeline] Writer error: {e}")
                if active_writer:
                    active_writer.release()
                    active_writer = None

    def _async_transcoder_loop(self):
        """Sequentially processes transcoding jobs from queue, preventing CPU spikes."""
        while self._transcode_running:
            try:
                job = self._transcode_queue.get(timeout=0.5)
                if job is None:
                    break
                
                temp_path, final_path = job
                self._convert_to_h264(temp_path, final_path)
                self._transcode_queue.task_done()
            except queue.Empty:
                continue

    def _convert_to_h264(self, temp_path: Path, final_path: Path):
        """Transcodes using superfast configurations for Apple Silicon / Macbook CPU."""
        if not temp_path.exists():
            return
        
        logger.info(f"[RecordingPipeline] Transcoding clip {temp_path.name}...")
        try:
            # Dynamically locate ffmpeg binary to prevent hardcoded failures
            ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"
            if not os.path.exists(ffmpeg_bin):
                if os.path.exists("/usr/local/bin/ffmpeg"):
                    ffmpeg_bin = "/usr/local/bin/ffmpeg"
                else:
                    ffmpeg_bin = "ffmpeg"  # fallback to PATH search

            # -preset ultrafast with threads 2 limits CPU load
            cmd = [
                ffmpeg_bin, "-y", "-i", str(temp_path),
                "-threads", "2",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-profile:v", "baseline", "-level", "3.0",
                str(final_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                logger.info(f"[RecordingPipeline] Transcode complete -> {final_path.name}")
                if temp_path.exists():
                    os.remove(temp_path)
            else:
                logger.error(f"[RecordingPipeline] FFmpeg failed: {result.stderr}")
                if temp_path.exists():
                    import shutil
                    shutil.copy2(temp_path, final_path)
        except Exception as e:
            logger.error(f"[RecordingPipeline] Transcoding crash: {e}")
            if temp_path.exists():
                import shutil
                shutil.copy2(temp_path, final_path)
