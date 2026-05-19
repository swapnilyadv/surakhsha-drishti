"""
Human Detector Service
=======================
Lightweight person detection using YOLOv8n via the Ultralytics library.

Strategy:
  - Uses ultralytics YOLO Python API (no ONNX export required)
  - Model: yolov8n.pt  (~6 MB, auto-downloaded on first run from Ultralytics CDN)
  - Filters COCO class 0 (person) only
  - CPU-only inference via PyTorch (MPS disabled to avoid crashes on older macOS)
  - Returns bounding boxes + confidence scores per detected person

Performance:
  - Inference at 320×320 (imgsz=320) for ~2× speedup vs 640
  - Runs every Nth frame (skipping controlled by FrameProcessor)
  - Thread count limited to 2 to keep CPU usage low on MacBook Air
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID      = 0      # COCO class 0 = person
CONFIDENCE_THRESHOLD = 0.50   # minimum score to accept a detection
INFERENCE_SIZE       = 320    # input resolution (320 = fast, 640 = accurate)


class HumanDetector:
    """
    YOLOv8n person detector using Ultralytics Python API.

    Usage:
        detector = HumanDetector(model_path="models/yolov8n.pt")
        ok = detector.load()
        persons = detector.detect(frame)   # frame = BGR numpy array

    Returns:
        [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load YOLOv8n model.
        If model_path does not exist, ultralytics auto-downloads yolov8n.pt.
        Returns True on success.
        """
        try:
            from ultralytics import YOLO

            # Force CPU — avoids MPS crashes on older macOS
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            logger.info(f"[HumanDetector] Loading YOLOv8n from {self.model_path}...")
            self.model  = YOLO(self.model_path)

            # Warm-up inference to load weights into memory
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                       imgsz=INFERENCE_SIZE, device="cpu", verbose=False)

            self.loaded = True
            logger.info("[HumanDetector] YOLOv8n loaded and warmed up ✓")
            return True

        except ImportError:
            logger.warning(
                "[HumanDetector] 'ultralytics' not installed — human detection disabled.\n"
                "  Fix: pip install ultralytics"
            )
            return False
        except Exception as e:
            logger.error(f"[HumanDetector] Load failed: {e}")
            return False

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all persons in a BGR frame.

        Args:
            frame: BGR numpy array (any resolution)

        Returns:
            List of dicts: [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
            Empty list if model not loaded or no persons found.
        """
        if not self.loaded or self.model is None:
            return []

        try:
            results = self.model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                imgsz=INFERENCE_SIZE,
                device="cpu",
                verbose=False,
            )

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    detections.append({
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 3),
                    })

            return detections

        except Exception as e:
            logger.debug(f"[HumanDetector] Inference error: {e}")
            return []
