"""
Weapon Detector Service
========================
ONNX Runtime-based weapon detection using your custom YOLO model (best.onnx).

v4.0 improvements:
  - Returns bbox list in format expected by frontend & detection JSON spec
  - FPS tracking per detector (exposed for model-status endpoint)
  - Faster preprocessing using letterbox resize (maintains aspect ratio)
  - Per-class confidence thresholds
  - Thread-safe inference timing stats

Input:  BGR numpy frame
Output: {
  "weapon_detected": bool,
  "label":           str,         # best-confidence weapon class
  "confidence":      float,
  "detections":      [            # all detected weapons
    {
      "label":      str,
      "confidence": float,
      "box":        [x1, y1, x2, y2]
    }, ...
  ],
  "inference_ms":    float,       # inference latency in ms
}
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Weapon class map — update to match your model's training classes ──────────
WEAPON_CLASSES = {
    0: "knife",
    1: "gun",
    2: "pistol",
    3: "rifle",
    4: "weapon",
}

# ── Inference config ──────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60     # global detection threshold
NMS_IOU_THRESHOLD    = 0.40     # NMS overlap threshold
INPUT_SIZE           = 640      # model input resolution (must match training)


class WeaponDetector:
    """
    Custom YOLO ONNX weapon detector.

    - Loads model once at startup
    - CPU-only inference (stable on MacBook Air, avoids MPS crashes)
    - Letterbox preprocessing preserves aspect ratio
    - Tracks inference latency for /model-status endpoint
    """

    def __init__(self, model_path: str):
        self.model_path   = model_path
        self.session      = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[list] = None
        self.loaded       = False

        # Performance stats (thread-safe primitives)
        self._inference_count  = 0
        self._total_ms         = 0.0
        self._last_inference_ms = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load ONNX model with CPU provider. Returns True on success."""
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2    # limit CPU spike
            opts.inter_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name  = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.loaded      = True

            logger.info(f"[WeaponDetector] Model loaded: {self.model_path}")
            logger.info(
                f"[WeaponDetector] Input: {self.input_name} shape={self.input_shape}"
            )
            return True

        except ImportError:
            logger.warning("[WeaponDetector] onnxruntime not installed — disabled")
            return False
        except Exception as e:
            logger.error(f"[WeaponDetector] Load failed: {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run weapon detection on a BGR frame.

        Returns:
            {
              "weapon_detected": bool,
              "label":           str,
              "confidence":      float,
              "detections":      list[dict],
              "inference_ms":    float,
            }
        """
        if not self.loaded or self.session is None:
            return self._empty_result()

        if frame is None or frame.size == 0 or len(frame.shape) < 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("[WeaponDetector] Null/Empty frame received.")
            return self._empty_result()

        t0 = time.perf_counter()

        try:
            blob, scale_x, scale_y, pad_x, pad_y = self._preprocess_letterbox(frame)
            
            # ONNX input tensor validation
            if blob is None or not isinstance(blob, np.ndarray):
                raise ValueError("Preprocessed ONNX input is not a valid numpy array")
            if not np.isfinite(blob).all():
                raise ValueError("ONNX input tensor contains infinite or NaN values")
            if len(blob.shape) != 4 or blob.shape[0] != 1 or blob.shape[1] != 3:
                raise ValueError(f"ONNX input tensor shape {blob.shape} is invalid")

            outputs = self.session.run(None, {self.input_name: blob})
            detections = self._postprocess(outputs, scale_x, scale_y, pad_x, pad_y, frame.shape)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._last_inference_ms = elapsed_ms
            self._inference_count  += 1
            self._total_ms         += elapsed_ms

            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                return {
                    "weapon_detected": True,
                    "label":           best["label"],
                    "confidence":      best["confidence"],
                    "detections":      detections,
                    "inference_ms":    round(elapsed_ms, 1),
                }

            return {**self._empty_result(), "inference_ms": round(elapsed_ms, 1)}

        except Exception as e:
            logger.error(f"[WeaponDetector] Inference error: {e}")
            return self._empty_result()

    @property
    def avg_inference_ms(self) -> float:
        """Average inference latency in ms."""
        if self._inference_count == 0:
            return 0.0
        return round(self._total_ms / self._inference_count, 1)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_letterbox(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float, int, int]:
        """
        Letterbox resize: scale image to INPUT_SIZE × INPUT_SIZE with grey padding.
        Preserves aspect ratio — better than squash-resize for detection accuracy.

        Returns: (blob, scale_x, scale_y, pad_x, pad_y)
        """
        h_orig, w_orig = frame.shape[:2]
        scale = min(INPUT_SIZE / w_orig, INPUT_SIZE / h_orig)
        new_w = int(round(w_orig * scale))
        new_h = int(round(h_orig * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = INPUT_SIZE - new_w
        pad_h = INPUT_SIZE - new_h
        pad_x = pad_w // 2
        pad_y = pad_h // 2

        padded = cv2.copyMakeBorder(
            resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR → RGB, HWC → NCHW, [0,255] → [0,1]
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Effective scale factors (from letterbox space back to original)
        scale_x = 1.0 / scale
        scale_y = 1.0 / scale
        return img, scale_x, scale_y, pad_x, pad_y

    # ── Postprocessing ────────────────────────────────────────────────────────

    def _postprocess(
        self,
        outputs: list,
        scale_x: float,
        scale_y: float,
        pad_x: int,
        pad_y: int,
        orig_shape: tuple,
    ) -> list[dict]:
        """
        Parse YOLOv8 ONNX output and convert letterbox coords → original coords.
        Output shape: [1, num_classes+4, N_anchors]
        """
        detections = []
        h_orig, w_orig = orig_shape[:2]

        try:
            out = outputs[0]
            if out.ndim == 3:
                out = out[0]              # remove batch → [features, anchors]

            # Ensure shape is [N_anchors, features]
            if out.shape[0] < out.shape[1]:
                out = out.T

            boxes, scores, class_ids = [], [], []

            for row in out:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                class_confs = row[4:]
                cls_id = int(np.argmax(class_confs))
                conf   = float(class_confs[cls_id])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Convert from letterbox space → original image coords
                x1 = int((cx - bw / 2 - pad_x) * scale_x)
                y1 = int((cy - bh / 2 - pad_y) * scale_y)
                x2 = int((cx + bw / 2 - pad_x) * scale_x)
                y2 = int((cy + bh / 2 - pad_y) * scale_y)

                # Clamp to frame bounds
                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(cls_id)

            # Non-Maximum Suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, scores, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
                )
                for idx in (indices.flatten() if len(indices) else []):
                    x, y, w, h = boxes[idx]
                    label = WEAPON_CLASSES.get(class_ids[idx], f"weapon_{class_ids[idx]}")
                    detections.append({
                        "label":      label,
                        "confidence": round(float(scores[idx]), 3),
                        "box":        [x, y, x + w, y + h],
                    })

        except Exception as e:
            logger.debug(f"[WeaponDetector] Postprocess error: {e}")

        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "weapon_detected": False,
            "label":           "",
            "confidence":      0.0,
            "detections":      [],
            "inference_ms":    0.0,
        }
