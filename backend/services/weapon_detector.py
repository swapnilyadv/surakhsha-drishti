"""
Weapon Detector Service
========================
Uses YOLO ONNX model with ONNXRuntime (CPU provider).
Optimized for low CPU usage: resizes frames to 320x320 before inference.
"""

import numpy as np
import cv2
import time
import logging

logger = logging.getLogger(__name__)

# YOLO class names – update to match your model's classes
WEAPON_CLASSES = {
    0: "knife",
    1: "gun",
    2: "pistol",
    3: "rifle",
    4: "weapon",
}

# Confidence threshold – higher = less false positives
CONFIDENCE_THRESHOLD = 0.65
NMS_THRESHOLD = 0.4
INPUT_SIZE = 640  # must match model input shape [1, 3, 640, 640]


class WeaponDetector:
    """
    YOLO-based weapon detector using ONNX Runtime.
    - Loads model once at startup
    - Runs inference on CPU only (no GPU/MPS to avoid crashes)
    - Resizes frames to 320x320 for speed
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.loaded = False
        self._load_time = None

    def load(self) -> bool:
        """Load the ONNX model. Returns True on success."""
        try:
            import onnxruntime as ort

            # CPU-only provider — safe on MacBook Air, no WindowServer crashes
            providers = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2   # limit threads to reduce CPU spike
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.loaded = True
            self._load_time = time.time()
            logger.info(f"[WeaponDetector] Model loaded: {self.model_path}")
            logger.info(f"[WeaponDetector] Input: {self.input_name} shape={self.input_shape}")
            return True

        except ImportError:
            logger.warning("[WeaponDetector] onnxruntime not installed — weapon detection disabled")
            return False
        except Exception as e:
            logger.error(f"[WeaponDetector] Load failed: {e}")
            return False

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Resize + normalize frame for YOLO inference.
        Returns: (blob, scale_x, scale_y)
        """
        h_orig, w_orig = frame.shape[:2]

        # Resize to model input size (320x320 is fastest for YOLO)
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

        # BGR → RGB, HWC → CHW, normalize to [0,1]
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # CHW
        img = np.expand_dims(img, axis=0)            # NCHW

        scale_x = w_orig / INPUT_SIZE
        scale_y = h_orig / INPUT_SIZE
        return img, scale_x, scale_y

    def postprocess(self, outputs, scale_x: float, scale_y: float, orig_shape: tuple):
        """
        Parse YOLO output → list of detections.
        Returns list of dicts: {label, confidence, box: [x1,y1,x2,y2]}
        """
        detections = []
        try:
            # YOLOv8 output shape: [1, num_classes+4, num_anchors]
            # or legacy [1, num_anchors, 5+num_classes]
            out = outputs[0]
            if out.ndim == 3:
                out = out[0]  # remove batch dim → [features, anchors] or [anchors, features]

            # Handle both transpositions
            if out.shape[0] < out.shape[1]:
                out = out.T   # → [anchors, features]

            h_orig, w_orig = orig_shape[:2]
            boxes, scores, class_ids = [], [], []

            for row in out:
                # YOLOv8: [cx, cy, w, h, cls0_conf, cls1_conf, ...]
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                class_confs = row[4:]
                class_id = int(np.argmax(class_confs))
                conf = float(class_confs[class_id])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Scale back to original frame coords
                x1 = int((cx - bw / 2) * scale_x)
                y1 = int((cy - bh / 2) * scale_y)
                x2 = int((cx + bw / 2) * scale_x)
                y2 = int((cy + bh / 2) * scale_y)

                # Clamp to frame bounds
                x1 = max(0, min(x1, w_orig))
                y1 = max(0, min(y1, h_orig))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(class_id)

            # Non-Maximum Suppression to remove overlapping boxes
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                for idx in (indices.flatten() if len(indices) else []):
                    x, y, w, h = boxes[idx]
                    label = WEAPON_CLASSES.get(class_ids[idx], f"weapon_{class_ids[idx]}")
                    detections.append({
                        "label": label,
                        "confidence": round(float(scores[idx]), 3),
                        "box": [x, y, x + w, y + h],
                    })

        except Exception as e:
            logger.debug(f"[WeaponDetector] Postprocess error: {e}")

        return detections

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run weapon detection on a frame.
        Returns: {weapon_detected: bool, label: str, confidence: float, detections: list}
        """
        if not self.loaded or self.session is None:
            return {"weapon_detected": False, "label": "", "confidence": 0.0, "detections": []}

        try:
            blob, sx, sy = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: blob})
            detections = self.postprocess(outputs, sx, sy, frame.shape)

            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                return {
                    "weapon_detected": True,
                    "label": best["label"],
                    "confidence": best["confidence"],
                    "detections": detections,
                }
            return {"weapon_detected": False, "label": "", "confidence": 0.0, "detections": []}

        except Exception as e:
            logger.error(f"[WeaponDetector] Inference error: {e}")
            return {"weapon_detected": False, "label": "", "confidence": 0.0, "detections": []}
