"""
Violence Detector Service
==========================
Enhanced in v5.0:
  - Custom ONNX Temporal Model integration (MobileNetV3 + GRU).
  - 16-frame sliding sequence history window.
  - 5-step prediction probability smoothing.
  - Hysteresis triggering: threshold boundary high (0.75) to alert, low (0.35) to clear.
  - Kinematics joint-velocity context blending.
  - Zero-downtime graceful fallback to HuggingFace ViT + pose engine when custom ONNX model is compiling.
"""

import time
import logging
import os
import cv2
import numpy as np
import collections
from PIL import Image

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ONNX_MODEL_PATH = "/Users/swapnil/Desktop/my project/surakhsha-drishti/backend/models/violence_model.onnx"
INFERENCE_INTERVAL = 0.15          # Fast 150ms temporal sequence interval
VIOLENCE_CONFIDENCE_THRESHOLD = 0.58
MODEL_NAME = "jaranohaal/vit-base-violence-detection"

# Hysteresis Thresholds
HYSTERESIS_HIGH = 0.75
HYSTERESIS_LOW = 0.35

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class ViolenceDetector:
    """
    Stateful temporal violence classifier combining custom ONNX sequence learning,
    kinematics joint calculations, and image-based ViT backup.
    """

    def __init__(self):
        # Fallback ViT variables
        self.model = None
        self.processor = None
        self.vit_loaded = False
        self._last_run = 0.0
        self._cached_result = self._make_safe_result()

        # Custom ONNX variables
        self.onnx_session = None
        self.onnx_loaded = False
        self.frame_history = collections.deque(maxlen=16)
        self.pred_history = collections.deque(maxlen=5)
        self.last_violence_state = False

        # Stateful temporal kinematics engine
        from services.violence_classifier import ViolenceClassifier
        self.temporal_classifier = ViolenceClassifier()

        # Self load
        self.load()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Attempts to load custom ONNX temporal model first, then loads fallback ViT."""
        # 1. Custom ONNX temporal model auto-load
        if os.path.exists(ONNX_MODEL_PATH):
            try:
                import onnxruntime as ort
                logger.info(f"[ViolenceDetector] Custom temporal ONNX model found at {ONNX_MODEL_PATH}. Loading...")
                
                # Limit threads to respect MacBook CPU limits
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 2
                
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
                self.onnx_loaded = True
                logger.info("[ViolenceDetector] Custom temporal ONNX model loaded successfully!")
                return True
            except Exception as e:
                logger.error(f"[ViolenceDetector] Failed to load custom ONNX model: {e}")

        # 2. Fallback HuggingFace ViT load
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            logger.info(f"[ViolenceDetector] Loading fallback ViT model {MODEL_NAME}...")
            self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
            self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)
            self.model.eval()
            self.vit_loaded = True
            logger.info("[ViolenceDetector] Fallback ViT model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"[ViolenceDetector] Fallback ViT load failed (HuggingFace transformers not installed / network error): {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, pose_persons: list = None) -> dict:
        """
        Classifies current frame. Swaps instantly to custom temporal ONNX sequence
        classification if compiled; otherwise utilizes fallback ViT.
        """
        h_orig, w_orig = frame.shape[:2]

        # Always update the kinematics joint-velocity temporal parser first
        temp_action, temp_score = self.temporal_classifier.process_pose_data(
            pose_persons or [], h_orig, w_orig
        )

        # Re-check ONNX model compiled state in case it just became ready
        if not self.onnx_loaded and os.path.exists(ONNX_MODEL_PATH):
            self.load()

        # CASE A: CUSTOM ONNX TEMPORAL SEQUENCE MODEL
        if self.onnx_loaded and self.onnx_session is not None:
            return self._detect_custom_onnx(frame, temp_action, temp_score)

        # CASE B: FALLBACK HUGGINGFACE ViT + KINEMATICS HEURISTICS
        return self._detect_fallback_vit(frame, temp_action, temp_score)

    # ── Custom ONNX Execution Pipeline ────────────────────────────────────────

    def _detect_custom_onnx(self, frame: np.ndarray, temp_action: str, temp_score: float) -> dict:
        """Preprocesses frame, runs optimized classifier sequence prediction and kinematics fusion."""
        try:
            # 1. Delegate sequence prediction and smoothing logic to the temporal classifier
            raw_prob, smoothed_prob, active_thresh, is_violent = self.temporal_classifier.predict_frame_sequence(frame)

            # 2. Joint kinematics blend: boost score on high velocity peaks
            fused_agg = round(smoothed_prob * 0.5 + temp_score * 0.5, 3) if is_violent else round(temp_score * 0.3, 3)

            # 3. Action mapping based on threat parameters
            if is_violent:
                if temp_action in ["Fighting", "Chasing", "Fall Detected"]:
                    action_label = temp_action
                else:
                    action_label = "Fighting" if fused_agg > 0.7 else "Harassment"
            else:
                action_label = "Suspicious" if temp_score > 0.38 else "Normal"

            res = {
                "violence_detected": is_violent,
                "confidence":        round(raw_prob, 3),
                "smoothed_confidence": round(smoothed_prob, 3),
                "active_threshold":  active_thresh,
                "label":             "Violence" if is_violent else "Non Violence",
                "action":            action_label,
                "aggression_score":  fused_agg,
            }
            self._cached_result = res
            return res

        except Exception as e:
            logger.error(f"[ViolenceDetector] Custom ONNX execution pipeline failed: {e}")
            # Fall back to kinematic output directly
            is_violent = temp_score >= 0.65
            return {
                "violence_detected": is_violent,
                "confidence":        temp_score if is_violent else 0.0,
                "smoothed_confidence": temp_score if is_violent else 0.0,
                "active_threshold":  ACTIVATE_THRESHOLD,
                "label":             "Violence" if is_violent else "Non Violence",
                "action":            temp_action if is_violent else "Suspicious",
                "aggression_score":  temp_score,
            }

    # ── Fallback ViT Execution Pipeline ───────────────────────────────────────

    def _detect_fallback_vit(self, frame: np.ndarray, temp_action: str, temp_score: float) -> dict:
        """Classic image-based ViT inference with pose kinematics blending (used as fallback)."""
        if (time.time() - self._last_run) < INFERENCE_INTERVAL:
            return self._fuse_temporal(self._cached_result.copy(), temp_action, temp_score)

        if not self.vit_loaded or self.model is None:
            is_violent = temp_score >= 0.55
            res = {
                "violence_detected": is_violent,
                "confidence":        temp_score if is_violent else 0.0,
                "label":             "Violence" if is_violent else "Non Violence",
                "action":            temp_action if is_violent else ("Suspicious" if temp_score > 0.35 else "Normal"),
                "aggression_score":  temp_score,
            }
            self._cached_result = res
            return res

        try:
            import torch

            # Resize to ViT native size
            small = cv2.resize(frame, (224, 224))
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            inputs = self.processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_id = int(torch.argmax(probs).item())
                vit_conf = float(probs[pred_id].item())

            id2label = getattr(self.model.config, "id2label", {0: "Non Violence", 1: "Violence"})
            label = id2label.get(pred_id, str(pred_id))

            vit_is_violent = (
                "violence" in label.lower()
                and "non" not in label.lower()
                and vit_conf >= VIOLENCE_CONFIDENCE_THRESHOLD
            )

            # Blender decision
            is_violent = False
            if vit_is_violent:
                if temp_score > 0.35:
                    is_violent = True
            else:
                if temp_score > 0.70:
                    is_violent = True

            fused_agg = round(vit_conf * 0.4 + temp_score * 0.6, 3) if is_violent else round(temp_score * 0.3, 3)
            final_action = temp_action if is_violent else ("Suspicious" if temp_score > 0.38 else "Normal")

            self._last_run = time.time()
            self._cached_result = {
                "violence_detected": is_violent,
                "confidence":        round(vit_conf, 3) if is_violent else round(vit_conf * 0.2, 3),
                "label":             label,
                "action":            final_action,
                "aggression_score":  fused_agg,
            }
            return self._cached_result.copy()

        except Exception as e:
            logger.error(f"[ViolenceDetector] Fallback ViT failed: {e}")
            self._last_run = time.time()
            return self._fuse_temporal(self._cached_result.copy(), temp_action, temp_score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_safe_result(self) -> dict:
        return {
            "violence_detected": False,
            "confidence":        0.0,
            "label":             "Non Violence",
            "action":            "Normal",
            "aggression_score":  0.0,
        }

    def _fuse_temporal(self, result: dict, action: str, temp_score: float) -> dict:
        vit_agg = result["confidence"] if result["violence_detected"] else 0.0
        result["aggression_score"] = round(vit_agg * 0.4 + temp_score * 0.6, 3)
        
        is_violent = result["violence_detected"]
        if temp_score > 0.72:
            is_violent = True
            
        result["violence_detected"] = is_violent
        if is_violent:
            result["action"] = action
        else:
            result["action"] = "Suspicious" if temp_score > 0.38 else "Normal"
        return result
