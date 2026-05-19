"""
Violence Detector Service
==========================
Primary model: HuggingFace ViT  — jaranohaal/vit-base-violence-detection
                                  (binary: Violence vs Non-Violence)

Enhanced in v4.0:
  - Richer output: action_label, aggression_score, action enum
  - Pose-context fusion: if pose estimator detected high aggression,
    this is factored into the final confidence (no false-safe)
  - Strict rate limiting (0.5s interval) + result caching
  - Graceful fallback when model unavailable

Output dict:
  {
    "violence_detected": bool,
    "confidence": float,          # 0.0–1.0
    "label": str,                 # raw model label
    "action": str,                # "Fighting" | "Aggressive" | "Normal" | etc.
    "aggression_score": float,    # 0.0–1.0 fused with pose data
  }
"""

import time
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
INFERENCE_INTERVAL   = 0.5    # minimum seconds between ViT inferences
VIOLENCE_CONFIDENCE_THRESHOLD = 0.58   # min confidence to declare violence
MODEL_NAME = "jaranohaal/vit-base-violence-detection"

# Action label mapping based on violence confidence + pose aggression
def _map_action(violence_detected: bool, vit_conf: float, pose_agg: float) -> str:
    """Derive human-readable action label from model outputs."""
    if not violence_detected:
        if pose_agg > 0.5:
            return "Suspicious"
        return "Normal"
    if vit_conf > 0.85 or pose_agg > 0.8:
        return "Fighting"
    if pose_agg > 0.5:
        return "Aggressive"
    return "Violence"


class ViolenceDetector:
    """
    ViT-based binary violence classifier with pose-context fusion.

    - Loads model once, cached in memory
    - Rate-limited to 2 inferences/sec (configurable)
    - Accepts optional pose_aggression float for score boosting
    - Falls back safely when model unavailable
    """

    def __init__(self):
        self.model     = None
        self.processor = None
        self.loaded    = False

        self._last_run = 0.0
        self._cached_result = self._make_safe_result()
        
        # Stateful temporal violence classifier
        from services.violence_classifier import ViolenceClassifier
        self.temporal_classifier = ViolenceClassifier()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load ViT model from HuggingFace cache or download. Returns True on success."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor

            logger.info(
                f"[ViolenceDetector] Loading {MODEL_NAME}... "
                "(first run may download ~350MB)"
            )
            self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
            self.model     = ViTForImageClassification.from_pretrained(MODEL_NAME)
            self.model.eval()   # disable dropout/BN training behaviour

            self.loaded = True
            logger.info("[ViolenceDetector] Model loaded successfully")

            if hasattr(self.model.config, "id2label"):
                logger.info(f"[ViolenceDetector] Labels: {self.model.config.id2label}")
            return True

        except ImportError:
            logger.warning(
                "[ViolenceDetector] 'transformers' not installed — violence detection disabled"
            )
            return False
        except Exception as e:
            logger.error(f"[ViolenceDetector] Load failed: {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, pose_persons: list = None) -> dict:
        """
        Classify frame as violent / non-violent.

        Args:
            frame:         BGR numpy array
            pose_persons:  List of person dicts containing keypoints, aggression_score, etc.

        Returns:
            {
              "violence_detected": bool,
              "confidence": float,
              "label": str,
              "action": str,
              "aggression_score": float,
            }
        """
        h_orig, w_orig = frame.shape[:2]
        
        # 1. Update temporal sequence classifier (velocity, acceleration, angles, proximity)
        temp_action, temp_score = self.temporal_classifier.process_pose_data(
            pose_persons or [], h_orig, w_orig
        )

        # Rate limiter — return cached result if called too frequently
        if self._is_rate_limited():
            return self._fuse_temporal(self._cached_result.copy(), temp_action, temp_score)

        if not self.loaded or self.model is None:
            # Fallback to temporal classifier directly if ViT is unavailable
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

            # Resize to ViT native resolution (224x224)
            small   = cv2.resize(frame, (224, 224))
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            # Preprocess → inference
            inputs = self.processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs   = torch.softmax(outputs.logits, dim=-1)[0]
                pred_id = int(torch.argmax(probs).item())
                vit_conf = float(probs[pred_id].item())

            # Resolve label
            id2label = getattr(
                self.model.config, "id2label", {0: "Non Violence", 1: "Violence"}
            )
            label = id2label.get(pred_id, str(pred_id))

            # Determine violence flag from label string
            vit_is_violent = (
                "violence" in label.lower()
                and "non" not in label.lower()
                and vit_conf >= VIOLENCE_CONFIDENCE_THRESHOLD
            )

            # Highly stable fused decision logic:
            # - Suppress ViT false-positives (waving, stretching) if temporal pose activity is low
            # - Trigger immediately on extreme temporal markers (attacks, punches, kicks)
            is_violent = False
            if vit_is_violent:
                if temp_score > 0.35:
                    is_violent = True
            else:
                if temp_score > 0.70:
                    is_violent = True

            # Fused aggression confidence
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
            logger.error(f"[ViolenceDetector] Inference error: {e}")
            self._last_run = time.time()
            return self._fuse_temporal(self._cached_result.copy(), temp_action, temp_score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_rate_limited(self) -> bool:
        return (time.time() - self._last_run) < INFERENCE_INTERVAL

    def _make_safe_result(self) -> dict:
        """Return a safe 'no violence' default result."""
        return {
            "violence_detected": False,
            "confidence":        0.0,
            "label":             "Non Violence",
            "action":            "Normal",
            "aggression_score":  0.0,
        }

    def _fuse_temporal(self, result: dict, action: str, temp_score: float) -> dict:
        """Apply fresh temporal analysis to rate-limited / fallback results."""
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

