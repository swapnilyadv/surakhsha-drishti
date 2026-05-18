"""
Violence Detector Service
==========================
Uses HuggingFace ViT model: jaranohaal/vit-base-violence-detection
Optimized for LOW CPU usage:
  - Loads model ONCE at startup
  - Only processes 1 frame per second (rate-limited)
  - Falls back gracefully if transformers not installed
"""

import time
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Rate limit: only run violence detection once per N seconds
INFERENCE_INTERVAL = 0.5   # seconds between violence checks (faster detection)
MODEL_NAME = "jaranohaal/vit-base-violence-detection"


class ViolenceDetector:
    """
    ViT-based binary violence classifier.
    - Runs at 1 FPS maximum (rate limited)
    - Model loaded once, cached in memory
    - Falls back to 'No Violence' if model unavailable
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False
        self._last_run = 0.0
        self._cached_result = {
            "violence_detected": False,
            "confidence": 0.0,
            "label": "Non Violence",
        }

    def load(self) -> bool:
        """Load ViT model from HuggingFace cache or download. Returns True on success."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor

            logger.info(f"[ViolenceDetector] Loading {MODEL_NAME}... (first run may download ~350MB)")

            self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
            self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)
            self.model.eval()  # inference mode – disables dropout etc.

            self.loaded = True
            logger.info("[ViolenceDetector] Model loaded successfully")

            # Log class labels for debugging
            if hasattr(self.model.config, "id2label"):
                logger.info(f"[ViolenceDetector] Labels: {self.model.config.id2label}")
            return True

        except ImportError:
            logger.warning("[ViolenceDetector] transformers not installed — violence detection disabled")
            return False
        except Exception as e:
            logger.error(f"[ViolenceDetector] Load failed: {e}")
            return False

    def _is_rate_limited(self) -> bool:
        """Return True if we ran inference too recently (rate limiter)."""
        return (time.time() - self._last_run) < INFERENCE_INTERVAL

    def detect(self, frame: np.ndarray) -> dict:
        """
        Classify frame as violent / non-violent.
        Returns cached result if called more often than 1/sec.
        """
        # Rate limiter — return cached result without inference
        if self._is_rate_limited():
            return self._cached_result

        # If model not loaded, return safe default
        if not self.loaded or self.model is None:
            return self._cached_result

        try:
            import torch

            # Resize frame before passing to ViT (224x224 is the native size)
            small = cv2.resize(frame, (224, 224))
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            # Preprocess
            inputs = self.processor(images=pil_img, return_tensors="pt")

            # Inference — no gradient for speed
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_id = int(torch.argmax(probs).item())
                confidence = float(probs[pred_id].item())

            # Get label string
            id2label = getattr(self.model.config, "id2label", {0: "Non Violence", 1: "Violence"})
            label = id2label.get(pred_id, str(pred_id))

            # Map classes for jaranohaal/vit-base-violence-detection:
            # LABEL_0 usually 'non-violence', LABEL_1 usually 'violence'
            # We check the specific index for 'violence' string in config
            is_violent = False
            if "violence" in label.lower() and "non" not in label.lower():
                is_violent = True
            
            # Additional safety: ensure confidence is significant
            if is_violent and confidence < 0.60:
                is_violent = False

            self._last_run = time.time()
            self._cached_result = {
                "violence_detected": is_violent,
                "confidence": round(confidence, 3),
                "label": label,
            }
            return self._cached_result

        except Exception as e:
            logger.error(f"[ViolenceDetector] Inference error: {e}")
            self._last_run = time.time()  # still update timer to avoid error spam
            return self._cached_result
