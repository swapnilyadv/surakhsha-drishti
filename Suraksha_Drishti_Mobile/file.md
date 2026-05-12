Project: Suraksha Drishti — Requirements & System Specification (RM)

Purpose
-------
This document captures the full specification for the Suraksha Drishti system as present in this repository. It is intended to be machine-readable and human-friendly so an automated processor or an AI assistant can parse the project architecture, components, models, data flows, and mobile integration details.

High-level summary
------------------
- Goal: Real-time monitoring system for CCTV / camera streams that performs violence detection, weapon detection, gender classification, person counting, crowd density estimation and emits alerts and evidence clips.
- Input: MJPEG/RTSP/video/webcam streams; optionally images uploaded via API.
- Output: WebSocket events and REST endpoints with detection results, HUD overlays on streaming video, recorded evidence clips stored in `evidence/`.

Repository layout (relevant files)
---------------------------------
- backend/
  - server.py              # Flask + SocketIO server, inference pipeline and camera threads
  - gender_predict.py      # small helper using Caffe gender model (added)
  - test_gender.py         # simple gender-model loader
  - models/                # Caffe models and face detectors: gender_net.caffemodel, gender_deploy.prototxt, face_net.caffemodel, face_deploy.prototxt
  - requirements.txt       # (if present) Python requirements
- src/                     # frontend (TypeScript / React / Vite)
  - routes/                # dashboards and ai-monitoring pages
  - components/            # UI components and surveillance UI
- best_violence_model.keras # main violence detection Keras model
- gender-detection.ipynb   # Notebook that trains a Keras gender classifier (CelebA)
- evidence/                # saved video evidence

Functional features
-------------------
1. Camera ingest and stream
   - Accepts webcam index or RTSP/video file paths.
   - MJPEG streaming for frontend clients.

2. Violence detection
   - Temporal model (Keras) accepts sequences of frames (24 frames x 112x112) and outputs violence confidence. Model stored in `best_violence_model.keras`.
   - Falling back to simulation when model missing.

3. Weapon detection & person counting
   - YOLOv8 (ultralytics.PY) used to detect people and weapon-like objects (knife class used, simulation for firearms).

4. Face and gender detection
   - Face detection: OpenCV Haar cascades.
   - Gender detection: lightweight Caffe model (`gender_deploy.prototxt`, `gender_net.caffemodel`) loaded via OpenCV DNN.
   - There is also a notebook implementing a Keras-based gender classifier trained on CelebA for experimentation.

5. Overlay & UI
   - HUD overlay with counts, gender ratio, density level, alerts.
   - WebSocket emits `detection_result` objects which frontend consumes.

Non-functional requirements
--------------------------
- Real-time: ~8–15 FPS target depending on hardware; YOLO and face/gender DNN are the main CPU/GPU consumers.
- Privacy: store minimal PII; recorded clips are stored under `evidence/`.
- Extensibility: modular inference helpers allow swapping models.

Tech stack
----------
- Backend: Python 3.8+ (Flask, Flask-SocketIO, OpenCV, numpy, keras/tensorflow, ultralytics (YOLOv8)).
- Frontend: TypeScript + React + Vite; SocketIO client to receive detection events.
- Models:
  - Violence: Keras model (seq model) saved as `best_violence_model.keras`.
  - Face: OpenCV face detector (haarcascade or Caffe face detector available in backend/models/).
  - Gender: Caffe model (gender_net.caffemodel + gender_deploy.prototxt) and an optional Keras notebook model.

Data and dataset notes
----------------------
- CelebA is used in the notebook for training a gender classifier (images and list_attr_celeba.csv). The notebook expects the dataset at `../input/celeba-dataset/` (Kaggle layout).
- Training data / evaluation metrics should be stored separately (not committed) and referenced in README.

API & events (contracts)
------------------------
1. WebSocket: channel `detection_result`
   Payload (example):
   {
     cam_id: str,
     violence: bool,
     violence_score: float,
     weapon: bool,
     weapon_type: str,
     weapon_confidence: float,
     weapon_boxes: [ { type, confidence, box } ],
     males: int,
     females: int,
     crowd_count: int,
     density: float,
     density_level: str,
     timestamp: float
   }

2. REST endpoints (server.py exposes camera registry endpoints):
   - /camera/add, /camera/remove, /camera/list, /stream/:id (MJPEG stream), etc.

Gender-detection: diagnosis & recommended fixes
---------------------------------------------
Problem reported: gender detection always classifies faces as `Male` (never `Female`). Possible causes and checks:

1) Label mapping order mismatch
   - Check `GENDER_LIST` or label mapping used after model output. In `server.py` currently `GENDER_LIST = ['Male', 'Female']`. Many public Caffe gender nets (from learnopencv) use label order `['Male','Female']` or `['Female','Male']` depending on the network. Double-check with a small test: run model.forward() on a known male and known female image and log raw `gender_preds` values. If argmax picks index 0 always it might mean the model outputs are reversed or the dataset used to train it had different label order.

2) Preprocessing mismatch
   - The script uses `cv2.dnn.blobFromImage(face_crop, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)`. If the model expects RGB ordering or different mean values / scale, predictions will be garbage. Confirm model origin (the models in `backend/models` appear to be from the LearnOpenCV AgeGender sample: their mean values and (227,227) suggest correctness). If in doubt, try swapRB=True and inspect probabilities.

3) Model corrupted or wrong files
   - Validate that `gender_net.caffemodel` and `gender_deploy.prototxt` match each other (not swapped) and were downloaded intact. Try loading the model and printing shapes and example forward outputs in a test script.

4) All-faces look similar (resolution/quality)
   - Low face resolution or faces with occlusion or off-angle may bias predictions. Try cropping to larger face area (less padding) and resizing properly.

5) Post-processing bug
   - The server increments males if gender == 'Male' else females. If model yields floats but argmax indexing or shape usage is wrong, the mapped class may always be Male. Inspect `gender_preds[0].argmax()` and shapes. Add safe code that prints or logs `gender_preds` for debugging.

Quick reproduction test (recommended)
-----------------------------------
1. Add a tiny test script (already added `backend/gender_predict.py`) which:
   - Loads the Caffe model, preprocesses an image with multiple permutations (swapRB True/False, different mean values), and prints raw forward output.
2. Run tests on a few labeled sample images (one male, one female). Save outputs to `training_logs/` for analysis.

Suggested fixes (practical)
-------------------------
1. Add verbose debug logging around gender inference in `server.py` (one-time): log `gender_preds` and mapping for a few frames so you can quickly see whether predictions are degenerate.
2. Try `swapRB=True` and/or different `MODEL_MEAN_VALUES` (the notebook and `test_gender.py` share the same values; try the inverted mean or zero mean as a toggle) and re-evaluate.
3. If the Caffe model consistently fails, switch to the Keras CelebA model (notebook) by converting the trained Keras model to a lightweight format (TFLite or ONNX) for faster inference on CPU/mobile.
4. Add a simple calibration threshold: if predicted probability for top class < 0.6, mark as `Unknown` and skip counting (avoid wrong male counts). This reduces false positives.
5. Consider integrating DeepFace or a pre-trained modern gender model (InsightFace or DeepFace-based gender head) if accuracy remains poor (tradeoff: larger model and more dependencies).

Mobile integration plan
-----------------------
Goal: provide a mobile app that receives alerts, thumbnails, and can display live stream and detection overlays.

Architecture options:
- Option A — Thin mobile client (recommended):
  - Mobile app (iOS/Android/React Native or Flutter) connects to backend SocketIO or WebRTC for live MJPEG (less smooth). It receives `detection_result` WebSocket events and displays HUD elements locally. Evidence clips and snapshots are downloaded via HTTPS endpoints.
  - Advantage: minimal compute on-device.

- Option B — On-device inference (offline mode):
  - Convert models to mobile-friendly formats (TFLite for Android/iOS, or CoreML for iOS). Use a small face detector (MTCNN/RetinaFace-lite) + gender head (TFLite) for on-device gender classification and person detection.
  - Advantage: low-latency offline mode; disadvantage: engineering cost and platform-specific packaging.

Mobile data contract
- WebSocket for events (same payload as server emits).
- REST endpoints for camera registration, evidence download, health checks.

Security & privacy
------------------
- Use HTTPS for REST and WSS (secure WebSocket).
- Protect Supabase keys (do not expose private keys in frontend Vite env; use a secure server-side flow).
- Provide deletion policy for `evidence/` and access controls (who can download clips).

Testing & Validation
--------------------
- Unit tests for small helpers: gender model loader, blob preprocessing, result mapping.
- Integration tests: run the server locally against sample videos and validate `detection_result` outputs with expected labels.
- E2E tests: mobile client receiving events and rendering overlays.

Roadmap & priorities (short-term)
---------------------------------
1. Add diagnostic logs to gender inference and run the small test script on labeled images to identify whether labels are reversed or preprocessing is wrong.
2. Add a simple configuration toggle in `server.py` to switch `swapRB` and alter mean values and label order so we can quickly A/B test.
3. If Caffe model is broken, use the Keras notebook route and convert a trained Keras model to ONNX / TFLite and integrate.
4. Add an 'Unknown' fallback and calibration threshold to reduce false male predictions.

Appendix: Suggested minimal commands to run locally
-------------------------------------------------
1) Create a virtualenv and install minimal packages:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install opencv-python-headless numpy flask flask-socketio ultralytics tensorflow keras

2) Run sample gender test script (example):
   python backend/gender_predict.py path/to/labeled_female.jpg

3) Start backend server (in dev):
   python backend/server.py

Contacts & ownership
--------------------
- Repo owner: swapnilyadv (local workspace)
- Purpose owner: Suraksha Drishti application / tactical assistant for CCTV monitoring

Notes
-----
- This RM file is intentionally detailed and includes immediate diagnostic steps for the gender-only detection issue. After you run the quick tests and post outputs (raw gender_preds for a known female example), we can provide an exact patch (swap labels, change swapRB, or replace the model).

End of specification.
