# 👁️ SURAKSHA DRISHTI (सुरक्षा-दृष्टि)
### *Next-Gen AI Surveillance Evidence Management & Emergency Police Coordination Platform*

Suraksha Drishti is a real-world, high-performance tactical surveillance platform. It leverages live multi-threaded computer vision pipeline models to detect aggression, violence, and weapon threats in real-time, automatically trigger coordinated alarms, save transcoded evidence recordings, and synchronize emergency backup dispatches across multiple police station dashboard terminals.

---

## 🚀 Technological Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | **Next.js 16 (React 19)** | High-performance dashboard, maps routing, and interactive visual interface. |
| **Frontend Styling** | **Vanilla CSS & TailwindCSS** | Sleek glassmorphic dark mode, neon HUD alerts, and micro-animations. |
| **Backend Framework** | **FastAPI (Python 3.11)** | High-throughput asynchronous REST APIs & persistent WebSocket channels. |
| **Real-Time Pipeline** | **WebSockets** | Ultra-low latency event broadcasts (3Hz updates & instant alert coordination). |
| **Core AI Vision** | **YOLOv8n (ONNX Runtime)** | Fast multi-person bounding box and weapon coordinate inference on CPU. |
| **Pose Processing** | **MediaPipe Landmarker** | Real-time body skeleton joint tracking. |
| **Inference Models** | **ViT (Vision Transformer)** | Deep aggression and action classifier (HuggingFace `jaranohaal/vit-base-violence-detection`). |
| **Frame Operations** | **OpenCV (cv2)** | Camera capture, frame processing, and graphical HUD overlay drawings. |
| **Clip Transcoder** | **FFmpeg** | Asynchronous CLI sub-process that transcodes raw `.mp4` recordings into browser-playable **H264 AVC (yuv420p)** format. |
| **Database** | **Persistent JSON Store** | Light and fast persistent file-based evidence card tracker. |

---

## 🛠️ Complete Work Completed & Features Implemented (v5.0 Upgraded)

### 1. 🧠 Upgraded Temporal Violence Engine (v6.0 Close-Combat Grappling Upgraded)
*   **Wrestling, Grappling & Choking Support**: Incorporates state-of-the-art proximity, torso-overlap, and body-acceleration algorithms to automatically classify physical combat even in the absence of visible punch gestures.
*   **Person-Count & Separation Independence**: Violence and close-combat aggression heuristics will successfully trigger even when overlapping bodies are merged by YOLOv8 into a single tracked bounding box (`person_count = 1`).
*   **Hybrid Weighted Aggression Scoring**: Uses robust mathematical heuristic layers matching attacker wrist velocities, centroid accelerations, relative head proximities, and body jitter rates:
    - *Chaotic Motion*: **+40 points**
    - *Repeated Arm Strikes*: **+35 points**
    - *Skeleton Overlap*: **+30 points**
    - *Temporal Model Confidence*: **+40 points**
    - *Close Proximity Aggression*: **+15 points**
    - *Wrist Collision / Strikes*: **+15 points**
    - *Victim Recoil / Momentum*: **+10 points**
    - *Close Combat / Grappling Boost*: **+25 points**
*   **Aggression Override Mode**: If temporal confidence is elevated or chaotic motion is high, and rapid arm speeds are active, the threat forces **HIGH/CRITICAL** (minimum score of `75`) bypassing exact strike rules.
*   **30-Frame Sliding Window**: Replaced single-frame landmark analysis with a continuous 30-frame (~1.5s) temporal trajectory window storing joint vectors, bounding coordinates, and pixel differences.
*   **Joint Velocity Trackers**: Extracts kinematic velocity rates of elbows, wrists, and ankles to isolate flailing combat gestures.
*   **Repeated Arm Strike Cycle Counter**: Counts mathematical extrema velocity wave peaks. If 3 or more high-speed peaks occur within a sliding interval, flags a flailing strike pattern (punching fights).
*   **Centroid Fall Detection**: Tracks downward vertical velocity drop spikes in the body centroid. If followed by deceleration, flags a post-impact collapse/fall event.
*   **Suspicious Running & Chasing Tracker**: Measures shifts in relative centroid distances over 5-frame windows. If distance between two targets converges rapidly while velocity is high, flags aggressive chasing.
*   **Crowd Aggression Estimator**: Computes overlapping bounding boxes coupled with elevated motion pixel differences.

### 2. 📊 Dynamic Threat Scoring Engine (Phase 2)
*   Computes a real-time weighted threat index between `0.0` and `1.0`:
    $$\text{Score} = (\text{Aggression} \times 0.40) + (\text{Weapon Conf} \times 0.35) + (\text{Motion Intensity} \times 0.15) + (\text{Crowd Density} \times 0.10) + \text{Event Boosts}$$
*   Applies dynamic boosts for specialized gestures (+15% for repeated arm strikes, +10% for falls, +12% for aggressive chasing).
*   Categorizes threats into distinct levels: `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`.

### 3. 📹 Smart Recording Pipeline & Incident Merging (Phase 3)
*   **5-Second Pre-Event Buffer**: Maintains a thread-safe rolling queue of raw images so recordings capture the exact precursor to threat triggers.
*   **10-Second Incident Merging**: Re-detections within 10 seconds of stopping will bypass the cooldown and append frames to the *same* active evidence clip, preventing fragmented files.
*   **Minimum Duration Lock**: Holds the video writer active for at least 15 seconds to ensure evidence clarity.
*   **Asynchronous FFmpeg Queue**: Spawns transcoding subprocesses inside a background thread pool (preset `ultrafast` at 2 threads) to prevent CPU core saturation.

### 4. ⚡ CPU Performance Safeguards & Adaptive Skipping (Phase 4)
*   **Adaptive skipping**: Dynamically adjusts frame sampling step (`AI_EVERY_N_FRAMES`) between 4 and 10 based on measured inference latency to prevent CPU thread lockups on MacBook Air hardware.
*   **Sub-scale preprocessing**: Resizes raw frames before heavy classifier feeds to preserve memory bandwidth.

### 5. 👮 Responding Authority & Real-Time Dispatches
*   Double-dispatch lock prevents overlapping accept operations.
*   Reads logged-in operator metadata (`localStorage.getItem("sd_auth")`) to dynamically query and log responding stations (e.g., *"Bandra Station"*) rather than rendering static text templates.

### 6. 🧹 Clear Evidences & Reload Persistence
*   Dashboard clearing persists an `sd_seeded` flag in local storage, preventing sample mock cards from re-seeding on page refreshes.

### 7. 📺 Dynamic HUD Overlay & Events Timeline (Phase 6)
*   **MJPEG Overlays**: Renders live video feeds inside standard responsive `<img>` elements for real-time bounding box visualization.
*   **Surveillance Activity Log**: Integrates a live-updating visual event logging dashboard component by periodically querying our new `/api/events/timeline` REST API.
*   **Dynamic Pulse Display**: Renders glowing dark-mode indicators that change pulse frequencies and colors (cyan, yellow, orange, bright red) relative to the active threat state.

---

## 📁 Directory Structure

```text
surakhsha-drishti/
├── backend/
│   ├── main.py                         # FastAPI REST Endpoints, WebSocket server & Broadcaster Loop
│   ├── start_backend.sh                # Executable shell runner using localized pyenv Python 3.11
│   ├── models/                         # YOLO ONNX and MediaPipe pose models
│   ├── recordings/
│   │   ├── evidence_db.json            # Persistent JSON evidence database
│   │   └── *.mp4                       # Transcoded browser-playable MP4 clips
│   └── services/
│       ├── recording_manager.py        # Thread-safe rolling deque and background FFmpeg transcoder
│       ├── dispatch_manager.py         # Thread-safe locking dispatch coordinator
│       ├── evidence_manager.py         # High-level database manager and coordinate mapping
│       ├── alert_manager.py            # Global and local alarm muting coordinator
│       ├── frame_processor.py          # Vision AI multi-threaded inference loop
│       └── human_detector.py           # YOLO person scanner
└── frontend/
    ├── src/
    │   ├── app/
    │   │   └── page.tsx                # Main HUD, Loop alarm player, and alert banner
    │   ├── hooks/
    │   │   ├── useEvidenceStore.ts     # Persistent optimistic database store, REST, and WS sync
    │   │   ├── useBackendAI.ts         # Real-time WebSocket connection and Event router
    │   │   └── useCameraStore.ts       # Active camera configurations
    │   └── components/
    │       ├── EvidenceDetail.tsx      # Modal player, responding authority cards, dispatch controls
    │       └── ...
    └── public/
        └── sound/
            └── alarm.wav               # Real-world alarm voice sound file
```

---

## 📡 WebSocket Broadcast Specification

Connected dashboards synchronize in real-time by receiving structured events:

```json
/* Real-Time Violence Alarm Trigger */
{
  "event": "violence_detected",
  "alarm": true,
  "recording": true,
  "evidence_id": "EVD-REC-1718000000"
}

/* Police Dispatch Coordination */
{
  "event": "dispatch_accepted",
  "evidence_id": "EVD-REC-1718000000",
  "station": "Gateway Station",
  "officer": "Officer-47",
  "status": "POLICE_DISPATCHED"
}

/* Incident Resolve */
{
  "event": "evidence_resolved",
  "evidence_id": "EVD-REC-1718000000",
  "status": "Resolved"
}
```

---

## 🚦 Run Instructions

### 1. Start Vision AI Backend Server
```bash
cd backend
chmod +x start_backend.sh
./start_backend.sh
```
*The server will initialize models, start the capture threads, and listen at `http://localhost:8765`.*

### 2. Start Next.js Development Server
```bash
cd frontend
npm install
npm run dev
```
*Open `http://localhost:3000` in your browser.*

### 3. Authenticate Dashboard
*   **Operator ID**: `ADMIN_001`
*   **Passphrase**: `admin@123`
*   *Once authenticated, the platform will load system components, sync the database, and bind WebSocket streams.*
