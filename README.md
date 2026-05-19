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

## 🛠️ Complete Work Completed & Features Implemented

### 1. 📹 Thread-Safe Rolling Buffers & Auto-Recording System
*   Implemented a rolling buffer queue (`collections.deque`) capturing the preceding **5 seconds** of camera history at 20 FPS (prevents losing vital context leading up to an incident).
*   Enforces a **minimum recording duration of 15 seconds** per threat.
*   Employs a **5-second cooldown** buffer: if violence ceases, recording continues for 5 seconds to ensure temporal continuity and prevent clipping/flickering.
*   All frame writing operates asynchronously on a background worker thread queue (`queue.Queue`) to prevent model inference lag.

### 2. ⚡ Asynchronous FFmpeg H264 Transcoding Pipeline
*   OpenCV's native `VideoWriter` outputs raw non-web-playable `mp4v` codec frames.
*   Once recording stops, the system spawns a background thread system call to **`ffmpeg`** at `/opt/homebrew/bin/ffmpeg` to transcode the video into a standard HTML5 **H264/AVC** stream.
*   Serves playable `.mp4` files statically from `/recordings/`, enabling real-time playback inside the frontend modal and direct downloads.

### 3. 💾 Persistent Evidence Vault
*   Replaced in-memory states with a persistent database file (`backend/recordings/evidence_db.json`) that survives server restarts.
*   **Peak Stat Accumulator**: Dynamically logs peak gender counts (male/female classifications), weapon detections, highest confidence levels, and active GPS coordinates for Google Maps routing.

### 4. 🔗 Dynamic Responding Authority Dispatching
*   Added a safe double-dispatch lock inside `DispatchManager` to prevent multiple police stations from accepting the same incident.
*   Replaced hardcoded dispatch names by dynamically reading the logged-in user session metadata (`localStorage.getItem("sd_auth")`). Clicking **DISPATCH** now registers the actual, real station name (e.g., *"Bandra West Station"*) and timestamps the action.

### 5. 🆘 Tactical Emergency Escalations ("Need More Help")
*   Clicking **REQUEST MORE HELP** on a dispatched incident locks the state, alerts cooperating divisions, and forcefully re-loops the audio alert sound globally on all station terminals.
*   Synchronizes active coordinates dynamically for rapid tactical dispatch mapping.

### 6. 🧹 Clean Evidences & Reload Persistence
*   Addressed a loader bug: clearing the dashboard now correctly stores an `sd_seeded` flag in local storage.
*   Upon refresh, the system remembers that the operator explicitly cleared all cards, preventing sample mock cards from re-seeding.

### 7. 📺 Dynamic Media Element Player
*   Configured the detail player modal to distinguish between MJPEG live feeds and transcoded clips. 
*   **MJPEG stream**: Renders inside a responsive `<img src="..." />` element so AI camera coordinate bounding overlays play dynamically in real-time.
*   **Clips**: Renders inside `<video controls />` for perfect playback, scanning, and downloading.

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
