# Suraksha Drishti Project Details

## Overview

Suraksha Drishti is a full-stack AI surveillance and incident-coordination platform. The repository combines a Python FastAPI backend for real-time computer-vision inference, recording, evidence management, WebSocket broadcasting, and dispatch coordination with a Next.js frontend that acts as the operator dashboard.

The system is designed around these core behaviors:

- Live detection of humans, weapons, violence, aggressive motion, falls, chasing, and repeated strikes.
- Real-time streaming of annotated camera feeds through MJPEG.
- Real-time event updates through WebSockets.
- Persistent evidence tracking and playback.
- Alarm handling and dispatch/escalation workflows.
- A dashboard login gate with a hardcoded master admin bypass plus station-based credentials.

## Repository Layout

Top-level structure:

- backend
  - FastAPI application, AI services, models, recordings, and training scripts.
- frontend
  - Next.js dashboard, UI components, client-side stores, and Supabase integration.
- Dataset
  - Training/validation assets for normal and violence classes.
- raspberry_pi
  - Raspberry Pi focused detector scripts and TFLite/Keras assets.
- sound
  - Shared audio assets.
- README.md
  - High-level project description and quick start notes.
- setup_backend.sh
  - Backend dependency bootstrap script.
- start_backend.sh
  - Backend launcher script.

## Root Scripts

### setup_backend.sh

Backend dependency installer for macOS/Homebrew-style Python 3.11 environments.

What it does:

- Uses Python at /opt/homebrew/bin/python3.11.
- Upgrades pip.
- Installs the backend dependency stack.
- Verifies that key packages import correctly.

Packages installed by the script:

- fastapi 0.115.5
- uvicorn[standard] 0.32.1
- python-multipart 0.0.12
- websockets 14.1
- opencv-python 4.10.0.84
- numpy 1.26.4
- Pillow 10.4.0
- onnxruntime 1.20.1
- transformers 4.47.0
- torch 2.5.1
- python-dotenv 1.0.1
- mediapipe 0.10.35
- ultralytics >= 8.3.0

### start_backend.sh

Backend runtime launcher.

What it does:

- Uses the pyenv Python at /Users/swapnil/.pyenv/versions/3.11.9/bin/python3.11.
- Changes into the backend directory.
- Runs backend/main.py.

### frontend/package.json scripts

Frontend scripts:

- dev - next dev
- build - next build
- start - next start
- lint - eslint

## Backend Architecture

The backend is a FastAPI app in backend/main.py. It coordinates model loading, camera session management, frame processing, evidence persistence, and broadcasting.

### Backend startup flow

At startup, the application:

- Loads the human detector.
- Loads the weapon detector.
- Loads the pose estimator.
- Loads the violence detector.
- Starts the evidence manager.
- Pre-boots the default camera processor.
- Starts the WebSocket broadcaster loop.
- Exposes REST and WebSocket endpoints for the frontend.

### Core models and assets

Model files currently present under backend/models:

- best_model.pth
- pose_landmarker_lite.task
- violence_model.onnx
- violence_model.onnx.data
- yolov8n.pt

Important operational note:

- The backend weapon detector currently looks for backend/models/best.onnx.
- The repository contains backend/models/best_model.pth instead.
- In the current environment, this causes the weapon detector to boot as disabled while the rest of the backend starts normally.

### Backend service registry

backend/services contains the following service modules:

- alert_manager.py - alarm state, muting, acknowledgement, and emergency trigger handling.
- dispatch_manager.py - dispatch acceptance and escalation coordination.
- evidence_manager.py - persistent evidence database management.
- event_manager.py - recent incident/event timeline tracking.
- frame_processor.py - camera capture loop, inference pipeline, source switching, and MJPEG frame generation.
- human_detector.py - human/person detection model wrapper.
- pose_estimator.py - pose landmark estimation wrapper.
- recording_manager.py - recording lifecycle and file handling.
- recording_pipeline.py - background recording/transcoding pipeline.
- threat_engine.py - threat scoring and decision logic.
- violence_classifier.py - violence classification heuristics and model wrapper.
- violence_detector.py - violence detection logic and temporal model integration.
- weapon_detector.py - weapon detection model wrapper.
- websocket_broadcaster.py - push broadcasting support.
- **init**.py - service package marker.

### WebSocket manager and broadcast loop

The backend maintains a thread-safe WebSocket connection manager and a background broadcast loop.

Broadcast behavior:

- Aggregates the latest detection result from active frame processors.
- Sends the highest threat payload to connected clients.
- Emits new-evidence events when the evidence manager finalizes a clip.
- Emits recording-start events when active recording begins.
- Suppresses redundant updates unless meaningful values change.

### Backend routes

REST and WebSocket endpoints exposed by backend/main.py:

- GET /api/health
  - Health status, uptime, active camera state, and model readiness.
- GET /api/model-status
  - Loaded model metadata plus inference statistics.
- GET /api/detection/latest
  - Latest detection JSON for a camera id.
- GET /api/events/timeline
  - Recent event timeline from active processors.
- POST /api/upload-video
  - Accepts mp4, mov, avi, and mkv uploads and queues them for source mapping.
- POST /api/select-source
  - Dynamically changes the source used by a frame processor.
- GET /api/evidence
  - Returns persistent evidence records.
- POST /api/evidence/{evidence_id}/dispatch
  - Marks evidence as police dispatched and broadcasts the dispatch event.
- POST /api/evidence/{evidence_id}/escalate
  - Escalates an incident and requests more help.
- POST /api/evidence/{evidence_id}/resolve
  - Resolves an incident and clears active escalation state when appropriate.
- POST /api/evidence/clear
  - Clears persistent evidence records.
- POST /api/acknowledge
  - Acknowledges an active alarm and mutes audio alerts.
- GET /api/stream/mjpeg
  - Live MJPEG stream with AI overlays.
- WS /ws/detections
  - Real-time detection feed and keepalive channel.

### Backend response contracts

The WebSocket payload includes these fields:

- weapon_detected
- weapon_label
- weapon_confidence
- violence_detected
- violence_label
- violence_confidence
- action
- aggression_score
- fps
- person_count
- male_count
- female_count
- total_persons
- timestamp
- frame_count
- violence
- recording
- alarm_active
- event
- threat_level
- threat_score
- fall_detected
- repeated_strikes
- chasing_detected
- motion_intensity
- punch_detected
- attacker_bbox
- victim_bbox
- punch_arrow

Special broadcast events used by the frontend:

- new_evidence
- dispatch_accepted
- need_more_help
- evidence_resolved
- violence_detected

### Persistent backend storage

backend/recordings contains:

- evidence_db.json - persistent evidence record database.
- snapshots - saved still images.
- Many MP4 recordings named in the EVD_CAM-\* format.

### Backend runtime behavior observed in this workspace

Verified runtime notes from the live backend:

- The backend starts successfully on http://localhost:8765.
- The human detector loads successfully.
- The pose estimator loads successfully.
- The violence detector loads successfully.
- The weapon detector is disabled because backend/models/best.onnx is missing.
- The frame processor repeatedly logs frame-loss and reconnect attempts for the default camera source in this environment.

## Frontend Architecture

The frontend is a Next.js 16 application using React 19 and TypeScript.

### Frontend runtime and metadata

From frontend/package.json and frontend/src/app/layout.tsx:

- next 16.2.6
- react 19.2.4
- react-dom 19.2.4
- framer-motion 12.38.0
- leaflet 1.9.4
- react-leaflet 5.0.0
- lucide-react 1.14.0
- @supabase/supabase-js 2.105.4
- @tensorflow/tfjs 4.22.0
- @vladmandic/face-api 1.7.15

Metadata and theme:

- App title: SurakshaDrishti — Surveillance Command System
- App description: AI-powered real-time surveillance and harassment detection system
- Fonts used:
  - Rajdhani for body text
  - Share Tech Mono for monospace text
  - Orbitron loaded from Google Fonts for HUD styling

### Frontend routes

- /
  - Main dashboard and login gate.
- /surveillance
  - Redirects immediately back to /.

The /surveillance route exists as a redirect page to avoid module resolution errors from the previously empty route folder.

### Main dashboard flow in frontend/src/app/page.tsx

The root page coordinates the whole dashboard.

What it does:

- Shows the login screen until authentication succeeds.
- After login, renders the top bar and the active tab content.
- Subscribes to backend AI state through useBackendAI.
- Manages cameras through useCameraStore.
- Manages evidence through useEvidenceStore.
- Plays and stops the alarm sound through /sound/alarm.wav.
- Shows alert banners when violence, alarm, or recording are active.
- Handles event propagation from the backend via custom window events.

Tabs available in the main dashboard:

- dashboard
- evidence
- map
- admin

Admin tab availability is controlled by the isAdmin state.

### Frontend stores and hooks

#### useBackendAI

Purpose:

- Connects to ws://localhost:8765/ws/detections by default.
- Handles heartbeat pings.
- Reconnects using exponential backoff.
- Falls back to /api/health polling.
- Dispatches custom browser events for evidence and dispatch updates.

Important frontend events emitted from the hook:

- new-evidence-recorded
- ws-dispatch-accepted
- ws-need-more-help
- ws-evidence-resolved
- ws-violence-detected

#### useCameraStore

Purpose:

- Stores camera entries in localStorage under sd_cameras_v2.
- Supports add, remove, status update, and generic update operations.
- Sanitizes loaded localStorage data.

Camera entry fields:

- id
- type
- label
- url
- lat
- lng
- addedAt
- status

#### useEvidenceStore

Purpose:

- Stores evidence entries in localStorage under sd_evidence_v2.
- Syncs with backend /api/evidence on load.
- Seeds demo evidence when no backend data exists and no local cache is present.
- Syncs dispatch, escalation, resolve, and clear actions back to the backend.
- Handles custom browser events from the WebSocket hook.

Evidence entry fields:

- id
- cameraId
- cameraLabel
- timestamp
- isoTime
- confidence
- type
- thumbnail
- videoUrl
- snapshotUrl
- duration
- maleCount
- femaleCount
- weaponDetected
- weaponType
- lat
- lng
- locationName
- status
- authorityStation
- dispatchTime

Seed evidence examples created by the hook:

- LIVE WEBCAM TEST
- Main Entrance CCTV
- Parking Zone B

#### usePoliceStore

Purpose:

- Maintains police station account data.
- Uses localStorage key sd_police_stations.
- Falls back to seed accounts when the backend or Supabase is unavailable.
- Syncs the table police_stations to Supabase when possible.
- Listens for real-time postgres_changes events.

Seed police station accounts:

- ST-01 - Mumbai Headquarters - South Mumbai - adminpassword
- ST-02 - Bandra Police Station - West Bandra - bandrapassword
- ST-03 - Delhi Central Division - Central Delhi - delhipassword

### Frontend authentication flow

The login screen in frontend/src/components/LoginScreen.tsx uses several paths:

- Master admin bypass:
  - Operator ID: ADMIN_001
  - Passphrase: admin@123
- Supabase Auth if the input looks like an email address.
- Supabase table lookup against police_stations.
- Local storage fallback cache.
- Hardcoded seed station accounts if no cache exists.

Authentication state is saved in localStorage as sd_auth for 24 hours.

Important auth behavior:

- Email-based Supabase logins are treated as admin logins in the current UI.
- Station logins are treated as non-admin.
- The login UI accepts either an email or operator id in the first field.

### Frontend component inventory

The src/components directory contains:

- AddCameraModal.tsx
- AdminPanel.tsx
- AlertBanner.tsx
- Analysis.tsx
- CameraFeed.tsx
- CameraGrid.tsx
- CameraManagement.tsx
- EvidenceDetail.tsx
- EvidenceVault.tsx
- FakeCameraCanvas.tsx
- LiveIncidentCard.tsx
- LiveMap.tsx
- LoginScreen.tsx
- MapClient.tsx
- MiniMap.tsx
- StatsPanel.tsx
- Toast.tsx
- TopBar.tsx

Main usage from the dashboard page:

- LoginScreen handles authentication.
- TopBar renders navigation, status, clock, current user, and logout.
- CameraGrid shows live camera feeds and calls detection handlers.
- StatsPanel summarizes status, counts, and telemetry.
- EvidenceVault displays and manages evidence records.
- LiveMap renders map-based camera and incident views.
- AdminPanel exposes administrative controls for admins.
- Toast renders transient system notifications.

### Frontend visual design system

Defined in frontend/src/app/globals.css:

- Dark surveillance HUD theme.
- Custom CSS variables for background, borders, accent colors, safe/warning/danger states, and text levels.
- Scanline overlay on the body.
- Custom scrollbar styling.
- Corner bracket styles for card framing.
- Animated blink, pulse, alert-flash, and marker-pulse keyframes.

Color variables:

- --bg
- --bg2
- --bg3
- --panel
- --border
- --border-glow
- --accent
- --accent2
- --safe
- --warning
- --danger
- --danger2
- --text
- --text-dim
- --text-bright

## Data Flow Summary

1. The frontend login screen authenticates the operator.
2. Once logged in, the dashboard mounts and opens the backend WebSocket.
3. The backend broadcasts detection state and incident events.
4. The frontend reacts to updates, shows alerts, and plays the alarm.
5. Evidence records are stored locally and synchronized to the backend.
6. Dispatch and resolution actions are mirrored back to the backend API.
7. Camera and police-station state are cached locally for offline resilience.

## Dataset and Training Assets

Dataset directory:

- Dataset/normal
- Dataset/violence

Training scripts in backend/train:

- compile_model.py
- train_violence.py

These files indicate the project includes a training/export workflow for the violence model.

## Raspberry Pi Assets

The raspberry_pi directory contains edge-deployment and detector variants:

- best_violence_model.keras
- gui_detector_pi.py
- gui_violence_detector.py
- violence_detector_pi_simple.py
- violence_detector_pi.py
- violence_model.tflite

The grep scan of these scripts shows hardcoded email alert configuration and SMTP-based alert sending logic.

## Sound Assets

Sound assets are used for alarm playback.

Observed sound locations:

- frontend/public/sound
- frontend/sound
- root sound directory

The dashboard currently loads /sound/alarm.wav from the frontend public path.

## Current Verified Run State

In this workspace session, the project was successfully started locally:

- Backend: http://localhost:8765
- Frontend: http://localhost:3000

Verified login credential:

- Operator ID: ADMIN_001
- Passphrase: admin@123

The frontend also has demo station credentials available in the fallback path:

- Mumbai Headquarters / adminpassword
- Bandra Police Station / bandrapassword
- Delhi Central Division / delhipassword

## Known Operational Notes

- Supabase access is currently unresolved in this environment because the configured host does not resolve.
- The UI falls back to local storage and seed data when Supabase is unavailable.
- The backend still boots and serves the dashboard even when the weapon detector model file is missing.
- The backend frame processor is noisy in this environment because it is repeatedly reconnecting to the default webcam source.

## Practical Setup Summary

If you want the shortest working startup path:

1. Start the backend from the repository root with ./start_backend.sh.
2. Start the frontend from frontend with npm run dev.
3. Open the dashboard in the browser.
4. Sign in using ADMIN_001 / admin@123.

## Notes For Future Maintenance

- Consider aligning the weapon detector model filename with what backend/main.py expects.
- Replace hardcoded credentials with environment-backed or proper auth before production use.
- Decide whether the Supabase dependency should be mandatory or optional and document the expected environment variables.
- If the camera source is not present on a machine, the backend should be run with a real source or the processor should be configured for upload-only / test mode.
