"""
Suraksha Drishti — FastAPI Backend v4.0
========================================
Production-grade AI surveillance API server.

Serves:
  WebSocket  ws://localhost:8765/ws/detections     ← real-time detection events
  MJPEG      http://localhost:8765/api/stream/mjpeg ← live annotated video feed
  REST       http://localhost:8765/api/health        ← health check
  REST       http://localhost:8765/api/model-status  ← model load status + stats
  REST       http://localhost:8765/api/detection/latest ← latest JSON (polling)

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Thread 1: webcam-capture                                   │
  │    OpenCV VideoCapture → frame queue + MJPEG JPEG buffer    │
  └───────────────────────────┬─────────────────────────────────┘
                              │ frame queue (maxsize=3)
  ┌───────────────────────────▼─────────────────────────────────┐
  │  Thread 2: ai-process                                       │
  │    HumanDetector → PoseEstimator → ViolenceDetector        │
  │                  → WeaponDetector → DetectionResult         │
  └───────────────────────────┬─────────────────────────────────┘
                              │ shared DetectionResult (lock)
  ┌───────────────────────────▼─────────────────────────────────┐
  │  FastAPI (async, main thread)                               │
  │    /api/stream/mjpeg      → StreamingResponse (MJPEG)       │
  │    /ws/detections         → WebSocket broadcast             │
  │    /api/health            → JSON                            │
  │    /api/model-status      → JSON                            │
  └─────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import shutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODEL_DIR       = BASE_DIR / "models"
WEAPON_MODEL    = str(MODEL_DIR / "best.onnx")
HUMAN_MODEL     = str(MODEL_DIR / "yolov8n.pt")
POSE_MODEL      = str(MODEL_DIR / "pose_landmarker_lite.task")

sys.path.insert(0, str(BASE_DIR))
from services.human_detector   import HumanDetector
from services.weapon_detector  import WeaponDetector
from services.violence_detector import ViolenceDetector
from services.pose_estimator   import PoseEstimator
from services.frame_processor  import FrameProcessor
from services.evidence_manager import EvidenceManager
from services.alert_manager    import AlertManager
from services.dispatch_manager import DispatchManager
from fastapi.staticfiles       import StaticFiles

# ── Logging ───────────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("suraksha")

# ── Global service instances ──────────────────────────────────────────────────
human_detector    = HumanDetector(HUMAN_MODEL)
weapon_detector   = WeaponDetector(WEAPON_MODEL)
violence_detector = ViolenceDetector()
pose_estimator    = PoseEstimator(POSE_MODEL)
evidence_manager  = EvidenceManager(output_dir=str(BASE_DIR / "recordings"))
alert_manager     = AlertManager()
dispatch_manager  = DispatchManager()
frame_processors: dict[str, FrameProcessor] = {}
latest_unclaimed_uploads: list[str] = []

def get_processor(cam_id: str, lat: float = None, lng: float = None, label: str = None) -> FrameProcessor:
    """Retrieves or dynamically instantiates an independent FrameProcessor session."""
    global latest_unclaimed_uploads
    if cam_id in frame_processors:
        proc = frame_processors[cam_id]
        if lat is not None:
            proc.latitude = lat
        if lng is not None:
            proc.longitude = lng
        if label is not None:
            proc.location_name = label
        return proc

    # Dynamically resolve camera source
    source = 0  # Default to primary local webcam
    if cam_id != "default":
        if latest_unclaimed_uploads:
            source = latest_unclaimed_uploads.pop(0)
            logger.info(f"[Registry] Mapping upload camera card '{cam_id}' to source={source}")
        else:
            logger.info(f"[Registry] Mapping new camera card '{cam_id}' to default webcam source=0")
    else:
        logger.info(f"[Registry] Initializing default unit '{cam_id}' using source=0")

    proc = FrameProcessor(
        weapon_detector   = weapon_detector,
        violence_detector = violence_detector,
        human_detector    = human_detector,
        pose_estimator    = pose_estimator,
        camera_index      = source,
        evidence_manager  = evidence_manager,
        alert_manager     = alert_manager,
    )
    if lat is not None:
        proc.latitude = lat
    if lng is not None:
        proc.longitude = lng
    if label is not None:
        proc.location_name = label

    proc.start()
    frame_processors[cam_id] = proc
    return proc

frame_processor: FrameProcessor | None = None

# Track startup status
_startup_time = datetime.utcnow()
_model_status: dict = {}


# ── WebSocket Connection Manager ──────────────────────────────────────────────
class ConnectionManager:
    """Thread-safe WebSocket client manager."""

    def __init__(self):
        self.active: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        logger.info(f"[WS] Client connected. Total: {len(self.active)}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self.active:
                try:
                    self.active.remove(ws)
                except ValueError:
                    pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        """Send detection payload to all connected clients. Dead connections are pruned."""
        if not self.active:
            return
        msg  = json.dumps(data)
        dead = []
        async with self._lock:
            targets = list(self.active)

        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_processor, _model_status

    logger.info("=" * 60)
    logger.info("  SURAKSHA DRISHTI — AI Surveillance Backend v4.0")
    logger.info("=" * 60)

    # ── Load AI models ────────────────────────────────────────────────────────
    logger.info("[Boot] Loading human detector (YOLOv8n ONNX)...")
    h_ok = human_detector.load()
    logger.info(f"[Boot] Human detector:   {'✓ READY' if h_ok else '✗ DISABLED'}")

    logger.info("[Boot] Loading weapon detector (custom ONNX)...")
    w_ok = weapon_detector.load()
    logger.info(f"[Boot] Weapon detector:  {'✓ READY' if w_ok else '✗ DISABLED'}")

    logger.info("[Boot] Loading pose estimator (MediaPipe Lite)...")
    p_ok = pose_estimator.load()
    logger.info(f"[Boot] Pose estimator:   {'✓ READY' if p_ok else '✗ DISABLED'}")

    logger.info("[Boot] Loading violence detector (ViT)...")
    v_ok = violence_detector.load()
    logger.info(f"[Boot] Violence detector: {'✓ READY' if v_ok else '✗ DISABLED'}")

    _model_status = {
        "human_detector":    {"loaded": h_ok, "model": "yolov8n.pt"},
        "weapon_detector":   {"loaded": w_ok, "model": "best.onnx"},
        "pose_estimator":    {"loaded": p_ok, "model": "pose_landmarker_lite.task"},
        "violence_detector": {"loaded": v_ok, "model": "jaranohaal/vit-base-violence-detection"},
    }

    # ── Start evidence manager and frame processor registry ───────────────────
    evidence_manager.start()
    
    # Pre-boot the default webcam processor session
    get_processor("default")

    # ── Start background broadcaster ──────────────────────────────────────────
    broadcast_task = asyncio.create_task(broadcast_loop())

    logger.info("[Boot] Backend fully operational!")
    logger.info(f"  MJPEG Stream:   http://localhost:8765/api/stream/mjpeg")
    logger.info(f"  WebSocket:      ws://localhost:8765/ws/detections")
    logger.info(f"  Health:         http://localhost:8765/api/health")
    logger.info(f"  Model Status:   http://localhost:8765/api/model-status")

    yield   # ← Application runs here ←

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[Shutdown] Stopping active camera sessions...")
    broadcast_task.cancel()
    for proc in list(frame_processors.values()):
        proc.stop()
    frame_processors.clear()
    evidence_manager.stop()
    logger.info("[Shutdown] Clean shutdown complete")


# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Suraksha Drishti API",
    description = "Real-time AI surveillance backend — Human · Pose · Violence · Weapon",
    version     = "4.0.0",
    lifespan    = lifespan,
)

# CORS — allow the Next.js frontend (and any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve saved recordings dynamically
app.mount("/recordings", StaticFiles(directory=str(BASE_DIR / "recordings")), name="recordings")


# ── Background broadcaster ────────────────────────────────────────────────────
async def broadcast_loop():
    """
    Async task: reads latest detection results from all active processors,
    aggregates them, and broadcasts the most significant events to all WS clients.
    """
    logger.info("[Broadcaster] Started (3 Hz)")
    prev_payload: dict = {}
    prev_recording_state = False

    while True:
        try:
            await asyncio.sleep(0.333)

            processors = list(frame_processors.values())
            if not processors:
                continue

            # Aggregate active frame processor states and select the one showing the maximum threat score
            results = [p.get_detection_result() for p in processors]
            result = max(results, key=lambda r: r.get("threat_score", 0.0))

            # Check evidence lifecycle to stop active recordings and generate clips
            new_ev = evidence_manager.update_lifecycle()
            if new_ev:
                logger.info(f"[Broadcaster] Auto-recorded new evidence! Broadcasting event: {new_ev['id']}")
                asyncio.create_task(manager.broadcast({
                    "type": "new_evidence",
                    "evidence": new_ev
                }))

            # Global real-time violence detection start broadcast trigger
            is_rec = evidence_manager.is_recording
            if is_rec and not prev_recording_state:
                ev_id = f"EVD-REC-{int(evidence_manager.start_time)}"
                logger.info(f"[Broadcaster] Auto-recording started. Broadcasting violence_detected event for {ev_id}")
                asyncio.create_task(manager.broadcast({
                    "event": "violence_detected",
                    "alarm": True,
                    "recording": True,
                    "evidence_id": ev_id
                }))
            prev_recording_state = is_rec

            # Build the lightweight WebSocket payload (frontend contract)
            payload = {
                "weapon_detected":     result["weapon_detected"],
                "weapon_label":        result["weapon_label"],
                "weapon_confidence":   result["weapon_confidence"],
                "violence_detected":   result["violence_detected"],
                "violence_label":      result["violence_label"],
                "violence_confidence":  result["violence_confidence"],
                "action":              result["action"],
                "aggression_score":    result["aggression_score"],
                "fps":                 result["fps"],
                "person_count":        result["person_count"],
                "male_count":          result["male_count"],
                "female_count":        result["female_count"],
                "total_persons":       result["male_count"] + result["female_count"],
                "timestamp":           result["timestamp"],
                "frame_count":         result["frame_count"],
                
                # Alarm & Recording states
                "violence":            result["violence_detected"] or result["weapon_detected"],
                "recording":           evidence_manager.is_recording,
                "alarm_active":        alert_manager.alarm_active,
                "event":               alert_manager.threat_type,

                # Upgraded Temporal dynamic metrics
                "threat_level":        result["threat_level"],
                "threat_score":        result["threat_score"],
                "fall_detected":       result["fall_detected"],
                "repeated_strikes":    result["repeated_strikes"],
                "chasing_detected":    result["chasing_detected"],
                "motion_intensity":    result["motion_intensity"],

                # Interaction Aware fields (Task 1, 2, 6)
                "punch_detected":     result["punch_detected"],
                "attacker_bbox":      result["attacker_bbox"],
                "victim_bbox":        result["victim_bbox"],
                "punch_arrow":        result["punch_arrow"],
            }

            # Only broadcast when something meaningful changed
            changed = (
                payload.get("weapon_detected")     != prev_payload.get("weapon_detected")
                or payload.get("violence_detected") != prev_payload.get("violence_detected")
                or payload.get("person_count")      != prev_payload.get("person_count")
                or payload.get("recording")         != prev_payload.get("recording")
                or payload.get("alarm_active")      != prev_payload.get("alarm_active")
                or payload.get("threat_level")      != prev_payload.get("threat_level")
                or payload.get("fall_detected")     != prev_payload.get("fall_detected")
                or payload.get("repeated_strikes")  != prev_payload.get("repeated_strikes")
                or payload.get("chasing_detected")  != prev_payload.get("chasing_detected")
                or abs(payload.get("aggression_score", 0) - prev_payload.get("aggression_score", 0)) > 0.05
            )

            if changed or not prev_payload:
                await manager.broadcast(payload)
                prev_payload = payload.copy()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Broadcaster] Error in broadcast loop: {e}")
            await asyncio.sleep(1)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Health check")
async def health():
    """
    Returns system health status.
    Checked by the frontend every 10s to show backend indicator.
    """
    cameras_active = any(p._running for p in list(frame_processors.values()))
    return JSONResponse({
        "status":   "ok",
        "version":  "4.0.0",
        "uptime_s": round((datetime.utcnow() - _startup_time).total_seconds(), 1),
        "cameras_active": cameras_active,
        "weapon_detector":   weapon_detector.loaded,
        "violence_detector": (violence_detector.onnx_loaded or violence_detector.vit_loaded),
        "human_detector":    human_detector.loaded,
        "pose_estimator":    pose_estimator.loaded,
    })


@app.get("/api/model-status", summary="Model load status and inference stats")
async def model_status():
    """
    Returns detailed model status + inference performance metrics.
    Useful for debugging and the /admin panel.
    """
    fps = 0.0
    person_count = 0
    default_proc = frame_processors.get("default")
    if default_proc:
        r = default_proc.get_detection_result()
        fps          = r.get("fps", 0.0)
        person_count = r.get("person_count", 0)

    return JSONResponse({
        "models":       _model_status,
        "pipeline_fps": fps,
        "persons_in_frame": person_count,
        "weapon_avg_inference_ms": weapon_detector.avg_inference_ms,
        "startup_time": _startup_time.isoformat() + "Z",
    })


@app.get("/api/detection/latest", summary="Latest detection as JSON (polling fallback)")
async def latest_detection(camId: str = "default"):
    """
    Return the full latest detection result as JSON for a specific camera.
    Use this as a polling fallback when WebSocket is unavailable.
    """
    proc = get_processor(camId)
    if not proc:
        return JSONResponse({"error": f"Processor {camId} not found"}, status_code=404)
    return JSONResponse(proc.get_detection_result())


@app.get("/api/events/timeline", summary="Get recent AI surveillance events timeline")
async def get_events_timeline():
    """
    Returns the recent AI visual events timeline log (falls, chases, arm strikes) from all active processors.
    """
    events = []
    for proc in list(frame_processors.values()):
        events.extend(proc.event_manager.get_recent_events())
    # Sort descending by timestamp
    events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return JSONResponse(events[:50])


@app.post("/api/upload-video", summary="Upload a video file for real-time AI surveillance processing")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts video file uploads (mp4, mov, avi, mkv),
    saves them to a temporary path, and adds to unclaimed queue for dynamic session matching.
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        return JSONResponse({"status": "error", "message": f"Unsupported extension {ext}"}, status_code=400)

    # Save to a unique timestamped path to prevent multi-session collision
    temp_path = BASE_DIR / f"temp_upload_{int(time.time())}{ext}"

    # Save the file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"[Upload] Saved video file to {temp_path}. Queueing for multi-session camera card matching...")

    global latest_unclaimed_uploads
    latest_unclaimed_uploads.append(str(temp_path))

    return JSONResponse({
        "status": "success",
        "filename": file.filename,
        "path": str(temp_path),
        "message": "Video uploaded successfully. Queued for camera mapping."
    })


@app.post("/api/select-source", summary="Select frame processor camera source dynamically")
async def select_source(source: str, camId: str = "default"):
    """
    Dynamically switches the FrameProcessor input source.
    If source is '0', it parses as integer 0 (default webcam).
    """
    proc = get_processor(camId)
    if not proc:
        return JSONResponse({"status": "error", "message": f"Processor {camId} not found"}, status_code=404)

    try:
        actual_source = int(source)
    except ValueError:
        actual_source = source

    logger.info(f"[Source] Dynamically selecting input source for camera '{camId}' -> {actual_source}")
    proc.change_source(actual_source)
    return JSONResponse({"status": "success", "source": actual_source})


from pydantic import BaseModel

class DispatchRequest(BaseModel):
    station: str
    officer: str

class EscalateRequest(BaseModel):
    station: str
    lat: float
    lng: float

# ── Evidence DB & Dispatch Sync Routes ──────────────────────────────────────────

@app.get("/api/evidence", summary="Get all persistent evidence records")
async def get_evidence():
    return JSONResponse(evidence_manager.get_all_evidence())

@app.post("/api/evidence/{evidence_id}/dispatch", summary="Dispatch police to incident")
async def dispatch_evidence(evidence_id: str, req: DispatchRequest):
    dispatch_info = dispatch_manager.accept_dispatch(evidence_id, req.station, req.officer)
    if not dispatch_info:
        return JSONResponse({"status": "error", "message": "Incident already accepted or dispatched"}, status_code=409)
    
    # Update evidence card status in database
    updated_item = evidence_manager.update_evidence_status(evidence_id, {
        "status": "Police Dispatched",
        "authorityStation": req.station,
        "dispatchTime": datetime.now().strftime("%I:%M %p")
    })
    
    # Mute alarm globally
    alert_manager.mute_alarm_globally()
    
    # Broadcast to all websocket connections
    broadcast_data = {
        "event": "dispatch_accepted",
        "evidence_id": evidence_id,
        "station": req.station,
        "officer": req.officer,
        "status": "POLICE_DISPATCHED"
    }
    await manager.broadcast(broadcast_data)
    
    return JSONResponse({"status": "success", "evidence": updated_item})

@app.post("/api/evidence/{evidence_id}/escalate", summary="Escalate incident for more backup help")
async def escalate_evidence(evidence_id: str, req: EscalateRequest):
    dispatch_manager.escalate(evidence_id, req.station, req.lat, req.lng)
    
    # Update evidence card status in database
    updated_item = evidence_manager.update_evidence_status(evidence_id, {
        "status": "More Help Requested"
    })
    
    # Trigger emergency alarm globally again
    alert_manager.trigger_emergency("EMERGENCY_BACKUP")
    
    # Broadcast to all websocket connections
    broadcast_data = {
        "event": "need_more_help",
        "evidence_id": evidence_id,
        "station": req.station,
        "location": {
            "lat": req.lat,
            "lng": req.lng
        }
    }
    await manager.broadcast(broadcast_data)
    
    return JSONResponse({"status": "success", "evidence": updated_item})

@app.post("/api/evidence/{evidence_id}/resolve", summary="Mark incident as resolved")
async def resolve_evidence(evidence_id: str):
    # Update status in persistent store
    updated_item = evidence_manager.update_evidence_status(evidence_id, {
        "status": "Resolved"
    })
    
    # Clear tracking in dispatch manager
    dispatch_manager.reset_incident(evidence_id)
    
    # If no other incidents are active/unresolved, we can mute the alarm
    all_evs = evidence_manager.get_all_evidence()
    active_incidents = [e for e in all_evs if e.get("status") in ["Active", "More Help Requested"]]
    if not active_incidents:
        alert_manager.mute_alarm_globally()
        
    # Broadcast resolved state to all dashboards
    broadcast_data = {
        "event": "evidence_resolved",
        "evidence_id": evidence_id,
        "status": "Resolved"
    }
    await manager.broadcast(broadcast_data)
    
    return JSONResponse({"status": "success", "evidence": updated_item})

@app.post("/api/evidence/clear", summary="Clear persistent database records")
async def clear_evidence():
    evidence_manager.clear_all()
    return JSONResponse({"status": "success"})


@app.post("/api/acknowledge", summary="Acknowledge active alarm")
async def acknowledge_alarm():
    """
    Acknowledge the active alarm and mute audio alerts.
    """
    if alert_manager:
        success = alert_manager.acknowledge()
        return JSONResponse({"status": "success", "acknowledged": success})
    return JSONResponse({"status": "error", "message": "Alert manager offline"}, status_code=503)


@app.get("/api/stream/mjpeg", summary="Live annotated MJPEG video stream")
async def mjpeg_stream(camId: str = "default", lat: float = None, lng: float = None, label: str = None):
    """
    MJPEG video stream with AI overlays drawn by the backend for a specific camera card ID.
    """
    async def generate():
        while True:
            proc = get_processor(camId, lat, lng, label)
            if proc is None:
                await asyncio.sleep(0.05)
                continue

            jpg = proc.get_latest_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            # 20 FPS ceiling for MJPEG (smooth display without overloading)
            await asyncio.sleep(1 / 20)

    return StreamingResponse(
        generate(),
        media_type = "multipart/x-mixed-replace; boundary=frame",
        headers    = {
            "Cache-Control":    "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


# ── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection events.

    Protocol:
      → Client connects
      ← Server sends initial state snapshot
      → Client may send "ping" to keep connection alive
      ← Server responds with "pong"
      → Client may send gender JSON: {"type":"gender","male":2,"female":1}
      ← Server broadcasts detection updates whenever data changes

    Reconnect: handled by frontend (exponential backoff in useBackendAI.ts)
    Keepalive: 25s ping/pong interval in frontend
    """
    await manager.connect(websocket)

    try:
        # Send initial snapshot immediately on connect
        if frame_processor:
            initial = frame_processor.get_detection_result()
            await websocket.send_text(json.dumps(initial))

        # Listen for client messages (ping / gender updates)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)

                if data == "ping":
                    await websocket.send_text("pong")
                    continue

                # Handle gender count updates from frontend face-api
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "gender" and frame_processor:
                        frame_processor.update_gender_counts(
                            male   = int(msg.get("male", 0)),
                            female = int(msg.get("female", 0)),
                        )
                except (json.JSONDecodeError, ValueError):
                    pass  # ignore malformed messages

            except asyncio.TimeoutError:
                # Send keepalive when no client message received for 30s
                await websocket.send_text(json.dumps({"type": "keepalive"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WS] Connection error: {e}")
    finally:
        await manager.disconnect(websocket)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run(
        "main:app",
        host         = "0.0.0.0",
        port         = port,
        reload       = False,      # disable hot-reload in production
        workers      = 1,          # single worker — avoids model duplication in memory
        log_level    = "info",
        access_log   = False,      # disable per-request logs for performance
    )
