"""
Suraksha Drishti — FastAPI Backend
====================================
Serves:
  - WebSocket at  ws://localhost:8765/ws/detections
  - MJPEG stream  http://localhost:8765/api/stream/mjpeg
  - REST health   http://localhost:8765/api/health

Architecture:
  ┌──────────────┐    queue    ┌──────────────┐
  │ Capture Thd  │──────────►│ AI Proc Thd  │
  │ (OpenCV cam) │            │ ONNX + ViT   │
  └──────┬───────┘            └──────┬───────┘
         │  JPEG                     │  Result
         ▼                           ▼
  ┌──────────────────────────────────────────┐
  │         FastAPI (async)                  │
  │  /api/stream/mjpeg  /ws/detections       │
  └──────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "models" / "best.onnx")

sys.path.insert(0, str(BASE_DIR))
from services.weapon_detector import WeaponDetector
from services.violence_detector import ViolenceDetector
from services.frame_processor import FrameProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("suraksha")

# ── Global services ───────────────────────────────────────────────────────────
weapon_detector = WeaponDetector(MODEL_PATH)
violence_detector = ViolenceDetector()
frame_processor: FrameProcessor | None = None

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    """Manages all active WebSocket clients."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws) if ws in self.active else None
        logger.info(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        """Send detection payload to all connected clients."""
        if not self.active:
            return
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_processor

    logger.info("=" * 60)
    logger.info("  SURAKSHA DRISHTI — AI Surveillance Backend v3.0")
    logger.info("=" * 60)

    # Load AI models
    logger.info("[Boot] Loading weapon detector (ONNX)...")
    w_ok = weapon_detector.load()
    logger.info(f"[Boot] Weapon detector: {'✓ READY' if w_ok else '✗ DISABLED'}")

    logger.info("[Boot] Loading violence detector (ViT)...")
    v_ok = violence_detector.load()
    logger.info(f"[Boot] Violence detector: {'✓ READY' if v_ok else '✗ DISABLED'}")

    # Start frame processor
    camera_idx = int(os.getenv("CAMERA_INDEX", "0"))
    frame_processor = FrameProcessor(weapon_detector, violence_detector, camera_index=camera_idx)
    frame_processor.start()

    # Start background WebSocket broadcaster
    broadcast_task = asyncio.create_task(broadcast_loop())

    logger.info("[Boot] Backend ready!")
    logger.info("  MJPEG Stream: http://localhost:8765/api/stream/mjpeg")
    logger.info("  WebSocket:    ws://localhost:8765/ws/detections")
    logger.info("  Health:       http://localhost:8765/api/health")

    yield  # App runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[Shutdown] Stopping services...")
    broadcast_task.cancel()
    if frame_processor:
        frame_processor.stop()
    logger.info("[Shutdown] Done")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Suraksha Drishti API",
    description="Real-time AI surveillance backend",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background broadcaster ────────────────────────────────────────────────────
async def broadcast_loop():
    """
    Async task: reads detection results every 0.5s,
    broadcasts lightweight JSON to all WebSocket clients.
    """
    logger.info("[Broadcaster] Started")
    prev_result = {}

    while True:
        try:
            await asyncio.sleep(0.5)   # 2 updates/sec — low overhead

            if not frame_processor:
                continue

            result = frame_processor.get_detection_result()

            # Only broadcast when something changed (saves bandwidth)
            payload = {
                "weapon_detected": result["weapon_detected"],
                "violence_detected": result["violence_detected"],
                "weapon_label": result["weapon_label"],
                "weapon_confidence": result["weapon_confidence"],
                "violence_label": result["violence_label"],
                "violence_confidence": result["violence_confidence"],
                "male_count": result["male_count"],
                "female_count": result["female_count"],
                "timestamp": result["timestamp"],
                "frame_count": result["frame_count"],
            }

            if payload != prev_result:
                await manager.broadcast(payload)
                prev_result = payload.copy()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Broadcaster] Error: {e}")
            await asyncio.sleep(1)


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "weapon_detector": weapon_detector.loaded,
        "violence_detector": violence_detector.loaded,
        "cameras_active": frame_processor is not None and frame_processor._running,
    })


@app.get("/api/detection/latest")
async def latest_detection():
    """Return latest detection result as JSON (polling fallback)."""
    if not frame_processor:
        return JSONResponse({"error": "not started"}, status_code=503)
    return JSONResponse(frame_processor.get_detection_result())


@app.get("/api/stream/mjpeg")
async def mjpeg_stream():
    """
    MJPEG stream endpoint.
    Frontend displays this in an <img> tag for live webcam feed.
    AI overlays are drawn directly on frames by FrameProcessor.
    """
    async def generate():
        while True:
            if frame_processor is None:
                await asyncio.sleep(0.1)
                continue

            jpg = frame_processor.get_latest_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(1 / 20)  # 20 FPS MJPEG (smooth video)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection alerts.
    Sends lightweight JSON: weapon/violence status, confidence, counts.
    """
    await manager.connect(websocket)
    try:
        # Send initial state on connect
        if frame_processor:
            await websocket.send_text(json.dumps(frame_processor.get_detection_result()))

        # Keep connection alive — listen for pings
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Echo back pings
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "keepalive"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        manager.disconnect(websocket)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        reload=False,           # disable reload in production
        workers=1,              # single worker — avoids model duplication
        log_level="info",
        access_log=False,       # disable request logs for performance
    )
