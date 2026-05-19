"""
WebSocket Broadcaster — Suraksha Drishti AI
===========================================
Thread-safe async WebSocket channel broker coordinating real-time alerts,
police dispatches, and threat status across multiple dashboard clients.
"""

import json
import asyncio
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger("suraksha.websocket_broadcaster")

class WebSocketBroadcaster:
    """
    Asynchronous channel manager that handles dashboard connections
    and broadcasts threat events to multiple connected police stations.
    """
    def __init__(self):
        self.active_sockets: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket):
        """Registers a client terminal."""
        await websocket.accept()
        async with self._lock:
            self.active_sockets.append(websocket)
        logger.info(f"[WS Broadcaster] Connected station client. Pool count: {len(self.active_sockets)}")

    async def unregister(self, websocket: WebSocket):
        """Prunes disconnected client terminals."""
        async with self._lock:
            if websocket in self.active_sockets:
                self.active_sockets.remove(websocket)
        logger.info(f"[WS Broadcaster] Pruned client terminal. Pool count: {len(self.active_sockets)}")

    async def broadcast_event(self, data: dict):
        """Asynchronously sends synchronized threat and coordination events."""
        if not self.active_sockets:
            return
        
        payload = json.dumps(data)
        disconnected = []
        
        async with self._lock:
            current_clients = list(self.active_sockets)

        for ws in current_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            await self.unregister(ws)
