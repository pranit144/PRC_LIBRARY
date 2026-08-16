from __future__ import annotations

import asyncio
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    """Tracks active WebSocket connections per run_id and broadcasts
    newly received events to them."""

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, run_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(run_id, []).append(ws)

    async def disconnect(self, run_id: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self._connections.get(run_id, [])
            if ws in conns:
                conns.remove(ws)

    async def broadcast(self, run_id: str, message: dict) -> None:
        async with self._lock:
            conns = list(self._connections.get(run_id, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections.get(run_id, []):
                        self._connections[run_id].remove(ws)
