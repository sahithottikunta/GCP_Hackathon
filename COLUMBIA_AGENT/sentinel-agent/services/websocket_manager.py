"""
SENTINEL — WebSocket Connection Manager
Handles multiple client connections, room-based broadcasting, and message routing.
"""

from __future__ import annotations
import asyncio
import json
import time
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket
from utils import get_logger

log = get_logger("ws")


class ConnectionManager:
    """
    Manages WebSocket connections with support for:
    - Multiple concurrent clients
    - Channel-based subscription (market_data, alerts, voice, vision)
    - Targeted and broadcast messaging
    - Graceful disconnect handling
    """

    def __init__(self):
        # client_id → WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        # channel → set of client_ids
        self.channels: Dict[str, Set[str]] = {
            "market_data": set(),
            "alerts": set(),
            "voice": set(),
            "vision": set(),
            "agent_status": set(),
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str, channels: list[str] = None):
        """Accept connection and subscribe to channels."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
            subscribe_to = channels or list(self.channels.keys())
            for ch in subscribe_to:
                if ch in self.channels:
                    self.channels[ch].add(client_id)

        log.info(f"Client connected: {client_id} → channels: {subscribe_to}")

    async def disconnect(self, client_id: str):
        """Remove client from all channels and close connection."""
        async with self._lock:
            self.active_connections.pop(client_id, None)
            for ch in self.channels.values():
                ch.discard(client_id)
        log.info(f"Client disconnected: {client_id}")

    async def send_json(self, client_id: str, data: dict):
        """Send JSON message to a specific client."""
        ws = self.active_connections.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                log.debug(f"Send failed for {client_id}: {e}")
                # Don't disconnect on transient errors — only on connection closed
                try:
                    # Check if connection is still alive
                    await ws.send_json({"event": "ping"})
                except Exception:
                    await self.disconnect(client_id)

    async def send_bytes(self, client_id: str, data: bytes):
        """Send binary data to a specific client (for audio)."""
        ws = self.active_connections.get(client_id)
        if ws:
            try:
                await ws.send_bytes(data)
            except Exception:
                await self.disconnect(client_id)

    async def broadcast_channel(self, channel: str, data: dict):
        """Broadcast JSON message to all clients subscribed to a channel."""
        client_ids = self.channels.get(channel, set()).copy()
        disconnected = []
        for cid in client_ids:
            ws = self.active_connections.get(cid)
            if ws:
                try:
                    await ws.send_json(data)
                except Exception as e:
                    log.debug(f"Broadcast to {cid} failed: {e}")
                    disconnected.append(cid)

        for cid in disconnected:
            await self.disconnect(cid)

    async def broadcast_all(self, data: dict):
        """Broadcast to all connected clients."""
        for cid in list(self.active_connections.keys()):
            await self.send_json(cid, data)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)

    def get_channel_clients(self, channel: str) -> Set[str]:
        return self.channels.get(channel, set()).copy()


# Singleton
ws_manager = ConnectionManager()
