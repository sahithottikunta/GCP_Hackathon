"""
SENTINEL — Agent Orchestrator
Coordinates multiple agents and routes inputs to the correct handler.

This is the central brain that:
  1. Manages agent lifecycle per client
  2. Routes voice/text/image inputs
  3. Coordinates anomaly alerts with voice responses
  4. Handles the real-time market data broadcast
"""

from __future__ import annotations
import asyncio
import time
import base64
from typing import Dict, Optional
from agents.voice_agent import VoiceAgent
from agents.anomaly_agent import AnomalyAgent
from agents.vision_agent import VisionAgent
from services.websocket_manager import ws_manager
from services.data_stream import market_stream
from utils import get_logger

log = get_logger("orchestrator")


class AgentOrchestrator:
    """
    Multi-agent orchestrator managing:
    - One VoiceAgent per connected client
    - Shared AnomalyAgent for market surveillance
    - Shared VisionAgent for image analysis
    - Market data broadcasting to all clients
    """

    def __init__(self):
        self.voice_agents: Dict[str, VoiceAgent] = {}
        self.anomaly_agent = AnomalyAgent()
        self.vision_agent = VisionAgent()
        self._market_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        log.info("Agent Orchestrator initialized")

    # ── Client Lifecycle ──────────────────────────────

    async def register_client(self, client_id: str):
        """Create a voice agent for a new client connection."""
        agent = VoiceAgent(client_id)
        self.voice_agents[client_id] = agent
        log.info(f"Registered voice agent for client {client_id} (total: {len(self.voice_agents)})")

    async def unregister_client(self, client_id: str):
        """Remove a client's voice agent."""
        self.voice_agents.pop(client_id, None)
        log.info(f"Unregistered client {client_id} (remaining: {len(self.voice_agents)})")

    # ── Input Routing ─────────────────────────────────

    async def handle_audio(self, client_id: str, audio_bytes: bytes):
        """Route audio input to the client's voice agent."""
        agent = self.voice_agents.get(client_id)
        if agent:
            await agent.process_audio(audio_bytes)

    async def handle_text(self, client_id: str, text: str):
        """Route text input to the client's voice agent."""
        agent = self.voice_agents.get(client_id)
        if agent:
            await agent.process_text(text)

    async def handle_image(self, client_id: str, image_base64: str, query: str = ""):
        """Route image to vision agent, then speak the result via voice agent."""
        # Run vision analysis
        result = await self.vision_agent.analyze(
            image_base64=image_base64,
            query=query,
        )

        # Send structured result to client
        await ws_manager.send_json(client_id, {
            "event": "vision_analysis",
            "data": result,
        })

        # Have the voice agent speak the analysis
        agent = self.voice_agents.get(client_id)
        if agent:
            await agent.process_text(f"Based on the chart analysis: {result['analysis']}")

        return result

    # ── Market Data Broadcasting ──────────────────────

    async def start_market_stream(self):
        """Start the market data stream and wire up broadcasting."""

        async def broadcast_tick(combined: dict):
            """Send combined multi-ticker payload to all subscribed clients."""
            await ws_manager.broadcast_channel("market_data", {
                "event": "market_tick",
                "data": combined,
            })

        async def handle_anomaly(detection: dict):
            """When ML detects an anomaly, explain it and broadcast."""
            explanation = await self.anomaly_agent.explain_anomaly(detection)

            alert_data = {
                "event": "anomaly_alert",
                "data": {
                    "detection": detection,
                    "explanation": explanation,
                    "timestamp": time.time(),
                },
            }
            await ws_manager.broadcast_channel("alerts", alert_data)

            # Proactively speak critical alerts to all connected voice agents
            if detection.get("severity") in ("CRITICAL", "HIGH"):
                for cid, agent in self.voice_agents.items():
                    # Only interrupt idle agents
                    if agent.state.status.value in ("IDLE", "LISTENING"):
                        log.info(f"Proactive alert to client {cid}: {detection['primary_type']}")
                        await ws_manager.send_json(cid, {
                            "event": "proactive_alert",
                            "data": {
                                "explanation": explanation,
                                "severity": detection["severity"],
                            },
                        })
                        # Synthesize and send audio
                        from services.llm_service import llm_service
                        audio = await llm_service.synthesize_speech(explanation)
                        if audio:
                            await ws_manager.send_json(cid, {
                                "event": "agent_audio",
                                "data": {
                                    "audio_base64": base64.b64encode(audio).decode(),
                                    "format": "mp3",
                                    "text": explanation,
                                    "is_alert": True,
                                },
                            })

        market_stream.on_tick(broadcast_tick)
        market_stream.on_anomaly(handle_anomaly)

        self._market_task = asyncio.create_task(
            market_stream.start(interval_ms=500)
        )
        log.info("Market stream + anomaly pipeline started")

    def stop_market_stream(self):
        """Stop the market data stream."""
        market_stream.stop()
        if self._market_task:
            self._market_task.cancel()

    # ── Status ────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "connected_clients": len(self.voice_agents),
            "market_running": market_stream.running,
            "total_ticks": market_stream.tick_count,
            "anomaly_stats": self.anomaly_agent.get_stats(),
            "agents": {
                cid: agent.get_state()
                for cid, agent in self.voice_agents.items()
            },
        }


# Singleton
orchestrator = AgentOrchestrator()
