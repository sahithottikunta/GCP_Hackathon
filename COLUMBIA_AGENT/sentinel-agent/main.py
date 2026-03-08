"""
SENTINEL — Main Server
FastAPI application with WebSocket endpoints for real-time voice, vision, and market data.

Endpoints:
  GET  /                    → Test client (static HTML)
  GET  /health              → Health check
  GET  /status              → Agent orchestrator status
  POST /api/chat            → Text-based chat (non-voice)
  POST /api/analyze-image   → Image analysis via vision agent
  WS   /ws/agent/{id}       → Full voice + data WebSocket
  WS   /ws/market           → Market data only WebSocket
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from agents.orchestrator import orchestrator
from services.websocket_manager import ws_manager
from services.data_stream import market_stream
from utils import get_logger

log = get_logger("server")


# ─── Lifespan ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    log.info("=" * 60)
    log.info("  SENTINEL — Real-Time Financial Analyst Agent")
    log.info("=" * 60)
    log.info(f"  Server:    http://{settings.host}:{settings.port}")
    log.info(f"  Gemini:    {'✓ configured' if settings.has_gemini else '✗ not configured'}")
    log.info("=" * 60)

    # Start market data stream
    await orchestrator.start_market_stream()

    yield

    # Shutdown
    orchestrator.stop_market_stream()
    log.info("SENTINEL shut down cleanly")


# ─── App ──────────────────────────────────────────────

app = FastAPI(
    title="SENTINEL",
    description="Real-Time Financial Analyst Voice Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ───────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "clients": ws_manager.client_count,
    }


@app.get("/status")
async def status():
    return orchestrator.get_status()


@app.post("/api/chat")
async def chat(payload: dict):
    """Text-based chat endpoint (for non-voice interaction)."""
    text = payload.get("message", "")
    client_id = payload.get("client_id", f"api-{uuid.uuid4().hex[:6]}")

    if not text:
        return JSONResponse({"error": "No message provided"}, status_code=400)

    # Use the LLM service directly for REST chat
    from services.llm_service import llm_service
    from services.data_stream import market_stream

    response = await llm_service.generate(
        user_message=text,
        market_context=market_stream.get_snapshot(),
    )

    return {"response": response, "client_id": client_id}


@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    query: str = Form(""),
):
    """Upload an image for vision analysis."""
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode()

    media_type = file.content_type or "image/png"

    result = await orchestrator.vision_agent.analyze(
        image_base64=image_b64,
        query=query,
        media_type=media_type,
    )

    return result


# ─── WebSocket: Full Agent Connection ─────────────────

@app.websocket("/ws/agent/{client_id}")
async def ws_agent(websocket: WebSocket, client_id: str):
    """
    Primary WebSocket for voice agent interaction.

    Client sends:
      - {"event": "audio", "data": "<base64 PCM16 audio>"}
      - {"event": "text", "data": {"message": "..."}}
      - {"event": "image", "data": {"image": "<base64>", "query": "..."}}
      - {"event": "interrupt", "data": {}}
      - Binary frames: raw PCM16 audio bytes

    Server sends:
      - {"event": "agent_status", "data": {"status": "..."}}
      - {"event": "transcription", "data": {"text": "...", "is_final": true}}
      - {"event": "agent_token", "data": {"token": "...", "partial": true}}
      - {"event": "agent_audio", "data": {"audio_base64": "...", "format": "mp3"}}
      - {"event": "agent_response_complete", "data": {"text": "..."}}
      - {"event": "market_tick", "data": {...}}
      - {"event": "anomaly_alert", "data": {...}}
      - {"event": "vision_analysis", "data": {...}}
      - {"event": "interrupt", "data": {"message": "..."}}
    """
    await ws_manager.connect(websocket, client_id)
    await orchestrator.register_client(client_id)

    # Send initial welcome
    await ws_manager.send_json(client_id, {
        "event": "connected",
        "data": {
            "client_id": client_id,
            "message": "SENTINEL agent connected. Ready for voice or text input.",
            "capabilities": ["voice", "text", "vision", "market_data", "anomaly_alerts"],
        },
    })

    try:
        while True:
            # Handle both text (JSON) and binary (audio) frames
            message = await websocket.receive()

            if "text" in message:
                # JSON message
                try:
                    data = json.loads(message["text"])
                    event = data.get("event", "")

                    if event == "text":
                        text = data.get("data", {}).get("message", "")
                        if text:
                            await orchestrator.handle_text(client_id, text)

                    elif event == "audio":
                        # Base64-encoded audio
                        audio_b64 = data.get("data", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await orchestrator.handle_audio(client_id, audio_bytes)

                    elif event == "image":
                        img_data = data.get("data", {})
                        image_b64 = img_data.get("image", "")
                        query = img_data.get("query", "")
                        if image_b64:
                            await orchestrator.handle_image(client_id, image_b64, query)

                    elif event == "interrupt":
                        agent = orchestrator.voice_agents.get(client_id)
                        if agent:
                            await agent.audio._handle_interrupt()

                    elif event == "ping":
                        await ws_manager.send_json(client_id, {"event": "pong", "data": {}})

                except json.JSONDecodeError:
                    log.warning(f"Invalid JSON from {client_id}")

            elif "bytes" in message:
                # Raw binary audio
                await orchestrator.handle_audio(client_id, message["bytes"])

    except WebSocketDisconnect:
        log.info(f"Client {client_id} disconnected")
    except Exception as e:
        log.error(f"WebSocket error for {client_id}: {e}")
    finally:
        await ws_manager.disconnect(client_id)
        await orchestrator.unregister_client(client_id)


# ─── WebSocket: Market Data Only ──────────────────────

@app.websocket("/ws/market")
async def ws_market(websocket: WebSocket):
    """Lightweight WebSocket for market data streaming only."""
    client_id = f"market-{uuid.uuid4().hex[:6]}"
    await ws_manager.connect(websocket, client_id, channels=["market_data", "alerts"])

    try:
        while True:
            # Just keep connection alive; data is pushed via broadcast
            msg = await websocket.receive_text()
            if msg == "ping":
                await ws_manager.send_json(client_id, {"event": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(client_id)


# ─── Serve Test Client ───────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_client():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body style="background:#0a1628;color:#00ffc8;font-family:monospace;padding:40px">
        <h1>SENTINEL Agent Server Running</h1>
        <p>WebSocket endpoint: ws://localhost:8000/ws/agent/{client_id}</p>
        <p>REST chat: POST /api/chat</p>
        <p>Status: GET /status</p>
        </body></html>
        """)


# ─── Entry Point ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
