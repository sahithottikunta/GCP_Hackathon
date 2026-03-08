# 🛡️ SENTINEL — Live Real-Time Financial Analyst Voice Agent

> **Hackathon Track**: Live Agents — Real-time Interaction (Audio/Vision)

## What It Does

SENTINEL is a live, interruptible voice agent acting as a senior financial analyst.
Talk to it naturally, show it charts, and get real-time spoken analysis.

### Core Capabilities
- **Voice**: Natural conversation via WebSocket audio streaming
- **Interrupt**: Agent stops mid-sentence when you start speaking
- **Vision**: Upload/stream charts for AI visual analysis
- **ML Detection**: Real-time anomaly scoring on live market data
- **Multi-Agent**: Voice, Anomaly, Vision agents orchestrated together

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your API keys
python main.py              # → http://localhost:8000
```

## Architecture

```
Browser ──WebSocket──► FastAPI Server
  • mic audio chunks  ──►  Audio Pipeline (VAD → Whisper STT)
  • camera frames     ──►  Vision Agent (Claude/GPT-4o)
  • ◄── TTS audio     ◄──  Voice Agent (LLM → TTS)
  • ◄── market data   ◄──  Data Stream + ML Anomaly Detector
```
