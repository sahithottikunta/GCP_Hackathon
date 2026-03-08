"""
SENTINEL — Data Schemas
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ═══════════════════════════════════════════════════════
# MARKET DATA
# ═══════════════════════════════════════════════════════

class MarketTick(BaseModel):
    timestamp: float
    symbol: str = "SYNTH"
    price: float
    open: float
    high: float
    low: float
    volume: int
    volatility: float = 0.0
    anomaly_score: float = 0.0
    is_anomaly: bool = False


# ═══════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════

class AnomalySeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AnomalyType(str, Enum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    PRICE_DISLOCATION = "PRICE_DISLOCATION"
    VOLATILITY_REGIME = "VOLATILITY_REGIME"
    SPOOFING = "SPOOFING"
    WASH_TRADE = "WASH_TRADE"
    LAYERING = "LAYERING"
    CORRELATION_BREAK = "CORRELATION_BREAK"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"


class AnomalyAlert(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float = Field(ge=0.0, le=1.0)
    title: str
    description: str
    reasoning: str = ""
    metrics: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════
# VOICE / AUDIO
# ═══════════════════════════════════════════════════════

class TranscriptionResult(BaseModel):
    text: str
    confidence: float = 1.0
    is_partial: bool = False
    language: str = "en"


class AgentUtterance(BaseModel):
    text: str
    audio_base64: Optional[str] = None
    is_final: bool = True
    can_interrupt: bool = True


# ═══════════════════════════════════════════════════════
# VISION
# ═══════════════════════════════════════════════════════

class VisionAnalysis(BaseModel):
    description: str
    detected_elements: List[str] = []
    anomalies_spotted: List[str] = []
    recommendation: str = ""


# ═══════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class AgentState(BaseModel):
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []
    market_context: Dict[str, Any] = {}
    active_alerts: List[AnomalyAlert] = []


# ═══════════════════════════════════════════════════════
# WEBSOCKET MESSAGES
# ═══════════════════════════════════════════════════════

class WSMessage(BaseModel):
    event: str
    data: Dict[str, Any] = {}
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
