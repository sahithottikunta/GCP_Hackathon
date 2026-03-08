"""
SENTINEL — Vision Agent
Handles image/chart analysis via vision-capable LLMs.
Users can upload screenshots, chart images, or stream camera frames.
"""

from __future__ import annotations
import base64
import time
from typing import Optional
from services.llm_service import llm_service
from services.data_stream import market_stream
from utils import get_logger

log = get_logger("vision_agent")


class VisionAgent:
    """
    Analyzes financial charts and visual data.

    Supports:
    - Static image upload (base64)
    - Contextual analysis with current market data
    - Multiple query types (pattern recognition, anomaly spotting, etc.)
    """

    QUERY_TEMPLATES = {
        "general": (
            "Analyze this financial chart or trading screen. Describe key patterns, "
            "notable features, and any anomalies. Speak as a senior analyst."
        ),
        "anomaly": (
            "Focus on anomalies in this chart. Look for unusual volume bars, "
            "price dislocations, gaps, or pattern breaks. What stands out?"
        ),
        "trend": (
            "What is the primary trend in this chart? Identify support/resistance levels, "
            "trend lines, and potential reversal signals."
        ),
        "compare": (
            "Compare what you see in this chart against the current market context provided. "
            "Are there any divergences or confirmations?"
        ),
    }

    def __init__(self):
        self.analysis_history: list = []
        self.analysis_count = 0

    async def analyze(
        self,
        image_base64: str,
        query: Optional[str] = None,
        query_type: str = "general",
        media_type: str = "image/png",
        include_market_context: bool = True,
    ) -> dict:
        """
        Analyze an image and return structured results.

        Args:
            image_base64: Base64 encoded image data
            query: Custom query (overrides query_type template)
            query_type: One of general, anomaly, trend, compare
            media_type: MIME type of the image
            include_market_context: Whether to include current market data
        """
        self.analysis_count += 1
        start = time.time()

        # Build the query
        if query:
            final_query = query
        else:
            final_query = self.QUERY_TEMPLATES.get(query_type, self.QUERY_TEMPLATES["general"])

        # Add market context if requested
        if include_market_context:
            ctx = market_stream.get_snapshot()
            if ctx.get("status") != "no_data":
                final_query += (
                    f"\n\nCurrent market context: Price=${ctx.get('current_price', 'N/A')}, "
                    f"Volatility={ctx.get('current_volatility', 'N/A')}%, "
                    f"Recent anomalies={ctx.get('recent_anomalies', 0)}"
                )

        # Run vision analysis
        log.info(f"Running vision analysis (type={query_type})")
        analysis_text = await llm_service.analyze_image(
            image_base64=image_base64,
            query=final_query,
            media_type=media_type,
        )

        elapsed = time.time() - start

        result = {
            "analysis": analysis_text,
            "query_type": query_type,
            "processing_time_ms": round(elapsed * 1000),
            "timestamp": time.time(),
            "analysis_id": self.analysis_count,
        }

        self.analysis_history.append(result)
        log.info(f"Vision analysis complete in {elapsed:.2f}s")

        return result

    def get_history(self, n: int = 5) -> list:
        return self.analysis_history[-n:]
