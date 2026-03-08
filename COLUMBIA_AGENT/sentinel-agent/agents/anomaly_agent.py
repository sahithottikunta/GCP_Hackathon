"""
SENTINEL — Anomaly Reasoning Agent
Takes raw ML detection results and generates human-readable verbal explanations.
This is what makes SENTINEL an *analyst* rather than just a detector.
"""

from __future__ import annotations
import time
from typing import Dict, List
from services.llm_service import llm_service
from services.data_stream import market_stream
from utils import get_logger

log = get_logger("anomaly_agent")

# Pre-built explanation templates for fast responses
# (used when LLM is unavailable or for low-severity alerts)
TEMPLATES = {
    "VOLUME_SPIKE": (
        "I'm seeing a significant volume spike — current volume is {volume_ratio:.1f}x "
        "the rolling average. This kind of surge typically indicates either institutional "
        "positioning, a news catalyst, or potentially manipulative activity. "
        "The Z-score on volume hit {z_volume:.1f}, which puts this in the top {percentile}% of observations."
    ),
    "PRICE_DISLOCATION": (
        "There's a notable price dislocation here. The return just registered a {z_return:.1f} "
        "sigma move, which is statistically rare under normal conditions. "
        "This could indicate a large block trade, a stop-loss cascade, or an external shock."
    ),
    "VOLATILITY_REGIME": (
        "We're seeing a volatility regime shift. The short-term vol is now {ratio:.1f}x "
        "the longer-term baseline — that's a {regime} regime. When volatility transitions "
        "this sharply, it often precedes a larger directional move or signals changing market microstructure."
    ),
    "SPOOFING": (
        "The order flow pattern is flagging potential spoofing behavior. There's extreme "
        "directional imbalance at {z_imbalance:.1f} sigma — large orders appearing on one side "
        "then rapidly disappearing. This warrants closer investigation of the order book."
    ),
    "WASH_TRADE": (
        "I'm detecting a wash trading pattern. The order imbalance is oscillating rapidly — "
        "{wash_score:.0%} sign-flip rate in the recent window. This back-and-forth pattern "
        "with no net directional conviction is characteristic of self-dealing activity."
    ),
    "LAYERING": (
        "There are indications of layering behavior in the order flow. Multiple levels of "
        "directional pressure are building and collapsing in a coordinated pattern. "
        "The flow imbalance is abnormal at {z_imbalance:.1f} standard deviations."
    ),
}


class AnomalyAgent:
    """
    Converts raw ML anomaly detections into spoken analyst commentary.
    For high-severity alerts, uses LLM for richer reasoning.
    For lower severity, uses fast templates.
    """

    def __init__(self):
        self.recent_explanations: list = []
        self.alert_count = 0

    async def explain_anomaly(self, detection: dict) -> str:
        """
        Generate a verbal explanation of an anomaly detection.

        Args:
            detection: Output from AnomalyDetectorML.process_tick()

        Returns:
            Human-readable spoken explanation string
        """
        severity = detection.get("severity", "LOW")
        atype = detection.get("primary_type", "STATISTICAL_OUTLIER")
        score = detection.get("ensemble_score", 0)
        models = detection.get("models", {})
        features = detection.get("features", {})

        self.alert_count += 1

        # ── High severity → LLM-powered deep explanation ──
        if severity in ("CRITICAL", "HIGH"):
            explanation = await self._llm_explain(detection)
            if explanation:
                self.recent_explanations.append({
                    "time": time.time(),
                    "severity": severity,
                    "type": atype,
                    "explanation": explanation,
                })
                return explanation

        # ── Lower severity → fast template ──
        explanation = self._template_explain(atype, models, features, score)
        self.recent_explanations.append({
            "time": time.time(),
            "severity": severity,
            "type": atype,
            "explanation": explanation,
        })
        return explanation

    async def _llm_explain(self, detection: dict) -> str:
        """Use LLM for rich, contextual anomaly explanation."""
        try:
            market_ctx = market_stream.get_snapshot()

            prompt = f"""An anomaly has been detected by the ML pipeline. Explain it verbally 
as a senior analyst would to a colleague. Be concise (3-4 sentences max).

DETECTION:
- Type: {detection['primary_type']}
- Severity: {detection['severity']}
- Ensemble Score: {detection['ensemble_score']}
- Model Scores: {detection['models']}
- Features: {detection['features']}

MARKET CONTEXT:
- Current Price: ${market_ctx.get('current_price', 'N/A')}
- Volatility: {market_ctx.get('current_volatility', 'N/A')}%
- Recent Anomaly Count: {market_ctx.get('recent_anomalies', 0)}

Explain what's happening, why it matters, and what to watch for next."""

            response = await llm_service.generate(user_message=prompt)
            return response

        except Exception as e:
            log.error(f"LLM anomaly explanation failed: {e}")
            return ""

    def _template_explain(self, atype: str, models: dict, features: dict, score: float) -> str:
        """Fast template-based explanation."""
        template = TEMPLATES.get(atype)
        if not template:
            return (
                f"The ensemble detector flagged an anomaly with a score of {score:.2f}. "
                f"The primary classification is {atype.replace('_', ' ').lower()}. "
                f"This warrants monitoring but doesn't match a specific manipulation pattern."
            )

        # Build template variables
        z_data = models.get("zscore", {}).get("details", {})
        of_data = models.get("order_flow", {}).get("details", {})
        vr_data = models.get("vol_regime", {}).get("details", {})

        vars = {
            "volume_ratio": features.get("volume_ratio", 1.0),
            "z_volume": z_data.get("z_volume", 0),
            "z_return": z_data.get("z_return", 0),
            "z_imbalance": of_data.get("z_imbalance", 0),
            "wash_score": of_data.get("wash_score", 0),
            "ratio": vr_data.get("ratio", 1.0),
            "regime": vr_data.get("regime", "NORMAL"),
            "percentile": max(1, int((1 - score) * 100)),
        }

        try:
            return template.format(**vars)
        except KeyError:
            return f"Anomaly detected: {atype} with ensemble score {score:.2f}."

    def get_recent_alerts(self, n: int = 5) -> List[dict]:
        """Get the N most recent alert explanations."""
        return self.recent_explanations[-n:]

    def get_stats(self) -> dict:
        return {
            "total_alerts": self.alert_count,
            "recent_count": len(self.recent_explanations),
        }
