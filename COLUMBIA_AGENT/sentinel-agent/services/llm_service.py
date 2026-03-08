"""
SENTINEL — LLM Service
Unified interface using Google Gemini as the primary LLM.
Supports text generation, streaming, and vision analysis.
"""

from __future__ import annotations
import asyncio
import base64
from typing import AsyncGenerator, Optional, List, Dict
from config import settings
from utils import get_logger

log = get_logger("llm")

# ── System Prompts ────────────────────────────────────

FINANCIAL_ANALYST_SYSTEM = """You are SENTINEL, an elite real-time financial analyst AI voice agent.

ROLE: Senior quantitative analyst + compliance investigator at a top-tier hedge fund.

VOICE STYLE:
- Speak naturally as if talking to a colleague on a trading floor
- Be concise but thorough — every word should add value
- Use phrases like "What stands out here is...", "This is atypical because...", "If we look at the data..."
- Maintain a calm, authoritative, confident tone
- Keep responses to 2-4 sentences for simple queries, up to a short paragraph for complex analysis

CAPABILITIES:
- Analyze real-time market data, price action, volume patterns
- Detect and explain anomalies (spoofing, wash trades, unusual flow)
- Interpret charts and visual data when images are provided
- Reason step-by-step through complex market scenarios
- Assess correlation shifts, volatility regimes, risk factors

RULES:
- Never hallucinate specific numbers unless provided in context
- State uncertainty explicitly when data is incomplete
- Never provide trading advice or specific trade recommendations
- Focus on analysis, interpretation, and risk awareness
- If interrupted, gracefully acknowledge and pivot

When given market context data, incorporate it naturally into your spoken response."""

VISION_ANALYST_SYSTEM = """You are SENTINEL's vision module — a financial chart analysis expert.
When shown an image of a chart, trading screen, or financial data:
1. Describe what you see concisely
2. Identify key patterns (support/resistance, trend lines, volume patterns)
3. Flag anything anomalous or noteworthy
4. Speak as if narrating to a colleague looking at the same screen
Keep it to 3-5 sentences unless asked for more detail."""


class LLMService:
    """Unified LLM interface powered by Google Gemini."""

    MAX_RETRIES = 3

    def __init__(self):
        self._gemini = None
        self._gemini_model = settings.gemini_model or "gemini-2.0-flash"
        self._init_clients()

    def _init_clients(self):
        if settings.has_gemini:
            from google import genai

            # Strategy 1: Service account + Vertex AI (paid, higher quota)
            if settings.google_service_account_file and settings.google_cloud_project:
                try:
                    from google.oauth2 import service_account as sa

                    creds = sa.Credentials.from_service_account_file(
                        settings.google_service_account_file,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    client = genai.Client(
                        vertexai=True,
                        credentials=creds,
                        project=settings.google_cloud_project,
                        location="us-central1",
                    )
                    # Quick validation — will throw if Vertex AI API is not enabled
                    client.models.generate_content(
                        model=self._gemini_model,
                        contents="ping",
                        config={"max_output_tokens": 5},
                    )
                    self._gemini = client
                    log.info(f"✅ Gemini via Vertex AI / Service Account (model: {self._gemini_model})")
                except Exception as e:
                    log.warning(f"Vertex AI init failed (will try API key fallback): {e}")

            # Strategy 2: API key (free tier, lower quota)
            if not self._gemini and settings.gemini_api_key:
                try:
                    self._gemini = genai.Client(api_key=settings.gemini_api_key)
                    log.info(f"✅ Gemini via API key (model: {self._gemini_model})")
                except Exception as e:
                    log.warning(f"Gemini API key init failed: {e}")

        if not self._gemini:
            log.warning("⚠️ No LLM configured — using mock responses")

    async def _call_with_retry(self, func, *args, **kwargs):
        """Call a Gemini API function with exponential backoff on rate limits."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    log.warning(f"Rate limited (attempt {attempt+1}/{self.MAX_RETRIES}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise e
        raise Exception("Max retries exceeded for Gemini API call")

    # ── Text Generation (Gemini) ──────────────────────

    async def generate(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = FINANCIAL_ANALYST_SYSTEM,
        market_context: dict = None,
    ) -> str:
        """Generate a text response using Gemini."""

        # Inject market context
        augmented_system = system_prompt
        if market_context:
            augmented_system += f"\n\nCURRENT MARKET CONTEXT:\n{_format_context(market_context)}"

        if self._gemini:
            try:
                # Build Gemini contents from conversation history
                contents = _build_gemini_contents(conversation_history, user_message)
                resp = await self._call_with_retry(
                    self._gemini.models.generate_content,
                    model=self._gemini_model,
                    contents=contents,
                    config={
                        "system_instruction": augmented_system,
                        "max_output_tokens": 1024,
                    },
                )
                return resp.text
            except Exception as e:
                log.error(f"Gemini error: {e}")

        return _mock_response(user_message)

    # ── Streaming Generation ──────────────────────────

    async def generate_stream(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = FINANCIAL_ANALYST_SYSTEM,
        market_context: dict = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text response token-by-token for real-time display."""

        augmented_system = system_prompt
        if market_context:
            augmented_system += f"\n\nCURRENT MARKET CONTEXT:\n{_format_context(market_context)}"

        if self._gemini:
            contents = _build_gemini_contents(conversation_history, user_message)
            error_to_raise = None

            try:
                # Use asyncio.Queue bridged from a thread for robust streaming
                aqueue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _stream_worker():
                    """Run the blocking Gemini streaming call in a thread."""
                    try:
                        response = self._gemini.models.generate_content_stream(
                            model=self._gemini_model,
                            contents=contents,
                            config={
                                "system_instruction": augmented_system,
                                "max_output_tokens": 1024,
                            },
                        )
                        for chunk in response:
                            if chunk.text:
                                # Thread-safe put onto the asyncio queue
                                loop.call_soon_threadsafe(aqueue.put_nowait, ("text", chunk.text))
                        loop.call_soon_threadsafe(aqueue.put_nowait, ("done", None))
                    except Exception as e:
                        log.error(f"Gemini stream worker error: {type(e).__name__}: {e}")
                        loop.call_soon_threadsafe(aqueue.put_nowait, ("error", e))

                import threading
                thread = threading.Thread(target=_stream_worker, daemon=True)
                thread.start()

                while True:
                    try:
                        kind, value = await asyncio.wait_for(aqueue.get(), timeout=45)
                    except asyncio.TimeoutError:
                        log.error("Gemini stream timed out after 45s")
                        error_to_raise = Exception("LLM response timed out")
                        break
                    if kind == "done":
                        break
                    elif kind == "error":
                        error_to_raise = value
                        break
                    elif kind == "text":
                        yield value

            except Exception as e:
                log.error(f"Gemini stream setup error: {e}")
                error_to_raise = e

            # If there was an error and we yielded nothing, yield the error as text
            if error_to_raise:
                error_msg = str(error_to_raise)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    yield "I'm currently rate-limited by the API. Please wait a moment and try again."
                else:
                    yield f"Sorry, I encountered an error: {error_msg[:200]}"
            return

        # Mock fallback
        for word in _mock_response(user_message).split():
            yield word + " "
            await asyncio.sleep(0.05)

    # ── Vision Analysis ───────────────────────────────

    async def analyze_image(
        self,
        image_base64: str,
        query: str = "Analyze this financial chart and describe what you see.",
        media_type: str = "image/png",
    ) -> str:
        """Analyze a chart/image using Gemini vision."""

        if self._gemini:
            try:
                from google.genai import types

                image_bytes = base64.b64decode(image_base64)

                contents = [
                    types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                    query,
                ]
                resp = await asyncio.to_thread(
                    self._gemini.models.generate_content,
                    model=self._gemini_model,
                    contents=contents,
                    config={
                        "system_instruction": VISION_ANALYST_SYSTEM,
                        "max_output_tokens": 1024,
                    },
                )
                return resp.text
            except Exception as e:
                log.error(f"Gemini vision error: {e}")

        return "Vision analysis unavailable — no vision-capable API configured."

    # ── Speech to Text (placeholder — Gemini doesn't have a Whisper equivalent) ──

    async def transcribe_audio(self, audio_bytes: bytes, format: str = "wav") -> str:
        """Transcribe audio. Uses Gemini's multimodal audio understanding."""
        if self._gemini:
            try:
                from google.genai import types

                mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "webm": "audio/webm", "ogg": "audio/ogg"}
                mime_type = mime_map.get(format, "audio/wav")

                contents = [
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    "Transcribe this audio exactly. Return only the transcription text, nothing else.",
                ]
                resp = await asyncio.to_thread(
                    self._gemini.models.generate_content,
                    model=self._gemini_model,
                    contents=contents,
                    config={"max_output_tokens": 512},
                )
                return resp.text.strip()
            except Exception as e:
                log.error(f"Gemini STT error: {e}")
                return ""
        log.warning("No STT service available")
        return ""

    # ── Text to Speech (placeholder — no TTS without OpenAI) ──

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio bytes. TTS not available without OpenAI key."""
        log.warning("TTS not available — Gemini doesn't provide a TTS endpoint. Text will be displayed instead.")
        return None

    async def synthesize_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio. Not available without OpenAI key."""
        log.warning("TTS streaming not available without OpenAI key")
        return
        yield  # make it a generator


def _build_gemini_contents(conversation_history: List[Dict[str, str]] | None, user_message: str) -> list:
    """Build Gemini-compatible contents from conversation history."""
    contents = []
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            # Gemini uses "user" and "model" roles
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def _format_context(ctx: dict) -> str:
    """Format market context dict into readable text for the LLM."""
    lines = []
    for key, val in ctx.items():
        if isinstance(val, dict):
            lines.append(f"  {key}:")
            for k2, v2 in val.items():
                lines.append(f"    {k2}: {v2}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _mock_response(query: str) -> str:
    """Fallback mock response when no LLM is configured."""
    return (
        f"I'm analyzing your query about '{query[:50]}'. "
        "In a production environment, this would route through Gemini "
        "for real-time financial analysis. The ML anomaly detection pipeline is "
        "still running and scoring market data — you'd see those results on the dashboard. "
        "To enable full analysis, configure your GEMINI_API_KEY in the .env file."
    )


# Singleton
llm_service = LLMService()
