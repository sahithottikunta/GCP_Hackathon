"""
SENTINEL — Voice Agent
The primary conversational agent that handles real-time voice interaction.

Responsibilities:
  - Maintain conversation history
  - Route queries that need market context
  - Generate spoken responses via LLM + TTS
  - Handle interruption gracefully
  - Provide thinking/status indicators to the UI
"""

from __future__ import annotations
import asyncio
import time
import base64
from typing import Optional, Dict, List
from models.schemas import AgentStatus, AgentState
from services.llm_service import llm_service
from services.audio_pipeline import AudioPipeline, AudioState
from services.websocket_manager import ws_manager
from services.data_stream import market_stream
from utils import get_logger

log = get_logger("voice_agent")


class VoiceAgent:
    """
    Real-time voice interaction agent.

    Loop:
      1. Audio in → VAD → STT → text
      2. Text → LLM (with market context) → response text
      3. Response text → TTS → audio chunks → client
      4. If interrupted: stop TTS, listen to new input
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.state = AgentState()
        self.audio = AudioPipeline()
        self._setup_callbacks()
        self._current_response_task: Optional[asyncio.Task] = None
        log.info(f"Voice agent created for client {client_id}")

    def _setup_callbacks(self):
        """Wire up audio pipeline callbacks."""
        self.audio.on_transcription(self._handle_transcription)
        self.audio.on_state_change(self._handle_state_change)
        self.audio.on_interrupt(self._handle_interrupt)

    async def _handle_state_change(self, new_state: AudioState):
        """Broadcast agent status to the client."""
        status_map = {
            AudioState.IDLE: AgentStatus.IDLE,
            AudioState.LISTENING: AgentStatus.LISTENING,
            AudioState.PROCESSING: AgentStatus.THINKING,
            AudioState.SPEAKING: AgentStatus.SPEAKING,
            AudioState.INTERRUPTED: AgentStatus.IDLE,
        }
        self.state.status = status_map.get(new_state, AgentStatus.IDLE)
        await ws_manager.send_json(self.client_id, {
            "event": "agent_status",
            "data": {"status": self.state.status.value, "audio_state": new_state.value},
        })

    async def _handle_interrupt(self):
        """Called when user interrupts the agent mid-speech."""
        log.info(f"Agent interrupted by client {self.client_id}")

        # Cancel any ongoing response generation
        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()
            log.debug("Cancelled ongoing response task")

        # Notify client to stop audio playback
        await ws_manager.send_json(self.client_id, {
            "event": "interrupt",
            "data": {"message": "Agent interrupted — listening..."},
        })

        # Add interruption note to conversation
        self.state.conversation_history.append({
            "role": "assistant",
            "content": "[Response interrupted by user]",
        })

    async def _handle_transcription(self, text: str):
        """Called when speech is transcribed — trigger the response pipeline."""
        log.info(f"Processing voice query: \"{text}\"")

        # Notify client of transcription (source=voice so frontend shows it)
        await ws_manager.send_json(self.client_id, {
            "event": "transcription",
            "data": {"text": text, "is_final": True, "source": "voice"},
        })

        # Add to conversation history
        self.state.conversation_history.append({"role": "user", "content": text})

        # Kick off response generation (non-blocking)
        self._current_response_task = asyncio.create_task(
            self._generate_and_speak(text)
        )

    async def _generate_and_speak(self, user_text: str):
        """Generate LLM response and stream it as TTS audio."""
        try:
            # ── Get market context ──
            market_ctx = market_stream.get_snapshot()
            self.state.market_context = market_ctx

            # ── Notify: thinking ──
            await ws_manager.send_json(self.client_id, {
                "event": "agent_thinking",
                "data": {"query": user_text},
            })

            # ── Stream LLM response ──
            full_response = ""
            sentence_buffer = ""

            await self.audio.start_speaking()

            async for token in llm_service.generate_stream(
                user_message=user_text,
                conversation_history=self.state.conversation_history[-10:],  # last 10 turns
                market_context=market_ctx,
            ):
                # Check for interruption
                if self.audio.is_interrupted:
                    log.info("Response generation stopped due to interrupt")
                    break

                full_response += token
                sentence_buffer += token

                # Send text token to client for live display
                await ws_manager.send_json(self.client_id, {
                    "event": "agent_token",
                    "data": {"token": token, "partial": True},
                })

                # ── Sentence-level TTS ──
                # When we hit a sentence boundary, synthesize and send audio
                if any(sentence_buffer.rstrip().endswith(p) for p in [".", "!", "?", "—"]):
                    sentence = sentence_buffer.strip()
                    if sentence and len(sentence) > 10:
                        await self._speak_sentence(sentence)
                    sentence_buffer = ""

            # Speak any remaining text
            if sentence_buffer.strip() and not self.audio.is_interrupted:
                await self._speak_sentence(sentence_buffer.strip())

            await self.audio.stop_speaking()

            # ── Final response ──
            if full_response and not self.audio.is_interrupted:
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": full_response,
                })
                await ws_manager.send_json(self.client_id, {
                    "event": "agent_response_complete",
                    "data": {"text": full_response},
                })

            self.audio.reset_interrupt()

        except asyncio.CancelledError:
            log.info("Response task cancelled")
            await self.audio.stop_speaking()
        except Exception as e:
            log.error(f"Response generation error: {e}")
            await self.audio.stop_speaking()
            await ws_manager.send_json(self.client_id, {
                "event": "agent_error",
                "data": {"error": str(e)},
            })

    async def _speak_sentence(self, text: str):
        """Synthesize a sentence to audio and send to client."""
        if self.audio.is_interrupted:
            return

        audio_bytes = await llm_service.synthesize_speech(text)
        if audio_bytes and not self.audio.is_interrupted:
            audio_b64 = base64.b64encode(audio_bytes).decode()
            await ws_manager.send_json(self.client_id, {
                "event": "agent_audio",
                "data": {
                    "audio_base64": audio_b64,
                    "format": "mp3",
                    "text": text,
                },
            })

    async def process_audio(self, audio_bytes: bytes):
        """Feed audio chunk from WebSocket into the pipeline."""
        await self.audio.process_audio_chunk(audio_bytes)

    async def process_text(self, text: str):
        """Handle a text message (typed instead of spoken).
        Bypasses audio pipeline — no VAD, no TTS, just stream text response."""
        log.info(f"Processing text query: \"{text}\"")

        # NOTE: Don't send 'transcription' event — the frontend already
        # displays the user message immediately on send. Sending it again
        # would create a duplicate bubble in the chat.

        # Add to conversation history
        self.state.conversation_history.append({"role": "user", "content": text})

        # Generate and stream response directly (no audio pipeline involvement)
        self._current_response_task = asyncio.create_task(
            self._generate_text_response(text)
        )
        # Surface any unhandled exceptions from the task
        self._current_response_task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: asyncio.Task):
        """Log any unhandled exceptions from background response tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error(f"Background response task failed: {type(exc).__name__}: {exc}")

    async def _generate_text_response(self, user_text: str):
        """Generate LLM response for text input — stream tokens, no TTS."""
        try:
            market_ctx = market_stream.get_snapshot()
            self.state.market_context = market_ctx

            # Notify: thinking
            await ws_manager.send_json(self.client_id, {
                "event": "agent_thinking",
                "data": {"query": user_text},
            })

            # Stream LLM response
            full_response = ""
            token_count = 0

            log.info(f"Starting LLM stream for client {self.client_id}...")

            async for token in llm_service.generate_stream(
                user_message=user_text,
                conversation_history=self.state.conversation_history[-10:],
                market_context=market_ctx,
            ):
                full_response += token
                token_count += 1

                # Send text token to client for live display
                await ws_manager.send_json(self.client_id, {
                    "event": "agent_token",
                    "data": {"token": token, "partial": True},
                })

            log.info(f"LLM stream complete: {token_count} tokens, {len(full_response)} chars")

            # Final response
            if full_response:
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": full_response,
                })
                await ws_manager.send_json(self.client_id, {
                    "event": "agent_response_complete",
                    "data": {"text": full_response},
                })
            else:
                log.warning("LLM returned empty response — sending fallback")
                await ws_manager.send_json(self.client_id, {
                    "event": "agent_response_complete",
                    "data": {"text": "I wasn't able to generate a response. Please try again."},
                })

        except asyncio.CancelledError:
            log.info("Text response task cancelled")
        except Exception as e:
            log.error(f"Text response generation error: {e}")
            await ws_manager.send_json(self.client_id, {
                "event": "agent_error",
                "data": {"error": str(e)},
            })

    async def process_image(self, image_base64: str, query: str = "") -> str:
        """Handle an uploaded image for vision analysis."""
        log.info(f"Processing image with query: {query or '(default analysis)'}")

        await ws_manager.send_json(self.client_id, {
            "event": "agent_thinking",
            "data": {"query": f"Analyzing image: {query}"},
        })

        analysis = await llm_service.analyze_image(
            image_base64=image_base64,
            query=query or "Analyze this financial chart. What patterns, anomalies, or notable features do you see?",
        )

        self.state.conversation_history.append({"role": "user", "content": f"[Image uploaded] {query}"})
        self.state.conversation_history.append({"role": "assistant", "content": analysis})

        await ws_manager.send_json(self.client_id, {
            "event": "vision_analysis",
            "data": {"analysis": analysis, "query": query},
        })

        # Also speak the analysis
        audio = await llm_service.synthesize_speech(analysis)
        if audio:
            await ws_manager.send_json(self.client_id, {
                "event": "agent_audio",
                "data": {
                    "audio_base64": base64.b64encode(audio).decode(),
                    "format": "mp3",
                    "text": analysis,
                },
            })

        return analysis

    def get_state(self) -> dict:
        return {
            "status": self.state.status.value,
            "history_length": len(self.state.conversation_history),
            "market_context": bool(self.state.market_context),
        }
