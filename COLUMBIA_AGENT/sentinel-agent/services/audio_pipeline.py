"""
SENTINEL — Real-Time Audio Pipeline
Handles the full voice interaction loop:
  1. Receive audio chunks from WebSocket
  2. Voice Activity Detection (VAD) to detect speech boundaries
  3. Buffer speech → send to Whisper STT
  4. Generate response via LLM
  5. Stream TTS audio back
  6. Handle interruption: if user speaks during TTS, stop immediately

This is the core of the "interruptible live agent" requirement.
"""

from __future__ import annotations
import asyncio
import time
import struct
import io
import wave
from collections import deque
from typing import Callable, Awaitable, Optional
from enum import Enum
from config import settings
from utils import get_logger

log = get_logger("audio")


class AudioState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class VoiceActivityDetector:
    """
    Simple energy-based VAD with adaptive threshold.
    For production, you'd use webrtcvad or Silero VAD.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        speech_pad_ms: int = 300,
        silence_trigger_ms: int = 800,
        sample_rate: int = 16000,
    ):
        self.base_threshold = energy_threshold
        self.threshold = energy_threshold
        self.speech_pad_ms = speech_pad_ms
        self.silence_trigger_ms = silence_trigger_ms
        self.sample_rate = sample_rate

        self._is_speech = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._energy_history: deque = deque(maxlen=100)

    def _compute_energy(self, audio_bytes: bytes) -> float:
        """Compute RMS energy of a PCM16 audio chunk."""
        if len(audio_bytes) < 2:
            return 0.0
        n_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", audio_bytes[:n_samples * 2])
        rms = (sum(s ** 2 for s in samples) / n_samples) ** 0.5
        return rms / 32768.0  # Normalize to [0, 1]

    def _update_threshold(self, energy: float):
        """Adaptive threshold based on recent energy levels."""
        self._energy_history.append(energy)
        if len(self._energy_history) >= 20:
            noise_floor = sorted(self._energy_history)[:10]
            avg_noise = sum(noise_floor) / len(noise_floor)
            self.threshold = max(self.base_threshold, avg_noise * 3.0)

    def process_chunk(self, audio_bytes: bytes) -> dict:
        """
        Process an audio chunk and return VAD state.
        Returns: {
            "is_speech": bool,
            "speech_started": bool,   # transition: silence → speech
            "speech_ended": bool,     # transition: speech → silence (after pad)
            "energy": float,
        }
        """
        energy = self._compute_energy(audio_bytes)
        self._update_threshold(energy)
        now = time.time()

        speech_started = False
        speech_ended = False

        if energy > self.threshold:
            if not self._is_speech:
                self._is_speech = True
                self._speech_start = now
                speech_started = True
                log.debug(f"Speech started (energy={energy:.4f}, threshold={self.threshold:.4f})")
            self._silence_start = None
        else:
            if self._is_speech:
                if self._silence_start is None:
                    self._silence_start = now
                elif (now - self._silence_start) * 1000 > self.silence_trigger_ms:
                    self._is_speech = False
                    speech_ended = True
                    duration = now - (self._speech_start or now)
                    log.debug(f"Speech ended (duration={duration:.2f}s)")
                    self._speech_start = None
                    self._silence_start = None

        return {
            "is_speech": self._is_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "energy": round(energy, 5),
        }


class AudioPipeline:
    """
    Full audio pipeline managing the voice interaction loop.

    Flow:
      Audio In → VAD → Buffer → STT → [callback with text]
                                          ↓
      Audio Out ← TTS ← LLM Response ← Agent
                  ↑
      Interrupt ← VAD detects speech during playback
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.state = AudioState.IDLE
        self._audio_buffer: list[bytes] = []
        self._is_playing = False
        self._interrupt_event = asyncio.Event()

        # Callbacks
        self._on_transcription: Optional[Callable[[str], Awaitable]] = None
        self._on_state_change: Optional[Callable[[AudioState], Awaitable]] = None
        self._on_interrupt: Optional[Callable[[], Awaitable]] = None

    def on_transcription(self, callback: Callable[[str], Awaitable]):
        self._on_transcription = callback

    def on_state_change(self, callback: Callable[[AudioState], Awaitable]):
        self._on_state_change = callback

    def on_interrupt(self, callback: Callable[[], Awaitable]):
        self._on_interrupt = callback

    async def _set_state(self, new_state: AudioState):
        if new_state != self.state:
            old = self.state
            self.state = new_state
            log.info(f"Audio state: {old} → {new_state}")
            if self._on_state_change:
                await self._on_state_change(new_state)

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        Process incoming audio chunk from the client.
        This is called for every chunk received via WebSocket.
        """
        vad_result = self.vad.process_chunk(audio_bytes)

        # ── Interrupt detection ──
        # If agent is SPEAKING and user starts talking → INTERRUPT
        if self.state == AudioState.SPEAKING and vad_result["speech_started"]:
            log.info(">>> INTERRUPT DETECTED — stopping TTS playback")
            await self._set_state(AudioState.INTERRUPTED)
            self._interrupt_event.set()
            self._is_playing = False
            self._audio_buffer = []
            if self._on_interrupt:
                await self._on_interrupt()
            # Immediately start listening to the new utterance
            await self._set_state(AudioState.LISTENING)
            self._audio_buffer.append(audio_bytes)
            return

        # ── Normal listening flow ──
        if vad_result["speech_started"]:
            await self._set_state(AudioState.LISTENING)
            self._audio_buffer = [audio_bytes]

        elif vad_result["is_speech"] and self.state == AudioState.LISTENING:
            self._audio_buffer.append(audio_bytes)

        elif vad_result["speech_ended"] and self._audio_buffer:
            await self._set_state(AudioState.PROCESSING)
            # Combine buffer into a WAV and send to STT
            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer = []
            wav_bytes = self._pcm_to_wav(audio_data)

            # Transcribe
            from services.llm_service import llm_service
            text = await llm_service.transcribe_audio(wav_bytes, format="wav")

            if text and text.strip():
                log.info(f"Transcribed: \"{text}\"")
                if self._on_transcription:
                    await self._on_transcription(text)
            else:
                log.debug("Empty transcription — returning to idle")
                await self._set_state(AudioState.IDLE)

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM16 bytes to WAV format for Whisper."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    async def start_speaking(self):
        """Mark the agent as speaking (for interrupt detection)."""
        self._interrupt_event.clear()
        self._is_playing = True
        await self._set_state(AudioState.SPEAKING)

    async def stop_speaking(self):
        """Mark the agent as done speaking."""
        self._is_playing = False
        await self._set_state(AudioState.IDLE)

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_event.is_set()

    def reset_interrupt(self):
        self._interrupt_event.clear()
