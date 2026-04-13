"""
Speech-to-Text module with pluggable providers.
Default path uses Groq Whisper for lower latency; Gemini remains as fallback.
"""

import asyncio
import base64
import math
import struct
import time

from google import genai
from groq import Groq

from backend.config import settings

gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

STT_PROMPT = """Transcribe the following audio accurately. Output ONLY the transcribed text, nothing else.
If the audio is unclear or silent, output "[inaudible]".
If there is no speech, output "[silence]"."""


def _gemini_stt_config() -> genai.types.GenerateContentConfig:
    cfg: dict = {
        "temperature": 0.1,
        "max_output_tokens": min(settings.gemini_text_max_tokens, 256),
        "top_p": settings.gemini_top_p,
        "top_k": settings.gemini_top_k,
    }
    thinking_cls = getattr(genai.types, "ThinkingConfig", None)
    if thinking_cls is not None:
        cfg["thinking_config"] = thinking_cls(thinking_budget=settings.gemini_thinking_budget)
    return genai.types.GenerateContentConfig(**cfg)


def _groq_transcribe_sync(audio_bytes: bytes, mime_type: str) -> str:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes, mime_type),
        model=settings.stt_model,
        language=settings.stt_language or None,
        temperature=0.0,
        prompt="Return only the spoken transcript text.",
        response_format="json",
    )

    transcript = getattr(response, "text", "")
    if not transcript and isinstance(response, dict):
        transcript = response.get("text", "")
    return (transcript or "").strip()


async def _groq_transcribe(audio_bytes: bytes, mime_type: str) -> str:
    return await asyncio.to_thread(_groq_transcribe_sync, audio_bytes, mime_type)


async def _gemini_transcribe(audio_bytes: bytes, mime_type: str) -> str:
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    response = await gemini_client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": STT_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_b64,
                        }
                    },
                ],
            }
        ],
        config=_gemini_stt_config(),
    )

    return (response.text or "").strip()


async def transcribe(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe raw audio into text."""
    if not audio_bytes:
        return "[silence]"

    provider = (settings.stt_provider or "groq").lower()
    try:
        if provider == "groq":
            transcript = await _groq_transcribe(audio_bytes, mime_type)
            if transcript:
                return transcript
            if gemini_client is not None:
                transcript = await _gemini_transcribe(audio_bytes, mime_type)
                return transcript if transcript else "[silence]"
            return "[inaudible]"

        transcript = await _gemini_transcribe(audio_bytes, mime_type)
        return transcript if transcript else "[silence]"

    except Exception as e:
        print(f"STT Error: {e}")
        return f"[transcription error: {str(e)}]"


class AudioAccumulator:
    """
    Accumulates streaming audio chunks and triggers transcription
    when a silence gap is detected.
    """

    def __init__(self, silence_threshold_rms: int = 500, silence_duration_sec: float = 1.1, min_chunk_seconds: float = 0.4):
        """
        Args:
            silence_threshold_rms: RMS volume below which is considered silence
            silence_duration_sec: How long silence must last to trigger processing
        """
        self.buffer = bytearray()
        self.silence_threshold = silence_threshold_rms
        self.silence_duration = silence_duration_sec
        
        self.last_speech_time = time.time()
        self.is_speaking = False
        # Smaller minimum chunk reduces end-of-speech latency.
        self.min_chunk_bytes = int(16000 * 2 * max(0.2, min_chunk_seconds))

    def calculate_rms(self, chunk: bytes) -> float:
        """Calculate RMS volume of 16-bit PCM chunk."""
        count = len(chunk) // 2
        if count == 0: return 0.0
        
        # Unpack as little-endian signed shorts
        try:
            shorts = struct.unpack(f'<{count}h', chunk)
            sum_squares = sum(s * s for s in shorts)
            return math.sqrt(sum_squares / count)
        except Exception:
            return 0.0

    def add_chunk(self, chunk: bytes) -> bool:
        """
        Add an audio chunk to the buffer.

        Returns:
            True if buffer is ready for transcription (silence detected after speech)
        """
        self.buffer.extend(chunk)
        
        rms = self.calculate_rms(chunk)
        current_time = time.time()
        
        if rms > self.silence_threshold:
            self.last_speech_time = current_time
            self.is_speaking = True
        
        # If we were speaking, but have been silent for the duration, and have minimum bytes
        if self.is_speaking and (current_time - self.last_speech_time) > self.silence_duration:
            if len(self.buffer) > self.min_chunk_bytes:
                return True
            else:
                # Reset if it was just a false start
                self.buffer.clear()
                self.is_speaking = False

        return False

    def get_audio_and_reset(self) -> bytes:
        """Get accumulated audio and reset buffer."""
        audio = bytes(self.buffer)
        
        # Create WAV container in memory for provider STT input
        import io
        import wave
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio)
            
        wav_bytes = wav_io.getvalue()
        
        self.buffer = bytearray()
        self.is_speaking = False
        self.last_speech_time = time.time()
        
        return wav_bytes

    def flush(self) -> bytes:
        """Get any remaining audio formatted as WAV."""
        if len(self.buffer) == 0:
            return b""
        return self.get_audio_and_reset()
