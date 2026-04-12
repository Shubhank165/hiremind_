"""
Speech-to-Text module using Gemini multimodal API.
Converts audio bytes to text transcription.
"""

import base64
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

STT_PROMPT = """Transcribe the following audio accurately. Output ONLY the transcribed text, nothing else.
If the audio is unclear or silent, output "[inaudible]".
If there is no speech, output "[silence]"."""


async def transcribe(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """
    Transcribe audio bytes to text using Gemini multimodal model.

    Args:
        audio_bytes: Raw audio data
        mime_type: MIME type of the audio (default: audio/wav)

    Returns:
        Transcribed text string
    """
    if not audio_bytes:
        return "[silence]"

    try:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": STT_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_b64
                            }
                        }
                    ]
                }
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2000,
            )
        )

        transcript = response.text.strip()
        return transcript if transcript else "[silence]"

    except Exception as e:
        print(f"STT Error: {e}")
        return f"[transcription error: {str(e)}]"


import time
import struct
import math

class AudioAccumulator:
    """
    Accumulates streaming audio chunks and triggers transcription
    when a silence gap is detected.
    """

    def __init__(self, silence_threshold_rms: int = 500, silence_duration_sec: float = 1.1):
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
        self.min_chunk_bytes = 16000 * 2 * 1  # At least 1 second of audio (16kHz 16-bit)

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
        
        # Create WAV container in memory for Gemini STT
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
