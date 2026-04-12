"""
Text-to-Speech module using Gemini TTS API.
Converts interviewer text responses to natural-sounding speech.
"""

import base64
import struct
import json
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)


async def synthesize(text: str) -> bytes:
    """
    Synthesize speech from text using Gemini TTS.

    Args:
        text: The text to convert to speech

    Returns:
        WAV audio bytes
    """
    if not text or text.strip() in ["[silence]", "[inaudible]"]:
        return b""

    try:
        response = await client.aio.models.generate_content(
            model=settings.tts_model,
            contents=text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai.types.SpeechConfig(
                    voice_config=genai.types.VoiceConfig(
                        prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(
                            voice_name="Kore"  # Professional, clear voice
                        )
                    )
                ),
            )
        )

        # Extract audio data from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    audio_data = part.inline_data.data
                    if isinstance(audio_data, str):
                        audio_data = base64.b64decode(audio_data)
                    return _ensure_wav_header(audio_data)

        return b""

    except Exception as e:
        print(f"TTS Error: {e}")
        return b""


def _ensure_wav_header(audio_data: bytes) -> bytes:
    """
    Ensure audio data has a proper WAV header.
    If it's raw PCM, wrap it in a WAV container.
    """
    # Check if already has RIFF header
    if audio_data[:4] == b"RIFF":
        return audio_data

    # Assume raw PCM: 24kHz, 16-bit, mono (Gemini TTS default)
    sample_rate = 24000
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio_data)
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,                  # PCM format chunk size
        1,                   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )

    return header + audio_data
