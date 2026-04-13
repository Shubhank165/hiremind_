"""
Text-to-Speech module with pluggable providers.
Default path uses local Piper on CPU for lower latency and no remote TTS calls.
"""

import asyncio
import base64
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from google import genai

from backend.config import settings

gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None


async def synthesize(text: str) -> bytes:
    """
    Synthesize speech from text.

    Args:
        text: The text to convert to speech

    Returns:
        WAV audio bytes
    """
    if not text or text.strip() in ["[silence]", "[inaudible]"]:
        return b""

    try:
        provider = (settings.tts_provider or "piper").lower()
        if provider == "piper":
            wav_audio = await asyncio.to_thread(_synthesize_piper_sync, text)
            if wav_audio:
                return wav_audio
            if gemini_client is not None:
                return await _synthesize_gemini(text)
            return b""

        return await _synthesize_gemini(text)

    except Exception as e:
        print(f"TTS Error: {e}")
        return b""


def _synthesize_piper_sync(text: str) -> bytes:
    binary = _resolve_piper_binary()
    if not binary:
        raise RuntimeError("Piper binary not found. Set PIPER_BINARY or install piper.")

    model_path = _resolve_project_path(settings.piper_model_path)
    if not model_path.exists():
        raise RuntimeError(f"Piper model not found at {model_path}")

    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_file.close()

    try:
        cmd = [
            binary,
            "--model", str(model_path),
            "--output_file", output_file.name,
            "--speaker", str(settings.piper_speaker_id),
            "--noise_scale", str(settings.piper_noise_scale),
            "--length_scale", str(settings.piper_length_scale),
            "--noise_w", str(settings.piper_noise_w),
        ]

        model_cfg = _resolve_project_path(settings.piper_model_config_path)
        if model_cfg.exists():
            cmd.extend(["--config", str(model_cfg)])

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip() or "piper synthesis failed")

        return Path(output_file.name).read_bytes()
    finally:
        try:
            os.unlink(output_file.name)
        except OSError:
            pass


def _resolve_piper_binary() -> str:
    configured = settings.piper_binary
    if configured and Path(configured).exists():
        return str(Path(configured))
    return shutil.which(configured or "piper") or ""


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


async def _synthesize_gemini(text: str) -> bytes:
    if gemini_client is None:
        return b""

    cfg: dict = {
        "response_modalities": ["AUDIO"],
        "speech_config": genai.types.SpeechConfig(
            voice_config=genai.types.VoiceConfig(
                prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(voice_name="Kore")
            )
        ),
        "top_p": settings.gemini_top_p,
        "top_k": settings.gemini_top_k,
    }
    thinking_cls = getattr(genai.types, "ThinkingConfig", None)
    if thinking_cls is not None:
        cfg["thinking_config"] = thinking_cls(thinking_budget=settings.gemini_thinking_budget)

    response = await gemini_client.aio.models.generate_content(
        model=settings.tts_model,
        contents=text,
        config=genai.types.GenerateContentConfig(**cfg),
    )

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_data = part.inline_data.data
                if isinstance(audio_data, str):
                    audio_data = base64.b64decode(audio_data)
                return _ensure_wav_header(audio_data)

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
