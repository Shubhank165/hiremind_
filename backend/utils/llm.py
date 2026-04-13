"""LLM provider abstraction (Groq or Gemini)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google import genai
from groq import Groq

from backend.config import settings
from backend.utils.gemini_fast import fast_config

_gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
_groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _groq_chat_sync(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> str:
    if _groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")

    kwargs: dict[str, Any] = {
        "model": settings.groq_llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = _groq_client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


async def _groq_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> str:
    return await asyncio.to_thread(
        _groq_chat_sync,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )


async def _gemini_text(
    prompt: str,
    *,
    system: str | None,
    max_tokens: int,
    temperature: float,
    response_mime_type: str | None = None,
) -> str:
    if _gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    if system:
        contents = [
            {"role": "user", "parts": [{"text": system}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
    else:
        contents = prompt

    response = await _gemini_client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=fast_config(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=response_mime_type,
        ),
    )
    return (response.text or "").strip()


async def generate_text(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.4,
) -> str:
    provider = (settings.llm_provider or "gemini").lower()
    max_tokens = max_tokens or settings.gemini_text_max_tokens

    if provider == "groq":
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await _groq_chat(messages, max_tokens=max_tokens, temperature=temperature)

    return await _gemini_text(
        prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def generate_json(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    provider = (settings.llm_provider or "gemini").lower()
    max_tokens = max_tokens or settings.gemini_json_max_tokens

    json_prompt = (
        f"{prompt}\n\n"
        "Return ONLY valid JSON. No markdown, no prose."
    )

    if provider == "groq":
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": json_prompt})
        content = await _groq_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(_strip_code_fence(content))

    content = await _gemini_text(
        json_prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        response_mime_type="application/json",
    )
    return json.loads(_strip_code_fence(content))
