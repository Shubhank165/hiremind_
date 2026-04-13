"""Helpers for low-latency Gemini generation config."""

from typing import Any

from google import genai

from backend.config import settings


def fast_config(
	*,
	temperature: float,
	max_output_tokens: int | None = None,
	response_mime_type: str | None = None,
	response_modalities: list[str] | None = None,
	speech_config: Any | None = None,
) -> genai.types.GenerateContentConfig:
	"""Build a Gemini config tuned for lower latency and predictable output size."""
	cfg: dict[str, Any] = {
		"temperature": temperature,
		"top_p": settings.gemini_top_p,
		"top_k": settings.gemini_top_k,
	}

	if max_output_tokens is not None:
		cfg["max_output_tokens"] = max_output_tokens
	if response_mime_type:
		cfg["response_mime_type"] = response_mime_type
	if response_modalities:
		cfg["response_modalities"] = response_modalities
	if speech_config is not None:
		cfg["speech_config"] = speech_config

	thinking_cls = getattr(genai.types, "ThinkingConfig", None)
	if thinking_cls is not None:
		cfg["thinking_config"] = thinking_cls(thinking_budget=settings.gemini_thinking_budget)

	return genai.types.GenerateContentConfig(**cfg)
