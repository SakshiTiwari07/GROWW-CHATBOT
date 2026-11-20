"""Build the Groq client. This assistant does not call any other LLM provider."""

from __future__ import annotations

from collections.abc import Callable

from app.config import Settings
from app.llm.base import LLMError
from app.llm.groq_client import GroqClient

Complete = Callable[[str, str], str]


def groq_complete(settings: Settings) -> Complete:
    if settings.llm_provider != "groq":
        raise LLMError("The LLM is Groq. Set LLM_PROVIDER=groq.")
    try:
        api_key = settings.provider_api_key()
    except RuntimeError as exc:
        raise LLMError("GROQ_API_KEY is empty.") from exc
    return GroqClient(api_key, settings.llm_model).complete
