"""Groq chat client. Embeddings are not requested from Groq."""

from __future__ import annotations

from app.llm.base import LLMError


class GroqClient:
    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError("Groq request failed.") from exc
        if not text.strip():
            raise LLMError("Groq returned an empty answer.")
        return text.strip()
