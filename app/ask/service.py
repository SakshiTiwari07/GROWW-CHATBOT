"""Answer a question with Hugging Face retrieval and a Groq reply."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from app.ask.compose import compose
from app.ask.guardrails import screen_pii
from app.ask.intent import classify_intent
from app.ask.retrieve import retrieve
from app.config import Settings
from app.llm.base import LLMError
from app.llm.factory import groq_complete


@dataclass(frozen=True)
class AskResponse:
    answer: str
    citation_url: str
    last_updated: str
    refused: bool


def ask(question: str, settings: Settings, complete=None, embed=None, search=None) -> AskResponse:
    updated = _corpus_date(settings)
    if screen_pii(question).blocked:
        url = _educational_url(settings)
        return AskResponse(
            answer=(
                "I can only answer scheme facts, and I cannot read personal identifiers. "
                f"Source: {url}\nLast updated from sources: {updated}"
            ),
            citation_url=url,
            last_updated=updated,
            refused=True,
        )

    intent = classify_intent(question)
    hits = ()
    if intent == "factual":
        finder = search or retrieve
        found = finder(question, settings, embed=embed)
        if found.corpus_empty:
            url = settings.sources[0].url
            return _save(
                settings,
                question,
                intent,
                AskResponse(
                    answer=f"Sources are not loaded yet. Source: {url}\nLast updated from sources: {updated}",
                    citation_url=url,
                    last_updated=updated,
                    refused=True,
                ),
            )
        hits = found.hits

    answer, citation, refused = compose(
        question,
        intent,
        hits,
        settings,
        _complete(settings, complete),
        updated,
    )
    return _save(
        settings,
        question,
        intent,
        AskResponse(answer=answer, citation_url=citation, last_updated=updated, refused=refused),
    )


def _complete(settings: Settings, complete):
    if complete is not None:
        return complete

    def _missing(system_prompt: str, user_prompt: str) -> str:
        try:
            return groq_complete(settings)(system_prompt, user_prompt)
        except LLMError as exc:
            raise LLMError(str(exc)) from exc

    return _missing


def _educational_url(settings: Settings) -> str:
    rows = settings.citations("educational")
    if rows:
        return rows[0].url
    return settings.sources[0].url


def _corpus_date(settings: Settings) -> str:
    dates: list[str] = []
    if settings.documents_dir.is_dir():
        for path in settings.documents_dir.glob("*.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("fetched_at"):
                dates.append(str(raw["fetched_at"]))
    if dates:
        return max(dates)
    return datetime.now(timezone.utc).date().isoformat()


def _save(settings: Settings, question: str, intent: str, response: AskResponse) -> AskResponse:
    try:
        settings.answers_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "question": question,
            "intent": intent,
            "answer": response.answer,
            "citation_url": response.citation_url,
            "last_updated": response.last_updated,
            "refused": response.refused,
            "llm": "groq",
            "embedding_model": settings.embedding_model,
        }
        (settings.answers_dir / f"{stamp}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return response
    return response
