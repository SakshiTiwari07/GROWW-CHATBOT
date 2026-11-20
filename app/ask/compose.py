"""Ask Groq to phrase a short cited answer from retrieved chunks or a refusal."""

from __future__ import annotations

from app.ask.retrieve import Hit
from app.ask.validate import valid_draft
from app.config import Settings
from app.llm.base import LLMError
from app.llm.factory import Complete

FACTUAL_SYSTEM = """You answer factual questions about HDFC Mutual Fund from the supplied excerpts only.
State the fact that was asked.
Use at most three sentences.
Include exactly one URL, copied from the excerpts.
Do not recommend buying, selling, holding, or allocating.
Do not calculate, rank, or compare returns.
If the excerpts do not contain the fact, say it is not in the loaded sources and still include one excerpt URL.
End with a line: Last updated from sources: YYYY-MM-DD
Use the fetched_at of the page you cited."""

REFUSAL_SYSTEM = """You answer only with scheme facts.
Use at most three sentences.
Include exactly one URL, copied from the instruction.
Do not recommend buying, selling, holding, or allocating.
Do not calculate, rank, or compare returns.
End with a line: Last updated from sources: YYYY-MM-DD
Use the date given in the instruction."""


def compose(
    question: str,
    intent: str,
    hits: tuple[Hit, ...],
    settings: Settings,
    complete: Complete,
    updated: str,
) -> tuple[str, str, bool]:
    """Return answer text, citation URL, and whether the response is a refusal."""
    allowed, citation, user_prompt = _prompt(question, intent, hits, settings, updated)
    try:
        draft = complete(FACTUAL_SYSTEM if intent == "factual" else REFUSAL_SYSTEM, user_prompt)
    except LLMError:
        draft = ""
    if draft and valid_draft(draft, allowed, intent):
        return draft, urls_first(draft), intent != "factual"
    body, citation = _template(intent, hits, settings, citation)
    text = f"{body}\nLast updated from sources: {updated}"
    return text, citation, intent != "factual"


def urls_first(text: str) -> str:
    from app.ask.validate import urls_in

    found = urls_in(text)
    return found[0] if found else ""


def _prompt(
    question: str,
    intent: str,
    hits: tuple[Hit, ...],
    settings: Settings,
    updated: str,
) -> tuple[set[str], str, str]:
    if intent == "factual" and hits:
        blocks = []
        for hit in hits:
            blocks.append(
                f"section: {hit.section}\nurl: {hit.url}\nfetched_at: {hit.fetched_at}\n{hit.text}"
            )
        allowed = {hit.url for hit in hits}
        return allowed, hits[0].url, "Question:\n" + question + "\n\nExcerpts:\n" + "\n\n".join(blocks)
    citation = _catalog_url(settings, intent, hits)
    instruction = {
        "advice": "Refuse investment advice. Do not suggest a buy or sell.",
        "performance": "Do not compare or calculate returns. Point to the factsheet.",
        "out_of_scope": "Say this assistant covers HDFC Mutual Fund and four HDFC schemes only.",
        "factual": "The sources are not loaded or do not contain the fact.",
    }[intent]
    user = f"Question:\n{question}\n\n{instruction}\nUse this URL: {citation}\nDate: {updated}"
    return {citation}, citation, user


def _template(intent: str, hits: tuple[Hit, ...], settings: Settings, citation: str) -> tuple[str, str]:
    if intent == "factual" and hits:
        return f"The loaded page covers this topic. Source: {hits[0].url}", hits[0].url
    if intent == "factual":
        url = settings.sources[0].url
        return f"Sources are not loaded yet. Source: {url}", url
    if intent == "advice":
        return (
            "I can share scheme facts only, and I cannot suggest what to buy or sell. "
            f"Source: {citation}"
        ), citation
    if intent == "performance":
        return (
            "I do not compare or calculate returns. The official factsheet is the source for "
            f"performance figures. Source: {citation}"
        ), citation
    return (
        "This assistant covers HDFC Mutual Fund and four HDFC schemes only. "
        f"Source: {citation}"
    ), citation


def _catalog_url(settings: Settings, intent: str, hits: tuple[Hit, ...]) -> str:
    if intent == "performance" and settings.citations("factsheet"):
        return settings.citations("factsheet")[0].url
    if settings.citations("educational"):
        return settings.citations("educational")[0].url
    if hits:
        return hits[0].url
    return settings.sources[0].url
