"""Check that a draft has one allowed citation and at most three sentences."""

from __future__ import annotations

import re

from app.ask.intent import ADVICE, PERFORMANCE

URLS = re.compile(r"https?://[^\s)>\]]+")
UPDATED = re.compile(r"Last updated from sources: \d{4}-\d{2}-\d{2}\s*$")


def sentence_count(text: str) -> int:
    cleaned = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    parts = re.split(r"[.!?]+", cleaned)
    return len([part for part in parts if part.strip()])


def urls_in(text: str) -> list[str]:
    return [url.rstrip(".,") for url in URLS.findall(text)]


def valid_draft(draft: str, allowed_urls: set[str], intent: str) -> bool:
    found = urls_in(draft)
    if sentence_count(draft) > 3:
        return False
    if len(found) != 1 or found[0] not in allowed_urls:
        return False
    if UPDATED.search(draft.strip()) is None:
        return False
    if intent in {"advice", "performance", "out_of_scope"}:
        lowered = draft.lower()
        if any(phrase in lowered for phrase in ADVICE + PERFORMANCE):
            return False
    return True
