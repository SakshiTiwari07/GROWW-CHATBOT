"""Embed a question with the Hugging Face model and search Chroma."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Callable, Sequence

from app.config import Settings
from app.ingest.embed import embed_texts

# Longest phrase first so "tax saver" is not read as the tax fact.
_SCHEMES = (
    ("tax saver", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("elss", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("flexi", "hdfc-equity-fund-direct-growth"),
    ("mid cap", "hdfc-mid-cap-fund-direct-growth"),
    ("midcap", "hdfc-mid-cap-fund-direct-growth"),
    ("large cap", "hdfc-large-cap-fund-direct-growth"),
    ("largecap", "hdfc-large-cap-fund-direct-growth"),
)
_FACTS = (
    ("expense ratio", "Expense ratio"),
    ("exit load", "Exit load"),
    ("stamp duty", "Stamp duty"),
    ("benchmark", "Benchmark"),
    ("lock-in", "Lock-in"),
    ("lock in", "Lock-in"),
    ("capital gain", "Capital gain"),
    ("riskometer", "Risk"),
    ("risk level", "Risk"),
    ("risk rating", "Risk"),
    ("minimum sip", "Min. for SIP"),
    ("min sip", "Min. for SIP"),
    ("minimum investment", "Min. for SIP"),
    ("fund size", "Fund size (AUM)"),
)

Embed = Callable[[Sequence[str], str], list[list[float]]]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    url: str
    title: str
    section: str
    fetched_at: str
    distance: float


@dataclass(frozen=True)
class Retrieval:
    question: str
    hits: tuple[Hit, ...]
    corpus_empty: bool


def retrieve(question: str, settings: Settings, embed: Embed | None = None) -> Retrieval:
    """Search stored chunks. The embedding model is the Hugging Face model from config."""
    from app.ingest.store import open_collection

    collection = open_collection(settings)
    if collection.count() == 0:
        return Retrieval(question=question, hits=(), corpus_empty=True)

    encoder = embed or embed_texts
    vector = encoder([question], settings.embedding_model)[0]
    doc_id, section = question_focus(question)
    hits = _search(collection, vector, settings.retrieval_top_k, doc_id, section, question)
    matched = section if any(hit.section == section for hit in hits) else None
    ranked = rank_hits(hits, matched)[: settings.retrieval_top_k]
    retrieval = Retrieval(question=question, hits=ranked, corpus_empty=False)
    _write_retrieval(settings, retrieval)
    return retrieval


def question_focus(question: str) -> tuple[str | None, str | None]:
    """Return the scheme document id and fact section named by the question."""
    lowered = question.lower()
    doc_id = next((doc for phrase, doc in _SCHEMES if phrase in lowered), None)
    if doc_id is None and "hdfc" in lowered:
        doc_id = "hdfc-mutual-funds"
    section = next((name for phrase, name in _FACTS if phrase in lowered), None)
    if section is None and re.search(r"\brisk\b", lowered) and "tax saver" not in lowered:
        section = "Risk"
    scrubbed = lowered.replace("tax saver", " ")
    if section is None and re.search(r"\btax\b", scrubbed):
        section = "Tax"
    return doc_id, section


def rank_hits(hits: tuple[Hit, ...] | list[Hit], section: str | None) -> tuple[Hit, ...]:
    """Prefer the chunk that states a figure for the asked fact."""

    def key(hit: Hit) -> tuple[int, int, float]:
        has_figure = bool(section and (re.search(r"[₹%]", hit.text) or re.search(r"\bnil\b", hit.text, re.I)))
        dated = 1 if re.search(r"\b(?:19|20)\d{2}\b", hit.text) else 0
        return (0 if has_figure else 1, dated, hit.distance)

    ordered = sorted(hits, key=key)
    return tuple(ordered)


def _search(
    collection,
    vector: list[float],
    top_k: int,
    doc_id: str | None,
    section: str | None,
    question: str,
) -> list[Hit]:
    n_results = min(max(top_k, 8), collection.count())
    if doc_id and section:
        hits = _query(collection, vector, n_results, {"$and": [{"doc_id": doc_id}, {"section": section}]})
        if hits:
            return hits
    needle = _document_needle(question)
    if doc_id and needle:
        hits = _query(
            collection,
            vector,
            n_results,
            {"doc_id": doc_id},
            {"$contains": needle},
        )
        if hits:
            return hits
    if doc_id:
        hits = _query(collection, vector, n_results, {"doc_id": doc_id})
        if hits:
            return hits
    return _query(collection, vector, n_results, None)


def _document_needle(question: str) -> str | None:
    """A phrase that appears in the chunk when the section label does not."""
    lowered = question.lower()
    if "fund manager" in lowered or "who manages" in lowered:
        return "Fund Manager"
    if "objective" in lowered:
        return "Investment Objective"
    if re.search(r"\brisk\b", lowered):
        return "rated"
    return None


def _query(
    collection,
    vector: list[float],
    n_results: int,
    where: dict | None,
    where_document: dict | None = None,
) -> list[Hit]:
    kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document
    result = collection.query(**kwargs)
    hits: list[Hit] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        meta = metadata or {}
        hits.append(
            Hit(
                chunk_id=str(chunk_id),
                text=str(document or ""),
                url=str(meta.get("url", "")),
                title=str(meta.get("title", "")),
                section=str(meta.get("section", "")),
                fetched_at=str(meta.get("fetched_at", "")),
                distance=float(distance),
            )
        )
    return hits


def _write_retrieval(settings: Settings, retrieval: Retrieval) -> None:
    settings.retrieval_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload = {
        "question": retrieval.question,
        "embedding_model": settings.embedding_model,
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "score": round(1 - hit.distance, 6),
                "url": hit.url,
                "section": hit.section,
            }
            for hit in retrieval.hits
        ],
    }
    path = settings.retrieval_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
