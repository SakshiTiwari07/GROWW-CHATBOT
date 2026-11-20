"""Embed chunks with the configured sentence-transformers model."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.config import Settings
from app.ingest.chunk import Chunk, read_chunks

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384

_MODELS: dict[str, object] = {}
Encoder = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    model: str
    dimension: int
    vector: tuple[float, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "model": self.model,
            "dimension": self.dimension,
            "vector": list(self.vector),
        }


def get_model(model_name: str) -> object:
    """Load one SentenceTransformer per model name and reuse it for questions."""
    if os.environ.get("VERCEL"):
        os.environ["HF_HOME"] = "/tmp/huggingface"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence-transformers"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def embed_texts(texts: Sequence[str], model_name: str) -> list[list[float]]:
    """Embed chunk text or a user question with the same model."""
    if not texts:
        return []
    model = get_model(model_name)
    encoded = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return encoded.tolist()


def embed_chunks(
    chunks: Sequence[Chunk],
    model_name: str,
    encode: Encoder | None = None,
) -> list[EmbeddingRecord]:
    _require_unique_ids(chunks)
    if not chunks:
        return []
    encoder = encode or (lambda texts: embed_texts(texts, model_name))
    vectors = encoder([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("The encoder returned a different number of vectors than chunks.")
    expected = DEFAULT_DIMENSION if model_name == DEFAULT_MODEL else None
    records: list[EmbeddingRecord] = []
    for chunk, vector in zip(chunks, vectors):
        if expected is not None and len(vector) != expected:
            raise ValueError(
                f"{model_name} must produce {expected}-dimension vectors, got {len(vector)}."
            )
        if not vector:
            raise ValueError(f"Empty vector for {chunk.chunk_id}.")
        records.append(
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                model=model_name,
                dimension=len(vector),
                vector=tuple(float(value) for value in vector),
            )
        )
    return records


def write_embeddings(settings: Settings, encode: Encoder | None = None) -> list[EmbeddingRecord]:
    chunks = read_chunks(settings.chunks_path)
    records = embed_chunks(chunks, settings.embedding_model, encode=encode)
    path = settings.embeddings_path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(record.to_json()) + "\n" for record in records)
    path.write_text(body, encoding="utf-8")
    return records


def _require_unique_ids(chunks: Sequence[Chunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
