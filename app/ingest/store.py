"""Upsert chunk text and phase-5 vectors into ChromaDB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingest.chunk import Chunk, read_chunks


@dataclass(frozen=True)
class StoreReport:
    chunk_count: int
    chroma_count: int
    missing_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "chroma_count": self.chroma_count,
            "missing_ids": list(self.missing_ids),
        }


class StoredVectors:
    """Chroma calls this only when embeddings are omitted. This store always passes them."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "stored-vectors"

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise RuntimeError("Pass the phase-5 vectors. This collection does not embed text.")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        raise RuntimeError("Pass the phase-5 vectors. This collection does not embed text.")

    def get_config(self) -> dict[str, str]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, str]) -> "StoredVectors":
        return StoredVectors()

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        return False


def store_chunks(settings: Settings) -> StoreReport:
    chunks = read_chunks(settings.chunks_path)
    vectors = _read_vectors(settings.embeddings_path)
    collection = open_collection(settings)
    _delete_stale(collection, chunks)
    ready = [chunk for chunk in chunks if chunk.chunk_id in vectors]
    if ready:
        collection.upsert(
            ids=[chunk.chunk_id for chunk in ready],
            documents=[chunk.text for chunk in ready],
            embeddings=[vectors[chunk.chunk_id] for chunk in ready],
            metadatas=[_metadata(chunk) for chunk in ready],
        )
    missing = _missing_ids(collection, [chunk.chunk_id for chunk in chunks])
    report = StoreReport(
        chunk_count=len(chunks),
        chroma_count=collection.count(),
        missing_ids=tuple(missing),
    )
    _write_report(settings.ingest_report_path, report)
    return report


def open_collection(settings: Settings):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import register_embedding_function

    register_embedding_function(StoredVectors)

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=StoredVectors(),
        metadata={"hnsw:space": "cosine"},
    )


def _delete_stale(collection, chunks: list[Chunk]) -> None:
    produced: dict[str, set[str]] = {}
    for chunk in chunks:
        produced.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
    stale: list[str] = []
    for doc_id, new_ids in produced.items():
        existing = collection.get(where={"doc_id": doc_id}, include=[])
        stale.extend(chunk_id for chunk_id in existing["ids"] if chunk_id not in new_ids)
    if stale:
        collection.delete(ids=stale)


def _missing_ids(collection, chunk_ids: list[str]) -> list[str]:
    if not chunk_ids:
        return []
    found = set(collection.get(ids=chunk_ids, include=[])["ids"])
    return [chunk_id for chunk_id in chunk_ids if chunk_id not in found]


def _metadata(chunk: Chunk) -> dict[str, str]:
    return {
        "url": chunk.url,
        "title": chunk.title,
        "section": chunk.section,
        "fetched_at": chunk.fetched_at,
        "doc_id": chunk.doc_id,
    }


def _read_vectors(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    vectors: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        vectors[str(raw["chunk_id"])] = [float(value) for value in raw["vector"]]
    return vectors


def _write_report(path: Path, report: StoreReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
