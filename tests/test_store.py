"""Chroma stores every chunk that has a vector and drops ids the new run removed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings, Source
from app.ingest.chunk import Chunk
from app.ingest.store import store_chunks

PAGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
DOC_ID = "hdfc-large-cap-fund-direct-growth"


def one_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        url=PAGE,
        title="HDFC Large Cap Fund Direct Growth",
        section="Expense ratio",
        text=text,
        fetched_at="2026-11-20",
    )


def settings(root: Path) -> Settings:
    return Settings(
        sources=(Source(DOC_ID, "HDFC Large Cap Fund Direct Growth", PAGE),),
        citation_catalog=(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chroma_path=root / "chroma",
        chroma_collection="hdfc_mf_facts",
        chunk_max_chars=800,
        retrieval_top_k=4,
        html_dir=root / "html",
        documents_dir=root / "documents",
        chunks_path=root / "chunks.jsonl",
        embeddings_path=root / "embeddings.jsonl",
        retrieval_dir=root,
        answers_dir=root,
        ingest_report_path=root / "ingest_report.json",
        disclaimer_path=root / "disclaimer.txt",
        llm_provider="groq",
        llm_model="qwen/qwen3.8-27b",
    )


def write_inputs(config: Settings, chunks: list[Chunk], vectors: dict[str, list[float]]) -> None:
    config.chunks_path.write_text(
        "".join(json.dumps(chunk.to_json()) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    config.embeddings_path.write_text(
        "".join(
            json.dumps({"chunk_id": chunk_id, "model": "test", "dimension": len(vector), "vector": vector}) + "\n"
            for chunk_id, vector in vectors.items()
        ),
        encoding="utf-8",
    )


class StoreTests(unittest.TestCase):
    def test_upsert_keeps_url_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = settings(Path(tmp))
            chunk = one_chunk("fund:expense-ratio:00", "Expense ratio\n0.99%")
            write_inputs(config, [chunk], {chunk.chunk_id: [0.1, 0.2, 0.3]})
            report = store_chunks(config)
            saved = json.loads(config.ingest_report_path.read_text(encoding="utf-8"))
            from app.ingest.store import open_collection

            row = open_collection(config).get(ids=[chunk.chunk_id], include=["documents", "metadatas"])

        self.assertEqual(report.missing_ids, ())
        self.assertEqual(report.chunk_count, 1)
        self.assertEqual(report.chroma_count, 1)
        self.assertEqual(saved["missing_ids"], [])
        self.assertEqual(row["documents"][0], "Expense ratio\n0.99%")
        self.assertEqual(row["metadatas"][0]["url"], PAGE)

    def test_removed_chunk_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = settings(Path(tmp))
            first = one_chunk("fund:expense-ratio:00", "Expense ratio\n0.99%")
            second = one_chunk("fund:exit-load:00", "Exit load\n1%")
            write_inputs(
                config,
                [first, second],
                {first.chunk_id: [0.1, 0.2, 0.3], second.chunk_id: [0.4, 0.5, 0.6]},
            )
            store_chunks(config)
            write_inputs(config, [first], {first.chunk_id: [0.1, 0.2, 0.3]})
            report = store_chunks(config)
            from app.ingest.store import open_collection

            remaining = open_collection(config).get(include=[])["ids"]

        self.assertEqual(report.missing_ids, ())
        self.assertEqual(remaining, [first.chunk_id])

    def test_chunk_without_a_vector_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = settings(Path(tmp))
            chunk = one_chunk("fund:expense-ratio:00", "Expense ratio\n0.99%")
            write_inputs(config, [chunk], {})
            report = store_chunks(config)

        self.assertEqual(report.missing_ids, (chunk.chunk_id,))
        self.assertEqual(report.chroma_count, 0)


if __name__ == "__main__":
    unittest.main()
