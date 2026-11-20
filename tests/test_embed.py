"""Embedding writes one vector per chunk id and rejects a bad vector size."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings, Source
from app.ingest.chunk import Chunk
from app.ingest.embed import DEFAULT_MODEL, embed_chunks, write_embeddings

PAGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="hdfc-large-cap-fund-direct-growth",
        url=PAGE,
        title="HDFC Large Cap Fund Direct Growth",
        section="Expense ratio",
        text=text,
        fetched_at="2026-11-20",
    )


def settings(root: Path) -> Settings:
    return Settings(
        sources=(Source("hdfc-large-cap-fund-direct-growth", "HDFC Large Cap Fund Direct Growth", PAGE),),
        citation_catalog=(),
        embedding_model=DEFAULT_MODEL,
        chroma_path=root,
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


class EmbedTests(unittest.TestCase):
    def test_one_record_per_chunk(self) -> None:
        chunks = [chunk("fund:expense-ratio:00", "Expense ratio\n0.99%"), chunk("fund:exit-load:00", "Exit load\n1%")]

        def encode(texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 0.5] + [0.0] * 382 for text in texts]

        records = embed_chunks(chunks, DEFAULT_MODEL, encode=encode)
        self.assertEqual([record.chunk_id for record in records], [item.chunk_id for item in chunks])
        self.assertEqual(records[0].model, DEFAULT_MODEL)
        self.assertEqual(records[0].dimension, 384)

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        chunks = [chunk("fund:expense-ratio:00", "one"), chunk("fund:expense-ratio:00", "two")]
        with self.assertRaises(ValueError):
            embed_chunks(chunks, DEFAULT_MODEL, encode=lambda texts: [[0.0] * 384 for _ in texts])

    def test_default_model_rejects_the_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError):
            embed_chunks(
                [chunk("fund:expense-ratio:00", "Expense ratio")],
                DEFAULT_MODEL,
                encode=lambda texts: [[0.1, 0.2]],
            )

    def test_write_embeddings_matches_the_chunk_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = settings(root)
            rows = [chunk("fund:expense-ratio:00", "Expense ratio\n0.99%"), chunk("fund:exit-load:00", "Exit load")]
            config.chunks_path.write_text(
                "".join(json.dumps(item.to_json()) + "\n" for item in rows),
                encoding="utf-8",
            )
            records = write_embeddings(config, encode=lambda texts: [[0.0] * 384 for _ in texts])
            saved = [json.loads(line) for line in config.embeddings_path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(saved), len(records))
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["chunk_id"], "fund:expense-ratio:00")
        self.assertEqual(saved[0]["dimension"], 384)
        self.assertEqual(len(saved[0]["vector"]), 384)


if __name__ == "__main__":
    unittest.main()
