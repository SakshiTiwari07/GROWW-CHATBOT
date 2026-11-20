"""Retrieval uses the Hugging Face embedding model. The wording comes from Groq."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from app.ask.intent import classify_intent
from app.ask.retrieve import Hit, Retrieval
from app.ask.service import ask
from app.config import Citation, Settings, Source
from app.ingest.embed import DEFAULT_MODEL

PAGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
EDU = "https://investor.sebi.gov.in/understanding_mf.html"
FACTSHEET = "https://www.hdfcfund.com/mutual-funds/factsheets"


def settings(root: Path) -> Settings:
    return Settings(
        sources=(Source("hdfc-large-cap-fund-direct-growth", "HDFC Large Cap Fund Direct Growth", PAGE),),
        citation_catalog=(
            Citation("HDFC Large Cap Fund Direct Growth", PAGE, "corpus"),
            Citation("HDFC factsheets", FACTSHEET, "factsheet"),
            Citation("SEBI understanding mutual funds", EDU, "educational"),
        ),
        embedding_model=DEFAULT_MODEL,
        chroma_path=root / "chroma",
        chroma_collection="hdfc_mf_facts",
        chunk_max_chars=800,
        retrieval_top_k=4,
        html_dir=root / "html",
        documents_dir=root / "documents",
        chunks_path=root / "chunks.jsonl",
        embeddings_path=root / "embeddings.jsonl",
        retrieval_dir=root / "retrieval",
        answers_dir=root / "answers",
        ingest_report_path=root / "ingest_report.json",
        disclaimer_path=root / "disclaimer.txt",
        llm_provider="groq",
        llm_model="qwen/qwen3.8-27b",
    )


def hit() -> Hit:
    return Hit(
        chunk_id="fund:expense-ratio:00",
        text="Expense ratio\n0.99%",
        url=PAGE,
        title="HDFC Large Cap Fund Direct Growth",
        section="Expense ratio",
        fetched_at="2026-11-20",
        distance=0.2,
    )


class AskTests(unittest.TestCase):
    def test_personal_data_skips_retrieval_and_groq(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = settings(Path(tmp))

            def explode(*_args, **_kwargs):
                raise AssertionError("retrieval or Groq was called")

            response = ask(
                "My PAN is ABCDE1234F",
                config,
                complete=explode,
                search=explode,
            )

        self.assertTrue(response.refused)
        self.assertEqual(response.citation_url, EDU)
        self.assertNotIn("ABCDE1234F", response.answer)
        self.assertFalse(config.retrieval_dir.exists())
        self.assertFalse(config.answers_dir.exists())

    def test_advice_does_not_search_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = settings(Path(tmp))

            def explode(*_args, **_kwargs):
                raise AssertionError("Chroma was searched")

            response = ask(
                "Should I buy the mid cap fund?",
                config,
                complete=lambda _system, _user: "You should buy it.",
                search=explode,
            )

        self.assertTrue(response.refused)
        self.assertEqual(response.citation_url, EDU)
        self.assertNotIn("should buy", response.answer.lower())

    def test_performance_cites_the_factsheet(self) -> None:
        self.assertEqual(classify_intent("Which of these four funds has the best return?"), "performance")
        with tempfile.TemporaryDirectory() as tmp:
            response = ask(
                "Which of these four funds has the best return?",
                settings(Path(tmp)),
                complete=lambda _system, _user: "Fund A ranks first.",
                search=lambda *_args, **_kwargs: Retrieval("", (), False),
            )
        self.assertTrue(response.refused)
        self.assertEqual(response.citation_url, FACTSHEET)

    def test_factual_answer_uses_groq_text_and_hugging_face_model_name(self) -> None:
        seen: dict[str, str] = {}

        def embed(texts, model_name):
            seen["model"] = model_name
            return [[0.1, 0.2, 0.3]]

        def search(question, config, embed=None):
            embed(["question"], config.embedding_model)
            return Retrieval(question, (hit(),), False)

        def complete(system, user):
            seen["system"] = system
            return (
                "The expense ratio of HDFC Large Cap Fund Direct Growth is 0.99%. "
                f"Source: {PAGE}\nLast updated from sources: 2026-11-20"
            )

        with tempfile.TemporaryDirectory() as tmp:
            response = ask(
                "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
                settings(Path(tmp)),
                complete=complete,
                embed=embed,
                search=search,
            )

        self.assertFalse(response.refused)
        self.assertEqual(response.citation_url, PAGE)
        self.assertIn("0.99%", response.answer)
        self.assertEqual(seen["model"], DEFAULT_MODEL)
        self.assertIn("supplied excerpts", seen["system"])


class DocumentRetrievalTests(unittest.TestCase):
    def test_vercel_reads_the_saved_page_without_chroma(self) -> None:
        from app.ask.retrieve import retrieve

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = settings(root)
            config.documents_dir.mkdir()
            page = {
                "doc_id": "hdfc-large-cap-fund-direct-growth",
                "url": PAGE,
                "title": "HDFC Large Cap Fund Direct Growth",
                "text": "Expense ratio\n0.99%\nExit load\n1% if redeemed within 1 year",
                "fetched_at": "2026-11-20",
            }
            (config.documents_dir / "hdfc-large-cap-fund-direct-growth.json").write_text(
                json.dumps(page),
                encoding="utf-8",
            )
            previous = os.environ.get("VERCEL")
            os.environ["VERCEL"] = "1"
            try:
                found = retrieve(
                    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
                    config,
                )
            finally:
                if previous is None:
                    os.environ.pop("VERCEL", None)
                else:
                    os.environ["VERCEL"] = previous

        self.assertFalse(found.corpus_empty)
        self.assertIn("0.99%", found.hits[0].text)
        self.assertEqual(found.hits[0].section, "Expense ratio")
        self.assertFalse((root / "chroma").exists())


if __name__ == "__main__":
    unittest.main()
