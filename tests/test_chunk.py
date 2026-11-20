"""Chunking keeps a fact label with its value and stays within max_chars."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings, Source
from app.ingest.chunk import chunk_documents, write_chunks
from app.ingest.load import Document

PAGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
DOC_ID = "hdfc-large-cap-fund-direct-growth"


def document(text: str) -> Document:
    return Document(
        doc_id=DOC_ID,
        url=PAGE,
        title="HDFC Large Cap Fund Direct Growth",
        text=text,
        fetched_at="2026-11-20",
    )


class ChunkTests(unittest.TestCase):
    def test_expense_ratio_stays_with_its_value(self) -> None:
        chunks = chunk_documents(
            [document("About the fund\nExpense ratio\n0.99%\nExit load\n1% within 1 year")],
            max_chars=800,
        )
        expense = next(chunk for chunk in chunks if chunk.section == "Expense ratio")
        self.assertEqual(expense.chunk_id, f"{DOC_ID}:expense-ratio:00")
        self.assertIn("Expense ratio", expense.text)
        self.assertIn("0.99%", expense.text)
        self.assertNotIn("Exit load", expense.text)
        self.assertEqual(expense.url, PAGE)
        self.assertEqual(chunks[0].section, "Overview")
        self.assertIn("About the fund", chunks[0].text)

    def test_long_section_splits_and_repeats_the_title(self) -> None:
        body = "\n".join(f"line {index} " + ("word " * 12) for index in range(30))
        chunks = chunk_documents([document(f"Expense ratio\n{body}")], max_chars=200)
        expense = [chunk for chunk in chunks if chunk.section == "Expense ratio"]
        self.assertGreater(len(expense), 1)
        self.assertEqual(expense[0].chunk_id, f"{DOC_ID}:expense-ratio:00")
        self.assertEqual(expense[1].chunk_id, f"{DOC_ID}:expense-ratio:01")
        for chunk in expense:
            self.assertTrue(chunk.text.startswith("Expense ratio"))
            self.assertLessEqual(len(chunk.text), 200)

    def test_blank_lines_split_a_long_section(self) -> None:
        text = "Expense ratio\n" + ("alpha " * 30) + "\n\n" + ("beta " * 30)
        chunks = chunk_documents([document(text)], max_chars=120)
        expense = [chunk for chunk in chunks if chunk.section == "Expense ratio"]
        self.assertGreater(len(expense), 1)
        self.assertTrue(any("alpha" in chunk.text for chunk in expense))
        self.assertTrue(any("beta" in chunk.text for chunk in expense))
        for chunk in expense:
            self.assertTrue(chunk.text.startswith("Expense ratio"))
            self.assertLessEqual(len(chunk.text), 120)

    def test_fund_benchmark_line_becomes_a_benchmark_section(self) -> None:
        chunks = chunk_documents(
            [document("Overview line\nFund benchmark\nNIFTY 100 Total Return Index")],
            max_chars=800,
        )
        benchmark = next(chunk for chunk in chunks if chunk.section == "Benchmark")
        self.assertIn("NIFTY 100 Total Return Index", benchmark.text)
        self.assertTrue(benchmark.text.startswith("Benchmark"))

    def test_write_chunks_reads_document_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            documents.mkdir()
            payload = document("Expense ratio\n0.99%").to_json()
            (documents / f"{DOC_ID}.json").write_text(json.dumps(payload), encoding="utf-8")
            settings = Settings(
                sources=(Source(DOC_ID, "HDFC Large Cap Fund Direct Growth", PAGE),),
                citation_catalog=(),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chroma_path=root,
                chroma_collection="hdfc_mf_facts",
                chunk_max_chars=800,
                retrieval_top_k=4,
                html_dir=root / "html",
                documents_dir=documents,
                chunks_path=root / "chunks" / "chunks.jsonl",
                embeddings_path=root / "embeddings.jsonl",
                retrieval_dir=root,
                answers_dir=root,
                ingest_report_path=root / "ingest_report.json",
                disclaimer_path=root / "disclaimer.txt",
                llm_provider="groq",
                llm_model="qwen/qwen3.8-27b",
            )
            chunks = write_chunks(settings)
            saved = settings.chunks_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(saved), len(chunks))
        self.assertEqual(json.loads(saved[-1])["url"], PAGE)

    def test_fact_section_stops_at_the_next_heading(self) -> None:
        text = "\n".join(
            [
                "Nav item",
                "Expense ratio",
                "0.99%",
                "Return calculator",
                "Monthly SIP",
                "₹5,000",
                "Fund benchmark",
                "NIFTY 100 Total Return Index",
                "Scheme Information Document(SID)",
            ]
        )
        chunks = chunk_documents([document(text)], max_chars=800)
        expense = next(chunk for chunk in chunks if chunk.section == "Expense ratio")
        benchmark = next(chunk for chunk in chunks if chunk.section == "Benchmark")
        self.assertIn("0.99%", expense.text)
        self.assertIn("HDFC Large Cap Fund Direct Growth", expense.text)
        self.assertNotIn("Return calculator", expense.text)
        self.assertNotIn("₹5,000", expense.text)
        self.assertIn("NIFTY 100 Total Return Index", benchmark.text)
        self.assertNotIn("Scheme Information Document", benchmark.text)
        covered = "\n".join(chunk.text for chunk in chunks)
        for line in text.splitlines():
            self.assertIn(line, covered)

    def test_every_line_of_a_fact_page_is_kept(self) -> None:
        text = "Intro line\nMin. for SIP\n₹100\nFund size (AUM)\n₹37,466.30 Cr\nExit load of 1% if redeemed within 1 year"
        chunks = chunk_documents([document(text)], max_chars=800)
        covered = "\n".join(chunk.text for chunk in chunks)
        for line in text.splitlines():
            self.assertIn(line, covered)

    def test_exit_load_keeps_nil_with_the_label(self) -> None:
        chunks = chunk_documents([document("Exit load\nNil\nStamp duty\n0.005%")], max_chars=800)
        exit_load = next(chunk for chunk in chunks if chunk.section == "Exit load")
        self.assertIn("Nil", exit_load.text)
        self.assertNotIn("0.005%", exit_load.text)


if __name__ == "__main__":
    unittest.main()
