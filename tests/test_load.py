"""Loading writes document JSON and leaves the previous file when HTML is missing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import Settings, Source
from app.ingest.load import extract_document, load_documents
from app.ingest.scrape import ScrapeResult

PAGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
DOC_ID = "hdfc-large-cap-fund-direct-growth"


def settings(root: Path) -> Settings:
    return Settings(
        sources=(Source(DOC_ID, "HDFC Large Cap Fund Direct Growth", PAGE),),
        citation_catalog=(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
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


HTML = """
<html>
  <head><title>Ignored title</title></head>
  <header><a href="https://groww.in/secret">Home</a></header>
  <nav><a href="https://groww.in/other-fund">Other fund</a></nav>
  <body>
    <h1>HDFC Large Cap Fund Direct Growth</h1>
    <script>var secret = "https://internal.example/api";</script>
    <p>Expense ratio</p>
    <p>0.99%</p>
    <h3>Exit load</h3>
    <p>Exit load of 1% if redeemed within 1 year</p>
    <a href="https://www.hdfcfund.com/sid">Scheme Information Document</a>
    <footer>Footer links</footer>
  </body>
</html>
"""


class LoadTests(unittest.TestCase):
    def test_html_becomes_document_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = settings(root)
            html_path = config.html_dir / f"{DOC_ID}.html"
            html_path.parent.mkdir()
            html_path.write_text(HTML, encoding="utf-8")

            results = load_documents(config)
            document = results[0].document
            saved = json.loads((config.documents_dir / f"{DOC_ID}.json").read_text(encoding="utf-8"))

        self.assertFalse(results[0].kept_previous)
        self.assertEqual(saved["url"], PAGE)
        self.assertEqual(saved["title"], "HDFC Large Cap Fund Direct Growth")
        self.assertEqual(saved["fetched_at"], document.fetched_at)
        self.assertIn("Expense ratio", saved["text"])
        self.assertIn("0.99%", saved["text"])
        self.assertNotIn("https://groww.in/secret", saved["text"])
        self.assertNotIn("https://groww.in/other-fund", saved["text"])
        self.assertNotIn("https://internal.example/api", saved["text"])
        self.assertNotIn("https://www.hdfcfund.com/sid", saved["text"])
        self.assertNotIn("Footer links", saved["text"])
        self.assertNotIn("Home", saved["text"])
        expense_index = saved["text"].splitlines().index("Expense ratio")
        self.assertEqual(saved["text"].splitlines()[expense_index + 1], "0.99%")

    def test_header_lock_in_badge_is_kept(self) -> None:
        html = """
        <html><body>
          <header><a href="/mutual-funds/filter">Home</a><span>ELSS • 3Y Lock-in</span></header>
          <h1>HDFC ELSS Tax Saver Fund Direct Plan Growth</h1>
          <p>Expense ratio</p><p>1.19%</p>
        </body></html>
        """
        _title, text = extract_document(html, "HDFC ELSS Tax Saver Fund Direct Plan Growth")
        self.assertIn("ELSS • 3Y Lock-in", text)
        self.assertNotIn("Home", text)

    def test_missing_html_keeps_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = settings(root)
            config.documents_dir.mkdir()
            previous = {
                "doc_id": DOC_ID,
                "url": PAGE,
                "title": "Previous",
                "text": "previous text",
                "fetched_at": "2026-11-19",
            }
            path = config.documents_dir / f"{DOC_ID}.json"
            path.write_text(json.dumps(previous), encoding="utf-8")

            results = load_documents(config)

            self.assertTrue(results[0].kept_previous)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), previous)
            self.assertEqual(results[0].document.text, "previous text")

    def test_failed_scrape_keeps_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = settings(root)
            config.html_dir.mkdir()
            config.documents_dir.mkdir()
            (config.html_dir / f"{DOC_ID}.html").write_text(HTML, encoding="utf-8")
            previous = {
                "doc_id": DOC_ID,
                "url": PAGE,
                "title": "Previous",
                "text": "previous text",
                "fetched_at": "2026-11-19",
            }
            path = config.documents_dir / f"{DOC_ID}.json"
            path.write_text(json.dumps(previous), encoding="utf-8")
            failed = ScrapeResult(
                doc_id=DOC_ID,
                url=PAGE,
                ok=False,
                status_code=302,
                fetched_at="",
                error="HTTP 302",
                ignored_link_count=0,
                kept_previous=True,
            )

            results = load_documents(config, [failed])

            self.assertTrue(results[0].kept_previous)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["text"], "previous text")

    def test_extract_uses_fallback_title_when_h1_is_missing(self) -> None:
        title, text = extract_document("<p>Minimum SIP</p><p>₹100</p>", "Configured title")
        self.assertEqual(title, "Configured title")
        self.assertIn("Minimum SIP", text)


if __name__ == "__main__":
    unittest.main()
