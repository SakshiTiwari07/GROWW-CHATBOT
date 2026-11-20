"""The scraper requests the allowlist only and keeps a previous file on failure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from app.config import Settings, Source
from app.ingest.scrape import fetch_allowlisted, scrape_sources

LARGE = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
ELSS = "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"
OTHER = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


def settings(html_dir: Path) -> Settings:
    root = Path("/tmp/unused")
    return Settings(
        sources=(
            Source("hdfc-large-cap-fund-direct-growth", "HDFC Large Cap Fund Direct Growth", LARGE),
            Source("hdfc-elss-tax-saver-fund-direct-plan-growth", "HDFC ELSS Tax Saver Fund Direct Plan Growth", ELSS),
        ),
        citation_catalog=(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chroma_path=root,
        chroma_collection="hdfc_mf_facts",
        chunk_max_chars=800,
        retrieval_top_k=4,
        html_dir=html_dir,
        documents_dir=root,
        chunks_path=root / "chunks.jsonl",
        embeddings_path=root / "embeddings.jsonl",
        retrieval_dir=root,
        answers_dir=root,
        ingest_report_path=root / "ingest_report.json",
        disclaimer_path=root / "disclaimer.txt",
        llm_provider="groq",
        llm_model="qwen/qwen3.8-27b",
    )


class ScrapeTests(unittest.TestCase):
    def test_fixture_links_are_not_fetched(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if str(request.url) == OTHER:
                return httpx.Response(599, text="should not be fetched")
            html = f"<html><h1>HDFC</h1><a href='{OTHER}'>other scheme</a></html>"
            return httpx.Response(200, text=html)

        with tempfile.TemporaryDirectory() as tmp:
            html_dir = Path(tmp)
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                results = scrape_sources(settings(html_dir), client)

            self.assertEqual(requested, [LARGE, ELSS])
            self.assertTrue(all(result.ok for result in results))
            self.assertEqual(results[0].ignored_link_count, 1)
            saved = (html_dir / "hdfc-large-cap-fund-direct-growth.html").read_text(encoding="utf-8")
            self.assertIn(OTHER, saved)

    def test_redirect_keeps_previous_html(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == LARGE:
                return httpx.Response(302, headers={"location": OTHER})
            return httpx.Response(200, text="<html><p>ELSS</p></html>")

        with tempfile.TemporaryDirectory() as tmp:
            html_dir = Path(tmp)
            previous = html_dir / "hdfc-large-cap-fund-direct-growth.html"
            previous.write_text("previous page", encoding="utf-8")
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                results = scrape_sources(settings(html_dir), client)

            self.assertFalse(results[0].ok)
            self.assertTrue(results[0].kept_previous)
            self.assertEqual(previous.read_text(encoding="utf-8"), "previous page")
            self.assertTrue(results[1].ok)

    def test_non_allowlisted_url_is_refused_before_the_client_is_called(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"client was called for {request.url}")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ValueError):
                fetch_allowlisted(OTHER, frozenset({LARGE}), client)


if __name__ == "__main__":
    unittest.main()
