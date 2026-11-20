"""Run the offline ingest stages in order."""

from __future__ import annotations

from app.config import Settings
from app.ingest.embed import write_embeddings
from app.ingest.chunk import write_chunks
from app.ingest.load import load_documents
from app.ingest.scrape import scrape_sources
from app.ingest.store import StoreReport, store_chunks


def run_ingest(settings: Settings) -> StoreReport:
    """Scrape, load, chunk, embed, and store. A failed page keeps its previous document."""
    scrape_results = scrape_sources(settings)
    load_documents(settings, scrape_results)
    write_chunks(settings)
    write_embeddings(settings)
    return store_chunks(settings)
