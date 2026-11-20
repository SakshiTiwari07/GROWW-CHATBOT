"""Fetch the five allowlisted pages. Links inside those pages are not requested."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from app.config import Settings, Source

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


@dataclass(frozen=True)
class ScrapeResult:
    doc_id: str
    url: str
    ok: bool
    status_code: int | None
    fetched_at: str
    error: str | None
    ignored_link_count: int
    kept_previous: bool


def allowlist(settings: Settings) -> frozenset[str]:
    return frozenset(source.url for source in settings.sources)


def doc_id_for(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def fetch_allowlisted(url: str, allowed: frozenset[str], client: httpx.Client) -> httpx.Response:
    if url not in allowed:
        raise ValueError(f"Refusing to fetch a URL outside the allowlist: {url}")
    return client.get(url, follow_redirects=False, headers=HEADERS)


def scrape_sources(settings: Settings, client: httpx.Client | None = None) -> list[ScrapeResult]:
    """Fetch each source once. The request list is the allowlist and nothing else."""
    allowed = allowlist(settings)
    requests = [source.url for source in settings.sources if source.url in allowed]
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=False, headers=HEADERS)
    try:
        return [_scrape_one(source, allowed, client, settings.html_dir) for source in settings.sources if source.url in requests]
    finally:
        if owns_client:
            client.close()


def _scrape_one(
    source: Source,
    allowed: frozenset[str],
    client: httpx.Client,
    html_dir: Path,
) -> ScrapeResult:
    doc_id = doc_id_for(source.url)
    destination = html_dir / f"{doc_id}.html"
    try:
        response = fetch_allowlisted(source.url, allowed, client)
    except httpx.HTTPError as exc:
        return _failed(doc_id, source.url, None, str(exc), destination)
    except ValueError as exc:
        return _failed(doc_id, source.url, None, str(exc), destination)

    if response.status_code != 200:
        return _failed(
            doc_id,
            source.url,
            response.status_code,
            f"HTTP {response.status_code}",
            destination,
        )

    body = response.text
    if not body.strip():
        return _failed(doc_id, source.url, response.status_code, "Empty response body", destination)

    html_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return ScrapeResult(
        doc_id=doc_id,
        url=source.url,
        ok=True,
        status_code=response.status_code,
        fetched_at=date.today().isoformat(),
        error=None,
        ignored_link_count=len(_hrefs(body)),
        kept_previous=False,
    )


def _failed(
    doc_id: str,
    url: str,
    status_code: int | None,
    error: str,
    destination: Path,
) -> ScrapeResult:
    return ScrapeResult(
        doc_id=doc_id,
        url=url,
        ok=False,
        status_code=status_code,
        fetched_at="",
        error=error,
        ignored_link_count=0,
        kept_previous=destination.exists(),
    )


def _hrefs(html: str) -> list[str]:
    """Read anchor targets so they can be counted. They are not fetched."""
    soup = BeautifulSoup(html, "html.parser")
    return [anchor.get("href", "") for anchor in soup.find_all("a") if anchor.get("href")]
