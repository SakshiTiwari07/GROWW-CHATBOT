"""Turn saved HTML into document records. Href values are not read."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from app.config import Settings, Source
from app.ingest.scrape import ScrapeResult, doc_id_for

REMOVED_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "svg", "form")
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass(frozen=True)
class Document:
    doc_id: str
    url: str
    title: str
    text: str
    fetched_at: str

    def to_json(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
        }


@dataclass(frozen=True)
class LoadResult:
    doc_id: str
    url: str
    kept_previous: bool
    document: Document | None


def load_documents(
    settings: Settings,
    scrape_results: list[ScrapeResult] | None = None,
) -> list[LoadResult]:
    """Write one JSON record per page. A failed scrape leaves the previous JSON in place."""
    scrape_by_id = None
    if scrape_results is not None:
        scrape_by_id = {doc_id_for(result.url): result for result in scrape_results}
    return [_load_one(settings, source, scrape_by_id) for source in settings.sources]


def extract_document(html: str, fallback_title: str) -> tuple[str, str]:
    """Return the h1 title and visible text. Attributes such as href are ignored."""
    soup = BeautifulSoup(html, "html.parser")
    header_facts = _header_facts(soup)
    for tag in soup.find_all(REMOVED_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        tag.attrs = {}

    title = fallback_title
    heading = soup.find("h1")
    if heading is not None:
        heading_text = " ".join(heading.get_text(" ", strip=True).split())
        if heading_text:
            title = heading_text

    for heading_tag in soup.find_all(HEADING_TAGS):
        heading_tag.insert_before("\n")
        heading_tag.insert_after("\n")

    lines: list[str] = []
    for line in soup.get_text("\n").splitlines():
        clean = " ".join(line.split())
        if clean:
            lines.append(clean)
    return title, "\n".join(header_facts + lines)


def _header_facts(soup) -> list[str]:
    """Keep a fact badge such as '3Y Lock-in' that Groww places in the page header."""
    from app.ingest.chunk import matching_label

    facts: list[str] = []
    seen: set[str] = set()
    for header in soup.find_all("header"):
        for text in header.stripped_strings:
            clean = " ".join(text.split())
            if not clean or clean in seen or matching_label(clean) is None:
                continue
            facts.append(clean)
            seen.add(clean)
    return facts


def _load_one(
    settings: Settings,
    source: Source,
    scrape_by_id: dict[str, ScrapeResult] | None,
) -> LoadResult:
    doc_id = doc_id_for(source.url)
    html_path = settings.html_dir / f"{doc_id}.html"
    json_path = settings.documents_dir / f"{doc_id}.json"

    if scrape_by_id is not None:
        scrape = scrape_by_id.get(doc_id)
        if scrape is None or not scrape.ok:
            return _kept(doc_id, source.url, json_path)
        fetched_at = scrape.fetched_at
    else:
        fetched_at = _file_date(html_path)

    if not html_path.is_file() or not fetched_at:
        return _kept(doc_id, source.url, json_path)

    title, text = extract_document(
        html_path.read_text(encoding="utf-8", errors="replace"),
        source.title,
    )
    document = Document(
        doc_id=doc_id,
        url=source.url,
        title=title,
        text=text,
        fetched_at=fetched_at,
    )
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(document.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return LoadResult(doc_id=doc_id, url=source.url, kept_previous=False, document=document)


def _kept(doc_id: str, url: str, json_path: Path) -> LoadResult:
    return LoadResult(
        doc_id=doc_id,
        url=url,
        kept_previous=json_path.is_file(),
        document=_read_document(json_path),
    )


def _read_document(path: Path) -> Document | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Document(
        doc_id=str(raw["doc_id"]),
        url=str(raw["url"]),
        title=str(raw["title"]),
        text=str(raw["text"]),
        fetched_at=str(raw["fetched_at"]),
    )


def _file_date(path: Path) -> str:
    if not path.is_file():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
