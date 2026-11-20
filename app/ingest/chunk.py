"""Split loaded documents into fact chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.config import Settings
from app.ingest.load import Document
from app.ingest.scrape import doc_id_for

LABELS = (
    "Minimum SIP",
    "Min. for SIP",
    "Expense ratio",
    "Riskometer",
    "Capital gain",
    "Stamp duty",
    "Exit load",
    "Lock-in",
    "Lock in",
    "Benchmark",
    "Statement",
    "Tax",
    "Risk",
)
OVERVIEW = "Overview"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    section: str
    text: str
    fetched_at: str

    def to_json(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "fetched_at": self.fetched_at,
        }


def chunk_documents(documents: list[Document], max_chars: int) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("chunk.max_chars must be a positive integer")
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(_chunk_document(document, max_chars))
    return chunks


def write_chunks(settings: Settings, chunks: list[Chunk] | None = None) -> list[Chunk]:
    """Read phase-3 documents and rewrite data/chunks/chunks.jsonl."""
    if chunks is None:
        chunks = chunk_documents(_read_documents(settings), settings.chunk_max_chars)
    path = settings.chunks_path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n" for chunk in chunks)
    path.write_text(body, encoding="utf-8")
    return chunks


def matching_label(line: str) -> str | None:
    """Return the fact label that starts a section, if this line is one."""
    stripped = line.strip()
    lowered = stripped.lower()
    if not lowered:
        return None
    words = lowered.split()
    for label in sorted(LABELS, key=len, reverse=True):
        lab = label.lower()
        if lowered.startswith(lab):
            return label
        if lowered.endswith(lab) and len(words) <= 4 and len(stripped) <= 80:
            return label
    return None


def section_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _chunk_document(document: Document, max_chars: int) -> list[Chunk]:
    sections = _sections(document.text)
    used: dict[str, int] = {}
    chunks: list[Chunk] = []
    for title, lines in sections:
        base = section_slug(title)
        used[base] = used.get(base, 0) + 1
        slug = base if used[base] == 1 else f"{base}-{used[base]}"
        for index, text in enumerate(_pieces(title, lines, max_chars, document.title)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{slug}:{index:02d}",
                    doc_id=document.doc_id,
                    url=document.url,
                    title=document.title,
                    section=title,
                    text=text,
                    fetched_at=document.fetched_at,
                )
            )
    return chunks


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split on fact labels and, after the first label, on the next heading.

    A heading does not close a section until that section has a value line.
    Every non-empty line stays in a section.
    """
    sections: list[tuple[str, list[str]]] = []
    title = OVERVIEW
    lines: list[str] = []
    seen_label = False
    for line in text.splitlines():
        label = matching_label(line)
        if label is not None:
            seen_label = True
            sections.append((title, lines))
            title = label
            lines = [line.strip()]
            continue
        if (
            seen_label
            and any(part.strip() for part in lines)
            and _is_heading(line)
            and not _label_only(title, lines)
        ):
            sections.append((title, lines))
            title = line.strip()
            lines = []
            continue
        lines.append(line)
    sections.append((title, lines))
    return [
        (name, body)
        for name, body in sections
        if name != OVERVIEW or any(part.strip() for part in body)
    ]


def _label_only(title: str, lines: list[str]) -> bool:
    """True when the section is still just the label and the next line is its value."""
    body = [part.strip() for part in lines if part.strip()]
    return len(body) == 1 and body[0].lower() == title.lower()


def _is_heading(line: str) -> bool:
    """A short topic line, not a value like 0.99% or ₹100."""
    stripped = line.strip()
    if not stripped or matching_label(stripped):
        return False
    if re.search(r"[₹%\d]", stripped):
        return False
    words = stripped.split()
    if not 1 <= len(words) <= 6 or len(stripped) > 60:
        return False
    return not stripped.endswith((".", "?", "!"))


def _pieces(title: str, lines: list[str], max_chars: int, scheme: str = "") -> list[str]:
    head = title
    if scheme and scheme not in title:
        head = f"{title}\n{scheme}"
    budget = max_chars - len(head) - 1
    if budget < 1:
        return [title[:max_chars]]
    parts = _pack(lines, budget)
    if not parts:
        return [head]
    return [f"{head}\n{part}" if part else head for part in parts]


def _pack(lines: list[str], budget: int) -> list[str]:
    paragraphs = _paragraphs(lines)
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        blocks = [paragraph] if len(paragraph) <= budget else _split_paragraph(paragraph, budget)
        for block in blocks:
            if not current:
                current = block
                continue
            candidate = current + "\n\n" + block
            if len(candidate) <= budget:
                current = candidate
                continue
            pieces.append(current)
            current = block
    if current:
        pieces.append(current)
    return pieces


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
            continue
        if current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _split_paragraph(paragraph: str, budget: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for line in paragraph.split("\n"):
        segments = [line[index : index + budget] for index in range(0, len(line), budget)] or [""]
        for segment in segments:
            if not current:
                current = segment
                continue
            candidate = current + "\n" + segment
            if len(candidate) <= budget:
                current = candidate
                continue
            pieces.append(current)
            current = segment
    if current:
        pieces.append(current)
    return pieces


def read_chunks(path) -> list[Chunk]:
    if not path.is_file():
        return []
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        chunks.append(
            Chunk(
                chunk_id=str(raw["chunk_id"]),
                doc_id=str(raw["doc_id"]),
                url=str(raw["url"]),
                title=str(raw["title"]),
                section=str(raw["section"]),
                text=str(raw["text"]),
                fetched_at=str(raw["fetched_at"]),
            )
        )
    return chunks


def _read_documents(settings: Settings) -> list[Document]:
    documents: list[Document] = []
    for source in settings.sources:
        path = settings.documents_dir / f"{doc_id_for(source.url)}.json"
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        documents.append(
            Document(
                doc_id=str(raw["doc_id"]),
                url=str(raw["url"]),
                title=str(raw["title"]),
                text=str(raw["text"]),
                fetched_at=str(raw["fetched_at"]),
            )
        )
    return documents
