"""Load config.yaml and .env. Ingest does not need an API key."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

VALID_PROVIDERS = ("groq", "gemini", "openai", "anthropic", "mistral")
PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
VALID_KINDS = {"corpus", "factsheet", "educational"}
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str


@dataclass(frozen=True)
class Citation:
    title: str
    url: str
    kind: str


@dataclass(frozen=True)
class Settings:
    sources: tuple[Source, ...]
    citation_catalog: tuple[Citation, ...]
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    chunk_max_chars: int
    retrieval_top_k: int
    html_dir: Path
    documents_dir: Path
    chunks_path: Path
    embeddings_path: Path
    retrieval_dir: Path
    answers_dir: Path
    ingest_report_path: Path
    disclaimer_path: Path
    llm_provider: str
    llm_model: str

    def provider_key_name(self) -> str:
        return PROVIDER_KEYS[self.llm_provider]

    def provider_api_key(self) -> str:
        """Return the selected provider key. Call this on the ask path only."""
        name = self.provider_key_name()
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"{name} is empty.")
        return value

    def citations(self, kind: str) -> tuple[Citation, ...]:
        return tuple(item for item in self.citation_catalog if item.kind == kind)


def _require(mapping: dict, key: str, where: str) -> object:
    if key not in mapping:
        raise ValueError(f"{where} is missing {key}")
    return mapping[key]


def _as_path(root: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def load_settings(config_path: Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    path = config_path or (ROOT / "config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must be a mapping")

    sources = _load_sources(raw)
    catalog = _load_catalog(raw)
    _check_catalog(sources, catalog)

    embedding = _require(raw, "embedding", "config")
    chroma = _require(raw, "chroma", "config")
    chunk = _require(raw, "chunk", "config")
    retrieval = _require(raw, "retrieval", "config")
    paths = _require(raw, "paths", "config")
    for name, section in (
        ("embedding", embedding),
        ("chroma", chroma),
        ("chunk", chunk),
        ("retrieval", retrieval),
        ("paths", paths),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip() or DEFAULT_PROVIDER
    if provider not in VALID_PROVIDERS:
        allowed = ", ".join(VALID_PROVIDERS)
        raise ValueError(f"LLM_PROVIDER must be one of: {allowed}")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    max_chars = _require(chunk, "max_chars", "chunk")
    top_k = _require(retrieval, "top_k", "retrieval")
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("chunk.max_chars must be a positive integer")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k must be a positive integer")

    model_name = _require(embedding, "model", "embedding")
    collection = _require(chroma, "collection", "chroma")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("embedding.model must be a non-empty string")
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError("chroma.collection must be a non-empty string")

    settings = Settings(
        sources=sources,
        citation_catalog=catalog,
        embedding_model=model_name,
        chroma_path=_as_path(ROOT, _require(chroma, "path", "chroma"), "chroma.path"),
        chroma_collection=collection,
        chunk_max_chars=max_chars,
        retrieval_top_k=top_k,
        html_dir=_as_path(ROOT, _require(paths, "html", "paths"), "paths.html"),
        documents_dir=_as_path(ROOT, _require(paths, "documents", "paths"), "paths.documents"),
        chunks_path=_as_path(ROOT, _require(paths, "chunks", "paths"), "paths.chunks"),
        embeddings_path=_as_path(ROOT, _require(paths, "embeddings", "paths"), "paths.embeddings"),
        retrieval_dir=_as_path(ROOT, _require(paths, "retrieval", "paths"), "paths.retrieval"),
        answers_dir=_as_path(ROOT, _require(paths, "answers", "paths"), "paths.answers"),
        ingest_report_path=_as_path(
            ROOT, _require(paths, "ingest_report", "paths"), "paths.ingest_report"
        ),
        disclaimer_path=_as_path(ROOT, _require(paths, "disclaimer", "paths"), "paths.disclaimer"),
        llm_provider=provider,
        llm_model=model,
    )
    return _runtime_settings(settings)


def _runtime_settings(settings: Settings) -> Settings:
    """On Vercel, copy Chroma to writable /tmp and keep logs there too."""
    if not os.environ.get("VERCEL"):
        return settings
    import shutil

    runtime = Path("/tmp/hdfc-mf")
    chroma_copy = runtime / "chroma"
    source = settings.chroma_path
    if source.is_dir() and not (chroma_copy / "chroma.sqlite3").is_file():
        if chroma_copy.exists():
            shutil.rmtree(chroma_copy)
        shutil.copytree(source, chroma_copy)
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return replace(
        settings,
        chroma_path=chroma_copy if chroma_copy.is_dir() else settings.chroma_path,
        retrieval_dir=runtime / "retrieval",
        answers_dir=runtime / "answers",
    )


def _load_sources(raw: dict) -> tuple[Source, ...]:
    rows = _require(raw, "sources", "config")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("sources must list exactly five pages")
    sources: list[Source] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"sources[{index}] must be a mapping")
        source = Source(
            id=str(_require(row, "id", f"sources[{index}]")),
            title=str(_require(row, "title", f"sources[{index}]")),
            url=str(_require(row, "url", f"sources[{index}]")),
        )
        if source.id in seen_ids:
            raise ValueError(f"duplicate source id: {source.id}")
        if source.url in seen_urls:
            raise ValueError(f"duplicate source url: {source.url}")
        seen_ids.add(source.id)
        seen_urls.add(source.url)
        sources.append(source)
    return tuple(sources)


def _load_catalog(raw: dict) -> tuple[Citation, ...]:
    rows = _require(raw, "citation_catalog", "config")
    if not isinstance(rows, list) or not 15 <= len(rows) <= 25:
        raise ValueError("citation_catalog must contain 15 to 25 URLs")
    catalog: list[Citation] = []
    seen_urls: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"citation_catalog[{index}] must be a mapping")
        item = Citation(
            title=str(_require(row, "title", f"citation_catalog[{index}]")),
            url=str(_require(row, "url", f"citation_catalog[{index}]")),
            kind=str(_require(row, "kind", f"citation_catalog[{index}]")),
        )
        if item.kind not in VALID_KINDS:
            raise ValueError(f"citation kind must be corpus, factsheet, or educational: {item.url}")
        if item.url in seen_urls:
            raise ValueError(f"duplicate citation url: {item.url}")
        seen_urls.add(item.url)
        catalog.append(item)
    return tuple(catalog)


def _check_catalog(sources: tuple[Source, ...], catalog: tuple[Citation, ...]) -> None:
    source_urls = {source.url for source in sources}
    corpus_urls = {item.url for item in catalog if item.kind == "corpus"}
    if corpus_urls != source_urls:
        missing = source_urls - corpus_urls
        extra = corpus_urls - source_urls
        raise ValueError(
            "each source URL must appear once in citation_catalog with kind corpus. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    overlap = [
        item.url for item in catalog if item.kind != "corpus" and item.url in source_urls
    ]
    if overlap:
        raise ValueError(f"factsheet and educational URLs must not be scrape targets: {overlap}")
