"""Command entry point. Ingest is added after the later pipeline phases."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["scrape"]:
        raise SystemExit(_scrape())
    if args == ["load"]:
        raise SystemExit(_load())
    if args == ["chunk"]:
        raise SystemExit(_chunk())
    if args == ["embed"]:
        raise SystemExit(_embed())
    if args == ["store"]:
        raise SystemExit(_store())
    if args == ["ingest"]:
        raise SystemExit(_ingest())
    if args and args[0] == "ask":
        raise SystemExit(_ask(args[1:]))
    raise SystemExit(
        'Run python -m app.cli ask "your question". '
        "Embeddings use the Hugging Face model in config.yaml. The reply uses Groq."
    )


def _scrape() -> int:
    from app.config import load_settings
    from app.ingest.scrape import scrape_sources

    results = scrape_sources(load_settings())
    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"{status}\t{result.doc_id}\t{result.status_code}\t{result.url}")
        if result.error:
            print(f"  {result.error}")
        if result.ok:
            print(f"  ignored_links={result.ignored_link_count}")
    return 1 if any(not result.ok for result in results) else 0


def _load() -> int:
    from app.config import load_settings
    from app.ingest.load import load_documents

    results = load_documents(load_settings())
    for result in results:
        if result.kept_previous:
            print(f"kept\t{result.doc_id}")
            continue
        document = result.document
        print(f"loaded\t{result.doc_id}\t{document.fetched_at}\t{document.title}")
    return 0


def _chunk() -> int:
    from app.config import load_settings
    from app.ingest.chunk import write_chunks

    chunks = write_chunks(load_settings())
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    for doc_id, count in counts.items():
        print(f"chunked\t{doc_id}\t{count}")
    print(f"chunks\t{len(chunks)}")
    return 0


def _embed() -> int:
    from app.config import load_settings
    from app.ingest.embed import write_embeddings

    records = write_embeddings(load_settings())
    dimension = records[0].dimension if records else 0
    model = records[0].model if records else ""
    print(f"embedded\t{len(records)}\t{model}\t{dimension}")
    return 0


def _store() -> int:
    from app.config import load_settings
    from app.ingest.store import store_chunks

    report = store_chunks(load_settings())
    print(f"stored\t{report.chroma_count}\tchunks\t{report.chunk_count}\tmissing\t{len(report.missing_ids)}")
    return 1 if report.missing_ids else 0


def _ingest() -> int:
    from app.config import load_settings
    from app.ingest.run import run_ingest

    report = run_ingest(load_settings())
    print(f"stored\t{report.chroma_count}\tchunks\t{report.chunk_count}\tmissing\t{len(report.missing_ids)}")
    return 1 if report.missing_ids else 0


def _ask(words: list[str]) -> int:
    from app.ask.service import ask
    from app.config import load_settings

    question = " ".join(words).strip()
    if not question:
        print('Usage: python -m app.cli ask "your question"')
        return 1
    response = ask(question, load_settings())
    print(response.answer)
    print(f"citation\t{response.citation_url}")
    print(f"refused\t{response.refused}")
    return 0


if __name__ == "__main__":
    main()
