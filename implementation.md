# Implementation: HDFC Mutual Fund Facts Assistant

**Source:** `architecture.md`

Build in this order. Each phase reads the files the previous phase wrote. Ingest stays offline. The chat page only calls `POST /ask`.

Load documents sits between scraping and chunking. Scraping saves HTML. Chunking reads the text records produced by the load step.

| Phase | What it produces |
| --- | --- |
| 1. Create the directory | Project tree, config, dependencies |
| 2. Scrape documents | `data/html/<doc_id>.html` for the five URLs |
| 3. Load documents | `data/documents/<doc_id>.json` |
| 4. Chunk | `data/chunks/chunks.jsonl` |
| 5. Embed | `data/embeddings/embeddings.jsonl` |
| 6. Store in the vector DB | Chroma collection and `data/ingest_report.json` |
| 7. Retrieval logic | `POST /ask` answer with one citation |
| 8. UI | Chat page |
| 9. Guardrail edge cases and tests | Refusals for advice, returns, and personal data |

One command, `python -m app.cli ingest`, runs phases 2–6 in that order.

## 1. Create the directory

```text
config.yaml
.env.example
.gitignore
requirements.txt
app/
  config.py
  disclaimer.txt
  cli.py
  api.py
  ingest/          scrape.py  load.py  chunk.py  embed.py  store.py  run.py
  ask/             guardrails.py  intent.py  retrieve.py  compose.py  validate.py  service.py
  llm/             base.py  factory.py  and one client per provider
  static/          index.html  styles.css  app.js
data/
  html/  documents/  chunks/  embeddings/  chroma/  retrieval/  answers/
doc/
  sources.md
tests/
```

`requirements.txt`: `fastapi`, `uvicorn`, `httpx`, `beautifulsoup4`, `pyyaml`, `python-dotenv`, `sentence-transformers`, `chromadb`, `groq`, `google-genai`, `openai`, `anthropic`, `mistralai`.

`.gitignore` ignores `.env`, `.venv/`, `__pycache__/`, and everything under `data/` except `.gitkeep` files.

`app/disclaimer.txt` is exactly: `Facts-only. No investment advice.`

`config.yaml` holds the non-secret settings:

- `sources`: the five Groww URLs (AMC, Large Cap, Flexi Cap, ELSS, Mid Cap).
- `citation_catalog`: those five as `corpus`, plus official HDFC factsheet and SEBI/AMFI `educational` URLs, 15–25 rows total. The loader never fetches factsheet or educational rows.
- `embedding.model`: `sentence-transformers/all-MiniLM-L6-v2`
- `chroma.path`: `data/chroma`
- `chroma.collection`: `hdfc_mf_facts`
- `chunk.max_chars`: `800`
- `retrieval.top_k`: `4`

`.env.example` lists `LLM_PROVIDER`, `LLM_MODEL`, and the five API keys with empty values. Default provider is `groq`. Default model id is `qwen/qwen3.8-27b`. The running model id is whatever `.env` sets.

`app/config.py` loads the YAML and `.env`. Ingest does not need an API key.

## 2. Scrape documents

Module: `app/ingest/scrape.py`.

Fetch only the five URLs in `sources`. The request list is that allowlist. An exact string match is required before `httpx` is called. Leave redirects off. A redirect or a non-200 response marks that page failed and leaves any previous HTML file in place.

Save the body to `data/html/<doc_id>.html`. `doc_id` is the last path segment, for example `hdfc-large-cap-fund-direct-growth`.

Href values in the HTML are discarded. They are never added to the request list.

## 3. Load documents

Module: `app/ingest/load.py`. Runs after scraping and before chunking.

For each saved HTML file:

1. Remove `script`, `style`, `noscript`, `nav`, `footer`, `header`, `svg`, and `form`.
2. Keep heading breaks as newlines.
3. Write `data/documents/<doc_id>.json` with `doc_id`, `url`, `title`, `text`, and `fetched_at` (`YYYY-MM-DD`).

`fetched_at` is the date later printed as `Last updated from sources:`.

If this run failed to scrape a page, skip the write and keep the existing JSON. Chunking then still has that scheme’s previous text.

The loader does not read `href` or any other URL out of the HTML.

## 4. Chunk

Module: `app/ingest/chunk.py`. Input is the document JSON from phase 3.

Start a new section on a heading or when a line begins with one of these labels: Expense ratio, Exit load, Minimum SIP, Min. for SIP, Riskometer, Risk, Benchmark, Lock-in, Lock in, Stamp duty, Tax, Capital gain, Statement. Text before the first label is the `Overview` section.

A section of at most `chunk.max_chars` (800) is one chunk. The text starts with the section title, so the label and the value stay together. A longer section splits on blank lines, and each piece repeats the section title.

Chunk id: `{doc_id}:{section-slug}:{00}`. The same id is reused on the next ingest.

Write all chunks to `data/chunks/chunks.jsonl`:

```json
{
  "chunk_id": "hdfc-large-cap-fund-direct-growth:expense-ratio:00",
  "doc_id": "hdfc-large-cap-fund-direct-growth",
  "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "title": "HDFC Large Cap Fund Direct Growth",
  "section": "Expense ratio",
  "text": "Expense ratio\n0.99%",
  "fetched_at": "2026-11-19"
}
```

800 characters stays inside the 256-token window of `all-MiniLM-L6-v2`.

## 5. Embed

Module: `app/ingest/embed.py`.

Load `SentenceTransformer` with `embedding.model`. Encode every chunk text. Write `data/embeddings/embeddings.jsonl` with `chunk_id`, `model`, `dimension`, and `vector`. The default model produces 384-dimension vectors.

The ask path embeds a question with this same model name. It does not load a different model.

Line count of the embeddings file equals the chunk count.

## 6. Store in the vector DB

Module: `app/ingest/store.py`.

Open a persistent Chroma client at `chroma.path` and get or create `chroma.collection`. Upsert each chunk:

| Chroma field | Value |
| --- | --- |
| id | `chunk_id` |
| document | chunk text |
| embedding | vector from phase 5 |
| metadata.url | page URL, used as the citation |
| metadata.title | scheme or AMC name |
| metadata.section | heading or label |
| metadata.fetched_at | load date |
| metadata.doc_id | document id |

For each `doc_id`, delete collection ids that this run no longer produced, then upsert the new ids. Questions are never upserted.

Write `data/ingest_report.json` with `chunk_count`, `chroma_count`, and `missing_ids`. An empty `missing_ids` list means every chunk is in Chroma.

`app/ingest/run.py` runs scrape, load, chunk, embed, and store. A failed page continues with the other pages and that page’s previous document.

## 7. Retrieval logic

`POST /ask` accepts `{ "question": "..." }` and returns:

```json
{
  "answer": "...",
  "citation_url": "https://...",
  "last_updated": "2026-11-19",
  "refused": false
}
```

Order inside `app/ask/service.py`:

1. **Personal data.** If the question contains a PAN, Aadhaar number, email, phone number, account number, or OTP, return a refusal with one educational URL. Do not embed it, retrieve, or write the question to disk.
2. **Intent.** Classify the remaining question:
   - `advice` — buy, sell, recommend, which is better, best fund, portfolio.
   - `performance` — compare or calculate returns, rank funds, CAGR.
   - `out_of_scope` — an AMC or scheme other than HDFC and the four in-scope schemes.
   - `factual` — everything else.
3. **Retrieve, factual only.** Embed the question with the ingest model. Query the same Chroma collection for `top_k` chunks. Save `data/retrieval/<timestamp>.json` with the question, chunk ids, scores, and URLs.
4. **Cite.**
   - Factual: one URL from the retrieved chunks, the page that states the fact.
   - Advice or out of scope: one educational URL from the catalog. No retrieval.
   - Performance: one factsheet URL from the catalog. No retrieval and no ranking.
   - Factual but the chunks do not contain the fact: say it is not in the loaded sources, and cite the closest retrieved page.
   - Empty collection: say sources are not loaded yet, and cite the AMC overview URL.
5. **Compose.** The configured LLM (`groq` by default; also `gemini`, `openai`, `anthropic`, `mistral`) writes at most three sentences from the supplied chunks only, plus one allowed URL, and ends with `Last updated from sources: <fetched_at>`.
6. **Check.** Reject the draft when it has more than three sentences, a URL outside the allowed set, a missing last-updated line, or advice/ranking language on a refusal. Replace a rejected draft, or an LLM error, with a one-sentence template that still has one catalog link and the last-updated line.

`refused` is `true` for personal data, advice, performance, out of scope, and an empty collection.

A labeled figure already on the page may be quoted when the user asks for that figure. The composer does no new arithmetic.

Provider clients implement `complete(system_prompt, user_prompt) -> str`. `LLM_MODEL` is the model id. A missing key raises an error that the handler turns into the template. The error text contains no API key.

## 8. UI

`GET /` serves `app/static/index.html`. The page has:

- Title `HDFC Mutual Fund Facts` and a scope line naming the AMC overview plus Large Cap, Flexi Cap, ELSS, and Mid Cap.
- The disclaimer `Facts-only. No investment advice.`
- Three example buttons:
  - What is the expense ratio of HDFC Large Cap Fund Direct Growth?
  - What is the ELSS lock-in?
  - What is the minimum SIP for HDFC Flexi Cap Direct Plan Growth?
- A transcript that shows the answer, one citation link, and the last-updated line.
- One question box and a send button. No login and no personal-identifier fields.

`app.js` sends `POST /ask` and renders the JSON. Colors are tokens: light background `#f4f7f6`, white surface, dark text, Groww green `#00b386` on buttons and the citation link.

## 9. Guardrail edge cases and tests

| Case | Expected result |
| --- | --- |
| Expense ratio of HDFC Large Cap Fund Direct Growth | Fact from the stored chunk, that scheme’s URL, `refused: false` |
| ELSS lock-in | Fact from the ELSS page |
| Minimum SIP for HDFC Mid Cap | Fact from the mid-cap page |
| Exit load of HDFC Flexi Cap | Fact from the flexi-cap page |
| Riskometer and benchmark of HDFC Large Cap | Both facts, large-cap URL |
| How do I download a capital-gains statement? | Steps if the loaded text contains them. Otherwise say the fact is not in the loaded sources and still cite one corpus URL |
| Should I buy the mid cap fund? | Polite refusal, one educational URL, `refused: true`, no recommendation |
| Which of these four funds has the best return? | No ranking and no calculated return. One factsheet URL, `refused: true` |
| Expense ratio of SBI Bluechip Fund | Out of scope, one educational or AMC URL, `refused: true` |
| Question containing a PAN, Aadhaar, email, phone, account number, or OTP | Refusal with one educational URL. No new file under `data/retrieval/` or `data/answers/`, and the value is absent from Chroma |
| Page HTML that links to other schemes | Scraper requests only the allowlisted URL |
| One of the five pages fails to load | Previous document and chunks for that page stay. The other four still update |
| Chroma collection missing or empty | “Sources are not loaded yet.” AMC overview URL. No invented fact |
| Model draft has four sentences, two URLs, or says “you should buy” | Draft is replaced by the template. The user still gets one allowed link |
| LLM key missing or provider error | Same template. Response body contains no API key |

Tests:

- `tests/test_scrape.py` — fixture HTML with extra links; the fetch list stays the five URLs.
- `tests/test_load.py` — a saved HTML file becomes a document JSON with `url`, `text`, and `fetched_at`. A missing HTML file leaves the old JSON unchanged.
- `tests/test_chunk.py` — `Expense ratio` and `0.99%` land in one chunk. A long section splits into pieces that each start with the section title.
- `tests/test_store.py` — after upsert, `missing_ids` is empty and one known chunk id returns the same URL and text.
- `tests/test_guardrails.py` — one sample of each personal-data type. The raw question is not written to retrieval, answers, or Chroma.
- `tests/test_intent.py` — “Should I buy the mid cap fund?” is `advice`. “Which of these four funds has the best return?” is `performance`.
- `tests/test_validate.py` — four sentences, two URLs, a URL outside the allowed set, and “you should buy” on a refusal all fall back to the template.
- `tests/test_ask.py` — the table above through `POST /ask`, with a fake LLM that returns an invalid draft so refusal tests assert the template.

`doc/sources.md` lists the catalog (15–25 official URLs). `doc/sample_qa.md` stores 5–10 real answers and their links from the table. The README covers ingest, the API command, scope, and the limits: facts match the last load, returns are not calculated, and the live model id is `LLM_MODEL`.
