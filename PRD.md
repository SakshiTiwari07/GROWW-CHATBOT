# PRD: HDFC Mutual Fund Facts Assistant

**Status:** Draft  
**Source:** `doc/problemStatement.txt`  
**Product type:** Working prototype of a small retrieval-augmented FAQ assistant

## 1. Summary

Build a small FAQ assistant that answers factual questions about HDFC Mutual Fund and four of its schemes. Answers come only from official public pages. Every answer includes one source link. The assistant does not give investment advice.

This is a RAG demo. Each pipeline stage must persist its output so the flow can be inspected: load, chunk, embed, store, retrieve, answer.

## 2. Problem

Retail users comparing schemes, and support or content teams answering the same mutual-fund questions, need short factual answers with a citation. They should not get opinions, return comparisons, or answers drawn from unofficial blogs.

## 3. Users

| User | Need |
| --- | --- |
| Retail investor comparing schemes | Look up a scheme fact (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark) without being told what to buy or sell. |
| Support or content team | Answer repetitive factual questions with one official source link. |

## 4. Goals

- Answer factual questions about the in-scope AMC and schemes.
- Cite exactly one source link on every answer, including refusals.
- Refuse advice, portfolio, and performance-comparison questions with a short, polite, facts-only message and one relevant educational link.
- Keep every answer to three sentences or fewer, plus a last-updated line.
- Show the RAG pipeline as stored artifacts at each stage.
- Make the embedding model, vector database, and LLM selectable from configuration.

## 5. Non-goals

- Investment advice, buy/sell recommendations, or portfolio construction.
- Computing, ranking, or comparing returns.
- Scraping pages other than the five listed below, including links inside those pages.
- Using screenshots of the Groww app back end, or third-party blogs, as sources.
- Accepting or storing personal data.
- A full Groww product, account login, transactions, or statement download on the user’s behalf.

## 6. Scope

### 6.1 Corpus

Ingest only these five public pages. Do not follow links on them.

| # | Page | URL |
| --- | --- | --- |
| 1 | AMC overview — HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| 2 | Large Cap — HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | Flexi Cap — HDFC Flexi Cap Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 4 | ELSS (tax saver) — HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 5 | Mid Cap — HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

The submitted source list is a separate deliverable of 15–25 official public URLs. That list may include these five pages plus official pages used only as citation targets (for example an official factsheet, or an SEBI or AMFI educational page on a refusal). Those extra URLs are not scrape targets.

### 6.2 Questions the assistant answers

Factual queries only, for example:

- Expense ratio of a named in-scope scheme
- ELSS lock-in
- Minimum SIP
- Exit load
- Riskometer and benchmark
- How to download a capital-gains statement, as a factual how-to from public pages

If the fact is not in the corpus, say so and still cite the closest official page. Do not invent the fact.

### 6.3 Questions the assistant refuses

Refuse opinionated or portfolio questions, for example “Should I buy?” or “Should I sell?”. The refusal is polite, facts-only, at most three sentences, and includes one relevant educational link.

If the user asks for returns or a performance comparison, do not compute or compare. Point them to the official factsheet with one link.

## 7. Answer contract

Every response, including refusals, must:

1. State the exact fact asked, or state that the question is outside facts-only scope.
2. Stay within three sentences.
3. Include exactly one clear citation link.
4. End with `Last updated from sources: <date>`, where the date is the date those sources were loaded.

Example shape:

> The expense ratio of HDFC Large Cap Fund Direct Growth is 0.99%. Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth  
> Last updated from sources: 2026-11-19

Numbers in examples are illustrative. Live answers must use the loaded page, not this document.

## 8. Guardrails

| Rule | Required behavior |
| --- | --- |
| Public sources only | Cite only official public pages (the corpus, plus an official factsheet or SEBI/AMFI educational page when the answer is a refusal or a factsheet redirect). Never cite a third-party blog or an app back-end screenshot. |
| No advice | Do not recommend buy, sell, hold, allocation, or “best fund”. |
| No performance claims | Do not calculate, rank, or compare returns. Link the official factsheet when returns are asked. |
| No PII | Do not accept or store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers. If a message contains any of these, refuse it and do not write the value to logs, the vector store, or sample files. |
| Missing fact | If retrieval does not support the fact, say it is not in the loaded sources. Do not fill the gap from model memory. |
| Out of corpus | Questions about schemes or AMCs outside the five pages are refused or answered only to say they are out of scope, with one in-corpus or official educational link. |

## 9. User interface

A small chat page:

- A welcome line that names the assistant and the scope (HDFC Mutual Fund facts: AMC overview plus Large Cap, Flexi Cap, ELSS, and Mid Cap direct-growth schemes).
- Three example questions the user can click or send. They must be factual, for example expense ratio, ELSS lock-in, and minimum SIP.
- A persistent note: **Facts-only. No investment advice.**
- Each assistant message shows the answer text, one citation link, and the last-updated line.
- The same disclaimer snippet used in the UI is saved as a deliverable so it can be copied into the README.

Visual direction for the page: professional, using Groww’s public site colors (green on a light background). No account login and no data-entry fields for personal identifiers.

## 10. RAG pipeline

Each stage writes its output to disk so a reviewer can open it. Stages run in this order.

| Stage | What it does | What it stores |
| --- | --- | --- |
| 1. Load | Fetch only the five URLs. Save the raw page text with the source URL and fetch time. Do not crawl outbound or in-page links. | One document record per page. |
| 2. Chunk | Split each loaded document with a strategy chosen for this corpus (scheme fact blocks such as expense ratio, exit load, minimum SIP, riskometer, benchmark, and lock-in should stay intact when the page structure allows it). | Chunk records with chunk id, parent URL, and text. |
| 3. Embed | Embed every chunk with `sentence-transformers/all-MiniLM-L6-v2`. | Embeddings aligned to chunk ids. |
| 4. Store | Upsert chunks, embeddings, and metadata into ChromaDB. | A queryable collection. Metadata must include source URL so citations are exact. |
| 5. Retrieve | Embed the user question with the same model and search the same collection. | The retrieved chunks used for that answer (enough to audit the citation). |
| 6. Answer | The configured LLM writes a response that obeys the answer contract and guardrails. | The final answer text and the single citation URL. |

Retrieval must use the same embedding model and the same vector database as ingestion. The model and database names live in configuration so they can be changed without editing pipeline code.

### 10.1 Configuration

| Setting | Requirement |
| --- | --- |
| Embedding model | Default `sentence-transformers/all-MiniLM-L6-v2`. Same value for ingest and retrieve. Set in config. |
| Vector database | ChromaDB for this exercise. Collection name and path set in config. |
| LLM provider | Groq is the default. The provider interface must also allow Gemini, OpenAI, Anthropic, and Mistral. |
| LLM model id | Read from `.env`. The brief names `qwen/qwen3.8-27b` as the Groq model id; use the value in `.env` so it can be corrected without a code change. |
| API keys | Read from `.env` only. `.env.example` lists the variable names and contains no secrets. |

## 11. Functional requirements

1. A loader fetches the five URLs and refuses to enqueue any other URL found in the HTML.
2. Loaded documents, chunks, and vector-store contents can be inspected after a run.
3. Every stored chunk carries its source URL.
4. A question that maps to a fact in the corpus returns that fact, one citation, and the last-updated line.
5. A buy/sell or “which fund is better” question returns a refusal plus one educational link, and no recommendation.
6. A returns or comparison question returns no computed performance, and links an official factsheet.
7. A message containing PAN, Aadhaar, an account number, an OTP, an email, or a phone number is refused and the value is not stored.
8. The UI shows the welcome line, three example questions, and the facts-only disclaimer.
9. Changing the embedding model, Chroma path, or LLM model id is done in config or `.env`, not by editing stage code.

## 12. Acceptance criteria

- Asking “What is the expense ratio of HDFC Large Cap Fund Direct Growth?” returns the figure from the loaded page, one Groww URL for that scheme, and the last-updated line, in no more than three sentences.
- Asking “What is the ELSS lock-in?” returns the lock-in fact from the ELSS page with that page as the citation.
- Asking “Should I buy the mid cap fund?” does not recommend buying. It returns a polite facts-only refusal and one educational link.
- Asking “Which of these four funds has the best return?” does not rank them. It points to an official factsheet.
- Pasting a PAN, Aadhaar, account number, OTP, email, or phone number produces a refusal and leaves that value out of stored artifacts.
- A source list of 15–25 official URLs exists, and the five ingested pages are in it.
- A sample Q&A file contains 5–10 queries with the assistant’s answers and links.
- Inspecting stage outputs shows loaded documents, chunks, and that those chunks are present in ChromaDB.

## 13. Deliverables

| Deliverable | Contents |
| --- | --- |
| Working prototype | A hosted app link, or a demo video of three minutes or less if hosting is not possible. |
| Source list | CSV or Markdown of the 15–25 official URLs used. |
| README | Setup steps, scope (AMC plus the four schemes), and known limits. |
| Sample Q&A | 5–10 queries with the assistant’s answers and links. |
| Disclaimer snippet | The exact facts-only, no-advice text shown in the UI. |

## 14. Known limits

- Facts are only as fresh as the last load of the five pages.
- Scheme pages on Groww include return figures. The assistant may quote a labeled figure that appears on the page when the user asks for that figure, and must not compute a new return, rank funds, or turn a figure into a recommendation.
- Capital-gains statement steps are answered only if the loaded public pages describe them. The assistant does not log into an account or fetch a user’s statement.
- The Groq model id in the brief (`qwen/qwen3.8-27b`) may not match a live Groq model name. The running id is whatever `.env` sets.

## 15. Skills this prototype is meant to show

- Identify the exact fact asked, then decide to answer or refuse.
- Prompt for short answers, polite refusals, and explicit citation wording.
- Retrieve from a small corpus and cite the AMC, SEBI, or AMFI page the chunk came from.
