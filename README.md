# RAGifier AI

Drop a PDF, Word file or spreadsheet into the chat and ask about it. Answers
cite the document and page.

```
you    [attaches handbook.pdf]  what does error NWL-4471 mean?

        NWL-4471 means the Route Planner could not reach the Telemetry
        Collector within the configured timeout (Logistics Handbook, p13).
```

- **Small documents are read in full. Large ones are indexed.** RAGifier picks
  per file, by size. Nothing to configure.
- **Works with no embedding server.** Keyword search (BM25) is built in, so it
  runs the moment you install it. Add embeddings later as an upgrade.
- **Bring your own model.** Any OpenAI-compatible endpoint, or your own Ollama.
  One variable switches between them.
- **It never pretends to see images.** Ask about a diagram and it tells you the
  page number instead of inventing a description.
- Everything stays on your machine, apart from whatever you send your model.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/ragifier-ai.git
cd ragifier-ai

python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env     # set a model - see below
python ragifier.py            # opens http://127.0.0.1:8808
```

Requires Python 3.10+. The only thing you must configure is a model:

```bash
LLM_PROVIDER=cloud
CLOUD_BASE_URL=https://api.openai.com/v1
CLOUD_API_KEY=sk-...
CLOUD_MODEL=gpt-4o-mini
```

or your own box:

```bash
LLM_PROVIDER=local
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL=qwen2.5:14b
```

Then attach a file in the chat and ask a question. That's the whole workflow.

> Your model must support **tool calling**, or it will chat happily and never
> read anything. Known good locally: `qwen2.5`, `qwen3`, `llama3.1`/`3.3`,
> `mistral-nemo`.

---

## The small-vs-large decision

This is the part worth understanding, because it is what makes answers
trustworthy.

| | when | how it is answered |
|---|---|---|
| **CONTEXT** | fits the token budget | the model reads **every word** |
| **INDEXED** | too big | chunked, searched per question |

Retrieval can miss. Reading cannot. So anything small enough is handed over
whole — no chunking, no embedding, no retrieval step that might return the
wrong paragraph. A two-page memo is answered from the entire memo.

`CONTEXT_TOKEN_BUDGET` (default 6000 ≈ 10–12 pages of prose) is both the
threshold **and a shared pool**. Several small documents share it; once the
total would exceed it, later ones are indexed instead — so uploading a fifth
short file cannot quietly evict the first four.

Both paths feed a single answer. Ask one question across a 300-page manual and
a one-page summary, and the model gets the whole summary plus the best excerpts
from the manual.

Raise the budget if your model has a large context window and you prefer
full-text accuracy; lower it if you are paying per token. Override per document
with `mode` if you must, but the automatic choice is usually right.

---

## Search

`EMBED_PROVIDER` decides how indexed documents are searched:

| value | what you get | needs |
|---|---|---|
| `none` *(default)* | keyword search (BM25) | nothing |
| `local` | vector + keyword | Ollama, `ollama pull nomic-embed-text` |
| `cloud` | vector + keyword | an embeddings API key |

**`none` is a real option, not a stub.** On a 30-page technical handbook,
keyword-only search put the correct passage at **rank 1 for 8 out of 8** test
questions. BM25 is strong on the things people actually look up in documents:
error codes, part numbers, names, exact phrases.

Where it struggles is paraphrase — asking "how do I restore factory settings"
when the manual says "revert to shipped configuration". That is what embeddings
fix. When they are configured, **both** run and the rankings are fused, because
they fail in different ways: embeddings blur exact tokens, BM25 misses synonyms.

> `EMBED_PROVIDER=cloud` **sends your document text to that provider.** Use
> `local` or `none` for anything confidential.

---

## Supported files

| | notes |
|---|---|
| `.pdf` | text, page numbers, embedded images |
| `.docx` | headings and tables (no page numbers — Word has none until rendered) |
| `.pptx` | slide text and speaker notes; page = slide |
| `.xlsx` | cell values; page = sheet |
| `.csv` `.tsv` | parsed as rows |
| `.html` `.epub` | text, with `img` alt text as captions |
| `.md` `.txt` `.json` + source code | as-is |

Readers are independent — a missing package disables only that type and tells
you what to install.

**Scanned PDFs have no extractable text.** RAGifier detects this and says so
rather than silently indexing nothing; run them through OCR (e.g. `ocrmypdf`)
first.

Old `.doc` and Apple `.pages` are not supported — export to `.docx` or PDF.

---

## Images

There is no vision here. RAGifier will not describe a chart, and is instructed
never to guess from surrounding text.

What it does instead: records which **page** every embedded image is on,
matches nearby `Figure 3: ...` captions, and extracts the image files. Ask
"what does the architecture diagram show?" and you get:

```
Logistics Handbook   page 4    Figure 1: Service topology overview
```

— which is what you need in order to go and look.

One limitation worth knowing: **charts exported from Office are often vector
graphics, not embedded images**, and are not detected. The text around them is
still searchable.

---

## Endpoints

| route | |
|---|---|
| `/` | the chat UI |
| `/config` | model and document counts |
| `/api/documents` | what is loaded, how each is searched, budget used |
| `/api/debug` | what is running, backend health, retrieval settings |
| `/api/stop` | abort everything — works without the websocket |
| `/api/figure/{doc_id}/{file}` | an extracted image |

---

## Security

- **It binds to `127.0.0.1` by default. Keep it there.** The websocket is plain
  `ws://`, and your documents' contents travel over it. On `0.0.0.0` anyone who
  can reach the port can read everything you have uploaded.
- **There is no authentication.**
- `storage/` holds the **full extracted text** of every document, and `.env`
  holds your keys. Both are gitignored. Keep them that way.
- Your document text goes to whichever model endpoint you configure — and to
  your embeddings provider if you set `EMBED_PROVIDER=cloud`.
- `remove_document` deletes a document's folder, including its extracted text.

---

## Troubleshooting

**It chats but never reads my file.** The model doesn't support tool calling.

**"No text could be extracted."** A scanned PDF. It needs OCR.

**A question finds nothing.** With `EMBED_PROVIDER=none`, try the document's own
wording. Or ask for more depth — "search harder" raises the number of passages.

**A small document got indexed anyway.** The context budget was already used by
other documents. Check `/api/documents`, or remove one.

**Answers cite the wrong page.** Only PDF, PPTX and XLSX have real page numbers.
Word reports sections instead, because `.docx` has no fixed pages until it is
rendered.

**Something is stuck.** `curl -X POST http://127.0.0.1:8808/api/stop`

---

## How it fits together

```
ragifier.py         server, chat loop, tool orchestration
core/extract.py     file -> text with page anchors + figure locations
core/store.py       chunking, the small-vs-large routing, persistence
core/embed.py       embedding providers + BM25 (the no-server path)
core/retrieve.py    hybrid search, answer-context assembly, figure lookup
core/tools.py       the tools the model can call
core/llm.py         backend selection, failover, timeouts
core/bridge.py      runs tools on threads, streams their output
static/index.html   the UI (single file, no build step)
```

Page anchors are the thread running through all of it. `extract` inserts
`<!-- page: N -->`, `store` carries the page per line through chunking so each
chunk resolves the page at its own offset, and `retrieve` surfaces it as a
citation. Without that last part, a PDF with no markdown headings would have
every chunk labelled page 1.

## Contributing

Issues and PRs welcome. No build step — edit, run `python ragifier.py`, reload.

Useful additions: OCR for scanned PDFs, a vision model for figures, reranking,
per-document collections, conversation memory.

## License

MIT — see [LICENSE](LICENSE).
