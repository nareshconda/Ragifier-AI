"""
Document storage, chunking, and the small-vs-large routing decision.

THE ROUTING RULE
----------------
A document is handled one of two ways, decided by its size:

  CONTEXT  small enough to hand the model whole. No chunking, no embedding,
           no retrieval step -- the model sees every word, so it cannot miss
           something because a search failed. This is strictly better than RAG
           when it fits, and it is instant to ingest.

  INDEXED  too big for that. Split into overlapping chunks, embedded if an
           embedding provider is configured, and searched per question.

The threshold is a token budget (CONTEXT_TOKEN_BUDGET, default 6000 tokens
~= 24k characters, roughly 10-12 pages of prose). Several small documents can
share the budget; once the total would exceed it, later ones are indexed
instead, so uploading a fifth short file cannot quietly evict the first four.

Everything is stored under storage/, one folder per document, so you can
delete a document by deleting its folder.
"""

import json
import os
import re
import shutil
import time
import uuid

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    np = None

from . import embed, extract

STORAGE_DIR = os.path.abspath(
    os.environ.get("STORAGE_DIR")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "storage"))

# ~4 characters per token is close enough for routing decisions.
CHARS_PER_TOKEN = 4

CONTEXT_TOKEN_BUDGET = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "6000"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1100"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))


def est_tokens(text):
    return max(1, len(text or "") // CHARS_PER_TOKEN)


def _slug(name):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(name or "doc")).strip("._")
    return (s or "doc")[:60]


# ---------------------------------------------------------------------------
# Chunking - heading aware, page aware
# ---------------------------------------------------------------------------
_H = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk(text, source):
    """Split into overlapping chunks that each carry their heading trail and
    the page they START on.

    The buffer holds (line, page) pairs rather than bare lines. That matters:
    a PDF has page anchors but usually NO markdown headings, so the whole
    document would otherwise accumulate into one buffer and every chunk would
    inherit the page of its first line -- page 1 for all of them. Tracking the
    page per line lets each hard-split piece resolve the page at its own offset,
    which is what makes citations correct for plain PDFs.
    """
    lines = text.split("\n")
    stack, buf, chunks = [], [], []
    page = 0

    def crumb():
        return " > ".join(t for _, t in stack)

    def flush():
        nonlocal buf
        if not buf:
            return
        pairs, buf = buf, []

        # body text plus, for each line, the offset where it starts
        offsets, parts, pos = [], [], 0
        for ln, pg in pairs:
            offsets.append((pos, pg))
            parts.append(ln)
            pos += len(ln) + 1              # +1 for the newline we join with
        body = "\n".join(parts)

        stripped = body.strip()
        if not stripped:
            return
        lead = len(body) - len(body.lstrip())   # keep offsets aligned after strip

        def page_at(off):
            """Page of the character at `off` within the stripped body."""
            real = off + lead
            found = pairs[0][1] if pairs else 0
            for start, pg in offsets:
                if start <= real:
                    found = pg
                else:
                    break
            return found

        head = crumb()
        start = 0
        step = max(1, CHUNK_CHARS - CHUNK_OVERLAP)
        while start < len(stripped):
            piece = stripped[start:start + CHUNK_CHARS]
            chunks.append({
                "text": (f"[Section: {head}]\n{piece}" if head else piece),
                "heading": head,
                "page": page_at(start),
                "source": source,
            })
            if start + CHUNK_CHARS >= len(stripped):
                break
            start += step

    for ln in lines:
        m = extract.PAGE_RE.search(ln)
        if m:
            page = int(m.group(1))
            rest = extract.PAGE_RE.sub("", ln).strip()
            if rest:
                buf.append((rest, page))
            continue
        h = _H.match(ln)
        if h:
            flush()
            lvl, title = len(h.group(1)), h.group(2).strip()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
            continue
        buf.append((ln, page))
    flush()

    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _doc_dir(doc_id):
    return os.path.join(STORAGE_DIR, doc_id)


def list_documents():
    docs = []
    if not os.path.isdir(STORAGE_DIR):
        return docs
    for d in sorted(os.listdir(STORAGE_DIR)):
        meta_path = os.path.join(STORAGE_DIR, d, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    docs.append(json.load(f))
            except Exception:
                continue
    docs.sort(key=lambda m: m.get("added_at", 0))
    return docs


def get_document(doc_id):
    p = os.path.join(_doc_dir(doc_id), "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve(name_or_id):
    """Find a document by id, exact title, or a distinctive part of its name."""
    if not name_or_id:
        return None
    docs = list_documents()
    key = str(name_or_id).strip().lower()
    for d in docs:
        if d["id"].lower() == key:
            return d
    for d in docs:
        if d["title"].lower() == key or d["filename"].lower() == key:
            return d
    hits = [d for d in docs
            if key in d["title"].lower() or key in d["filename"].lower()]
    return hits[0] if len(hits) == 1 else None


def context_tokens_used():
    return sum(d.get("tokens", 0) for d in list_documents()
               if d.get("mode") == "context")


def load_text(doc_id):
    p = os.path.join(_doc_dir(doc_id), "text.txt")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return f.read()


def load_chunks(doc_id):
    p = os.path.join(_doc_dir(doc_id), "chunks.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_vectors(doc_id):
    p = os.path.join(_doc_dir(doc_id), "vectors.npy")
    if np is None or not os.path.exists(p):
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def load_figures(doc_id):
    p = os.path.join(_doc_dir(doc_id), "figures.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def remove(doc_id):
    d = _doc_dir(doc_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def decide_mode(tokens, force=""):
    """-> ('context'|'indexed', reason)"""
    force = (force or "").strip().lower()
    if force in ("context", "indexed"):
        return force, f"forced to {force} mode"

    used = context_tokens_used()
    if tokens <= CONTEXT_TOKEN_BUDGET and (used + tokens) <= CONTEXT_TOKEN_BUDGET:
        return "context", (
            f"{tokens:,} tokens fits the {CONTEXT_TOKEN_BUDGET:,}-token context "
            f"budget ({used:,} already used), so the model reads all of it")
    if tokens <= CONTEXT_TOKEN_BUDGET:
        return "indexed", (
            f"{tokens:,} tokens would fit on its own, but {used:,} of the "
            f"{CONTEXT_TOKEN_BUDGET:,}-token context budget is already taken by "
            f"other documents")
    return "indexed", (
        f"{tokens:,} tokens exceeds the {CONTEXT_TOKEN_BUDGET:,}-token context "
        f"budget, so it is chunked and searched per question")


def ingest(path, title="", mode="", on_progress=None):
    """Extract, route, store. Returns the document's metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"That file is {size_mb:.0f} MB, over the {MAX_UPLOAD_MB} MB limit "
            f"(MAX_UPLOAD_MB in .env).")

    say = on_progress or (lambda *_: None)
    filename = os.path.basename(path)
    say(f"Reading {filename} ({size_mb:.1f} MB)...")

    data = extract.extract(path)
    text = data["text"]
    tokens = est_tokens(text)
    say(f"Extracted {len(text):,} characters (~{tokens:,} tokens)"
        + (f" from {data['pages']} pages" if data["pages"] else ""))

    for w in data.get("warnings", []):
        say(f"Note: {w}")

    chosen, reason = decide_mode(tokens, mode)
    say(f"Mode: {chosen.upper()} - {reason}")

    doc_id = f"{_slug(title or filename)}-{uuid.uuid4().hex[:6]}"
    ddir = _doc_dir(doc_id)
    os.makedirs(ddir, exist_ok=True)

    with open(os.path.join(ddir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    figures = data.get("figures", [])
    n_chunks, vectors_built = 0, False

    if chosen == "indexed":
        chunks = chunk(text, filename)
        n_chunks = len(chunks)
        say(f"Split into {n_chunks} overlapping chunks.")
        with open(os.path.join(ddir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        if embed.available():
            say(f"Embedding with {embed.describe()['model']}...")
            try:
                def prog(done, total):
                    say(f"  embedded {done}/{total}")
                vecs = embed.embed_texts([c["text"] for c in chunks],
                                         progress=prog)
                np.save(os.path.join(ddir, "vectors.npy"), vecs)
                vectors_built = True
                say(f"Built {vecs.shape[0]} vectors ({vecs.shape[1]} dims).")
            except Exception as e:
                # Not fatal: BM25 still works, so the document stays usable.
                say(f"Embedding failed ({e}).")
                say("Falling back to keyword search for this document - it is "
                    "still searchable, just less good at paraphrased questions.")
        else:
            say("No embedding provider configured; using keyword (BM25) search.")

    # Figures: save what we can, but the page number is the part that matters.
    if figures and data["kind"] == "pdf":
        saved = extract.extract_pdf_images(path, os.path.join(ddir, "figures"))
        by_key = {(s["page"], s["index"]): s["file"] for s in saved}
        for fig in figures:
            fig["file"] = by_key.get((fig["page"], fig["index"]), "")
        say(f"Found {len(figures)} image(s); noted their page numbers"
            + (f" and saved {len(saved)} of them." if saved else "."))
    if figures:
        with open(os.path.join(ddir, "figures.json"), "w", encoding="utf-8") as f:
            json.dump(figures, f, ensure_ascii=False)

    meta = {
        "id": doc_id,
        "title": (title or os.path.splitext(filename)[0]).strip(),
        "filename": filename,
        "kind": data["kind"],
        "mode": chosen,
        "mode_reason": reason,
        "pages": data["pages"],
        "chars": len(text),
        "tokens": tokens,
        "chunks": n_chunks,
        "vectors": vectors_built,
        "figures": len(figures),
        "warnings": data.get("warnings", []),
        "added_at": time.time(),
        "size_mb": round(size_mb, 2),
    }
    with open(os.path.join(ddir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def describe():
    docs = list_documents()
    return {
        "storage_dir": STORAGE_DIR,
        "documents": len(docs),
        "context_budget_tokens": CONTEXT_TOKEN_BUDGET,
        "context_tokens_used": context_tokens_used(),
        "chunk_chars": CHUNK_CHARS,
        "max_upload_mb": MAX_UPLOAD_MB,
        "embedding": embed.describe(),
        "docs": [{"id": d["id"], "title": d["title"], "mode": d["mode"],
                  "tokens": d["tokens"], "pages": d["pages"],
                  "chunks": d["chunks"], "vectors": d["vectors"],
                  "figures": d["figures"]} for d in docs],
    }
