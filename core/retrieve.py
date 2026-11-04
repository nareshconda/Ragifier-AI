"""
Search across every ingested document and assemble an answer context.

Two kinds of material come back:

  * CONTEXT documents are included whole. They were small enough to fit, so
    there is nothing to retrieve -- the model reads all of them. Retrieval can
    miss; reading cannot.

  * INDEXED documents are searched. Vector similarity and BM25 are run
    separately and fused, because they fail differently: embeddings handle
    paraphrase, BM25 handles exact tokens like error codes and part numbers.

Everything carries its document, page and heading, so every answer can be
cited and checked.
"""

import os

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    np = None

from . import embed, extract, store

TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "8"))
MAX_CONTEXT_TOKENS = int(os.environ.get("ANSWER_CONTEXT_TOKENS", "10000"))

# Rank-fusion constant (standard Reciprocal Rank Fusion). Fusing RANKS rather
# than scores avoids having to normalise two scales that are not comparable.
RRF_K = 60

_BM25_CACHE = {}


def _bm25_for(doc_id, chunks):
    entry = _BM25_CACHE.get(doc_id)
    if entry and entry["n"] == len(chunks):
        return entry["bm25"]
    bm = embed.BM25([embed.tokenize(c["text"]) for c in chunks])
    _BM25_CACHE[doc_id] = {"bm25": bm, "n": len(chunks)}
    return bm


def clear_cache(doc_id=None):
    if doc_id:
        _BM25_CACHE.pop(doc_id, None)
    else:
        _BM25_CACHE.clear()


def _rank(scores):
    """-> {index: rank} with rank 0 the best, ignoring zero scores."""
    pairs = [(i, s) for i, s in enumerate(scores) if s > 0]
    pairs.sort(key=lambda p: -p[1])
    return {i: r for r, (i, _) in enumerate(pairs)}


def search(query, k=TOP_K, doc=None):
    """Hybrid search over indexed documents. -> [hit, ...] best first."""
    docs = store.list_documents()
    if doc:
        d = store.resolve(doc)
        docs = [d] if d else []
    docs = [d for d in docs if d.get("mode") == "indexed"]
    if not docs:
        return []

    qvec = None
    if embed.available():
        try:
            qvec = embed.embed_texts([query])[0]
        except Exception:
            qvec = None          # fall back to BM25 silently; it still works

    hits = []
    for meta in docs:
        chunks = store.load_chunks(meta["id"])
        if not chunks:
            continue

        vec_scores = None
        if qvec is not None and meta.get("vectors"):
            vecs = store.load_vectors(meta["id"])
            if vecs is not None and vecs.shape[0] == len(chunks):
                vec_scores = (vecs @ qvec).tolist()

        bm_scores = _bm25_for(meta["id"], chunks).scores(query)

        vec_rank = _rank(vec_scores) if vec_scores else {}
        bm_rank = _rank(bm_scores)

        for i, c in enumerate(chunks):
            if i not in vec_rank and i not in bm_rank:
                continue
            score = 0.0
            if i in vec_rank:
                score += 1.0 / (RRF_K + vec_rank[i])
            if i in bm_rank:
                score += 1.0 / (RRF_K + bm_rank[i])
            hits.append({
                "doc_id": meta["id"],
                "doc_title": meta["title"],
                "page": c.get("page", 0),
                "heading": c.get("heading", ""),
                "text": c["text"],
                "score": score,
                "vector_score": (vec_scores[i] if vec_scores else None),
                "bm25_score": bm_scores[i],
                "matched_by": ("both" if i in vec_rank and i in bm_rank
                               else "vector" if i in vec_rank else "keyword"),
            })

    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def context_documents():
    """Documents small enough to be read whole."""
    out = []
    for meta in store.list_documents():
        if meta.get("mode") != "context":
            continue
        text = store.load_text(meta["id"])
        if text.strip():
            out.append({"meta": meta, "text": text})
    return out


def build_answer_context(query, k=TOP_K, doc=None):
    """Everything the model should read before answering.

    -> {"blocks": [...], "citations": [...], "tokens": n, "used_search": bool}
    """
    blocks, citations = [], []
    budget = MAX_CONTEXT_TOKENS

    # 1. Whole small documents first: perfect recall, no retrieval risk.
    for item in context_documents():
        meta = item["meta"]
        if doc:
            want = store.resolve(doc)
            if not want or want["id"] != meta["id"]:
                continue
        # Turn the machine-readable anchors into something the model can quote
        # as a citation, rather than leaving raw HTML comments in the text.
        text = extract.PAGE_RE.sub(lambda m: f"\n[page {m.group(1)}]", item["text"])
        cost = store.est_tokens(text)
        if cost > budget:
            text = text[:budget * store.CHARS_PER_TOKEN]
            cost = budget
        if cost <= 0:
            break
        blocks.append(
            f"===== FULL DOCUMENT: {meta['title']} "
            f"({meta['filename']}) =====\n{text}")
        citations.append({"doc": meta["title"], "doc_id": meta["id"],
                          "page": None, "why": "included in full"})
        budget -= cost

    # 2. Then the best chunks from the large ones.
    hits = search(query, k=k, doc=doc) if budget > 500 else []
    for h in hits:
        cost = store.est_tokens(h["text"])
        if cost > budget:
            break
        where = f"page {h['page']}" if h["page"] else "no page number"
        head = f" | {h['heading']}" if h["heading"] else ""
        blocks.append(
            f"===== EXCERPT from {h['doc_title']} ({where}{head}) "
            f"| relevance {h['score']:.4f} | matched by {h['matched_by']} "
            f"=====\n{extract.strip_anchors(h['text'])}")
        citations.append({"doc": h["doc_title"], "doc_id": h["doc_id"],
                          "page": h["page"] or None, "why": h["matched_by"]})
        budget -= cost

    return {"blocks": blocks, "citations": citations,
            "tokens": MAX_CONTEXT_TOKENS - budget,
            "used_search": bool(hits)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
_FIGURE_WORDS = ("figure", "fig", "image", "picture", "photo", "diagram",
                 "chart", "graph", "screenshot", "illustration", "drawing",
                 "table", "exhibit", "plot", "schematic", "map")


def looks_like_figure_question(q):
    ql = (q or "").lower()
    return any(w in ql for w in _FIGURE_WORDS)


def find_figures(query="", doc=None, limit=25):
    """Locate images by page.

    We cannot read what an image shows -- nothing here does vision. What we can
    do is say exactly which page it is on, which is what someone asking
    "what does the architecture diagram show?" actually needs in order to go
    and look at it.
    """
    docs = store.list_documents()
    if doc:
        d = store.resolve(doc)
        docs = [d] if d else []

    q_tokens = set(embed.tokenize(query)) if query else set()
    results = []
    for meta in docs:
        figs = store.load_figures(meta["id"])
        if not figs:
            continue
        for f in figs:
            cap = f.get("caption", "") or ""
            score = 0.0
            if q_tokens:
                cap_tokens = set(embed.tokenize(cap))
                overlap = q_tokens & cap_tokens
                score = len(overlap) / (len(q_tokens) or 1)
            results.append({
                "doc": meta["title"], "doc_id": meta["id"],
                "page": f.get("page", 0), "caption": cap,
                "file": f.get("file", ""), "score": score,
                "pages_total": meta.get("pages", 0),
            })
    results.sort(key=lambda r: (-r["score"], r["doc"], r["page"]))
    return results[:limit]
