"""
Embeddings, with a path that needs no embedding server at all.

    EMBED_PROVIDER=local    Ollama (or any /api/embeddings endpoint)
    EMBED_PROVIDER=cloud    any OpenAI-compatible /v1/embeddings endpoint
    EMBED_PROVIDER=none     lexical search only

Why 'none' exists: requiring someone to install Ollama and pull a model before
they can ask a question about a PDF is a hard stop for most people trying this
out. BM25 is implemented here in pure Python, so RAGifier works immediately,
and vectors become an upgrade rather than a prerequisite.

When vectors ARE available we use both and fuse the rankings, which retrieves
better than either alone: embeddings catch paraphrase ("how do I reset it" vs
"restore factory defaults"), BM25 catches exact tokens embeddings blur over
(error codes, part numbers, surnames).
"""

import math
import os
import re
import time
from collections import Counter

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    np = None

import httpx

PROVIDER = (os.environ.get("EMBED_PROVIDER", "none").strip().lower() or "none")

LOCAL_URL = os.environ.get("EMBED_LOCAL_URL",
                           "http://localhost:11434").rstrip("/")
LOCAL_MODEL = os.environ.get("EMBED_LOCAL_MODEL", "nomic-embed-text").strip()

CLOUD_URL = os.environ.get("EMBED_CLOUD_URL",
                           "https://api.openai.com/v1").rstrip("/")
CLOUD_KEY = os.environ.get("EMBED_CLOUD_API_KEY", "").strip()
CLOUD_MODEL = os.environ.get("EMBED_CLOUD_MODEL", "text-embedding-3-small").strip()

TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "120"))
BATCH = int(os.environ.get("EMBED_BATCH", "64"))

STATE = {"last_error": None, "embedded": 0, "last_at": None}


def available():
    """Can we produce vectors right now?"""
    if np is None or PROVIDER == "none":
        return False
    if PROVIDER == "cloud":
        return bool(CLOUD_KEY and CLOUD_MODEL)
    return bool(LOCAL_MODEL)


def describe():
    return {"provider": PROVIDER, "vectors_available": available(),
            "model": (CLOUD_MODEL if PROVIDER == "cloud" else LOCAL_MODEL)
                     if PROVIDER != "none" else None,
            "endpoint": (CLOUD_URL if PROVIDER == "cloud" else LOCAL_URL)
                        if PROVIDER != "none" else None,
            "numpy": np is not None,
            "embedded_total": STATE["embedded"],
            "last_error": STATE["last_error"]}


def check():
    """Try one embedding so configuration problems surface at ingest time,
    not silently as bad search results later."""
    if not available():
        return False, ("no embedding provider configured - using lexical "
                       "search only")
    try:
        v = embed_texts(["probe"])
        return True, f"{len(v[0])}-dimensional vectors"
    except Exception as e:
        STATE["last_error"] = str(e)
        return False, str(e)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def _embed_local(texts):
    out = []
    with httpx.Client(timeout=TIMEOUT) as client:
        for t in texts:
            # /api/embeddings is Ollama's native endpoint and takes one prompt.
            r = client.post(f"{LOCAL_URL}/api/embeddings",
                            json={"model": LOCAL_MODEL, "prompt": t})
            if r.status_code == 404:
                raise RuntimeError(
                    f"{LOCAL_URL}/api/embeddings returned 404. Is Ollama "
                    f"running, and have you pulled the model? "
                    f"ollama pull {LOCAL_MODEL}")
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding") or (data.get("data") or [{}])[0].get("embedding")
            if not vec:
                raise RuntimeError(f"No embedding in the response: {str(data)[:200]}")
            out.append(vec)
    return out


def _embed_cloud(texts):
    out = []
    headers = {"Authorization": f"Bearer {CLOUD_KEY}"}
    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            r = client.post(f"{CLOUD_URL}/embeddings", headers=headers,
                            json={"model": CLOUD_MODEL, "input": batch})
            if r.status_code == 401:
                raise RuntimeError("Embedding endpoint rejected the API key "
                                   "(401). Check EMBED_CLOUD_API_KEY.")
            r.raise_for_status()
            data = r.json().get("data", [])
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embeddings, got {len(data)}.")
            out.extend(d["embedding"] for d in sorted(
                data, key=lambda d: d.get("index", 0)))
    return out


def embed_texts(texts, progress=None):
    """-> list of L2-normalised vectors. Raises if the provider fails."""
    if not available():
        raise RuntimeError("No embedding provider configured.")
    texts = [t if t.strip() else " " for t in texts]
    vecs = []
    step = max(1, len(texts) // 10)
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs.extend(_embed_cloud(batch) if PROVIDER == "cloud"
                    else _embed_local(batch))
        if progress and (i // BATCH) % max(1, step // BATCH or 1) == 0:
            progress(min(i + len(batch), len(texts)), len(texts))
    arr = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms                  # so a dot product IS cosine similarity
    STATE["embedded"] += len(texts)
    STATE["last_at"] = time.time()
    STATE["last_error"] = None
    return arr


# ---------------------------------------------------------------------------
# BM25 - the no-server path
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9_]+")

# Deliberately short: in technical documents, words like "not", "no" and "all"
# change the meaning of a sentence, so we keep almost everything.
_STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
         "on", "at", "by", "it", "this", "that", "as", "be", "was", "were",
         "with", "from"}


def tokenize(text):
    out = []
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or len(w) > 40:
            continue
        out.append(w)
        # split identifiers so "error_code_42" also matches "error" and "42"
        if "_" in w:
            out.extend(p for p in w.split("_") if p and p not in _STOP)
    return out


class BM25:
    """Standard Okapi BM25 over tokenised chunks."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.lens = [len(d) for d in docs_tokens]
        self.avg = (sum(self.lens) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

    def scores(self, query):
        q = tokenize(query)
        out = [0.0] * self.n
        if not q or not self.n:
            return out
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if not f:
                    continue
                dl = self.lens[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1))
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out

    def to_json(self):
        return {"docs": self.docs, "k1": self.k1, "b": self.b}

    @classmethod
    def from_json(cls, data):
        return cls(data["docs"], data.get("k1", 1.5), data.get("b", 0.75))
