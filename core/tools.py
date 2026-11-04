"""
The tools the model can call.

Deliberately few. A document assistant that works well needs: get a file in,
find the relevant parts, say where a picture is, and manage the shelf.
"""

import difflib
import inspect
import os

from . import embed, retrieve, store

try:
    from tabulate import tabulate
except ImportError:                                  # pragma: no cover
    tabulate = None

_NO_PARAMS = {"type": "object", "properties": {}}


def normalize_args(fn, args):
    """Map model-supplied argument names onto real parameter names, so a
    harmless rename (file vs file_path) is not a TypeError."""
    if not args:
        return args or {}
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return args
    params = list(sig.parameters)
    if not params:
        return {}
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return args

    def norm(s):
        return str(s).lower().replace("_", "").replace("-", "").replace(" ", "")

    pnorm = {p: norm(p) for p in params}
    out, leftover = {}, {}
    for k, v in args.items():
        (out if k in params else leftover)[k] = v
    for k in list(leftover):
        hit = next((p for p in params if p not in out and pnorm[p] == norm(k)), None)
        if hit:
            out[hit] = leftover.pop(k)
    for k in list(leftover):
        pool = [pnorm[p] for p in params if p not in out]
        cand = difflib.get_close_matches(norm(k), pool, n=1, cutoff=0.6)
        if cand:
            hit = next(p for p in params if p not in out and pnorm[p] == cand[0])
            out[hit] = leftover.pop(k)
    required = [n for n, prm in sig.parameters.items()
                if prm.default is inspect.Parameter.empty
                and prm.kind in (prm.POSITIONAL_OR_KEYWORD, prm.KEYWORD_ONLY)]
    missing = [p for p in required if p not in out]
    if len(missing) == 1 and len(leftover) == 1:
        out[missing[0]] = next(iter(leftover.values()))
    return out


def _grid(rows, headers):
    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="grid")
    return "\n".join("  ".join(str(c) for c in r) for r in [headers] + rows)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def add_document(file_path: str, title: str = "", mode: str = ""):
    """Ingest a file: extract, route by size, index if needed."""
    path = (file_path or "").strip().strip('"').strip("'")
    if not path:
        return "ERROR: no file path given."
    if not os.path.exists(path):
        print(f"-> No file at: {path}")
        print("   If you attached a file, use the path from the [system note] "
              "in the message.")
        return f"ERROR: no such file: {path}"

    print(f"-> Adding {os.path.basename(path)}")
    print()
    try:
        meta = store.ingest(path, title=title, mode=mode,
                            on_progress=lambda m: print(f"   {m}"))
    except Exception as e:
        print()
        print(f"-> Could not add that document: {e}")
        return f"ERROR: {e}"

    retrieve.clear_cache(meta["id"])
    print()
    print(_grid([["title", meta["title"]],
                 ["type", meta["kind"]],
                 ["pages", meta["pages"] or "n/a"],
                 ["size", f"{meta['chars']:,} chars (~{meta['tokens']:,} tokens)"],
                 ["mode", meta["mode"].upper()],
                 ["chunks", meta["chunks"] or "n/a"],
                 ["search", "vector + keyword" if meta["vectors"]
                            else ("keyword only" if meta["mode"] == "indexed"
                                  else "read in full")],
                 ["images", meta["figures"] or "none"]],
                ["", ""]))
    print()
    if meta["mode"] == "context":
        print("-> Added. It is small enough that I read it in full, so nothing "
              "can be missed by a search.")
    else:
        print(f"-> Added and indexed into {meta['chunks']} chunks. Ask away.")
    return (f"Added '{meta['title']}' in {meta['mode']} mode "
            f"({meta['tokens']:,} tokens, {meta['pages'] or 0} pages, "
            f"{meta['figures']} images). Now answer the user's question about "
            f"it if they asked one, using ask_documents.")


def ask_documents(question: str, document: str = "", depth: int = 0):
    """Retrieve the passages needed to answer a question, with citations."""
    q = (question or "").strip()
    if not q:
        return "ERROR: no question given."

    docs = store.list_documents()
    if not docs:
        print("-> No documents have been added yet.")
        print("   Attach a PDF, Word file, spreadsheet or text file and I will "
              "read it.")
        return "ERROR: no documents loaded."

    k = int(depth) if depth else retrieve.TOP_K
    k = max(1, min(k, 30))

    ctx = retrieve.build_answer_context(q, k=k, doc=document)
    if not ctx["blocks"]:
        print(f"-> Nothing in the loaded documents matched: {q}")
        print(f"   Loaded: {', '.join(d['title'] for d in docs)}")
        return ("No relevant passage found. Tell the user plainly that the "
                "documents do not appear to cover this - do NOT answer from "
                "your own knowledge and present it as being from the document.")

    whole = sum(1 for c in ctx["citations"] if c["page"] is None
                and c["why"] == "included in full")
    excerpts = len(ctx["citations"]) - whole
    bits = []
    if whole:
        bits.append(f"{whole} document(s) read in full")
    if excerpts:
        bits.append(f"{excerpts} excerpt(s) retrieved")
    print(f"-> {' and '.join(bits)} for: {q}")
    print()

    for b in ctx["blocks"]:
        print(b)
        print()

    pages = sorted({c["page"] for c in ctx["citations"] if c["page"]})
    if pages:
        print(f"-> Pages referenced: {', '.join(str(p) for p in pages)}")

    if retrieve.looks_like_figure_question(q):
        print()
        print("-> This looks like a question about an image. I cannot see "
              "images; call find_figures to give the user a page number.")

    return (f"Retrieved {len(ctx['citations'])} source(s) (~{ctx['tokens']:,} "
            f"tokens). Answer ONLY from the material above and cite the "
            f"document and page for each claim. If it does not contain the "
            f"answer, say so.")


def find_figures(query: str = "", document: str = ""):
    """Locate images, charts and diagrams by page number."""
    figs = retrieve.find_figures(query, doc=document)
    if not figs:
        docs = store.list_documents()
        if not docs:
            print("-> No documents loaded.")
            return "ERROR: no documents loaded."
        print("-> No images were found in the loaded document(s).")
        print("   Note: images drawn as vector graphics (many charts exported "
              "from Office) are not detected as embedded images. The text "
              "around them is still searchable.")
        return ("No images found. Tell the user, and offer to search the text "
                "for a caption or a mention instead.")

    rows = [[f["doc"], f["page"] or "n/a",
             (f["caption"] or "(no caption found)")[:60],
             "saved" if f["file"] else "-"] for f in figs]
    print(_grid(rows, ["document", "page", "caption", "extracted"]))
    print()
    print("-> I cannot see what these images contain - nothing here does "
          "vision. The page numbers are so you can go and look.")
    return (f"Found {len(figs)} image(s). Give the user the PAGE NUMBERS and "
            f"captions so they can look themselves. State plainly that you "
            f"cannot see image contents. Never guess what an image shows.")


def list_documents():
    """Show what has been loaded, and how each one is being searched."""
    docs = store.list_documents()
    if not docs:
        print("-> Nothing loaded yet. Attach a file and I will read it.")
        return "No documents loaded."

    rows = [[d["title"][:34], d["kind"], d["pages"] or "-",
             f"{d['tokens']:,}", d["mode"],
             "vector+keyword" if d["vectors"]
             else ("keyword" if d["mode"] == "indexed" else "full text"),
             d["figures"] or "-"] for d in docs]
    print(_grid(rows, ["title", "type", "pages", "tokens", "mode",
                       "search", "images"]))
    used = store.context_tokens_used()
    print()
    print(f"-> Context budget: {used:,} / {store.CONTEXT_TOKEN_BUDGET:,} tokens "
          f"used by full-text documents.")
    if not embed.available():
        print("-> Keyword search only (no embedding provider configured). "
              "Set EMBED_PROVIDER in .env for semantic search.")
    return f"{len(docs)} document(s) loaded."


def remove_document(document: str):
    """Forget a document and delete its index."""
    d = store.resolve(document)
    if not d:
        print(f"-> No document matching {document!r}.")
        names = [x["title"] for x in store.list_documents()]
        if names:
            print(f"   Loaded: {', '.join(names)}")
        return f"ERROR: no document matching {document!r}."
    store.remove(d["id"])
    retrieve.clear_cache(d["id"])
    print(f"-> Removed '{d['title']}' and deleted its index.")
    return f"Removed '{d['title']}'."


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------
TOOLS = {
    "add_document": {
        "fn": add_document,
        "declaration": {
            "name": "add_document",
            "description": (
                "Read a file and make it searchable: PDF, Word (.docx), "
                "PowerPoint, Excel, CSV, HTML, EPUB, Markdown or plain text.\n"
                "CALL THIS as soon as the user attaches a file - the saved "
                "path appears in a [system note] in their message; pass that "
                "path verbatim. Small documents are read in full; large ones "
                "are chunked and indexed. You do not choose - the tool decides "
                "by size and tells you which happened.\n"
                "After it succeeds, if the user asked a question in the same "
                "message, call ask_documents to answer it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string",
                                  "description": "Path from the [system note]."},
                    "title": {"type": "string",
                              "description": "Optional friendly name."},
                    "mode": {"type": "string",
                             "description": ("Leave empty. Only set to "
                                             "'context' or 'indexed' if the "
                                             "user explicitly asks to override "
                                             "the automatic choice.")},
                },
                "required": ["file_path"],
            },
        },
    },
    "ask_documents": {
        "fn": ask_documents,
        "declaration": {
            "name": "ask_documents",
            "description": (
                "Find the passages needed to answer a question about the "
                "loaded documents. USE THIS FOR EVERY question about document "
                "content - never answer from memory.\n"
                "Pass the user's question as `question`, rephrased into a "
                "retrieval-friendly form if that helps. The passages are "
                "printed above your reply; answer ONLY from them and cite the "
                "document and page for each claim. If they do not contain the "
                "answer, say so plainly rather than filling the gap from your "
                "own knowledge.\n"
                "Raise `depth` (up to 30) for broad questions like 'summarise "
                "this' or when a first attempt retrieved too little."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "What to look for."},
                    "document": {"type": "string",
                                 "description": ("Optional: restrict to one "
                                                 "document by title or id.")},
                    "depth": {"type": "integer",
                              "description": "Passages to retrieve (default 8)."},
                },
                "required": ["question"],
            },
        },
    },
    "find_figures": {
        "fn": find_figures,
        "declaration": {
            "name": "find_figures",
            "description": (
                "Find images, charts, diagrams, screenshots or photos in the "
                "loaded documents and report WHICH PAGE each is on.\n"
                "USE THIS whenever the user asks about anything visual - "
                "'what does the diagram show', 'where is the architecture "
                "chart', 'show me figure 3'.\n"
                "IMPORTANT: this does not read image contents - there is no "
                "vision here. Report the page numbers and any caption, and say "
                "plainly that the user needs to look at that page. NEVER "
                "describe or guess what an image contains."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What the image is about, if known."},
                    "document": {"type": "string",
                                 "description": "Optional: one document."},
                },
            },
        },
    },
    "list_documents": {
        "fn": list_documents,
        "declaration": {
            "name": "list_documents",
            "description": ("Show which documents are loaded, how big each is, "
                            "and whether it is read in full or searched."),
            "parameters": _NO_PARAMS,
        },
    },
    "remove_document": {
        "fn": remove_document,
        "declaration": {
            "name": "remove_document",
            "description": ("Forget a document and delete its index. Only call "
                            "this when the user asks."),
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {"type": "string",
                                 "description": "Title or id."},
                },
                "required": ["document"],
            },
        },
    },
}

FUNCTION_DECLARATIONS = [t["declaration"] for t in TOOLS.values()]
OPENAI_TOOLS = [{"type": "function", "function": d}
                for d in FUNCTION_DECLARATIONS]


def cancel_all():
    return []
