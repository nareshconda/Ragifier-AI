"""
RAGifier AI - chat server.

Upload a PDF, Word file, spreadsheet or any large document and ask questions
about it. Small documents are read in full; large ones are chunked, indexed and
searched. Answers cite the document and page.

    python app.py          then open http://127.0.0.1:8808

Everything configurable lives in .env (copy .env.example).
"""

import asyncio
import json
import os
import queue
import sys
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env from NEXT TO THIS FILE, explicitly. Bare load_dotenv() walks up from
# the caller's frame, which fails when there is no frame (python -c, some
# embedding hosts) and silently finds nothing when the app is started from
# another directory.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from core import bridge, embed, llm, render, retrieve, store, tools  # noqa: E402

APP_NAME = os.environ.get("APP_NAME", "RAGifier AI").strip()

# Loopback by default. The websocket is plain ws://, not wss://, and your
# documents' contents travel over it. On 0.0.0.0 that crosses your network in
# the clear, and anyone who can reach the port can read everything you have
# uploaded. Only change this if you have put TLS in front of it.
HOST = os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.environ.get("PORT", "8808"))

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
UPLOAD_DIR = os.path.join(HERE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_OUTPUT_FOR_MODEL = int(os.environ.get("MAX_OUTPUT_FOR_MODEL", "14000"))
TOOL_BUDGET = int(os.environ.get("TOOL_BUDGET", "8"))
ANSWER_TIMEOUT = bridge.ANSWER_TIMEOUT

bridge.install()


# ---------------------------------------------------------------------------
# System prompt - built from what is actually configured
# ---------------------------------------------------------------------------
def build_system_prompt():
    docs = store.list_documents()
    lines = [
        f"You are {APP_NAME}, an assistant that answers questions about the "
        f"user's own documents.",
        "",
        "Rules:",
        "- When the user attaches a file, call add_document with the path from "
        "the [system note] in their message. Do it immediately, without asking.",
        "- For ANY question about document content, call ask_documents. Never "
        "answer from your own knowledge and present it as coming from the "
        "document.",
        "- Answer ONLY from the retrieved passages, and CITE the document and "
        "page for each claim, like (Handbook, p12). If the passages do not "
        "contain the answer, say so plainly and suggest what might help - a "
        "rephrased question, a higher depth, or the right document.",
        "- Never invent a page number. Use the ones in the retrieved material.",
        "- You CANNOT see images. For any question about a figure, diagram, "
        "chart, screenshot or photo, call find_figures and give the user the "
        "PAGE NUMBER so they can look. Say plainly that you cannot see the "
        "image. Never describe or guess its contents, even if the surrounding "
        "text hints at it - you may quote the caption and the text near it, "
        "clearly labelled as such.",
        "- A tool's output is already shown to the user above your reply. Do "
        "not repeat passages verbatim at length; answer the question and quote "
        "only what you need.",
        "- Keep replies tight. Light markdown is fine.",
        "- If the user asks something unrelated to the documents, just answer "
        "normally and say you are answering from general knowledge.",
    ]

    if docs:
        lines += ["", "Documents currently loaded:"]
        for d in docs[:40]:
            how = ("read in full" if d["mode"] == "context"
                   else f"indexed, {d['chunks']} chunks")
            pages = f", {d['pages']} pages" if d["pages"] else ""
            figs = f", {d['figures']} images" if d["figures"] else ""
            lines.append(f"  - '{d['title']}' ({d['kind']}{pages}{figs}) - {how}")
    else:
        lines += ["", "No documents are loaded yet. Invite the user to attach "
                      "one - PDF, Word, PowerPoint, Excel, CSV, HTML, EPUB, "
                      "Markdown or text."]

    if not embed.available():
        lines += ["", "Note: semantic search is not configured, so retrieval "
                      "is keyword-based. If a question finds nothing, suggest "
                      "the user try the document's own wording."]

    extra = os.environ.get("SYSTEM_PROMPT_EXTRA", "").strip()
    if extra:
        lines += ["", extra]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
def _is_cancel(msg):
    """Accept every spelling of Stop. A Stop the server does not recognise is
    indistinguishable from one that was never pressed."""
    if not isinstance(msg, dict):
        return False
    return str(msg.get("type", "")).strip().lower() in (
        "cancel", "stop", "abort", "interrupt")


LIVE_QUEUES = set()
WS_STATE = {"connections": 0, "messages_in": 0, "last_message": None,
            "last_message_at": None, "answers_delivered": 0}


def stop_everything():
    """Abort everything running. Safe from an HTTP request too."""
    try:
        tools.cancel_all()
    except Exception:
        pass
    stopped = bridge.abandon_all()
    for q in list(LIVE_QUEUES):
        try:
            q.put_nowait({"type": "cancel"})
        except Exception:
            pass
    return stopped


# ---------------------------------------------------------------------------
# Tool execution over the websocket
# ---------------------------------------------------------------------------
async def run_tool(name, args, ws, user_q):
    """Execute a tool, streaming output and questions to the browser.

    Returns (output_for_model, result, error_or_None).
    """
    spec = tools.TOOLS.get(name)
    if spec is None:
        return "", None, f"Unknown tool: {name}"

    await ws.send_json({"type": "tool_start", "name": name})
    ex = bridge.ToolExecution(
        spec["fn"], tools.normalize_args(spec["fn"], dict(args or {})),
        name=name)
    ex.start()

    raw_parts, line_buf, table_block, in_table = [], "", [], False

    async def flush_table():
        nonlocal in_table, table_block
        if not in_table:
            return
        parsed = render.parse_grid_table(table_block)
        if parsed:
            await ws.send_json({"type": "tool_table",
                                "headers": parsed["headers"],
                                "rows": parsed["rows"]})
        table_block, in_table = [], False

    async def process_line(line):
        nonlocal in_table
        await ws.send_json({"type": "tool_output", "text": line + "\n"})
        raw_parts.append(line)
        if render.BORDER_RE.match(line) or render.ROW_RE.match(line):
            in_table = True
            table_block.append(line)
        else:
            await flush_table()

    async def feed(chunk):
        nonlocal line_buf
        line_buf += chunk
        while "\n" in line_buf:
            line, line_buf = line_buf.split("\n", 1)
            await process_line(line)

    async def flush_all():
        nonlocal line_buf
        if line_buf:
            await process_line(line_buf)
            line_buf = ""
        await flush_table()

    async def check_cancel():
        try:
            msg = user_q.get_nowait()
        except asyncio.QueueEmpty:
            return False
        if _is_cancel(msg):
            return True
        await user_q.put(msg)
        return False

    async def do_cancel():
        try:
            tools.cancel_all()
        except Exception:
            pass
        ex.abandon()
        await flush_all()
        note = "[Stopped by user]"
        raw_parts.append(note)
        await ws.send_json({"type": "tool_output", "text": note + "\n"})
        await ws.send_json({"type": "tool_end", "name": name, "ok": False})
        return (render.prepare_for_model("\n".join(raw_parts)), None,
                "Cancelled by the user. Do not retry. Summarise what you have "
                "and ask how to proceed.")

    while True:
        if await check_cancel():
            return await do_cancel()
        try:
            kind, payload = ex.out_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if kind == "stdout":
            await feed(payload)

        elif kind == "question":
            await flush_all()
            # A prompt marked [secret] asks for a credential. Strip the marker
            # for display and record the ANSWER as **** so it never reaches
            # the model.
            secret = payload.startswith(bridge.SECRET_PREFIX)
            shown = payload[len(bridge.SECRET_PREFIX):] if secret else payload
            raw_parts.append(shown)
            await ws.send_json({"type": "tool_question", "text": shown,
                                "secret": secret})
            try:
                msg = await asyncio.wait_for(user_q.get(),
                                             timeout=ANSWER_TIMEOUT)
            except asyncio.TimeoutError:
                ex.abandon()
                note = (f"[No answer within {ANSWER_TIMEOUT:.0f}s - giving up. "
                        f"Nothing was run.]")
                raw_parts.append(note)
                await ws.send_json({"type": "tool_output", "text": note + "\n"})
                await ws.send_json({"type": "tool_end", "name": name,
                                    "ok": False})
                return (render.prepare_for_model("\n".join(raw_parts)), None,
                        "The user never answered the tool's prompt. Tell them "
                        "it timed out and ask whether to retry.")
            if _is_cancel(msg):
                return await do_cancel()
            answer = (msg.get("text") or "").strip()
            raw_parts.append("********" if secret else answer)
            WS_STATE["answers_delivered"] += 1
            ex.answer_q.put(answer)
            await ws.send_json({"type": "answer_ack", "secret": secret,
                                "chars": len(answer)})

        elif kind == "done":
            await flush_all()
            text = str(payload or "")
            ok = not text.startswith(("ERROR:", "STOPPED"))
            await ws.send_json({"type": "tool_end", "name": name, "ok": ok})
            return (render.prepare_for_model("\n".join(raw_parts)), payload,
                    None if ok else text)

        elif kind == "error":
            await flush_all()
            raw_parts.append(payload)
            await ws.send_json({"type": "tool_output", "text": payload})
            await ws.send_json({"type": "tool_end", "name": name, "ok": False})
            return (render.prepare_for_model("\n".join(raw_parts)), None,
                    payload)


def _truncate_for_model(text):
    if len(text) <= MAX_OUTPUT_FOR_MODEL:
        return text
    head, tail = text[:2000], text[-(MAX_OUTPUT_FOR_MODEL - 2000):]
    return (head + "\n...[output truncated for the model; the user saw all of "
                   "it]...\n" + tail)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/config")
async def config():
    primary = llm.BACKENDS[0] if llm.BACKENDS else None
    return JSONResponse({
        "app_name": APP_NAME,
        "provider": llm.PROVIDER,
        "model": primary.model if primary else None,
        "backend": primary.label if primary else None,
        "fallback": llm.BACKENDS[1].model if len(llm.BACKENDS) > 1 else None,
        "documents": len(store.list_documents()),
        "embedding": embed.describe()["provider"],
        "vectors": embed.available(),
    })


@app.get("/api/debug")
async def api_debug():
    """Live backend state: what is running and what it is waiting for.

    Contains no credentials -- prompts are shown, answers never are.
    If a tool looks stuck, compare ws.answers_delivered against
    executions[-1].pending_question to tell a browser problem from a backend one.
    """
    return JSONResponse({
        "now": time.time(),
        "app": {"name": APP_NAME, "pid": os.getpid(), "cwd": os.getcwd(),
                "host": HOST, "port": PORT,
                "answer_timeout_s": ANSWER_TIMEOUT},
        "llm": llm.describe(),
        "store": store.describe(),
        "retrieval": {"top_k": retrieve.TOP_K,
                      "answer_context_tokens": retrieve.MAX_CONTEXT_TOKENS},
        "ws": dict(WS_STATE),
        "executions": [e.snapshot() for e in bridge.EXECUTIONS],
        "live_executions": len(bridge.live_executions()),
    })


@app.post("/api/stop")
@app.get("/api/stop")
async def api_stop():
    """Abort everything, without needing the websocket.

    The Stop button goes over the websocket; if that path breaks the button
    looks dead. This does not depend on it:
        curl -X POST http://127.0.0.1:8800/api/stop
    """
    stopped = stop_everything()
    return JSONResponse({"stopped": stopped, "count": len(stopped)})


@app.get("/api/documents")
async def api_documents():
    """What is loaded, how each one is searched, and the context budget."""
    return JSONResponse(store.describe())


@app.get("/api/figure/{doc_id}/{filename}")
async def api_figure(doc_id: str, filename: str):
    """Serve an image extracted from a PDF.

    Path-confined on purpose: doc_id and filename are taken from a URL, so the
    resolved path is checked to be inside that document's figures folder before
    anything is served.
    """
    base = os.path.realpath(os.path.join(store.STORAGE_DIR, doc_id, "figures"))
    target = os.path.realpath(os.path.join(base, os.path.basename(filename)))
    if not target.startswith(base + os.sep) or not os.path.isfile(target):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Save an uploaded file, streaming it to disk.

    This app exists for large documents, so the whole file must never be held
    in memory: a 200 MB PDF read with await file.read() is 200 MB of RSS per
    upload. Copy it through in chunks and enforce the size limit as we go,
    rather than after we have already buffered it all.
    """
    safe = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename or 'upload')}"
    dest = os.path.join(UPLOAD_DIR, safe)
    limit = store.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    out.close()
                    os.remove(dest)
                    return JSONResponse(
                        {"error": f"File is larger than the "
                                  f"{store.MAX_UPLOAD_MB} MB limit "
                                  f"(MAX_UPLOAD_MB in .env)."},
                        status_code=413)
                out.write(chunk)
    except Exception as e:
        try:
            os.remove(dest)
        except Exception:
            pass
        return JSONResponse({"error": f"Upload failed: {e}"}, status_code=500)

    return {"path": dest, "name": file.filename,
            "size_mb": round(written / (1024 * 1024), 2)}


# ---------------------------------------------------------------------------
# Websocket chat
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    if not llm.BACKENDS:
        await ws.send_json({"type": "error", "text":
            "No LLM backend configured. Copy .env.example to .env and fill in "
            "either the CLOUD_* block (and set LLM_PROVIDER=cloud) or the "
            "LOCAL_* block (LLM_PROVIDER=local)."})
        await ws.close()
        return

    messages = [{"role": "system", "content": build_system_prompt()}]
    user_q = asyncio.Queue()
    LIVE_QUEUES.add(user_q)
    WS_STATE["connections"] += 1

    async def receiver():
        while True:
            data = await ws.receive_json()
            WS_STATE["messages_in"] += 1
            # type and size only -- never the text, which may be a credential
            WS_STATE["last_message"] = {
                "type": str(data.get("type", "?")),
                "chars": len(data.get("text") or "")}
            WS_STATE["last_message_at"] = time.time()
            await user_q.put(data)

    recv_task = asyncio.create_task(receiver())

    async def notify(text):
        await ws.send_json({"type": "tool_output", "text": text + "\n"})

    try:
        while True:
            msg = await user_q.get()

            if _is_cancel(msg):
                killed = stop_everything()
                if killed:
                    await notify(f"[Stopped {len(killed)} leftover job(s)]")
                continue

            if msg.get("type") == "answer":
                # An answer that arrived with no tool waiting for it. DROP it.
                # It must never become a chat message: these carry credentials.
                continue

            text = (msg.get("text") or "").strip()
            if not text:
                continue

            messages.append({"role": "user", "content": text})
            tools_used = 0

            while True:
                await ws.send_json({"type": "thinking"})
                try:
                    resp = await llm.call(messages, tools.OPENAI_TOOLS,
                                          on_failover=notify)
                except llm.AllBackendsFailed as e:
                    note = (f"The AI model didn't respond.\n\n{e}\n\nNothing "
                            f"was run. Please wait a few minutes and try "
                            f"again.")
                    await ws.send_json({"type": "error", "text": note})
                    # Also send a normal assistant turn: the spinner clears on
                    # this in every code path, so a backend failure can never
                    # leave the UI stuck on "thinking".
                    await ws.send_json({"type": "assistant", "text": note})
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()
                    break
                except Exception as e:
                    note = f"LLM backend error: {e}\n\nNothing was run."
                    await ws.send_json({"type": "error", "text": note})
                    await ws.send_json({"type": "assistant", "text": note})
                    break

                choice = resp.choices[0].message
                tool_calls = choice.tool_calls or []

                if not tool_calls:
                    reply = render.strip_markdown_tables(
                        render.strip_think(choice.content or "")) or "(no response)"
                    messages.append({"role": "assistant", "content": reply})
                    await ws.send_json({"type": "assistant", "text": reply})
                    break

                messages.append({
                    "role": "assistant",
                    "content": render.strip_think(choice.content) or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tool_calls],
                })

                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tools_used += 1
                    if tools_used > TOOL_BUDGET:
                        output, result, error = "", None, (
                            f"TOOL BUDGET EXCEEDED ({TOOL_BUDGET} calls this "
                            "message). Do not call more tools. Summarise what "
                            "you found and what failed, and ask how to proceed.")
                    else:
                        output, result, error = await run_tool(
                            tc.function.name, args, ws, user_q)
                    payload = {"output": _truncate_for_model(output),
                               "result": str(result) if result is not None else ""}
                    if error:
                        payload["error"] = error[-3000:]
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(payload)})

    except WebSocketDisconnect:
        pass
    finally:
        WS_STATE["connections"] = max(0, WS_STATE["connections"] - 1)
        LIVE_QUEUES.discard(user_q)
        recv_task.cancel()
        # A dropped socket must not leave a tool parked on an answer forever.
        for e in bridge.EXECUTIONS:
            if e.alive and e.pending_question:
                e.abandon()


def _banner():
    primary = llm.BACKENDS[0] if llm.BACKENDS else None
    print(f"\n  {APP_NAME}")
    print(f"  -> http://{HOST}:{PORT}")
    if primary:
        extra = (f"  (fallback: {llm.BACKENDS[1].model})"
                 if len(llm.BACKENDS) > 1 else "  (no fallback configured)")
        print(f"  provider: {llm.PROVIDER}  |  model: {primary.model}{extra}")
    else:
        print("  WARNING: no LLM backend configured -- see .env.example")
    docs = store.list_documents()
    e = embed.describe()
    print(f"  documents: {len(docs)} in {store.STORAGE_DIR}")
    print(f"  search:    " + ("vector (" + str(e["model"]) + ") + keyword"
                              if e["vectors_available"] else "keyword only "
                              "(set EMBED_PROVIDER for semantic search)"))
    print()


if __name__ == "__main__":
    import uvicorn
    _banner()
    if os.environ.get("OPEN_BROWSER", "true").strip().lower() in ("1", "true", "yes"):
        try:
            import webbrowser
            webbrowser.open(f"http://{'127.0.0.1' if HOST in ('0.0.0.0','') else HOST}:{PORT}")
        except Exception:
            pass
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
