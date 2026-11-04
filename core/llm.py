"""
LLM backends.

ONE variable decides everything:

    LLM_PROVIDER=cloud     ->  CLOUD_* is the primary, LOCAL_* is the fallback
    LLM_PROVIDER=local     ->  LOCAL_* is the primary, CLOUD_* is the fallback

Both blocks live in .env side by side, so switching between your own Ollama box
and a hosted API is a one-word edit. Whichever is not primary becomes the
automatic fallback, used only when the primary times out or errors. Configure
just one of them and you simply have no fallback.

Any OpenAI-compatible endpoint works: OpenAI, Azure OpenAI, NVIDIA NIM, Groq,
Together, OpenRouter, vLLM, LM Studio, Ollama.

TIMEOUTS -- three layers, and all three matter:
  REQUEST_TIMEOUT   the HTTP read timeout. The worker thread dies here.
  MAX_RETRIES       SDK-level retries. Defaults to 0: the OpenAI SDK retries
                    twice by default, which turns a 60s timeout into three
                    minutes of silence before anything is reported.
  HARD_TIMEOUT      an asyncio cap on the whole call. This is the one that
                    actually saves you. run_in_executor CANNOT be cancelled, so
                    if the worker thread wedges, the await never returns and no
                    exception is ever raised -- the UI would sit on "thinking"
                    forever. wait_for abandons the await regardless.
"""

import asyncio
import concurrent.futures
import json
import os
import time

import httpx
from openai import OpenAI


def _envf(name, default):
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else float(default)
    except ValueError:
        return float(default)


def _envj(name, default=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"WARNING: {name} is not valid JSON -- ignoring it.")
        return default


REQUEST_TIMEOUT = _envf("LLM_REQUEST_TIMEOUT", 60)
HARD_TIMEOUT = _envf("LLM_HARD_TIMEOUT", REQUEST_TIMEOUT + 15)
COOLDOWN = _envf("LLM_COOLDOWN", 120)
try:
    MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "0").strip() or 0)
except ValueError:
    MAX_RETRIES = 0

INSECURE = os.environ.get("LLM_INSECURE", "").strip().lower() in ("1", "true", "yes")
CA_BUNDLE = None            # set by app.py if a custom bundle was built


class AllBackendsFailed(Exception):
    """Every configured endpoint timed out or errored."""


class Backend:
    def __init__(self, tag, base_url, api_key, model, label="",
                 temperature=None, extra_body=None):
        self.tag = tag
        self.base_url = (base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.label = (label or "").strip() or self.base_url
        self.temperature = temperature
        self.extra_body = extra_body
        self._client = None
        self.calls = 0
        self.failures = 0
        self.last_error = None
        self.last_ok_at = None
        self.last_fail_at = None

    @property
    def configured(self):
        return bool(self.base_url and self.api_key and self.model)

    def client(self):
        if self._client is None:
            verify = False if INSECURE else (CA_BUNDLE or True)
            http_client = httpx.Client(
                verify=verify,
                timeout=httpx.Timeout(REQUEST_TIMEOUT,
                                      connect=min(15.0, REQUEST_TIMEOUT)),
            )
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  http_client=http_client,
                                  max_retries=MAX_RETRIES)
        return self._client

    def snapshot(self):
        return {"tag": self.tag, "label": self.label, "model": self.model,
                "base_url": self.base_url, "configured": self.configured,
                "calls": self.calls, "failures": self.failures,
                "last_error": self.last_error, "last_ok_at": self.last_ok_at,
                "last_fail_at": self.last_fail_at}


def _build():
    """Read .env and return (backends, provider)."""
    provider = (os.environ.get("LLM_PROVIDER", "cloud").strip().lower()
                or "cloud")
    if provider not in ("cloud", "local"):
        print(f"WARNING: LLM_PROVIDER={provider!r} is not 'cloud' or 'local' "
              f"-- defaulting to 'cloud'.")
        provider = "cloud"

    cloud = Backend(
        "cloud",
        os.environ.get("CLOUD_BASE_URL", ""),
        os.environ.get("CLOUD_API_KEY", ""),
        os.environ.get("CLOUD_MODEL", ""),
        os.environ.get("CLOUD_LABEL", ""),
        _envf("CLOUD_TEMPERATURE", 0.2),
        _envj("CLOUD_EXTRA_BODY"),
    )
    local = Backend(
        "local",
        os.environ.get("LOCAL_BASE_URL", ""),
        # Ollama and most local servers ignore the key, but the OpenAI SDK
        # insists on a non-empty one.
        os.environ.get("LOCAL_API_KEY", "").strip() or "local",
        os.environ.get("LOCAL_MODEL", ""),
        os.environ.get("LOCAL_LABEL", ""),
        _envf("LOCAL_TEMPERATURE", 0.2),
        _envj("LOCAL_EXTRA_BODY"),
    )

    order = [local, cloud] if provider == "local" else [cloud, local]
    order[0].tag = f"primary({order[0].tag})"
    order[1].tag = f"fallback({order[1].tag})"
    return [b for b in order if b.configured], provider


BACKENDS, PROVIDER = _build()
STATE = {"cooldown_until": 0.0, "last_served_by": None}

# A wedged request holds its thread until the HTTP timeout fires. Keep that off
# the default executor, which also serves other awaits.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8,
                                              thread_name_prefix="llm")


def describe():
    return {
        "provider": PROVIDER,
        "backends": [b.snapshot() for b in BACKENDS],
        "last_served_by": STATE["last_served_by"],
        "cooling_down": time.time() < STATE["cooldown_until"],
        "cooldown_remaining_s": max(
            0, round(STATE["cooldown_until"] - time.time(), 1)),
        "request_timeout_s": REQUEST_TIMEOUT,
        "hard_timeout_s": HARD_TIMEOUT,
        "sdk_retries": MAX_RETRIES,
    }


async def _call_one(backend, messages, tools):
    kwargs = dict(model=backend.model, messages=messages)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if backend.temperature is not None:
        kwargs["temperature"] = backend.temperature
    if backend.extra_body:
        kwargs["extra_body"] = backend.extra_body

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _POOL, lambda: backend.client().chat.completions.create(**kwargs))
    backend.calls += 1
    try:
        resp = await asyncio.wait_for(fut, timeout=HARD_TIMEOUT)
    except asyncio.TimeoutError:
        backend.failures += 1
        backend.last_fail_at = time.time()
        backend.last_error = f"timed out after {HARD_TIMEOUT:.0f}s"
        fut.cancel()
        raise TimeoutError(backend.last_error)
    except Exception as e:
        backend.failures += 1
        backend.last_fail_at = time.time()
        backend.last_error = f"{type(e).__name__}: {e}"
        raise
    backend.last_ok_at = time.time()
    backend.last_error = None
    return resp


async def call(messages, tools=None, on_failover=None):
    """Try the primary, then the fallback.

    on_failover(msg) is awaited with a one-line notice when we switch, so a slow
    turn reads as a failover rather than an unexplained pause.
    """
    now = time.time()
    if now < STATE["cooldown_until"]:
        wait = max(1, int(STATE["cooldown_until"] - now))
        raise AllBackendsFailed(
            f"The AI backends are still unavailable. Please wait about "
            f"{max(1, wait // 60)} more minute(s) and try again.")

    if not BACKENDS:
        raise AllBackendsFailed(
            "No LLM backend is configured. Copy .env.example to .env and fill "
            "in either the CLOUD_* or the LOCAL_* block.")

    errors = []
    for i, backend in enumerate(BACKENDS):
        try:
            resp = await _call_one(backend, messages, tools)
            STATE["last_served_by"] = backend.tag
            STATE["cooldown_until"] = 0.0
            if i > 0 and on_failover:
                await on_failover(
                    f"[primary model unavailable - answered by the fallback: "
                    f"{backend.model}]")
            return resp
        except Exception as e:
            errors.append(f"{backend.tag} ({backend.model}): {e}")
            nxt = BACKENDS[i + 1] if i + 1 < len(BACKENDS) else None
            if nxt is not None and on_failover:
                await on_failover(
                    f"[{backend.tag} failed ({e}) - switching to {nxt.model}]")

    # Everything failed: stand down so the next message fails instantly rather
    # than walking the user through every timeout again.
    STATE["cooldown_until"] = time.time() + COOLDOWN
    raise AllBackendsFailed(
        "All AI backends are unavailable right now.\n  " + "\n  ".join(errors))
