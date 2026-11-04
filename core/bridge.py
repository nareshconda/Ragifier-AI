"""
Tool execution bridge.

Runs a tool function on a worker thread while:
  * capturing everything it prints, so the chat can stream it live
  * relaying any input() it calls back to the browser as a question
  * letting the user stop it

The important detail is that stdout and input() are patched ONCE, globally, and
then routed per thread. The naive approach -- each execution patching the
globals for itself -- breaks as soon as one tool is stopped but its thread keeps
running (a wedged network call cannot be killed from Python). The next
execution overwrites the globals, the two share one hook, and an answer typed by
the user can be delivered to the wrong thread. With no timeout on the handoff,
whichever side loses the race waits forever.
"""

import builtins
import io
import queue
import sys
import threading
import time
import traceback

# How long a bridged input() waits for an answer before giving up. Without this,
# a lost answer parks the thread permanently.
ANSWER_TIMEOUT = 600.0

_REAL_STDOUT = sys.stdout
_REAL_INPUT = builtins.input

_ROUTES = {}                 # thread ident -> ToolExecution
_ROUTE_LOCK = threading.Lock()
_ACTIVE = None               # newest running execution (serves its helper threads)

EXECUTIONS = []              # recent executions, for /api/debug
MAX_TRACKED = 25


def _route():
    """The execution owning the calling thread, else the active one."""
    with _ROUTE_LOCK:
        ex = _ROUTES.get(threading.get_ident())
        if ex is not None:
            return ex
        ex = _ACTIVE
    # Helper threads a tool spawned are not registered; they belong to whichever
    # execution is currently live.
    return ex if (ex is not None and ex.alive) else None


class _RoutingStdout(io.TextIOBase):
    def write(self, s):
        if s:
            ex = _route()
            if ex is not None:
                ex.out_q.put(("stdout", s))
            else:
                try:
                    _REAL_STDOUT.write(s)
                except Exception:
                    pass
        return len(s)

    def flush(self):
        try:
            _REAL_STDOUT.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def _routing_input(prompt=""):
    ex = _route()
    if ex is None:
        return _REAL_INPUT(prompt)
    return ex.ask(str(prompt))


def install():
    """Patch stdout/input once. Safe to call more than once."""
    if not isinstance(sys.stdout, _RoutingStdout):
        sys.stdout = _RoutingStdout()
    if builtins.input is not _routing_input:
        builtins.input = _routing_input


# Prompts starting with this marker ask for a credential. The server strips the
# marker before display, shows a masked input, and records the ANSWER as ****
# so it is never sent to the language model.
SECRET_PREFIX = "[secret] "


def ask_secret(prompt):
    """Ask the user for a credential. Use this from scripts/tools."""
    return input(SECRET_PREFIX + prompt)


class ToolExecution:
    """Runs fn(**kwargs) on a worker thread.

    Events on .out_q:  ("stdout", text) | ("question", prompt)
                       | ("done", return_value) | ("error", message)
    """

    _seq = 0

    def __init__(self, fn, kwargs, name=""):
        ToolExecution._seq += 1
        self.id = ToolExecution._seq
        self.name = name or getattr(fn, "__name__", "tool")
        self.fn = fn
        self.kwargs = kwargs or {}
        self.out_q = queue.Queue()
        self.answer_q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive = False
        self.started_at = None
        self.finished_at = None
        self.abandoned = False
        self.pending_question = None
        self.pending_since = None
        self.questions_asked = 0
        self.answers_received = 0
        self.last_error = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        global _ACTIVE
        install()
        self.alive = True
        self.started_at = time.time()
        with _ROUTE_LOCK:
            _ACTIVE = self
        EXECUTIONS.append(self)
        del EXECUTIONS[:-MAX_TRACKED]
        self.thread.start()

    def abandon(self):
        """User pressed Stop: release anything waiting on an answer."""
        self.abandoned = True
        self.answer_q.put("")

    # -- bridged input -----------------------------------------------------
    def ask(self, prompt):
        self.questions_asked += 1
        self.pending_question = prompt
        self.pending_since = time.time()
        self.out_q.put(("question", prompt))
        try:
            answer = self.answer_q.get(timeout=ANSWER_TIMEOUT)
        except queue.Empty:
            self.pending_question = None
            self.last_error = f"no answer within {ANSWER_TIMEOUT:.0f}s"
            raise TimeoutError(
                f"No answer came back for {prompt.strip()[:60]!r} within "
                f"{ANSWER_TIMEOUT:.0f}s. Nothing was run. Ask again to retry.")
        self.answers_received += 1
        self.pending_question = None
        self.pending_since = None
        return answer

    # -- body --------------------------------------------------------------
    def _run(self):
        global _ACTIVE
        ident = threading.get_ident()
        with _ROUTE_LOCK:
            _ROUTES[ident] = self
        try:
            result = self.fn(**self.kwargs)
            self.out_q.put(("done", result))
        except SystemExit:
            self.out_q.put(("done", "Process exited."))
        except Exception:
            tb = traceback.format_exc()
            self.last_error = tb
            self.out_q.put(("error", tb))
        finally:
            self.alive = False
            self.finished_at = time.time()
            with _ROUTE_LOCK:
                _ROUTES.pop(ident, None)
                if _ACTIVE is self:
                    _ACTIVE = None

    # -- diagnostics -------------------------------------------------------
    def snapshot(self):
        now = time.time()
        return {
            "id": self.id,
            "tool": self.name,
            "alive": self.alive,
            "abandoned": self.abandoned,
            "started_at": self.started_at,
            "running_for_s": (round(now - self.started_at, 1)
                              if self.started_at else None),
            "finished_at": self.finished_at,
            "questions_asked": self.questions_asked,
            "answers_received": self.answers_received,
            "pending_question": self.pending_question,
            "waiting_for_answer_s": (round(now - self.pending_since, 1)
                                     if self.pending_since else None),
            "out_q_backlog": self.out_q.qsize(),
            "last_error": (self.last_error or "")[-600:] or None,
        }


def live_executions():
    return [e for e in EXECUTIONS if e.alive]


def abandon_all():
    """Stop every running execution. Returns what was stopped."""
    stopped = []
    for ex in EXECUTIONS:
        if ex.alive:
            ex.abandon()
            stopped.append({"id": ex.id, "tool": ex.name})
    return stopped
