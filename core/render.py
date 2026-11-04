"""
Console output -> chat rendering helpers.

Tools print plain text. These turn ASCII grid tables into structured tables for
the UI, and summarise them for the model so it has something to comment on
without re-listing thousands of rows it would only get wrong.
"""

import json
import re

BORDER_RE = re.compile(r"^\s*\+[-=+]+\+\s*$")
ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _cells(line):
    parts = line.strip().split("|")
    if len(parts) >= 2:
        parts = parts[1:-1]
    return [p.strip() for p in parts]


def parse_grid_table(block_lines):
    row_lines = [ln for ln in block_lines if ROW_RE.match(ln)]
    if not row_lines:
        return None
    headers = _cells(row_lines[0])
    ncol = len(headers)
    if ncol == 0:
        return None
    rows = []
    for ln in row_lines[1:]:
        c = _cells(ln)
        c = c + [""] * (ncol - len(c)) if len(c) < ncol else c[:ncol]
        rows.append(c)
    return {"headers": headers, "rows": rows}


def _num(v):
    try:
        f = float(str(v).replace(",", ""))
        return int(f) if f.is_integer() else f
    except (ValueError, AttributeError):
        return None


def summarize_columns(headers, rows):
    info = {}
    for c, h in enumerate(headers):
        vals = [r[c] for r in rows if c < len(r) and r[c] != ""]
        nums = [_num(v) for v in vals]
        if vals and all(x is not None for x in nums):
            entry = {"type": "numeric", "min": min(nums), "max": max(nums),
                     "sum": sum(nums)}
            distinct = {}
            for x in nums:
                distinct[x] = distinct.get(x, 0) + 1
            if len(distinct) <= 6:
                entry["counts"] = distinct
            info[h] = entry
        else:
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            info[h] = ({"type": "categorical", "counts": counts}
                       if len(counts) <= 6
                       else {"type": "categorical", "distinct": len(counts)})
    return info


def table_note(headers, rows, samples=5):
    cols = summarize_columns(headers, rows)
    sample = [dict(zip(headers, r)) for r in rows[:samples]]
    n = len(rows)
    return ("[TABLE ALREADY DISPLAYED TO THE USER - do not repeat, list or "
            "enumerate its rows. Use this only for a 1-3 sentence takeaway. "
            f"columns={headers}; row_count={n}; "
            f"column_summary={json.dumps(cols, ensure_ascii=False, default=str)}; "
            f"sample_rows={json.dumps(sample, ensure_ascii=False, default=str)}"
            + (f" ({samples} of {n} shown)" if n > samples else "") + "]")


def prepare_for_model(raw):
    """Replace each ASCII table in `raw` with stats + a few sample rows."""
    lines = raw.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        if BORDER_RE.match(lines[i]) or ROW_RE.match(lines[i]):
            j, block = i, []
            while j < n and (BORDER_RE.match(lines[j]) or ROW_RE.match(lines[j])):
                block.append(lines[j])
                j += 1
            parsed = parse_grid_table(block)
            out.append(table_note(parsed["headers"], parsed["rows"])
                       if parsed else "\n".join(block))
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _is_md_separator(line):
    s = line.strip()
    return bool(s) and "-" in s and "|" in s and not (set(s) - set("|:- "))


def strip_think(text):
    """Remove <think>...</think> blocks that reasoning models emit."""
    if not text:
        return text
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    t = re.sub(r"<think>.*$", "", t, flags=re.S | re.I)
    return t.strip()


def strip_markdown_tables(text):
    """Hard guard: no duplicated table reaches the UI."""
    lines = (text or "").split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        if ROW_RE.match(lines[i]) and i + 1 < n and _is_md_separator(lines[i + 1]):
            i += 2
            while i < n and ROW_RE.match(lines[i]):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
