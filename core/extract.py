"""
Turn a file into text the assistant can search.

Every extractor returns the same shape:

    {
      "text":    "...with <!-- page: N --> anchors...",
      "pages":   12,                 # 0 when the format has no pages
      "figures": [ {...}, ... ],     # images/charts we found, WITH page numbers
      "kind":    "pdf",
      "warnings": ["..."],
    }

Page anchors are the point. They survive chunking, so every answer can say
which page it came from -- and when someone asks about a diagram we cannot
read, we can still say exactly where to look.

Readers are optional dependencies. A missing one disables that file type and
says which package to install, rather than crashing the app.
"""

import csv
import io
import json
import os
import re
import zipfile

PAGE_ANCHOR = "<!-- page: {} -->"
PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")

# Extensions we will attempt. Anything else is read as plain text if it decodes.
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".log", ".json", ".yaml",
             ".yml", ".ini", ".cfg", ".toml", ".py", ".js", ".ts", ".java",
             ".c", ".h", ".cpp", ".go", ".rs", ".rb", ".sh", ".sql", ".xml"}

SUPPORTED = ({".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".tsv", ".html",
              ".htm", ".epub"} | TEXT_EXTS)


class ExtractError(Exception):
    pass


def _missing(pkg, ext):
    return ExtractError(
        f"Reading {ext} files needs the '{pkg}' package. "
        f"Install it with:  pip install {pkg}")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _extract_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise _missing("pypdf", ".pdf")

    reader = PdfReader(path)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")        # many PDFs are "encrypted" with no password
        except Exception:
            raise ExtractError(
                "This PDF is password-protected. Remove the password and "
                "upload it again.")

    parts, figures, warnings = [], [], []
    empty_pages = 0

    for i, page in enumerate(reader.pages, start=1):
        parts.append(PAGE_ANCHOR.format(i))
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = ""
            warnings.append(f"page {i}: could not extract text ({e})")
        if not text.strip():
            empty_pages += 1
        parts.append(text)

        # Images: we cannot read what they show, but we CAN record that one is
        # on page i, so a question about a diagram gets a page number instead
        # of a guess.
        try:
            images = list(getattr(page, "images", []) or [])
        except Exception:
            images = []
        for n, img in enumerate(images, start=1):
            figures.append({
                "page": i,
                "index": n,
                "name": getattr(img, "name", f"image{n}") or f"image{n}",
                "caption": "",          # filled in below from nearby text
            })

    text = "\n".join(parts)
    n_pages = len(reader.pages)

    if empty_pages == n_pages and n_pages:
        warnings.append(
            "No selectable text found on any page - this looks like a scanned "
            "PDF. You would need OCR (e.g. ocrmypdf) before it can be searched.")
    elif empty_pages > n_pages * 0.4 and n_pages > 2:
        warnings.append(
            f"{empty_pages} of {n_pages} pages had no selectable text; parts of "
            f"this document may be scanned images.")

    _attach_captions(text, figures)
    return {"text": text, "pages": n_pages, "figures": figures,
            "kind": "pdf", "warnings": warnings}


_CAPTION_RE = re.compile(
    r"^\s*((?:figure|fig\.?|table|chart|diagram|exhibit|image|plate)\s*"
    r"[-–]?\s*[\dA-Z][\w.\-]*)\s*[:.–-]?\s*(.{0,120})",
    re.IGNORECASE)


def _attach_captions(text, figures):
    """Find 'Figure 3: ...' style lines and attach them to figures on that page.

    Imperfect by nature -- a caption may sit above or below its image -- but a
    real caption is far more useful than 'image2.png' when someone asks which
    page a chart is on.
    """
    if not figures:
        return
    by_page = {}
    page = 0
    for line in text.split("\n"):
        m = PAGE_RE.search(line)
        if m:
            page = int(m.group(1))
            continue
        c = _CAPTION_RE.match(line.strip())
        if c:
            by_page.setdefault(page, []).append(
                (c.group(1).strip() + (": " + c.group(2).strip()
                                       if c.group(2).strip() else "")).strip())
    for f in figures:
        caps = by_page.get(f["page"], [])
        if caps:
            f["caption"] = caps[min(f["index"] - 1, len(caps) - 1)]


def extract_pdf_images(path, out_dir, max_images=200):
    """Save embedded PDF images to disk. Returns [{page, index, file}].

    Best effort: we still report the page number even if saving fails, because
    the page number is what the user actually needs.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    saved = []
    try:
        reader = PdfReader(path)
    except Exception:
        return []
    os.makedirs(out_dir, exist_ok=True)
    for i, page in enumerate(reader.pages, start=1):
        try:
            images = list(getattr(page, "images", []) or [])
        except Exception:
            continue
        for n, img in enumerate(images, start=1):
            if len(saved) >= max_images:
                return saved
            try:
                ext = os.path.splitext(getattr(img, "name", "") or "")[1] or ".png"
                fn = f"p{i:04d}_{n}{ext}"
                dest = os.path.join(out_dir, fn)
                with open(dest, "wb") as fh:
                    fh.write(img.data)
                saved.append({"page": i, "index": n, "file": fn})
            except Exception:
                continue
    return saved


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------
def _extract_docx(path):
    try:
        import docx
    except ImportError:
        raise _missing("python-docx", ".docx")

    d = docx.Document(path)
    lines, figures, warnings = [], [], []

    # Word has no fixed pages until it is rendered, so we anchor by section
    # instead and say so, rather than inventing page numbers.
    for p in d.paragraphs:
        t = (p.text or "").rstrip()
        style = (p.style.name or "") if p.style is not None else ""
        if t and style.startswith("Heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            lines.append("#" * min(6, int(level)) + " " + t)
        elif t:
            lines.append(t)

    for ti, table in enumerate(d.tables, start=1):
        lines.append(f"\n[Table {ti}]")
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))

    try:
        n_img = sum(1 for r in d.part.rels.values() if "image" in r.reltype)
        for n in range(1, n_img + 1):
            figures.append({"page": 0, "index": n, "name": f"image{n}",
                            "caption": ""})
        if n_img:
            warnings.append(
                f"{n_img} image(s) are embedded. Word files have no fixed page "
                f"numbers until rendered, so figures are reported by section "
                f"heading rather than page.")
    except Exception:
        pass

    return {"text": "\n".join(lines), "pages": 0, "figures": figures,
            "kind": "docx", "warnings": warnings}


# ---------------------------------------------------------------------------
# PowerPoint - slides ARE pages
# ---------------------------------------------------------------------------
def _extract_pptx(path):
    try:
        from pptx import Presentation
    except ImportError:
        raise _missing("python-pptx", ".pptx")

    prs = Presentation(path)
    parts, figures = [], []
    for i, slide in enumerate(prs.slides, start=1):
        parts.append(PAGE_ANCHOR.format(i))
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = (shape.text_frame.text or "").strip()
                if t:
                    parts.append(t)
            if shape.shape_type is not None and getattr(shape, "image", None):
                figures.append({"page": i, "index": len(figures) + 1,
                                "name": f"slide{i}_image", "caption": ""})
        try:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[Speaker notes] {notes}")
        except Exception:
            pass
    return {"text": "\n".join(parts), "pages": len(prs.slides),
            "figures": figures, "kind": "pptx",
            "warnings": ["Page numbers refer to slide numbers."] if figures else []}


# ---------------------------------------------------------------------------
# Excel - sheets are pages
# ---------------------------------------------------------------------------
def _extract_xlsx(path, max_rows_per_sheet=5000):
    try:
        import openpyxl
    except ImportError:
        raise _missing("openpyxl", ".xlsx")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts, warnings = [], []
    n_sheets = len(wb.worksheets)
    for i, ws in enumerate(wb.worksheets, start=1):
        parts.append(PAGE_ANCHOR.format(i))
        parts.append(f"[Sheet: {ws.title}]")
        n = 0
        for row in ws.iter_rows(values_only=True):
            if n >= max_rows_per_sheet:
                warnings.append(
                    f"Sheet '{ws.title}' truncated at {max_rows_per_sheet} rows.")
                break
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                parts.append(" | ".join(cells))
                n += 1
    wb.close()
    return {"text": "\n".join(parts), "pages": n_sheets,
            "figures": [], "kind": "xlsx",
            "warnings": warnings + ["Page numbers refer to sheet numbers."]}


# ---------------------------------------------------------------------------
# Simple formats
# ---------------------------------------------------------------------------
def _extract_csv(path):
    rows, warnings = [], []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except Exception:
            dialect = csv.excel
        for i, row in enumerate(csv.reader(f, dialect)):
            if i >= 20000:
                warnings.append("Truncated at 20,000 rows.")
                break
            if any((c or "").strip() for c in row):
                rows.append(" | ".join(c or "" for c in row))
    return {"text": "\n".join(rows), "pages": 0, "figures": [],
            "kind": "csv", "warnings": warnings}


def _extract_html(path):
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        figures = [{"page": 0, "index": n, "name": img.get("src", "")[:80],
                    "caption": img.get("alt", "")}
                   for n, img in enumerate(soup.find_all("img"), start=1)]
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s{2,}", " ", text)
        figures = []
    return {"text": text, "pages": 0, "figures": figures, "kind": "html",
            "warnings": []}


def _extract_epub(path):
    """EPUB is a zip of XHTML. Each document becomes a 'page'."""
    parts, warnings = [], []
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith((".xhtml", ".html", ".htm"))]
            names.sort()
            for i, n in enumerate(names, start=1):
                parts.append(PAGE_ANCHOR.format(i))
                raw = z.read(n).decode("utf-8", errors="replace")
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(raw, "html.parser")
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    parts.append(soup.get_text("\n"))
                except ImportError:
                    parts.append(re.sub(r"<[^>]+>", " ", raw))
    except zipfile.BadZipFile:
        raise ExtractError("That .epub could not be opened (not a valid EPUB).")
    warnings.append("Page numbers refer to chapter files, not printed pages.")
    return {"text": "\n".join(parts), "pages": len(parts) // 2,
            "figures": [], "kind": "epub", "warnings": warnings}


def _extract_text(path):
    raw = open(path, "rb").read()
    if b"\x00" in raw[:4096]:
        raise ExtractError(
            "That looks like a binary file, not a document. Supported types: "
            + ", ".join(sorted(SUPPORTED)))
    text = raw.decode("utf-8", errors="replace")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            pass
    return {"text": text, "pages": 0, "figures": [], "kind": ext.lstrip(".") or "txt",
            "warnings": []}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".xlsx": _extract_xlsx,
    ".csv": _extract_csv, ".tsv": _extract_csv,
    ".html": _extract_html, ".htm": _extract_html,
    ".epub": _extract_epub,
}


def extract(path):
    if not os.path.exists(path):
        raise ExtractError(f"No such file: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".doc":
        raise ExtractError(
            "Old .doc files are not supported. Open it in Word and 'Save As' "
            ".docx, then upload that.")
    if ext == ".pages":
        raise ExtractError(
            "Apple Pages files are not supported. Export to PDF or .docx first.")

    fn = _EXTRACTORS.get(ext)
    if fn is None:
        if ext in TEXT_EXTS or ext == "":
            fn = _extract_text
        else:
            raise ExtractError(
                f"Don't know how to read '{ext}' files. Supported: "
                + ", ".join(sorted(SUPPORTED)))

    result = fn(path)
    result["text"] = _tidy(result.get("text", ""))
    if not result["text"].strip():
        hint = ""
        if ext == ".pdf":
            hint = (" The PDF appears to contain only scanned images - it needs "
                    "OCR (e.g. ocrmypdf) before it can be searched.")
        raise ExtractError(f"No text could be extracted from that file.{hint}")
    return result


def _tidy(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def page_of(text, offset):
    """Which page anchor precedes `offset`? 0 if the format has no pages."""
    last = 0
    for m in PAGE_RE.finditer(text, 0, max(0, offset)):
        last = int(m.group(1))
    return last


def strip_anchors(text):
    return PAGE_RE.sub("", text)
