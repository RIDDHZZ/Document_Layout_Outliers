"""
Phase 2 - PDF -> layout elements.

Every page becomes a `PageLayout`: a list of `Element`s with boxes normalised to [0, 1] (top-left origin).

Two extraction paths, chosen per page:
  native : PyMuPDF text lines + image blocks (fast, exact). Used when the page has a text layer.
  ocr    : page rendered at `render_dpi` -> grayscale/denoise -> Tesseract `image_to_data` -> words grouped into
           lines (block/paragraph/line ids). Non-text ink (signature-like graphics) is found with OpenCV
           connected-component analysis after masking OCR text boxes. Used for scanned pages, or when forced.

Elements are LINES (not raw OCR tokens) plus non-text regions (`image` from native, `graphic` from OCR path).
Nothing here reads PDF metadata; nothing is executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

try:
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover
    pytesseract = None


class ExtractionError(Exception):
    """User-presentable extraction failure (no stack traces should reach the UI)."""


@dataclass
class Element:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str            # 'text' | 'image' | 'graphic'
    text: str = ""
    conf: float = 100.0


@dataclass
class PageLayout:
    page_number: int     # 1-based
    width_pt: float
    height_pt: float
    elements: list[Element] = field(default_factory=list)
    source: str = "native"   # 'native' | 'ocr'


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _norm_rect(r: pymupdf.Rect, W: float, H: float) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = max(r.x0, 0), max(r.y0, 0), min(r.x1, W), min(r.y1, H)
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return x1 / W, y1 / H, x2 / W, y2 / H


def render_page(page: pymupdf.Page, dpi: int = 150) -> np.ndarray:
    """Render to a BGR uint8 array (also reused by the visualisation layer)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def open_pdf(path: str | Path, max_pages: int | None = None) -> pymupdf.Document:
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise ExtractionError("The uploaded file could not be opened as a PDF.") from e
    if doc.needs_pass:
        doc.close()
        raise ExtractionError("Password-protected PDFs are not supported.")
    if doc.page_count == 0:
        doc.close()
        raise ExtractionError("The uploaded PDF contains no pages.")
    if max_pages and doc.page_count > max_pages:
        n = doc.page_count
        doc.close()
        raise ExtractionError(f"The PDF has {n} pages; the limit is {max_pages}.")
    return doc


# Shared "line" definition for BOTH paths: a run of text on one baseline whose internal gaps are at most
# LINE_GAP_MULT line-heights. Tesseract sometimes fragments a line when it drops low-confidence words; PyMuPDF may
# join distant columns. Applying one rule to both keeps native and OCR features comparable.
LINE_GAP_MULT = 6.0


def _merge_fragments(lines: list[tuple], mult: float = LINE_GAP_MULT) -> list[tuple]:
    """Merge same-baseline OCR fragments: (x1, y1, x2, y2, text, conf) tuples in pixel units."""
    lines = list(lines)
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = (lines[i], lines[j]) if lines[i][0] <= lines[j][0] else (lines[j], lines[i])
                ha, hb = a[3] - a[1], b[3] - b[1]
                v_overlap = min(a[3], b[3]) - max(a[1], b[1])
                if v_overlap >= 0.6 * min(ha, hb) and (b[0] - a[2]) <= mult * max(ha, hb):
                    lines[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                                a[4] + " " + b[4], (a[5] + b[5]) / 2)
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    return lines


# --------------------------------------------------------------------------------------
# native path
# --------------------------------------------------------------------------------------
def native_word_count(page: pymupdf.Page) -> int:
    return len(page.get_text("words"))


def _segments(line: dict) -> list[tuple[str, tuple]]:
    """Split a PyMuPDF line into runs whose horizontal gaps are <= LINE_GAP_MULT line heights."""
    lx0, ly0, lx1, ly1 = line["bbox"]
    height = max(ly1 - ly0, 1e-6)
    horizontal = abs(line.get("dir", (1, 0))[1]) < 0.5           # rotated text: keep the whole line
    spans = [sp for sp in line["spans"] if sp["text"].strip()]
    if not spans:
        return []
    if not horizontal:
        return [("".join(sp["text"] for sp in spans).strip(), tuple(line["bbox"]))]
    spans.sort(key=lambda sp: sp["bbox"][0])
    runs, cur = [], [spans[0]]
    for sp in spans[1:]:
        if sp["bbox"][0] - cur[-1]["bbox"][2] > LINE_GAP_MULT * height:
            runs.append(cur)
            cur = [sp]
        else:
            cur.append(sp)
    runs.append(cur)
    out = []
    for r in runs:
        bb = (min(sp["bbox"][0] for sp in r), ly0, max(sp["bbox"][2] for sp in r), ly1)
        out.append(("".join(sp["text"] for sp in r).strip(), bb))
    return out


def extract_native(page: pymupdf.Page, cfg: dict) -> PageLayout:
    W, H = page.rect.width, page.rect.height
    rm = page.rotation_matrix     # PyMuPDF returns text/image boxes in UNROTATED space
    els: list[Element] = []

    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for text, bbox in _segments(ln):
                nb = _norm_rect(pymupdf.Rect(bbox) * rm, W, H)
                if nb:
                    els.append(Element(*nb, kind="text", text=text))

    thr = cfg["extraction"]["full_page_image_threshold"]
    for info in page.get_image_info():
        nb = _norm_rect(pymupdf.Rect(info["bbox"]) * rm, W, H)
        if nb and (nb[2] - nb[0]) * (nb[3] - nb[1]) < thr:      # skip full-page scan backgrounds
            els.append(Element(*nb, kind="image"))
    return PageLayout(page.number + 1, W, H, els, "native")


# --------------------------------------------------------------------------------------
# OCR path
# --------------------------------------------------------------------------------------
def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 3)


def ocr_lines(gray: np.ndarray, cfg: dict) -> list[tuple[int, int, int, int, str, float]]:
    """Tesseract words -> grouped LINE boxes in pixels: (x1, y1, x2, y2, text, mean_conf)."""
    if pytesseract is None:
        raise ExtractionError("OCR is unavailable on this server (pytesseract/Tesseract not installed).")
    oc = cfg["extraction"]["ocr"]
    try:
        d = pytesseract.image_to_data(gray, lang=oc["lang"], config=f"--psm {oc['psm']}", output_type=Output.DICT)
    except Exception as e:
        raise ExtractionError("OCR failed while reading the document.") from e
    groups: dict[tuple, list] = {}
    for i, t in enumerate(d["text"]):
        t = t.strip()
        try:
            conf = float(d["conf"][i])
        except ValueError:
            continue
        if not t or conf < oc["min_conf"]:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append((d["left"][i], d["top"][i], d["left"][i] + d["width"][i],
                                           d["top"][i] + d["height"][i], t, conf))
    lines = []
    for ws in groups.values():
        ws.sort(key=lambda w: w[0])
        lines.append((min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws),
                      " ".join(w[4] for w in ws), float(np.mean([w[5] for w in ws]))))
    return _merge_fragments(lines)


def detect_graphic_regions(gray: np.ndarray, text_boxes_px, cfg: dict) -> list[tuple[int, int, int, int]]:
    """Non-text ink (e.g. an inserted signature): Otsu -> mask OCR text -> drop ruled lines -> close -> contours."""
    gc = cfg["extraction"]["graphic_blob"]
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pad = 3
    for x1, y1, x2, y2 in text_boxes_px:
        bw[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(0.06 * w), 3), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(0.04 * h), 3))))
    bw = cv2.subtract(cv2.subtract(bw, horiz), vert)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.03 * w) | 1, int(0.008 * h) | 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch / h >= gc["min_height_frac"] and (cw * ch) / (w * h) >= gc["min_area_frac"] \
                and (cw * ch) / (w * h) < cfg["extraction"]["full_page_image_threshold"]:
            out.append((x, y, x + cw, y + ch))
    return out


def extract_ocr(page: pymupdf.Page, cfg: dict) -> PageLayout:
    W, H = page.rect.width, page.rect.height
    img = render_page(page, cfg["extraction"]["render_dpi"])
    gray = _preprocess(img)
    ph, pw = gray.shape
    els: list[Element] = []
    boxes = []
    for x1, y1, x2, y2, text, conf in ocr_lines(gray, cfg):
        els.append(Element(x1 / pw, y1 / ph, x2 / pw, y2 / ph, "text", text, conf))
        boxes.append((x1, y1, x2, y2))
    for x1, y1, x2, y2 in detect_graphic_regions(gray, boxes, cfg):
        els.append(Element(x1 / pw, y1 / ph, x2 / pw, y2 / ph, "graphic"))
    return PageLayout(page.number + 1, W, H, els, "ocr")


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def extract_page(page: pymupdf.Page, cfg: dict, force_ocr: bool = False) -> PageLayout:
    if not force_ocr and native_word_count(page) >= cfg["extraction"]["min_native_words"]:
        return extract_native(page, cfg)
    return extract_ocr(page, cfg)


def extract_document(path: str | Path, cfg: dict, force_ocr: bool = False,
                     max_pages: int | None = None) -> list[PageLayout]:
    doc = open_pdf(path, max_pages)
    try:
        layouts = [extract_page(p, cfg, force_ocr) for p in doc]
    finally:
        doc.close()
    if not any(l.elements for l in layouts):
        raise ExtractionError("The uploaded PDF could not be processed because no readable layout elements "
                              "were detected.")
    return layouts
