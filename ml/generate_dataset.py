"""
Phase 1 - Synthetic legal-style PDF dataset.

Produces, for every ORIGINAL document `doc_XXX`:
  data/normal/doc_XXX.pdf                       normal layout
  data/altered_margin/doc_XXX_margin.pdf        same text, margins / content position altered
  data/inserted_signature/doc_XXX_signature.pdf same PDF + one signature-like image at an unusual place
  data/metadata/annotations.csv                 one row per (file, page) with ground truth

Design notes (worth citing in the report)
  * Normal documents already vary (fonts, margins jitter around 1 inch, justified/ragged text, footer or not,
    5 document types) so "normal" is a distribution, not a single template.
  * ~60% of normal documents carry a GENUINE signature image on the signature line. Otherwise "any image = anomaly"
    would be a trivial (and misleading) rule.
  * Margin variants are re-rendered from the same document spec, so only the layout changes, not the text.
    A page is labelled anomalous only if its text box moved measurably (see `min_measurable_shift`).
  * `source_doc_id` links an original to all its variants -> used later for leakage-free grouped splitting.

Run:  python -m ml.generate_dataset [--n 120] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from .config import data_dir, load_config

PAGE_SIZES = {"letter": (612.0, 792.0), "a4": (595.28, 841.89)}
FONT_FAMILIES = [("Times-Roman", "Times-Bold"), ("Helvetica", "Helvetica-Bold")]
MARGIN_MODES = ["left_increase", "left_decrease", "right_increase", "right_decrease",
                "shift_x", "shift_y", "top_change", "bottom_change"]
SIGNATURE_MODES = ["center", "bottom_right", "top_right", "mid_left", "offset_from_block"]

# --------------------------------------------------------------------------------------
# Text pools
# --------------------------------------------------------------------------------------
FIRST = ["Anita", "Rahul", "Maria", "James", "Priya", "Daniel", "Sofia", "Arjun", "Elena", "Omar", "Grace", "Vikram"]
LAST = ["Sharma", "Patel", "Nguyen", "Garcia", "Miller", "Khan", "Rossi", "Iyer", "Brown", "Singh", "Lopez", "Shah"]
COMPANIES = ["Harbor Ridge Holdings LLC", "Northwind Legal Services Pvt Ltd", "Bluepine Estates Inc.",
             "Crestview Consulting Group", "Meridian Trade Partners LLP", "Oakfield Property Trust"]
CITIES = ["Surat", "Ahmedabad", "Mumbai", "Pune", "Austin", "Denver", "Leeds", "Toronto"]
CLAUSE_TITLES = ["Definitions", "Term and Termination", "Payment Terms", "Confidentiality", "Governing Law",
                 "Indemnification", "Limitation of Liability", "Force Majeure", "Notices", "Entire Agreement",
                 "Assignment", "Dispute Resolution", "Amendments", "Severability", "Representations and Warranties",
                 "Delivery and Acceptance", "Intellectual Property", "Relationship of the Parties"]
SENTENCES = [
    "The {a} agrees to perform the obligations described in this document in accordance with all applicable laws and regulations.",
    "Any payment due under these terms shall be made within {days} days of receipt of a valid written invoice.",
    "Neither party shall disclose confidential information belonging to the other party without prior written consent.",
    "This document shall be governed by and construed in accordance with the laws of the State of {state}.",
    "Either party may terminate this arrangement by giving not less than {days} days written notice to the other party.",
    "The {b} shall be responsible for all costs arising from its own acts, omissions, and breaches of these terms.",
    "In the event of a dispute, the parties shall first attempt to resolve the matter through good faith negotiation.",
    "No amendment to these terms shall be effective unless made in writing and signed by both parties.",
    "The total consideration payable under this document shall not exceed the sum of {amt} unless otherwise agreed.",
    "If any provision is held to be unenforceable, the remaining provisions shall continue in full force and effect.",
    "All notices required hereunder shall be delivered in person or sent by registered mail to the addresses stated above.",
    "The {a} represents that it has full authority to enter into this document and to perform its obligations.",
    "Time shall be of the essence with respect to every obligation of the {b} arising under this document.",
    "Neither party shall be liable for delay caused by events beyond its reasonable control, including natural disasters.",
    "This document constitutes the entire understanding between the parties and supersedes all prior discussions.",
    "The {b} shall keep complete and accurate records relating to the subject matter and make them available on request.",
]
FORM_FIELDS = ["Full Name", "Father's / Spouse's Name", "Date of Birth", "Nationality", "Occupation", "Telephone",
               "Email Address", "Identification Number", "Place of Issue", "Date of Application"]


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2019, 2025)}"


def _sentences(rng: random.Random, ctx: dict, k: int) -> str:
    return " ".join(rng.choice(SENTENCES).format(**ctx) for _ in range(k))


# --------------------------------------------------------------------------------------
# Signature-like image (procedural scribble, no real signatures involved)
# --------------------------------------------------------------------------------------
def _catmull_rom(p: np.ndarray, n: int = 24) -> np.ndarray:
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    return np.array(out)


@lru_cache(maxsize=512)
def signature_png(seed: int) -> tuple[bytes, float]:
    """Return (PNG bytes with alpha, aspect ratio w/h) of a tightly cropped scribble."""
    rng = np.random.default_rng(seed)
    W, H, S = 600, 200, 4
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    npts = int(rng.integers(6, 10))
    xs = np.linspace(0.04, 0.96, npts) + rng.normal(0, 0.02, npts)
    ys = rng.uniform(0.12, 0.85, npts)
    curve = _catmull_rom(np.stack([xs * W * S, ys * H * S], axis=1))
    ink = (15, 25, 110, 255)
    d.line([tuple(pt) for pt in curve], fill=ink, width=3 * S, joint="curve")
    if rng.random() < 0.7:  # underline flourish
        x0, x1 = rng.uniform(0.05, 0.25), rng.uniform(0.7, 0.95)
        y = rng.uniform(0.82, 0.95)
        d.line([(x0 * W * S, y * H * S), ((x0 + x1) / 2 * W * S, (y - 0.04) * H * S), (x1 * W * S, y * H * S)],
               fill=ink, width=2 * S)
    img = img.resize((W, H), Image.LANCZOS)
    bbox = img.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    img = img.crop(bbox)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width / img.height


# --------------------------------------------------------------------------------------
# Document specification + renderer
# --------------------------------------------------------------------------------------
@dataclass
class DocSpec:
    doc_id: str
    doc_type: str
    page_w: float
    page_h: float
    margins: dict
    font: str
    bold: str
    size: float
    leading: float          # multiplier
    para_gap: float
    justify: bool
    footer: bool
    title_center: bool
    signature_seed: int | None   # None -> no genuine signature images
    sig_width_frac: float
    items: list = field(default_factory=list)


def build_spec(doc_id: str, rng: random.Random, cfg: dict) -> DocSpec:
    dcfg = cfg["dataset"]
    size_key = rng.choices(list(dcfg["page_size_weights"]), weights=list(dcfg["page_size_weights"].values()))[0]
    pw, ph = PAGE_SIZES[size_key]
    dtype = rng.choices(list(dcfg["doc_type_weights"]), weights=list(dcfg["doc_type_weights"].values()))[0]
    font, bold = rng.choice(FONT_FAMILIES)
    ctx = dict(a=_name(rng), b=rng.choice(COMPANIES), days=rng.choice([7, 10, 14, 15, 30, 45, 60]),
               state=rng.choice(["Gujarat", "Maharashtra", "Texas", "Colorado", "Ontario"]),
               amt=f"${rng.randint(2, 90) * 1000:,}")
    a, b = ctx["a"], ctx["b"]
    items: list = []

    if dtype == "contract":
        kind = rng.choice(["SERVICE", "CONSULTING", "SUPPLY", "LEASE"])
        items += [("title", f"{kind} AGREEMENT"), ("rule",),
                  ("para", f"This Agreement is made on {_date(rng)} in {rng.choice(CITIES)} between {a} "
                           f"(the \"First Party\") and {b} (the \"Second Party\")."), ("space", 4)]
        for i, t in enumerate(rng.sample(CLAUSE_TITLES, rng.randint(6, 16)), 1):
            items += [("heading", f"{i}. {t}"), ("para", _sentences(rng, ctx, rng.randint(2, 4)))]
        items += [("space", 8), ("para", "IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
                                          "date first written above."),
                  ("sigblock", [(a, "First Party", True), (b, "Second Party", True)])]
    elif dtype == "affidavit":
        items += [("title", "AFFIDAVIT"), ("rule",),
                  ("para", f"I, {a}, residing in {rng.choice(CITIES)}, being duly sworn, do hereby state as follows:")]
        for i in range(1, rng.randint(6, 16)):
            items += [("para", f"{i}. {_sentences(rng, ctx, rng.randint(1, 3))}", 14)]
        items += [("para", f"Sworn before me on {_date(rng)}."),
                  ("sigblock", [(a, "Deponent", True), (_name(rng), "Notary Public", False)])]
    elif dtype == "notice":
        items += [("title", "LEGAL NOTICE"), ("rule",), ("para", f"To: {a}"), ("para", f"From: {b}"),
                  ("para", f"Date: {_date(rng)}"), ("para", f"Re: Notice under clause {rng.randint(2, 12)}"),
                  ("space", 6)]
        for _ in range(rng.randint(3, 9)):
            items += [("para", _sentences(rng, ctx, rng.randint(3, 5)))]
        items += [("sigblock", [(b, "Authorised Signatory", True)])]
    elif dtype == "application_form":
        items += [("title", "APPLICATION FORM"), ("rule",)]
        fields = rng.sample(FORM_FIELDS, rng.randint(6, 10))
        for j in range(0, len(fields) - 1, 2):
            items.append(("fieldpair", fields[j], fields[j + 1]))
        items += [("field", "Residential Address"), ("field", "Purpose of Application"), ("space", 6),
                  ("para", "I hereby declare that the information furnished above is true and correct to the best of "
                           "my knowledge. " + _sentences(rng, ctx, rng.randint(1, 3))),
                  ("sigblock", [(a, "Applicant", True)])]
    else:  # certificate (centered, sparse layout)
        items += [("centered", "CERTIFICATE OF COMPLETION", 22, True), ("space", 26),
                  ("centered", "This is to certify that", 12, False), ("space", 10),
                  ("centered", a, 20, True), ("space", 10),
                  ("centered", f"has satisfactorily completed the programme offered by {b}", 12, False),
                  ("space", 6), ("centered", f"Issued on {_date(rng)} at {rng.choice(CITIES)}", 11, False),
                  ("space", 40),
                  ("sigblock", [(_name(rng), "Director", True), (_name(rng), "Registrar", True)])]

    m = {k: rng.uniform(64, 88) for k in ("left", "right", "top", "bottom")}   # ~1 inch, natural jitter
    has_sig = rng.random() < dcfg["signature_probability"]
    return DocSpec(doc_id=doc_id, doc_type=dtype, page_w=pw, page_h=ph, margins=m, font=font, bold=bold,
                   size=rng.choice([10.5, 11, 11.5, 12]), leading=rng.uniform(1.25, 1.5),
                   para_gap=rng.uniform(5, 10), justify=rng.random() < 0.5, footer=rng.random() < 0.7,
                   title_center=rng.random() < 0.6 or dtype == "certificate",
                   signature_seed=rng.randrange(10 ** 6) if has_sig else None,
                   sig_width_frac=rng.uniform(0.16, 0.24), items=items)


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines, cur = [], []
    for w in text.split():
        if cur and stringWidth(" ".join(cur + [w]), font, size) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


class Renderer:
    """Tiny flow-layout engine. Cursor `y` is measured from the TOP of the page (same as PyMuPDF)."""

    def __init__(self, spec: DocSpec, path: Path, alt: dict | None = None):
        self.s, self.alt = spec, alt
        self.W, self.H = spec.page_w, spec.page_h
        self.c = canvas.Canvas(str(path), pagesize=(self.W, self.H))
        self.page = 0
        self.sig_block: dict[int, list[float]] = {}   # page -> [x1,y1,x2,y2] of signature area (top-left origin)
        self._begin_page()

    # -- page handling
    def _begin_page(self):
        m = dict(self.s.margins)
        self.dx = self.dy = 0.0
        if self.alt and self.page in self.alt["pages"]:
            for k in ("left", "right", "top", "bottom"):
                m[k] += self.alt[k]
            self.dx, self.dy = self.alt["dx"], self.alt["dy"]
        self.x0, self.x1 = m["left"], self.W - m["right"]
        self.y, self.ylim = m["top"], self.H - m["bottom"]

    def _footer(self):
        if self.s.footer:
            self.c.setFont(self.s.font, 9)
            self.c.drawCentredString(self.W / 2, self.s.margins["bottom"] * 0.45, f"Page {self.page + 1}")

    def _newpage(self):
        self._footer()
        self.c.showPage()
        self.page += 1
        self._begin_page()

    def _need(self, h: float):
        if self.y + h > self.ylim:
            self._newpage()

    # -- primitives (dx/dy only used by 'shift' alterations; footer is never shifted)
    def _text(self, x, ybase, text, font, size):
        self.c.setFont(font, size)
        self.c.drawString(x + self.dx, self.H - (ybase + self.dy), text)

    def _hline(self, xa, xb, y, lw=0.7):
        self.c.setLineWidth(lw)
        self.c.line(xa + self.dx, self.H - (y + self.dy), xb + self.dx, self.H - (y + self.dy))

    # -- items
    def para(self, text, indent=0.0, font=None, size=None, after=None):
        s = self.s
        font, size = font or s.font, size or s.size
        lead, width = size * s.leading, self.x1 - self.x0 - indent
        lines = wrap(text, font, size, width)
        for i, ln in enumerate(lines):
            self._need(lead)
            base, x = self.y + size, self.x0 + indent
            if s.justify and i < len(lines) - 1 and " " in ln:
                t = self.c.beginText(x + self.dx, self.H - (base + self.dy))
                t.setFont(font, size)
                t.setWordSpace((width - stringWidth(ln, font, size)) / ln.count(" "))
                t.textOut(ln)
                self.c.drawText(t)
            else:
                self._text(x, base, ln, font, size)
            self.y += lead
        self.y += s.para_gap if after is None else after

    def heading(self, text):
        self._need(self.s.size * self.s.leading * 3)
        self.para(text, font=self.s.bold, after=2)

    def title(self, text):
        size = self.s.size + 5
        self._need(size * 2)
        w = stringWidth(text, self.s.bold, size)
        x = (self.x0 + self.x1 - w) / 2 if self.s.title_center else self.x0
        self._text(x, self.y + size, text, self.s.bold, size)
        self.y += size * 1.6

    def centered(self, text, size, bold):
        font = self.s.bold if bold else self.s.font
        self._need(size * 1.6)
        w = stringWidth(text, font, size)
        self._text((self.x0 + self.x1 - w) / 2, self.y + size, text, font, size)
        self.y += size * 1.5

    def field(self, label, x=None, width=None):
        x = self.x0 if x is None else x
        width = (self.x1 - x) if width is None else width
        lw = stringWidth(label + ": ", self.s.font, self.s.size)
        self._text(x, self.y + self.s.size, label + ":", self.s.font, self.s.size)
        self._hline(x + lw, x + width, self.y + self.s.size + 1.5, 0.5)

    def fieldpair(self, l1, l2):
        self._need(self.s.size * 2.6)
        half = (self.x1 - self.x0) / 2
        self.field(l1, self.x0, half - 14)
        self.field(l2, self.x0 + half + 6, half - 6)
        self.y += self.s.size * 2.2

    def sigblock(self, signers):
        s = self.s
        self._need(120)
        self.y += 16
        top, line_y = self.y, self.y + 54
        cols = max(len(signers), 2)
        colw = (self.x1 - self.x0) / cols
        first_rect = None
        for i, (name, role, signs) in enumerate(signers):
            x = self.x0 + i * colw
            self._hline(x, x + colw * 0.8, line_y, 0.7)
            if s.signature_seed is not None and signs:
                png, asp = signature_png(s.signature_seed + i)
                w = s.sig_width_frac * self.W
                h = min(w / asp, 52.0)
                w = h * asp if h == 52.0 else w
                self.c.drawImage(ImageReader(io.BytesIO(png)), x + 8 + self.dx,
                                 self.H - (line_y - 3 + self.dy), width=w, height=h, mask="auto")
                rect = [x + 8, line_y - 3 - h, x + 8 + w, line_y - 3]
            else:
                rect = [x, top, x + colw * 0.8, line_y]
            first_rect = first_rect or rect
            self._text(x, line_y + 12, "Signature", s.font, 8)
            self._text(x, line_y + 25, name, s.bold, 9.5)
            self._text(x, line_y + 37, role, s.font, 9)
            self._text(x, line_y + 51, "Date:", s.font, 9)
            self._hline(x + 28, x + colw * 0.55, line_y + 52.5, 0.5)
        self.sig_block[self.page] = first_rect
        self.y = line_y + 62

    def render(self):
        for it in self.s.items:
            k = it[0]
            if k == "title": self.title(it[1])
            elif k == "rule":
                self._hline(self.x0, self.x1, self.y - 4, 0.8); self.y += 8
            elif k == "para": self.para(it[1], indent=it[2] if len(it) > 2 else 0.0)
            elif k == "heading": self.heading(it[1])
            elif k == "field":
                self._need(self.s.size * 2.6); self.field(it[1]); self.y += self.s.size * 2.2
            elif k == "fieldpair": self.fieldpair(it[1], it[2])
            elif k == "space": self.y += it[1]
            elif k == "centered": self.centered(it[1], it[2], it[3])
            elif k == "sigblock": self.sigblock(it[1])
        self._footer()
        self.c.save()
        return self.page + 1, self.sig_block


def render_pdf(spec: DocSpec, path: Path, alt: dict | None = None):
    return Renderer(spec, path, alt).render()


# --------------------------------------------------------------------------------------
# Anomaly 1: margin alteration
# --------------------------------------------------------------------------------------
def plan_margin_alteration(spec: DocSpec, rng: random.Random, cfg: dict, n_pages: int) -> dict:
    mc = cfg["dataset"]["margin_alteration"]
    W, H, m = spec.page_w, spec.page_h, spec.margins
    mode = rng.choice(MARGIN_MODES)
    fx, fy = rng.uniform(mc["min_frac"], mc["max_frac"]) * W, rng.uniform(mc["min_frac"], mc["max_frac"]) * H
    sign = rng.choice([-1, 1])
    d = dict(left=0.0, right=0.0, top=0.0, bottom=0.0, dx=0.0, dy=0.0)
    if mode == "left_increase": d["left"] = fx
    elif mode == "left_decrease": d["left"] = -min(fx, m["left"] - 22)
    elif mode == "right_increase": d["right"] = fx
    elif mode == "right_decrease": d["right"] = -min(fx, m["right"] - 22)
    elif mode == "shift_x": d["dx"] = sign * min(fx, (m["left"] if sign < 0 else m["right"]) - 12)
    elif mode == "shift_y": d["dy"] = sign * min(fy, (m["top"] - 12) if sign < 0 else (m["bottom"] - 44))
    elif mode == "top_change": d["top"] = fy if sign > 0 else -min(fy, m["top"] - 22)
    elif mode == "bottom_change": d["bottom"] = fy if sign > 0 else -min(fy, m["bottom"] - 44)
    if rng.random() < mc["all_pages_probability"]:
        pages = set(range(n_pages))
    else:
        pages = {rng.randrange(n_pages)}
    return dict(mode=mode, pages=pages, **d)


def _text_union(page: "pymupdf.Page") -> list[float] | None:
    """Union bbox (points, top-left origin) of text lines + images on a page, normalised later by callers."""
    rects = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    rects += [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    if not rects:
        return None
    u = rects[0]
    for r in rects[1:]:
        u |= r
    return [u.x0, u.y0, u.x1, u.y1]


def _measurably_moved(new_bb, old_bb, W, H, thr) -> bool:
    if new_bb is None or old_bb is None:
        return True
    diffs = [abs(new_bb[0] - old_bb[0]) / W, abs(new_bb[1] - old_bb[1]) / H,
             abs(new_bb[2] - old_bb[2]) / W, abs(new_bb[3] - old_bb[3]) / H]
    return max(diffs) >= thr


def make_margin_variant(spec, normal_path, out_path, rng, cfg, n_pages):
    """Render altered variant; retry planning until >=1 page is measurably changed. Returns (plan, altered_pages, bboxes)."""
    thr = cfg["dataset"]["margin_alteration"]["min_measurable_shift"]
    W, H = spec.page_w, spec.page_h
    with pymupdf.open(normal_path) as ref:
        ref_bb = [_text_union(p) for p in ref]
    for _ in range(25):
        plan = plan_margin_alteration(spec, rng, cfg, n_pages)
        n_var, _ = render_pdf(spec, out_path, plan)
        altered, bboxes = [], {}
        with pymupdf.open(out_path) as doc:
            for p in sorted(plan["pages"]):
                if p >= n_var:
                    continue
                bb = _text_union(doc[p])
                old = ref_bb[p] if p < len(ref_bb) else None
                if _measurably_moved(bb, old, W, H, thr):
                    altered.append(p)
                    bboxes[p] = bb
        if altered:
            return plan, altered, bboxes, n_var
    raise RuntimeError(f"could not produce a measurable margin alteration for {spec.doc_id}")


# --------------------------------------------------------------------------------------
# Anomaly 2: inserted signature
# --------------------------------------------------------------------------------------
def insert_signature(src, dst, spec, sig_block, n_pages, rng, cfg):
    sc = cfg["dataset"]["signature_insertion"]
    W, H = spec.page_w, spec.page_h
    mode = rng.choice(SIGNATURE_MODES)
    if mode == "offset_from_block" and not sig_block:
        mode = "bottom_right"
    scale = rng.uniform(sc["min_scale"], sc["max_scale"])
    png, asp = signature_png(rng.randrange(10 ** 6))
    w = rng.uniform(0.16, 0.24) * W * scale
    h = w / asp
    if h > 90:
        h, w = 90.0, 90.0 * asp
    page_idx = rng.randrange(n_pages)
    if mode == "center":
        cx, cy = rng.uniform(0.4, 0.6) * W, rng.uniform(0.4, 0.6) * H
    elif mode == "bottom_right":
        cx, cy = rng.uniform(0.72, 0.88) * W, rng.uniform(0.86, 0.95) * H
    elif mode == "top_right":
        cx, cy = rng.uniform(0.72, 0.88) * W, rng.uniform(0.04, 0.09) * H
    elif mode == "mid_left":
        cx, cy = rng.uniform(0.12, 0.25) * W, rng.uniform(0.3, 0.7) * H
    else:  # offset_from_block: subtle - close to the real signature area but displaced
        page_idx = max(sig_block)
        bx1, by1, bx2, by2 = sig_block[page_idx]
        cx = (bx1 + bx2) / 2 + rng.choice([-1, 1]) * rng.uniform(0.10, 0.22) * W
        cy = (by1 + by2) / 2 + rng.choice([-1, 1]) * rng.uniform(0.05, 0.12) * H
    x1 = min(max(cx - w / 2, 6), W - 6 - w)
    y1 = min(max(cy - h / 2, 6), H - 6 - h)
    rect = pymupdf.Rect(x1, y1, x1 + w, y1 + h)
    with pymupdf.open(src) as doc:
        doc[page_idx].insert_image(rect, stream=png)
        doc.save(dst, garbage=3, deflate=True)
    return dict(page=page_idx, bbox=[rect.x0, rect.y0, rect.x1, rect.y1], mode=mode, scale=round(scale, 3))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def generate(cfg: dict, n: int | None = None, root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else data_dir(cfg)
    n = n or cfg["dataset"]["n_documents"]
    seed = cfg["random_state"]
    for sub in ("normal", "altered_margin", "inserted_signature", "metadata"):
        shutil.rmtree(root / sub, ignore_errors=True)
        (root / sub).mkdir(parents=True, exist_ok=True)

    rows, specs = [], []

    def row(doc_id, src, category, file, pageno, W, H, atype="normal", bbox=None, detail=None):
        bb = bbox or [np.nan] * 4
        rows.append(dict(document_id=doc_id, source_doc_id=src, category=category, file=file,
                         page_number=pageno, page_width=round(W, 2), page_height=round(H, 2),
                         anomaly_type=atype, x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3],
                         is_anomalous=int(atype != "normal"), alteration_detail=json.dumps(detail) if detail else ""))

    for i in range(1, n + 1):
        doc_id = f"doc_{i:03d}"
        spec = build_spec(doc_id, random.Random(seed * 100_003 + i), cfg)
        specs.append(asdict(spec))
        W, H = spec.page_w, spec.page_h

        # normal
        rel = f"normal/{doc_id}.pdf"
        n_pages, sig_block = render_pdf(spec, root / rel)
        for p in range(n_pages):
            row(doc_id, doc_id, "normal", rel, p + 1, W, H)

        # altered margin
        vid, rel_m = f"{doc_id}_margin", f"altered_margin/{doc_id}_margin.pdf"
        plan, altered, bboxes, n_var = make_margin_variant(spec, root / rel, root / rel_m,
                                                           random.Random(seed * 7_919 + i), cfg, n_pages)
        detail = {k: (sorted(v) if k == "pages" else round(v, 2) if isinstance(v, float) else v)
                  for k, v in plan.items()}
        for p in range(n_var):
            if p in altered:
                row(vid, doc_id, "altered_margin", rel_m, p + 1, W, H, "altered_margin", bboxes[p], detail)
            else:
                row(vid, doc_id, "altered_margin", rel_m, p + 1, W, H)

        # inserted signature
        vid, rel_s = f"{doc_id}_signature", f"inserted_signature/{doc_id}_signature.pdf"
        ins = insert_signature(root / rel, root / rel_s, spec, sig_block, n_pages,
                               random.Random(seed * 6_007 + i), cfg)
        for p in range(n_pages):
            if p == ins["page"]:
                row(vid, doc_id, "inserted_signature", rel_s, p + 1, W, H, "inserted_signature", ins["bbox"],
                    dict(mode=ins["mode"], scale=ins["scale"]))
            else:
                row(vid, doc_id, "inserted_signature", rel_s, p + 1, W, H)

    ann = pd.DataFrame(rows)
    ann.to_csv(root / "metadata" / "annotations.csv", index=False)
    with open(root / "metadata" / "specs.jsonl", "w", encoding="utf-8") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    return ann


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="number of original documents")
    ap.add_argument("--out", default=None, help="output data directory (default: from config)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ann = generate(cfg, a.n, a.out)
    print(f"Wrote {ann['file'].nunique()} PDFs / {len(ann)} page rows")
    print(ann.groupby(["category", "anomaly_type"]).size().rename("pages").to_string())


if __name__ == "__main__":
    main()
