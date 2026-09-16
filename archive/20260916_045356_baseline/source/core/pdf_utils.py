"""
core/pdf_utils.py
-----------------
Shared ReportLab setup for all PDF reports in the v5 suite.

Usage in any PDF-generating script:
    from core.pdf_utils import register_fonts, SANS, SANS_BOLD, SANS_ITAL, MONO
    from core.pdf_utils import NAVY, RED, BLUE, GREEN, PURP, ORANGE
    from core.pdf_utils import (sty, title_sty, head_sty, small_sty, note_sty,
                                 mkbox, pass_box, fail_box, part_box, note_box,
                                 warn_box, mk_table, divline)
    register_fonts()

Why DejaVu:
    ReportLab's built-in Helvetica is Latin-1 only.  Every character outside
    that range (Greek letters, arrows, check marks, subscript numbers, etc.)
    renders as a solid black rectangle.  DejaVu Sans covers the full Unicode
    Basic Multilingual Plane and is available on every Ubuntu system.
"""
from __future__ import annotations
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import Table, TableStyle, Paragraph, HRFlowable

# ── Font paths ────────────────────────────────────────────────────────
_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
_FONTS = {
    "DejaVu":       "DejaVuSans.ttf",
    "DejaVu-Bold":  "DejaVuSans-Bold.ttf",
    "DejaVu-Ital":  "DejaVuSans-Oblique.ttf",
    "DejaVu-Mono":  "DejaVuSansMono.ttf",
}

# Friendly aliases used throughout all report scripts
SANS      = "DejaVu"
SANS_BOLD = "DejaVu-Bold"
SANS_ITAL = "DejaVu-Ital"
MONO      = "DejaVu-Mono"


def register_fonts() -> None:
    """Register DejaVu fonts with ReportLab.  Call once per process."""
    for name, fname in _FONTS.items():
        path = os.path.join(_DEJAVU_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DejaVu font not found at {path}.\n"
                "Install with:  sudo apt-get install fonts-dejavu-core"
            )
        try:
            pdfmetrics.registerFont(TTFont(name, path))
        except Exception:
            pass  # already registered in this process


# ── Colour palette ────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1F4E79")
RED    = colors.HexColor("#C0392B")
BLUE   = colors.HexColor("#2E75B6")
GREEN  = colors.HexColor("#375623")
PURP   = colors.HexColor("#7B2D8B")
ORANGE = colors.HexColor("#ED7D31")
TEAL   = colors.HexColor("#1A7B6B")
BLACK  = colors.HexColor("#000000")
DGREY  = colors.HexColor("#444444")
MGREY  = colors.HexColor("#CCCCCC")
LGREY  = colors.HexColor("#F4F4F4")
ROWBLU = colors.HexColor("#EEF2F8")
WHITE  = colors.white

PASSBG = colors.HexColor("#D5F5D5")
FAILBG = colors.HexColor("#FDEBEA")
PARTBG = colors.HexColor("#FFF3CD")
NOTEBG = colors.HexColor("#DCE6F1")
TEALBG = colors.HexColor("#E8F5F3")
AMBDR  = colors.HexColor("#856404")


# ── Style factories ───────────────────────────────────────────────────
def sty(name: str, **kw) -> ParagraphStyle:
    """Base paragraph style using DejaVu (Unicode-safe)."""
    base = dict(fontName=SANS, fontSize=8.5, textColor=BLACK,
                leading=11.5, spaceAfter=2)
    base.update(kw)
    return ParagraphStyle(name, **base)


def title_sty(color=NAVY) -> ParagraphStyle:
    return sty("title", fontSize=14, fontName=SANS_BOLD, textColor=color,
               alignment=TA_CENTER, leading=20, spaceAfter=5)


def stitle_sty() -> ParagraphStyle:
    return sty("stitle", fontSize=10, textColor=DGREY,
               alignment=TA_CENTER, spaceAfter=3)


def src_sty() -> ParagraphStyle:
    return sty("src", fontSize=8, textColor=colors.HexColor("#888888"),
               alignment=TA_CENTER, spaceAfter=4)


def head_sty(color=NAVY) -> ParagraphStyle:
    return sty("head", fontSize=11, fontName=SANS_BOLD, textColor=color,
               spaceBefore=8, spaceAfter=4)


def head2_sty(color=NAVY) -> ParagraphStyle:
    return sty("head2", fontSize=9.5, fontName=SANS_BOLD, textColor=color,
               spaceBefore=5, spaceAfter=2)


def small_sty() -> ParagraphStyle:
    return sty("small", fontSize=7.5, leading=10)


def note_sty() -> ParagraphStyle:
    return sty("note", fontSize=7.5, fontName=SANS_ITAL,
               textColor=colors.HexColor("#555555"))


def body_sty() -> ParagraphStyle:
    return sty("body", fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY)


def pass_sty() -> ParagraphStyle:
    return sty("pass", fontSize=8.5, fontName=SANS_BOLD, textColor=GREEN)


def fail_sty() -> ParagraphStyle:
    return sty("fail", fontSize=8.5, fontName=SANS_BOLD, textColor=RED)


def part_sty() -> ParagraphStyle:
    return sty("part", fontSize=8.5, fontName=SANS_BOLD, textColor=AMBDR)


def warn_sty() -> ParagraphStyle:
    return sty("warn", fontSize=8.5, fontName=SANS_BOLD, textColor=RED)


def tag(text: str, col: colors.HexColor) -> str:
    """Inline coloured bold tag for classification labels in tables."""
    return f'<font color="{col.hexval()}"><b>{text}</b></font>'


# ── Box builders ──────────────────────────────────────────────────────
def mkbox(text: str, s: ParagraphStyle, bg, bc, width: float = 7.2) -> Table:
    t = Table([[Paragraph(text, s)]], colWidths=[width * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.8, bc),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return t


def pass_box(text: str) -> Table:
    return mkbox(text, pass_sty(), PASSBG, GREEN)


def fail_box(text: str) -> Table:
    return mkbox(text, fail_sty(), FAILBG, RED)


def part_box(text: str) -> Table:
    return mkbox(text, part_sty(), PARTBG, AMBDR)


def note_box(text: str) -> Table:
    return mkbox(text, sty("nb", fontSize=8.5, textColor=BLACK), NOTEBG, BLUE)


def warn_box(text: str) -> Table:
    return mkbox(text, warn_sty(), FAILBG, RED)


def teal_box(text: str) -> Table:
    return mkbox(text,
                 sty("tb", fontSize=8.5, fontName=SANS_BOLD, textColor=TEAL),
                 TEALBG, TEAL)


# ── Table builder ─────────────────────────────────────────────────────
def mk_table(headers: list, rows: list, widths: list,
             row_colors: list | None = None) -> Table:
    """Zebra-striped table with navy header row."""
    SM = small_sty()
    hrow = [Paragraph(h, sty("th", fontSize=7.5, fontName=SANS_BOLD, textColor=WHITE))
            for h in headers]
    drows = [[Paragraph(str(c), SM) for c in row] for row in rows]
    style = [
        ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTSIZE",    (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGREY]),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("GRID",        (0, 0), (-1, -1), 0.3, MGREY),
    ]
    if row_colors:
        for i, c in enumerate(row_colors):
            style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), c))
    t = Table([hrow] + drows, colWidths=widths)
    t.setStyle(TableStyle(style))
    return t


def ftbl(headers: list, rows: list, widths: list,
         sub_indices: set) -> Table:
    """Field-master table with spanning navy-background subheader rows."""
    SM    = small_sty()
    SH    = sty("sh", fontSize=7.5, fontName=SANS_BOLD, textColor=NAVY)
    ncols = len(headers)
    hrow  = [Paragraph(f"<b>{h}</b>", SM) for h in headers]
    data  = [hrow]
    style = [
        ("FONTSIZE",    (0, 0), (-1, -1), 7.5),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("LINEBELOW",   (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW",   (0,-1), (-1,-1), 0.4, MGREY),
    ]
    ri = 1
    for row_i, row in enumerate(rows):
        if row_i in sub_indices:
            cells = [Paragraph(f"<b>{row[0]}</b>", SH)]
            cells += [Paragraph("", SM)] * (ncols - 1)
            data.append(cells)
            style += [
                ("SPAN",       (0, ri), (-1, ri)),
                ("BACKGROUND", (0, ri), (-1, ri), ROWBLU),
                ("LINEBELOW",  (0, ri), (-1, ri), 0.3, MGREY),
                ("TOPPADDING", (0, ri), (-1, ri), 4),
                ("BOTTOMPADDING", (0, ri), (-1, ri), 4),
            ]
        else:
            cells = [Paragraph(str(c), SM) for c in row]
            while len(cells) < ncols:
                cells.append(Paragraph("", SM))
            data.append(cells)
            if ri % 2 == 0:
                style.append(("BACKGROUND", (0, ri), (-1, ri), LGREY))
        ri += 1
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle(style))
    return t


def itbl(headers: list, rows: list, widths: list) -> Table:
    """Simple info table (no spanning subheaders)."""
    SM   = small_sty()
    hrow = [Paragraph(f"<b>{h}</b>", SM) for h in headers]
    data = [hrow]
    style = [
        ("FONTSIZE",    (0, 0), (-1, -1), 7.5),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("LINEBELOW",   (0, 0), (-1, 0), 0.8, NAVY),
    ]
    for ri, row in enumerate(rows):
        data.append([Paragraph(str(c), SM) for c in row])
        if ri % 2 == 1:
            style.append(("BACKGROUND", (0, ri + 1), (-1, ri + 1), LGREY))
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle(style))
    return t


def divline(col=NAVY, thick: float = 0.6,
            before: int = 4, after: int = 3) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thick, color=col,
                      spaceBefore=before, spaceAfter=after)


def parse_rows(rows: list, ncols: int):
    """
    Split a mixed list of (subheader,) 1-tuples and normal row tuples
    into (flat_rows, sub_indices) for use with ftbl().
    """
    flat, sub = [], set()
    for ri, row in enumerate(rows):
        if len(row) == 1:
            sub.add(ri)
            flat.append([row[0]] + [""] * (ncols - 1))
        else:
            r = list(row)
            while len(r) < ncols:
                r.append("")
            flat.append(r)
    return flat, sub


def make_footer(title: str, date: str = "2026-09-15"):
    """Returns an onPage callback that draws a ruled footer."""
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(SANS, 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(0.55 * inch, 0.4 * inch, f"{title}  •  {date}")
        canvas.drawRightString(8.05 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.setStrokeColor(MGREY)
        canvas.setLineWidth(0.3)
        canvas.line(0.55 * inch, 0.5 * inch, 8.05 * inch, 0.5 * inch)
        canvas.restoreState()
    return _footer
