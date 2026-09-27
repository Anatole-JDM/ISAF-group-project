"""Render the deck to PDF with reportlab.

LibreOffice cannot rasterise in this environment, so the PDF is drawn directly
rather than converted from the .pptx. Both read scripts/deck_content.py, so the
two files carry identical text and identical numbers.

Fonts: DejaVu (shipped with matplotlib) rather than reportlab's built-in
Helvetica, whose WinAnsi encoding has no glyph for chi, delta, the minus sign or
Y-circumflex - all of which appear on these slides.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import deck_content as C

FIG = ROOT / "outputs" / "deck"
INCH = 72.0
PW, PH = 13.333 * INCH, 7.5 * INCH
M = 0.72 * INCH
CW = PW - 2 * M

INK = HexColor("#1A1A2E")
ACCENT = HexColor("#C0392B")
BLUE = HexColor("#2C7FB8")
GREY = HexColor("#8A8984")
PALE = HexColor("#F2F1EE")
WHITE = HexColor("#FFFFFF")
DIM = HexColor("#C9C8D4")

REG, BOLD, ITAL = "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"


def _register_fonts() -> None:
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont(REG, str(d / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD, str(d / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(ITAL, str(d / "DejaVuSans-Oblique.ttf")))


def Y(top: float) -> float:
    """Slides are laid out top-down; reportlab's origin is bottom-left."""
    return PH - top


def wrap(c, text, font, size, maxw):
    c.setFont(font, size)
    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= maxw or not line:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def draw_wrapped(c, text, x, top, maxw, font, size, color, leading=None):
    leading = leading or size * 1.32
    c.setFillColor(color)
    for i, line in enumerate(wrap(c, text, font, size, maxw)):
        c.setFont(font, size)
        c.drawString(x, Y(top + size + i * leading), line)
    return top + size + (len(wrap(c, text, font, size, maxw)) - 1) * leading


def rect(c, x, top, w, h, color):
    c.setFillColor(color)
    c.rect(x, Y(top + h), w, h, stroke=0, fill=1)


# ------------------------------------------------------------------- blocks
def draw_stats(c, stats, top, height=108.0):
    n = len(stats)
    gap = 0.22 * INCH
    bw = (CW - gap * (n - 1)) / n
    for i, (val, label) in enumerate(stats):
        x = M + i * (bw + gap)
        rect(c, x, top, bw, height, PALE)
        c.setFillColor(ACCENT if i == 2 and n == 3 else INK)
        c.setFont(BOLD, 27)
        c.drawCentredString(x + bw / 2, Y(top + 42), val)
        c.setFillColor(INK)
        yy = top + 62
        for ln in label.split("\n"):
            for sub in wrap(c, ln, REG, 10, bw - 16):
                c.setFont(REG, 10)
                c.drawCentredString(x + bw / 2, Y(yy), sub)
                yy += 13
    return top + height


def draw_bullets(c, items, top, size=15.0, width=None):
    width = width or CW
    y = top
    for it in items:
        if isinstance(it, tuple):
            head, body = it
            for ln in wrap(c, head, BOLD, size, width):
                c.setFillColor(INK); c.setFont(BOLD, size)
                c.drawString(M, Y(y + size), ln)
                y += size * 1.3
            y += 2
            for ln in wrap(c, body, REG, size - 2.5, width):
                c.setFillColor(GREY); c.setFont(REG, size - 2.5)
                c.drawString(M, Y(y + size - 2.5), ln)
                y += (size - 2.5) * 1.32
            y += 13
        else:
            c.setFillColor(ACCENT); c.setFont(BOLD, size)
            c.drawString(M, Y(y + size), "—")
            for j, ln in enumerate(wrap(c, it, REG, size, width - 18)):
                c.setFillColor(INK); c.setFont(REG, size)
                c.drawString(M + 18, Y(y + size), ln)
                y += size * 1.3
            y += 11
    return y


def draw_picture(c, name, top, height_in):
    img = ImageReader(str(FIG / name))
    iw, ih = img.getSize()
    h = height_in * INCH
    w = iw * h / ih
    if w > CW:
        w, h = CW, CW * ih / iw
    c.drawImage(img, (PW - w) / 2, Y(top + h), w, h, mask="auto")
    return top + h


def draw_table(c, spec, top, size=11.0):
    headers, rows = spec["headers"], spec["rows"]
    cw = spec.get("col_w") or [1] * len(headers)
    total = sum(cw)
    widths = [CW * v / total for v in cw]
    rh = 0.34 * INCH
    xs, x = [], M
    for w in widths:
        xs.append(x); x += w

    rect(c, M, top, CW, rh, INK)
    for j, h in enumerate(headers):
        c.setFillColor(WHITE); c.setFont(BOLD, size)
        if j:
            c.drawCentredString(xs[j] + widths[j] / 2, Y(top + rh / 2 + size * 0.36), h)
        else:
            c.drawString(xs[j] + 7, Y(top + rh / 2 + size * 0.36), h)

    y = top + rh
    hl_neg = spec.get("highlight_neg")
    hl_row = spec.get("highlight_row")
    for i, row in enumerate(rows):
        lines = max(len(wrap(c, str(v), REG, size, widths[j] - 12)) for j, v in enumerate(row))
        h = max(rh, lines * size * 1.3 + 10)
        rect(c, M, y, CW, h, WHITE if i % 2 == 0 else PALE)
        for j, v in enumerate(row):
            v = str(v)
            neg = hl_neg is not None and j == hl_neg and v.lstrip().startswith(("−", "-"))
            hot = neg or (hl_row is not None and i == hl_row and j >= 2) or v == "SIGNIFICANT"
            col = ACCENT if hot else INK
            fnt = BOLD if (hot or j == 0) else REG
            for k, ln in enumerate(wrap(c, v, fnt, size, widths[j] - 12)):
                c.setFillColor(col); c.setFont(fnt, size)
                yy = Y(y + (h - lines * size * 1.3) / 2 + size + k * size * 1.3)
                if j:
                    c.drawCentredString(xs[j] + widths[j] / 2, yy, ln)
                else:
                    c.drawString(xs[j] + 7, yy, ln)
        y += h
    return y


def draw_callout(c, text, top, color=ACCENT, height=None, size=13.5):
    lines = wrap(c, text, BOLD, size, CW - 34)
    height = height or max(0.95 * INCH, len(lines) * size * 1.34 + 18)
    rect(c, M, top, CW, height, PALE)
    rect(c, M, top, 4.5, height, color)
    y = top + (height - len(lines) * size * 1.34) / 2
    for ln in lines:
        c.setFillColor(INK); c.setFont(BOLD, size)
        c.drawString(M + 17, Y(y + size), ln)
        y += size * 1.34
    return top + height


def draw_caption(c, text, top, size=10.5, color=GREY):
    for ln in wrap(c, text, ITAL, size, CW):
        c.setFillColor(color); c.setFont(ITAL, size)
        c.drawString(M, Y(top + size), ln)
        top += size * 1.35
    return top


# -------------------------------------------------------------------- page
def cover(c, s):
    rect(c, 0, 0, PW, PH, INK)
    rect(c, M, 2.42 * INCH, 2.0 * INCH, 5, ACCENT)
    y = 2.75 * INCH
    for ln in s["headline"]:
        c.setFillColor(WHITE); c.setFont(BOLD, 34)
        c.drawString(M, Y(y + 34), ln)
        y += 46
    y += 14
    for ln in wrap(c, s["sub"], REG, 15, 10.6 * INCH):
        c.setFillColor(DIM); c.setFont(REG, 15)
        c.drawString(M, Y(y + 15), ln)
        y += 21
    y = PH / INCH * INCH - 1.5 * INCH
    for ln in s["foot"]:
        c.setFillColor(GREY); c.setFont(REG, 11)
        c.drawString(M, Y(y + 11), ln)
        y += 17


def page(c, s, num):
    if s.get("cover"):
        cover(c, s)
        return
    top = M
    if s.get("kicker"):
        c.setFillColor(ACCENT); c.setFont(BOLD, 10.5)
        c.drawString(M, Y(0.34 * INCH + 10.5), s["kicker"].upper())
        top = 0.72 * INCH
    if s.get("title"):
        c.setFillColor(INK); c.setFont(BOLD, 27)
        c.drawString(M, Y(top + 27), s["title"])
        rect(c, M, top + 0.82 * INCH, 1.5 * INCH, 3.2, ACCENT)
        top += 1.18 * INCH

    body = top
    if "stats" in s:
        body = draw_stats(c, s["stats"], body, height=1.42 * INCH if len(s["stats"]) == 3 else 108) + 20
    if "picture" in s:
        body = draw_picture(c, s["picture"], body, s.get("picture_h", 4.0)) + 10
    if "table" in s:
        body = draw_table(c, s["table"], body) + 14
    if "bullets" in s:
        body = draw_bullets(c, s["bullets"], body, size=s.get("bullet_size", 15.0)) + 6
    if "caption" in s:
        cap_top = body if "table" in s or "stats" in s else PH - 1.15 * INCH
        body = draw_caption(c, s["caption"], cap_top,
                            color=INK if "table" in s else GREY, size=10.5) + 8
    if "callout" in s:
        col = BLUE if s.get("callout_color") == "blue" else ACCENT
        h = (s["callout_h"] * INCH) if s.get("callout_h") else None
        lines = len(wrap(c, s["callout"], BOLD, 13.5, CW - 34))
        need = h or max(0.95 * INCH, lines * 13.5 * 1.34 + 18)
        draw_callout(c, s["callout"], PH - 0.45 * INCH - need, color=col, height=need)

    c.setFillColor(GREY); c.setFont(REG, 9.5)
    c.drawRightString(PW - M, Y(PH - 0.3 * INCH), str(num))


def build(dest: Path | None = None) -> Path:
    _register_fonts()
    dest = dest or (ROOT / "reports" / "ISAF_presentation.pdf")
    dest.parent.mkdir(exist_ok=True)
    c = canvas.Canvas(str(dest), pagesize=(PW, PH))
    c.setTitle("Should a police force let a model decide who gets searched?")
    c.setAuthor("HEC Paris MSc Data Science & AI for Business")
    for i, s in enumerate(C.SLIDES, start=1):
        page(c, s, i)
        c.showPage()
    c.save()
    print(f"wrote {dest.relative_to(ROOT)}  ({len(C.SLIDES)} pages)")
    return dest


if __name__ == "__main__":
    build()
