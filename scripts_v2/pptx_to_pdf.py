"""Render a .pptx to .pdf without LibreOffice.

soffice cannot rasterise in this environment (it exits 0, or times out, having
written nothing - the macOS XPC graphics services it needs are unavailable), so
this walks the shape tree with python-pptx and draws it with reportlab. It is a
faithful-enough renderer for review and for handing the deck to someone who only
needs to read it: text boxes, auto shapes, lines, tables and pictures. A native
pptx CHART is drawn as a labelled placeholder - open the .pptx to see it.

    python3 scripts_v2/pptx_to_pdf.py reports/deck/ISAF_deck.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

EMU = 12700.0          # EMU per point
REG, BOLD, ITAL, BI = "DV", "DV-B", "DV-I", "DV-BI"
UNI = "UNI"          # a face that carries the glyphs Arial is missing
MAC = Path("/System/Library/Fonts/Supplemental")
_ARIAL_CMAP = set()


def _fonts() -> None:
    """Prefer the deck's real font. Arial and DejaVu have different metrics, so
    rendering Arial content with DejaVu wraps lines early and invents overlaps
    that do not exist in PowerPoint - which would send you optimising phantoms."""
    mac = MAC
    arial = ((REG, mac / "Arial.ttf"), (BOLD, mac / "Arial Bold.ttf"),
             (ITAL, mac / "Arial Italic.ttf"), (BI, mac / "Arial Bold Italic.ttf"))
    if all(f.exists() for _, f in arial):
        for name, f in arial:
            pdfmetrics.registerFont(TTFont(name, str(f)))
    else:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        print("  (Arial not found - falling back to DejaVu; wrapping will differ)")
        for name, f in ((REG, "DejaVuSans.ttf"), (BOLD, "DejaVuSans-Bold.ttf"),
                        (ITAL, "DejaVuSans-Oblique.ttf"), (BI, "DejaVuSans-BoldOblique.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(d / f)))
    pdfmetrics.registerFontFamily(REG, normal=REG, bold=BOLD, italic=ITAL, boldItalic=BI)
    for cand in (MAC / "Arial Unicode.ttf", Path("/Library/Fonts/Arial Unicode.ttf")):
        if cand.exists():
            try:
                pdfmetrics.registerFont(TTFont(UNI, str(cand)))
                from fontTools.ttLib import TTFont as _FT
                f = _FT(str(MAC / "Arial.ttf"))
                for t in f["cmap"].tables:
                    _ARIAL_CMAP.update(t.cmap.keys())
            except Exception:                              # noqa: BLE001
                pass
            break


def _pick(bold, italic, text=""):
    """Arial has no superscript minus, check mark or ballot X; a line using one is
    drawn with the unicode face so it does not come out as a hollow box."""
    if _ARIAL_CMAP and UNI in pdfmetrics.getRegisteredFontNames():
        if any(ord(c) > 127 and ord(c) not in _ARIAL_CMAP for c in text):
            return UNI
    return {(0, 0): REG, (1, 0): BOLD, (0, 1): ITAL, (1, 1): BI}[(int(bool(bold)), int(bool(italic)))]


def _rgb(c):
    """A shape/font colour, or None when it is inherited from the theme."""
    try:
        if c is None or c.type is None:
            return None
        return HexColor("#%02X%02X%02X" % (c.rgb[0], c.rgb[1], c.rgb[2]))
    except Exception:                                    # noqa: BLE001 - theme colours raise
        return None


def _fill(sh):
    try:
        if sh.fill.type is not None and str(sh.fill.type) != "MSO_FILL_TYPE.BACKGROUND (5)":
            return _rgb(sh.fill.fore_color)
    except Exception:                                    # noqa: BLE001
        pass
    return None


def _line(sh):
    try:
        col = _rgb(sh.line.color)
        w = sh.line.width.pt if sh.line.width else 0.75
        return (col, w) if col else (None, 0)
    except Exception:                                    # noqa: BLE001
        return (None, 0)


def _wrap(c, text, font, size, maxw):
    if maxw <= 2:
        return [text]
    out, line = [], ""
    for word in text.split():
        t = f"{line} {word}".strip()
        if c.stringWidth(t, font, size) <= maxw or not line:
            line = t
        else:
            out.append(line); line = word
    if line:
        out.append(line)
    return out or [""]


def _draw_text_frame(c, tf, x, top, w, h, PH):
    """Lay a text frame out top-down inside its box, honouring per-run styling."""
    pad = 3.6
    x += pad; w -= 2 * pad
    lines = []                                           # (text, font, size, colour, align)
    for p in tf.paragraphs:
        runs = [r for r in p.runs if r.text]
        if not runs:
            lines.append((None, None, 10, None, None)); continue
        size = next((r.font.size.pt for r in runs if r.font.size), 12)
        align = p.alignment
        # A paragraph is laid out as one styled block; mixed styling inside a
        # paragraph keeps the first run's font, which is enough for review.
        f0 = runs[0].font
        font = _pick(f0.bold, f0.italic, "".join(r.text for r in runs))
        col = _rgb(f0.color) or HexColor("#000000")
        text = "".join(r.text for r in runs)
        for ln in _wrap(c, text, font, size, w):
            lines.append((ln, font, size, col, align))
    total = sum(s * 1.22 for _, _, s, _, _ in lines)
    try:
        anchor = str(tf.vertical_anchor)
    except Exception:                                    # noqa: BLE001
        anchor = "None"
    y = top + (max(0.0, (h - total)) / 2 if "MIDDLE" in anchor else pad)
    for text, font, size, col, align in lines:
        if text is None:
            y += size * 1.22; continue
        c.setFont(font, size); c.setFillColor(col)
        yy = PH - (y + size)
        if align == PP_ALIGN.CENTER:
            c.drawCentredString(x + w / 2, yy, text)
        elif align == PP_ALIGN.RIGHT:
            c.drawRightString(x + w, yy, text)
        else:
            c.drawString(x, yy, text)
        y += size * 1.22


def _draw_table(c, sh, PH):
    tbl = sh.table
    x0, y0 = sh.left / EMU, sh.top / EMU
    widths = [col.width / EMU for col in tbl.columns]
    heights = [row.height / EMU for row in tbl.rows]
    y = y0
    for i, row in enumerate(tbl.rows):
        x = x0
        for j, cell in enumerate(row.cells):
            w, h = widths[j], heights[i]
            try:
                if cell.fill.type is not None and str(cell.fill.type) != "MSO_FILL_TYPE.BACKGROUND (5)":
                    col = _rgb(cell.fill.fore_color)
                    if col:
                        c.setFillColor(col); c.rect(x, PH - (y + h), w, h, stroke=0, fill=1)
            except Exception:                            # noqa: BLE001
                pass
            c.setStrokeColor(Color(0.82, 0.82, 0.82)); c.setLineWidth(0.4)
            c.rect(x, PH - (y + h), w, h, stroke=1, fill=0)
            _draw_text_frame(c, cell.text_frame, x, y, w, h, PH)
            x += w
        y += heights[i]


def _draw(c, shapes, PH):
    for sh in shapes:
        try:
            st = sh.shape_type
            if st == 6:                                   # GROUP
                _draw(c, sh.shapes, PH); continue
            if sh.left is None or sh.top is None:
                continue
            x, y = sh.left / EMU, sh.top / EMU
            w = (sh.width or 0) / EMU
            h = (sh.height or 0) / EMU

            if st == 13:                                  # PICTURE
                c.drawImage(ImageReader(__import__("io").BytesIO(sh.image.blob)),
                            x, PH - (y + h), w, h, mask="auto")
                continue
            if st == 19:                                  # TABLE
                _draw_table(c, sh, PH); continue
            if st == 3:                                   # CHART
                c.setStrokeColor(Color(0.7, 0.7, 0.7)); c.setLineWidth(0.8)
                c.rect(x, PH - (y + h), w, h, stroke=1, fill=0)
                c.setFont(ITAL, 9); c.setFillColor(Color(0.45, 0.45, 0.45))
                c.drawCentredString(x + w / 2, PH - (y + h / 2),
                                    "[native pptx chart - open the .pptx to view]")
                continue
            if st == 9:                                   # LINE
                col, lw = _line(sh)
                c.setStrokeColor(col or Color(0, 0, 0)); c.setLineWidth(max(lw, 0.5))
                c.line(x, PH - y, x + w, PH - (y + h))
                continue

            fill = _fill(sh)
            col, lw = _line(sh)
            if fill is not None or col is not None:
                if fill is not None:
                    c.setFillColor(fill)
                if col is not None:
                    c.setStrokeColor(col); c.setLineWidth(max(lw, 0.4))
                c.rect(x, PH - (y + h), w, h,
                       stroke=1 if col is not None else 0, fill=1 if fill is not None else 0)
            if sh.has_text_frame and sh.text_frame.text.strip():
                _draw_text_frame(c, sh.text_frame, x, y, w, h, PH)
        except Exception as e:                            # noqa: BLE001
            print(f"    (skipped a shape: {type(e).__name__}: {e})", file=sys.stderr)


def convert(src: Path, dest: Path | None = None) -> Path:
    _fonts()
    prs = Presentation(str(src))
    PW, PH = prs.slide_width / EMU, prs.slide_height / EMU
    dest = dest or src.with_suffix(".pdf")
    c = canvas.Canvas(str(dest), pagesize=(PW, PH))
    for s in prs.slides:
        c.setFillColor(HexColor("#FFFFFF")); c.rect(0, 0, PW, PH, stroke=0, fill=1)
        _draw(c, s.shapes, PH)
        c.showPage()
    c.save()
    print(f"wrote {dest}  ({len(prs.slides)} pages, {PW/72:.2f}x{PH/72:.2f} in)")
    return dest


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/deck/ISAF_deck.pptx")
    convert(src, Path(sys.argv[2]) if len(sys.argv) > 2 else None)
