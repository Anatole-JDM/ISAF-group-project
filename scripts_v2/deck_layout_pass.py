"""Content-preserving layout pass over a generated .pptx.

The deck is laid out at absolute coordinates, so a text box sized for one line
silently collides with whatever sits beneath it when its text wraps to two. This
measures every text box with the deck's real font metrics and repairs the
geometry only - it never reads, rewrites, reorders or reflows a single character.

Two repairs, in this order:
  1. grow the box downward into space that is genuinely free, and
  2. failing that, step the font down, never below MIN_SCALE of its original size.

    python3 scripts_v2/deck_layout_pass.py reports/deck/ISAF_deck.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

EMU = 12700.0                  # EMU per point
PAD = 3.6                      # pptx inset, points
LEAD = 1.22                    # line height multiple
MIN_SCALE = 0.86               # never shrink text below this fraction
FOOT_Y = 6.5 * 72              # below this is the note/source/page-number band

MAC = Path("/System/Library/Fonts/Supplemental")
FONTS = {(0, 0): ("A", "Arial.ttf"), (1, 0): ("A-B", "Arial Bold.ttf"),
         (0, 1): ("A-I", "Arial Italic.ttf"), (1, 1): ("A-BI", "Arial Bold Italic.ttf")}


def _register():
    for (b, i), (name, f) in FONTS.items():
        p = MAC / f
        if p.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p)))
            except Exception:                                  # noqa: BLE001
                pass


def _fname(bold, italic):
    return FONTS[(int(bool(bold)), int(bool(italic)))][0]


def _lines(text, font, size, maxw):
    """How many wrapped lines this text needs in a box `maxw` points wide."""
    if maxw <= 2 or not text:
        return 1
    n, line = 1, ""
    for word in text.split():
        t = f"{line} {word}".strip()
        if pdfmetrics.stringWidth(t, font, size) <= maxw or not line:
            line = t
        else:
            n += 1
            line = word
    return n


def _needed(tf, w, scale=1.0):
    """Height in points this text frame needs at `scale` of its font sizes."""
    total = 0.0
    for p in tf.paragraphs:
        runs = [r for r in p.runs if r.text]
        if not runs:
            total += 10 * LEAD * scale
            continue
        size = next((r.font.size.pt for r in runs if r.font.size), 12) * scale
        f0 = runs[0].font
        font = _fname(f0.bold, f0.italic)
        text = "".join(r.text for r in runs)
        total += _lines(text, font, size, w - 2 * PAD) * size * LEAD
        if p.space_after is not None:
            total += p.space_after.pt * scale
        if p.space_before is not None:
            total += p.space_before.pt * scale
    return total


def _rects(slide):
    out = []
    for sh in slide.shapes:
        if sh.left is None or sh.top is None:
            continue
        out.append((sh, sh.left / EMU, sh.top / EMU,
                    (sh.width or 0) / EMU, (sh.height or 0) / EMU))
    return out


def _free_below(sh, x, y, w, h, rects):
    """Vertical room under this box before it would touch something else."""
    limit = FOOT_Y if y + h <= FOOT_Y else 7.5 * 72 - 4
    for other, ox, oy, ow, oh in rects:
        if other is sh or oy < y + h - 1:
            continue
        if min(x + w, ox + ow) - max(x, ox) > 8:           # horizontally overlapping
            limit = min(limit, oy - 2)
    return max(0.0, limit - (y + h))


def _band_below(sh, x, y, w, h, rects):
    """Everything sitting under this box inside its horizontal band.

    Whole band, not just the text: a bar and its label must move together or the
    chart desynchronises from what it is labelling.
    """
    out = []
    for other, ox, oy, ow, oh in rects:
        if other is sh or oy < y + h - 1:
            continue
        if oy >= FOOT_Y:                                   # note, source, page number
            continue                                       # page furniture never moves
        if min(x + w, ox + ow) - max(x, ox) > 8:
            out.append((other, ox, oy, ow, oh))
    return out


def _headroom(band, limit):
    """How far the whole band can slide down before it reaches `limit`."""
    if not band:
        return None
    return max(0.0, limit - max(oy + oh for _, _, oy, _, oh in band))


LABEL_MIN_SCALE = 0.78         # short labels may shrink further than body text


def _hspace(sh, x, y, w, h, rects):
    """Free room to the left and right of this box within its own row."""
    left, right = 0.0, 960.0
    for other, ox, oy, ow, oh in rects:
        if other is sh:
            continue
        if min(y + h, oy + oh) - max(y, oy) <= 4:          # not on the same row
            continue
        if ox + ow <= x:
            left = max(left, ox + ow)
        elif ox >= x + w:
            right = min(right, ox)
    return x - left, right - (x + w)


harmonised = [0]


def _fix_wide_labels(slide, rects, n):
    """A one-line label wider than its box reads as touching its neighbour.

    Widen symmetrically where the row has room; otherwise step the size down.
    Only for short single-paragraph labels - never body copy, which is meant to wrap.
    """
    widened = shrunk = 0
    resized = []
    for sh, x, y, w, h in rects:
        if not sh.has_text_frame or sh.has_table:
            continue
        t = sh.text_frame.text.strip()
        if not t or len(t) > 26 or "\n" in t or len(sh.text_frame.paragraphs) > 1:
            continue
        runs = [r for r in sh.text_frame.paragraphs[0].runs if r.text]
        if not runs:
            continue
        size = next((r.font.size.pt for r in runs if r.font.size), 12)
        font = _fname(runs[0].font.bold, runs[0].font.italic)
        need = pdfmetrics.stringWidth(t, font, size)
        avail = w - 2 * PAD
        if need <= avail:
            continue
        gl, gr = _hspace(sh, x, y, w, h, rects)
        grow = min(max(0.0, gl - 2) + max(0.0, gr - 2), need - avail + 2)
        if grow > 1:
            sh.left = Emu(int((x - min(max(0.0, gl - 2), grow / 2)) * EMU))
            sh.width = Emu(int((w + grow) * EMU))
            widened += 1
            avail += grow
        if need > avail:
            scale = 1.0
            while scale > LABEL_MIN_SCALE and pdfmetrics.stringWidth(t, font, size * scale) > avail:
                scale -= 0.02
            for r in sh.text_frame.paragraphs[0].runs:
                if r.font.size:
                    r.font.size = Pt(round(r.font.size.pt * scale, 1))
            shrunk += 1
            resized.append((sh, y, h, size, round(size * scale, 1)))
            print(f"  slide {n:>2}: label '{t}' {scale:.0%}")

    # A row of column headers must read as one row. If one of them had to come
    # down to fit, bring its row-mates to the same size rather than leaving a
    # single odd label - mixed sizes in a header row look worse than the clash did.
    for sh0, y0, h0, orig0, new0 in resized:
        for sh, x, y, w, h in rects:
            if sh is sh0 or not sh.has_text_frame or sh.has_table:
                continue
            if min(y0 + h0, y + h) - max(y0, y) <= 4:       # not the same row
                continue
            tt = sh.text_frame.text.strip()
            if not tt or len(tt) > 26 or len(sh.text_frame.paragraphs) > 1:
                continue
            for r in sh.text_frame.paragraphs[0].runs:
                if r.font.size and abs(r.font.size.pt - orig0) < 0.6:
                    r.font.size = Pt(new0)
                    harmonised[0] += 1
    return widened, shrunk


UNICODE_FONT = "Arial Unicode MS"


def _missing_from_arial():
    """Codepoints used in the deck that Arial has no glyph for."""
    try:
        from fontTools.ttLib import TTFont as _FT
        f = _FT(str(MAC / "Arial.ttf"))
        cmap = set()
        for t in f["cmap"].tables:
            cmap.update(t.cmap.keys())
        return cmap
    except Exception:                                      # noqa: BLE001
        return None


def _fix_missing_glyphs(prs):
    """A character with no glyph renders as a hollow box. Arial has no superscript
    minus, check mark or ballot X, all of which this deck uses. Re-point just those
    runs at Arial Unicode MS, which is metrically close and has them. The text is
    untouched - only the font that draws it."""
    cmap = _missing_from_arial()
    if not cmap:
        return 0
    n = 0

    def fix(tf):
        nonlocal n
        for para in tf.paragraphs:
            for r in para.runs:
                if any(ord(c) > 127 and ord(c) not in cmap for c in r.text):
                    r.font.name = UNICODE_FONT
                    n += 1

    for sl in prs.slides:
        for sh in sl.shapes:
            if sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        fix(cell.text_frame)
            elif sh.has_text_frame:
                fix(sh.text_frame)
    return n


def run(src: Path, dest: Path | None = None) -> Path:
    _register()
    prs = Presentation(str(src))
    grown = shrunk = pushed = 0
    wide_w = wide_s = 0
    for n, slide in enumerate(prs.slides, 1):
        rects = _rects(slide)
        a, b = _fix_wide_labels(slide, rects, n)
        wide_w += a; wide_s += b
        rects = _rects(slide)
        for sh, x, y, w, h in rects:
            if not sh.has_text_frame or not sh.text_frame.text.strip() or sh.has_table:
                continue
            if w < 12 or h < 6:
                continue
            need = _needed(sh.text_frame, w)
            if need <= h + 0.5:
                continue
            room = _free_below(sh, x, y, w, h, rects)
            give = min(room, need - h)
            if give > 0.5:
                sh.height = Emu(int((h + give) * EMU))
                grown += 1
                h += give
                if _needed(sh.text_frame, w) <= h + 0.5:
                    continue

            # Still short: slide the block underneath down, if it has somewhere to go.
            deficit = need - h
            band = _band_below(sh, x, y, w, h, rects)
            head = _headroom(band, FOOT_Y if y + h <= FOOT_Y else 7.5 * 72 - 4)
            if band and head is not None and head >= deficit - 0.5:
                for other, ox, oy, ow, oh in band:
                    other.top = Emu(int((oy + deficit) * EMU))
                sh.height = Emu(int((h + deficit) * EMU))
                pushed += 1
                h += deficit
                print(f"  slide {n:>2}: pushed {len(band)} shapes down {deficit:.0f}pt for "
                      f"'{sh.text_frame.text.strip().splitlines()[0][:46]}'")
                rects = _rects(slide)
                continue
            scale = 1.0
            while scale > MIN_SCALE and _needed(sh.text_frame, w, scale) > h + 0.5:
                scale -= 0.02
            if scale < 1.0:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size:
                            r.font.size = Pt(round(r.font.size.pt * scale, 1))
                shrunk += 1
                print(f"  slide {n:>2}: shrank to {scale:.0%}  "
                      f"'{sh.text_frame.text.strip().splitlines()[0][:52]}'")
    glyphs = _fix_missing_glyphs(prs)
    if glyphs:
        print(f"  {glyphs} runs re-pointed to {UNICODE_FONT} for glyphs Arial lacks "
              f"(superscript minus, check mark, ballot X)")

    dest = dest or src
    prs.save(str(dest))
    print(f"\n{grown} boxes grown, {pushed} blocks pushed down, {shrunk} text-size "
          f"reductions (floor {MIN_SCALE:.0%}); {wide_w} labels widened, {wide_s} "
          f"resized to fit their column, {harmonised[0]} row-mates matched to them. "
          f"No text changed.")
    print(f"wrote {dest}")
    return dest


if __name__ == "__main__":
    s = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/deck/ISAF_deck.pptx")
    run(s, Path(sys.argv[2]) if len(sys.argv) > 2 else None)
