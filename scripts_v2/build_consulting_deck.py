"""Build the consulting-style deck (15 main slides + 7 backups) as an editable pptx. Fits no model.

Design rules: .claude/skills/consulting-slides/SKILL.md. Storyline and numbers: MASTER.md, reports/deck/BLUEPRINT.md.
Numbers are read from data/*.json where a JSON exists; the others are quoted from MASTER.md, with the section
named next to them. Charts are drawn with native shapes (editable, exact alignment), except one native chart.

Input   data/matched_arms_diagnostics.json, data/officer_model_diagnostics.json,
        data/pltr_whitebox_report.json, data/pltr_terms.json
Output  reports/deck/ISAF_deck.pptx (or the path given as the first argument)
Needs   python-pptx  (pip install python-pptx)
"""
import json
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'reports' / 'deck' / 'ISAF_deck.pptx'
J = {k: json.loads((ROOT / 'data' / f'{k}.json').read_text()) for k in
     ['matched_arms_diagnostics', 'officer_model_diagnostics', 'pltr_whitebox_report', 'pltr_terms']}

FONT = 'Arial'
HEX = dict(main='1976D2', t1='5E9FE0', t2='98C1EB', t3='C6DDF4', panel='EEF3F6', rule='D9DEE3', ink='222222',
           note='6B7C85', white='FFFFFF', wb='1976D2', xgb='5E6B75', pfn='98C1EB', ref='B0B8BE',
           gW='4C6A92', gB='8C6BB1', gH='3A9A8A', accent='D97706')
C = {k: RGBColor.from_string(v) for k, v in HEX.items()}
DOT, DASH = MSO_LINE_DASH_STYLE.ROUND_DOT, MSO_LINE_DASH_STYLE.DASH
L, CW = 0.6, 12.133                                   # left margin and content width (inches)


def IN(v):
    return Emu(int(round(v * 914400)))


prs = Presentation()
prs.slide_width, prs.slide_height = IN(13.333), IN(7.5)
BLANK = prs.slide_layouts[6]
PAGE = [0]


# ---------------------------------------------------------------- primitives
def _fill_tf(tf, paras, size, color, bold, align, italic=False):
    """paras: str | list of paragraphs; a paragraph is str | list of runs | dict(runs=, bullet=, after=, size=, align=).
    A run is str or (text, dict(bold=, color=, size=, italic=))."""
    if isinstance(paras, (str, tuple)) or (isinstance(paras, dict)):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        opts = p if isinstance(p, dict) else {}
        runs = opts.get('runs', p) if isinstance(p, dict) else p
        if isinstance(runs, (str, tuple)):
            runs = [runs]
        para.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[opts.get('align', align)]
        if opts.get('after') is not None:
            para.space_after = Pt(opts['after'])
        if opts.get('before') is not None:
            para.space_before = Pt(opts['before'])
        for r in runs:
            text, st = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = text
            f = run.font
            f.name, f.size = FONT, Pt(st.get('size', opts.get('size', size)))
            f.bold = st.get('bold', opts.get('bold', bold))
            f.italic = st.get('italic', italic)
            f.color.rgb = C[st.get('color', opts.get('color', color))]
        if opts.get('bullet'):
            _bullet(para, opts.get('bullet_color'))


def _bullet(para, color=None, indent=0.17):
    pPr = para._p.get_or_add_pPr()
    pPr.set('marL', str(int(indent * 914400)))
    pPr.set('indent', str(-int(indent * 914400)))
    if color:
        clr = etree.SubElement(pPr, qn('a:buClr'))
        etree.SubElement(clr, qn('a:srgbClr')).set('val', HEX[color])
    etree.SubElement(pPr, qn('a:buChar')).set('char', '•')


def tb(sl, x, y, w, h, paras, size=12, color='ink', bold=False, align='l', anchor='t', italic=False):
    s = sl.shapes.add_textbox(IN(x), IN(y), IN(w), IN(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
    _fill_tf(tf, paras, size, color, bold, align, italic)
    return s


def box(sl, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE, dash=None):
    s = sl.shapes.add_shape(shape, IN(x), IN(y), IN(w), IN(h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = C[fill]
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = C[line]
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = dash
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def ln(sl, x1, y1, x2, y2, color='rule', w=1.0, dash=None):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, IN(x1), IN(y1), IN(x2), IN(y2))
    st = c._element.find(qn('p:style'))
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = C[color]
    c.line.width = Pt(w)
    if dash:
        c.line.dash_style = dash
    return c


def dot(sl, cx, cy, r, color, filled=True, shape=MSO_SHAPE.OVAL, lw=1.75):
    return box(sl, cx - r, cy - r, 2 * r, 2 * r, fill=color if filled else 'white', line=color, lw=lw, shape=shape)


def panel(sl, x, y, w, h):
    return box(sl, x, y, w, h, fill='panel')


def chart_title(sl, x, y, w, text, h=0.3):
    return tb(sl, x, y, w, h, text, size=12, bold=True)


def legend(sl, x, y, items, gap=0.25, size=10):
    """items: (label, color, kind) with kind in square | dot | ring | diamond."""
    for label, color, kind in items:
        if kind == 'square':
            box(sl, x, y + 0.04, 0.14, 0.14, fill=color)
        elif kind == 'diamond':
            dot(sl, x + 0.07, y + 0.11, 0.07, color, shape=MSO_SHAPE.DIAMOND, lw=1)
        else:
            dot(sl, x + 0.07, y + 0.11, 0.07, color, filled=(kind == 'dot'), lw=1.5)
        w = 0.075 * len(label) * size / 10 + 0.1
        tb(sl, x + 0.2, y, w, 0.22, label, size=size, color='ink')
        x += 0.2 + w + gap


# ---------------------------------------------------------------- tables
NO_STYLE = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'


def _borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
        for e in tcPr.findall(qn(tag)):
            tcPr.remove(e)
    for i, (tag, spec) in enumerate([('a:lnL', None), ('a:lnR', None), ('a:lnT', top), ('a:lnB', bottom)]):
        e = etree.Element(qn(tag))
        if spec is None:
            e.set('w', '0')
            etree.SubElement(e, qn('a:noFill'))
        else:
            e.set('w', str(int(spec[1] * 12700)))
            etree.SubElement(etree.SubElement(e, qn('a:solidFill')), qn('a:srgbClr')).set('val', HEX[spec[0]])
        tcPr.insert(i, e)


def table(sl, x, y, widths, rows, row_h=0.3, size=10.5, header=True, fills=None, aligns=None, bold_first_col=False,
          header_fills=None, header_colors=None, valign='m', header_h=0.4):
    """rows: list of lists of cell contents (same formats as tb paragraphs). fills: {(i, j): colour}."""
    gs = sl.shapes.add_table(len(rows), len(widths), IN(x), IN(y), IN(sum(widths)), IN(row_h * len(rows)))
    tbl = gs.table
    tblPr = tbl._tbl.tblPr
    sid = tblPr.find(qn('a:tableStyleId'))
    if sid is None:
        sid = etree.SubElement(tblPr, qn('a:tableStyleId'))
    sid.text = NO_STYLE
    tbl.first_row = tbl.horz_banding = False
    for j, w in enumerate(widths):
        tbl.columns[j].width = IN(w)
    for i in range(len(rows)):
        tbl.rows[i].height = IN(header_h if header and i == 0 else row_h)
    fills = fills or {}
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = IN(0.07)
            cell.margin_top = cell.margin_bottom = IN(0.04)
            cell.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE}[valign]
            is_head = header and i == 0
            color = (header_colors or {}).get(j, 'ink') if is_head else 'ink'
            al = (aligns or {}).get(j, 'l')
            tf = cell.text_frame
            tf.word_wrap = True
            if isinstance(val, list) and all(isinstance(e, (str, tuple)) for e in val):
                val = [val]                       # a list of runs is one paragraph
            _fill_tf(tf, val if val is not None else '', size, color, is_head or (bold_first_col and j == 0), al)
            f = (header_fills or {}).get(j) if is_head else fills.get((i, j), fills.get((None, j)))
            if f:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C[f]
            else:
                cell.fill.background()
            _borders(cell, bottom=('ink', 1.0) if is_head else ('rule', 0.75))
    return tbl


# ---------------------------------------------------------------- slide shell
def new_slide(section, title, source=None, note=None, notes=None, title_size=22):
    PAGE[0] += 1
    sl = prs.slides.add_slide(BLANK)
    tb(sl, L, 0.32, CW, 0.3, section, size=13, color='main', bold=True)
    tb(sl, L, 0.62, CW, 0.95, title, size=title_size, bold=True)
    if note:
        tb(sl, L, 6.6, CW, 0.4, note, size=9, color='note', anchor='b')
    if source:
        tb(sl, L, 7.08, CW - 0.7, 0.25, source, size=9, color='note')
    tb(sl, 12.233, 7.08, 0.5, 0.25, str(PAGE[0]), size=9, color='note', align='r')
    if notes:
        sl.notes_slide.notes_text_frame.text = notes
    return sl


def pct(v, d=1):
    return f'{v * 100:.{d}f}%'


def xs(v, lo, hi, x0, w):
    return x0 + (v - lo) / (hi - lo) * w


def axis(sl, lo, hi, ticks, x0, w, y, fmt, top=None, ref=None, ref_label=None, size=10):
    """Tick labels under a horizontal value axis; optional dotted reference line from `top` to y."""
    for t in ticks:
        tb(sl, xs(t, lo, hi, x0, w) - 0.4, y + 0.05, 0.8, 0.22, fmt(t), size=size, color='note', align='c')
    ln(sl, x0, y, x0 + w, y, 'rule', 0.75)
    if ref is not None:
        xr = xs(ref, lo, hi, x0, w)
        ln(sl, xr, top, xr, y, 'ref', 1.5, DOT)
        if ref_label:
            tb(sl, xr - 0.6, y + 0.27, 1.2, 0.22, ref_label, size=9, color='note', align='c')


# ================================================================= numbers from JSON
M = J['matched_arms_diagnostics']['arms']
D = J['officer_model_diagnostics']
F = D['F_within_officer_auc']
CF = D['D_officer_counterfactual']
G = D['G_first_stop_check']
WB = J['pltr_whitebox_report']
FP = WB['from_predictions']
LOC = WB['local_explanations']
TERMS = J['pltr_terms']['temporal/full_time_officer/pltr_sparse']
RACES = ['white', 'black', 'hispanic']
GCOL = {'white': 'gW', 'black': 'gB', 'hispanic': 'gH'}
B = {'bold': True}

# ================================================================= 1 · Title
PAGE[0] += 1
s = prs.slides.add_slide(BLANK)
tb(s, L, 1.55, 7.6, 0.3, 'HEC Paris · ISAF group project · Client: Metro Nashville Police Department (role play)',
   size=13, color='main', bold=True)
tb(s, L, 2.0, 7.6, 1.9, 'Can a model tell an officer whom to search?', size=40, bold=True)
tb(s, L, 3.95, 7.4, 0.9, 'Scoring consent searches in Nashville: three models, four tests, one recommendation',
   size=18, color='note')
tb(s, L, 6.55, 7.6, 0.5, ['Data: Stanford Open Policing Project, Nashville traffic stops 2010–2018',
                          '28 September 2026 · Team: xx'], size=10, color='note')
# the evaluation frame, previewing the deck's structure
gx, gy = 8.3, 2.05
tb(s, gx, gy, 4.0, 0.3, 'How we evaluate each model', size=12, bold=True)
dims = ['Performance', 'Interpret-ability', 'Stability', 'Fairness']
for k, dname in enumerate(dims):
    tb(s, gx + 1.3 + k * 0.82, gy + 0.45, 0.78, 0.45, dname, size=9.5, color='note', align='c', anchor='b')
for r, (mname, col) in enumerate([('White box', 'wb'), ('XGBoost', 'xgb'), ('TabPFN', 'pfn')]):
    yy = gy + 1.0 + r * 0.62
    dot(s, gx + 0.1, yy + 0.25, 0.08, col)
    tb(s, gx + 0.28, yy + 0.13, 1.1, 0.25, mname, size=11, bold=True)
    for k in range(4):
        box(s, gx + 1.34 + k * 0.82, yy + 0.02, 0.7, 0.46, fill='panel')
tb(s, gx, gy + 3.0, 4.0, 0.9, ['Then: the trade-offs, and which model to deploy',
                               'under trustworthy-AI requirements.'], size=10.5, color='note')
s.notes_slide.notes_text_frame.text = (
    "MNPD asked us a simple question: can a scoring model tell officers which traffic stops are worth a consent "
    "search? We built the score three ways, a white box, XGBoost and a tabular foundation model, and tested each on "
    "performance, interpretability, stability and fairness. The answer is not the one the client expected, and "
    "that is what makes it useful.")

# ================================================================= 3 · Client question
s = new_slide('The client\'s question',
              'Consent searches are MNPD\'s most discretionary searches, and 83% of them find nothing',
              source='Source: Stanford Open Policing Project, Nashville (Pierson et al., Nature Human Behaviour 2020), '
                     'ODC-By; our cleaning. MASTER §1.2, §2.',
              note='Consent search = consent recorded and no arrest, warrant, inventory or plain-view authority. '
                   'Race as perceived by the officer. MNPD is a role-play client.',
              notes="Nashville recorded 3.08 million traffic stops from 2010 to 2018. About 4% led to a search, and "
                    "almost half of those were consent searches: no warrant, no probable cause, the driver agrees. "
                    "83% of them found nothing. Each fruitless search costs officer time, the driver's time and "
                    "dignity, and community trust, and MNPD's searches have documented racial disparities. Our "
                    "success criteria: fewer fruitless searches, finds preserved, no discrimination, and a tool "
                    "explainable to a driver, a court and the oversight board.")
chart_title(s, L, 1.7, 7.6, 'Nashville traffic stops to consent searches, 2010–2018')
for k, (num, lab, col) in enumerate([('3.08 M', 'traffic stops', 'ink'),
                                     ('127,121', 'searches · 4.1% of stops', 'ink'),
                                     ('58,865', 'consent searches · 46% of searches', 'main')]):
    x0 = L + k * 2.6
    tb(s, x0, 2.05, 2.2, 0.6, num, size=30, bold=True, color=col)
    tb(s, x0, 2.7, 2.3, 0.3, lab, size=11, color='note')
    if k < 2:
        tb(s, x0 + 2.05, 2.1, 0.5, 0.5, '→', size=24, color='ref', align='c')
chart_title(s, L, 3.35, 7.6, 'Outcome of the 58,865 consent searches')
bw = 7.6
box(s, L, 3.72, bw * 0.169, 0.62, fill='main')
box(s, L + bw * 0.169, 3.72, bw * 0.831, 0.62, fill='t3')
tb(s, L, 3.72, bw * 0.169, 0.62, '16.9%', size=13, bold=True, color='white', align='c', anchor='m')
tb(s, L + bw * 0.169 + 0.15, 3.72, 4, 0.62, [[('83.1% ', B), 'found nothing']], size=13, anchor='m')
tb(s, L, 4.38, 3, 0.25, 'found contraband', size=10.5, color='main', bold=True)
chart_title(s, L, 4.85, 7.6, 'Drivers searched with consent, by race')
xx = L
for lab, share, col in [('Black', 0.549, 'gB'), ('White', 0.361, 'gW'), ('Hispanic', 0.079, 'gH'),
                        ('other', 0.012, 'ref')]:
    box(s, xx, 5.22, bw * share, 0.62, fill=col)
    if share > 0.3:
        tb(s, xx + 0.12, 5.22, bw * share, 0.62, [[(f'{lab} ', {}), (f'{share * 100:.1f}%', B)]], size=12,
           color='white', anchor='m')
    elif share > 0.05:
        tb(s, xx, 5.22, bw * share, 0.62, f'{share * 100:.1f}%', size=10, bold=True, color='white', align='c',
           anchor='m')
    xx += bw * share
tb(s, L + bw - 4, 5.88, 4, 0.25, 'Hispanic 7.9% · other or unknown 1.2%', size=10.5, color='note', align='r')
panel(s, 8.55, 1.7, 4.183, 4.85)
tb(s, 8.8, 1.9, 3.7, 4.5, [
    {'runs': 'The client\'s question', 'bold': True, 'color': 'main', 'size': 12, 'after': 6},
    {'runs': '"Can a scoring model tell our officers which stops are worth a consent request?"', 'bold': True,
     'size': 14, 'after': 8},
    {'runs': 'No warrant, no probable cause: the driver agrees. Each fruitless search costs officer time, the '
             'driver\'s time and dignity, and community trust.', 'size': 11, 'after': 14},
    {'runs': 'Success criteria (ours to state, theirs to accept)', 'bold': True, 'color': 'main', 'size': 12,
     'after': 6},
    {'runs': [('1  ', B), 'Fewer fruitless searches'], 'size': 11, 'after': 4},
    {'runs': [('2  ', B), 'Finds preserved'], 'size': 11, 'after': 4},
    {'runs': [('3  ', B), 'No discrimination'], 'size': 11, 'after': 4},
    {'runs': [('4  ', B), 'Explainable to a driver, a court and the oversight board'], 'size': 11}])

# ================================================================= 4 · Scope
s = new_slide('Scope and set-up',
              'Pooling all searches would reverse the disparity, so we score consent searches only',
              source='Source: FINDINGS.md §1; MASTER §2. Illinois and North Carolina: same test on Open Policing data.',
              note='Legal-basis table from the earlier cleaning (58,836 consent searches); our sample is 58,865. '
                   'Excluding plain-view and arrest cases, the consent gap is −5.16 pp (−4.60 pp with them).',
              notes="Why consent searches only? The outcome test, comparing hit rates by race, depends on the legal "
                    "basis. Where the officer judges the driver, probable cause or consent, searches of Black "
                    "drivers find less: minus 16 and minus 5 points. Mechanical searches after an arrest don't. "
                    "Pooled together, the sign flips: you would conclude, wrongly, that Black drivers are searched on "
                    "better evidence. We train on 2010 to 2015 and test on 2016 to 2018, with 20% of officers never "
                    "seen in training. And one fact shapes everything: we only observe contraband when an officer "
                    "chose to search.")
chart_title(s, L, 1.7, 6.4, 'Hit-rate gap, Black minus white drivers, by legal basis (pp)')
lo, hi, px0, pw = -20, 5, 2.95, 3.9
zx = xs(0, lo, hi, px0, pw)
for k, (lab, v, col, extra) in enumerate([('Probable cause', -15.9, 'ref', ''),
                                          ('Consent (our sample)', -5.2, 'main', ''),
                                          ('Arrest, warrant, inventory', 1.8, 'ref', ''),
                                          ('All searches pooled', 0.8, 'ref', '  wrong sign')]):
    y = 2.2 + k * 0.72
    tb(s, L, y, 2.3, 0.42, lab, size=11.5, bold=(col == 'main'), color='main' if col == 'main' else 'ink',
       anchor='m')
    x1 = xs(v, lo, hi, px0, pw)
    box(s, min(x1, zx), y, abs(x1 - zx), 0.42, fill=col)
    lbl = f'{v:+.1f}'.replace('-', '−')
    if v < 0:
        tb(s, x1 - 0.75, y, 0.7, 0.42, lbl, size=11.5, bold=True, align='r', anchor='m',
           color='main' if col == 'main' else 'ink')
    else:
        tb(s, x1 + 0.06, y, 1.6, 0.42, [[(lbl, B), (extra, {'color': 'note', 'size': 10})]], size=11.5,
           anchor='m')
ln(s, zx, 2.08, zx, 4.85, 'ink', 1.0)
tb(s, L, 5.1, 6.3, 1.3, 'Below zero: consent searches of Black drivers find contraband less often, consistent with '
                        'a lower bar for searching them. Discretionary searches show it, mechanical ones do not, and '
                        'pooling hides it. The same pattern appears in Illinois (−7.3 pp) and North Carolina '
                        '(−6.3 pp).', size=11)
rx, rw = 7.45, 5.283
chart_title(s, rx, 1.7, rw, 'How we test the models')
yw = rw / 9
box(s, rx, 2.15, 6 * yw, 0.95, fill='t3')
box(s, rx + 6 * yw, 2.15, 3 * yw, 0.95, fill='main')
tb(s, rx + 0.12, 2.15, 6 * yw - 0.2, 0.95, [{'runs': 'Train 2010–2015', 'bold': True, 'size': 12},
                                           '49,752 searches · 16.1% hit'], size=11, anchor='m')
tb(s, rx + 6 * yw + 0.1, 2.15, 3 * yw - 0.15, 0.95, [{'runs': 'Test 2016–2018', 'bold': True, 'size': 12},
                                                    '9,113 · 21.4% hit'], size=11, color='white', anchor='m')
for yr, xx in [('2010', rx), ('2016', rx + 6 * yw), ('2018', rx + rw)]:
    tb(s, xx - 0.3, 3.13, 0.6, 0.22, yr, size=9.5, color='note', align='c')
tb(s, rx, 3.5, rw, 1.3, [
    {'runs': [('Temporal split: ', B), 'no learning from the future, as in deployment. Consent searches fell from '
                                       '~9,000 a year (2011–2014) to 2,167 in 2018 as the hit rate rose.'],
     'bullet': True, 'after': 6},
    {'runs': [('Officer holdout: ', B), '20% of officers never seen in training.'], 'bullet': True}], size=11)
panel(s, rx, 4.95, rw, 1.2)
tb(s, rx + 0.2, 5.07, rw - 0.4, 1.0, [
    {'runs': 'Selective labels shape everything', 'bold': True, 'color': 'main', 'size': 12, 'after': 4},
    'Contraband is observed only when an officer chose to search. Every model learns from officers\' decisions, '
    'and every number is conditional on a search having been made.'], size=11)

# ================================================================= 5 · Performance
s = new_slide('Act I · The task as posed · Performance',
              'All three models rank searches barely better than chance, and the best adds 82 finds per 2,000 '
              'searches',
              source='Source: src, outputs/metrics__consent__matched.json and __full.json (release), DeLong tests; '
                     'TabPFN on all rows: hosted inference, FINDINGS.md §10. MASTER §4.',
              note='*Break-even cost ratio = hit rate / (1 − hit rate) of the searches kept: the cost of searching an '
                   'innocent driver, as a fraction of the value of a find, above which keeping the top 2,000 destroys '
                   'value. The price is a policy judgement: we show it, we do not pick it.',
              notes="We train on the 49,752 consent searches of 2010 to 2015 and test on the 9,113 of 2016 to 2018. "
                    "For a fair comparison, all three models are trained on the same random 2,000 training searches, "
                    "the most TabPFN could run on CPU; the right-hand column shows each model on all 49,752 rows. "
                    "AUCs are 0.53 to 0.55, next to the 0.50 of a coin flip, and none beats a constant "
                    "prediction on the Brier score. XGBoost and TabPFN are statistically tied; the white box is 0.015 "
                    "to 0.020 below. Economically, keeping the 2,000 top-scored searches, the best model finds 509 "
                    "where random ranking finds 427: 82 more finds, for 1,491 innocent drivers searched. With all the "
                    "training data the AUCs rise only to 0.55 to 0.57: it is the information, not the model. Note "
                    "the two different 2,000s: 2,000 training rows, and a budget of 2,000 searches out of 9,113.")
chart_title(s, L, 1.7, 5.9, 'Test AUC on the 9,113 consent searches of 2016–2018')
tb(s, L, 2.0, 5.9, 0.45, 'Dots: all three trained on the same random 2,000 of the 49,752 searches of 2010–2015 (the '
                         'most TabPFN could run on CPU), so only the model differs', size=10, color='note')
lo, hi, px0, pw = 0.40, 0.70, 2.4, 2.45
tb(s, 5.6, 2.5, 0.8, 0.42, 'AUC, all 49,752 rows', size=9.5, color='note', bold=True, align='c', anchor='b')
tb(s, 4.95, 2.5, 0.65, 0.42, 'Brier', size=9.5, color='note', bold=True, align='c', anchor='b')
arms = [('White box (logistic)', M['scorecard']['auc'], 'wb', '0.172', '0.552'),
        ('XGBoost', M['gbm']['auc'], 'xgb', '0.182', '0.566'),
        ('TabPFN', M['tabpfn']['auc'], 'pfn', '0.169', '0.573'),
        ('Constant prediction', 0.5, 'ref', '0.168', '0.500')]
for k, (lab, v, col, brier, full) in enumerate(arms):
    y = 3.15 + k * 0.53
    tb(s, L, y - 0.15, 1.9, 0.3, lab, size=11.5, bold=(k < 3), anchor='m')
    ln(s, px0, y, px0 + pw, y, 'rule', 0.5)
    dot(s, xs(v, lo, hi, px0, pw), y, 0.1, col)
    tb(s, xs(v, lo, hi, px0, pw) + 0.14, y - 0.15, 0.6, 0.3, f'{v:.3f}', size=11, bold=True, anchor='m')
    tb(s, 5.6, y - 0.15, 0.8, 0.3, full, size=11, align='c', anchor='m')
    tb(s, 4.95, y - 0.15, 0.65, 0.3, brier, size=11, bold=(k == 3), align='c', anchor='m')
axis(s, lo, hi, [0.4, 0.5, 0.6, 0.7], px0, pw, 5.05, lambda t: f'{t:.2f}', top=2.95, ref=0.5, ref_label='random')
tb(s, L, 5.68, 5.9, 0.85, [
    {'runs': [('DeLong: ', B), 'XGBoost vs TabPFN −0.005 (p = 0.41), tied. The white box is below both (−0.015, '
                               'p = 0.03; −0.020, p < 0.001).'], 'after': 3},
    {'runs': [('Brier: ', B), 'none beats a constant prediction (0.168). With all 49,752 rows, AUC rises by only '
                              '0.02 and the ranking is unchanged.']}], size=10)
rx = 6.9
chart_title(s, rx, 1.7, 5.8, 'Finds if MNPD searched only the 2,000 top-scored of the 9,113 test stops')
tb(s, rx, 2.0, 5.8, 0.25, 'A search budget of ≈ 22%; same models as on the left (2,000 training rows)', size=10, color='note')
base, top_, vmax = 4.95, 2.35, 600
cols = [('Random', 427, 'ref', '1,573', '', '0.27'), ('White box', 460, 'wb', '1,540', '+33', '0.30'),
        ('XGBoost', 501, 'xgb', '1,499', '+74', '0.33'), ('TabPFN', 509, 'pfn', '1,491', '+82', '0.34')]
cx0, pitch, bwid = 8.75, 1.2, 0.7
ln(s, 8.2, base, 12.73, base, 'rule', 0.75)
yr = base - 427 / vmax * (base - top_)
ln(s, 8.2, yr, 12.73, yr, 'ref', 1.25, DOT)
for k, (lab, v, col, fruitless, extra, be) in enumerate(cols):
    cx = cx0 + k * pitch
    h = v / vmax * (base - top_)
    box(s, cx - bwid / 2, base - h, bwid, h, fill=col)
    tb(s, cx - 0.5, base - h - 0.3, 1.0, 0.26, str(v), size=12, bold=True, align='c')
    tb(s, cx - 0.6, base + 0.07, 1.2, 0.25, lab, size=11, bold=True, align='c')
    tb(s, cx - 0.6, base + 0.42, 1.2, 0.25, fruitless, size=11, align='c')
    tb(s, cx - 0.6, base + 0.72, 1.2, 0.25, extra or '·', size=11, bold=True, align='c',
       color='main' if extra else 'note')
    tb(s, cx - 0.6, base + 1.02, 1.2, 0.25, be, size=11, align='c')
for yy, lab in [(0.42, 'Fruitless'), (0.72, 'Extra finds'), (1.02, 'Break-even*')]:
    tb(s, rx, base + yy, 1.3, 0.25, lab, size=10, color='note', bold=True)
tb(s, rx, yr - 0.24, 1.3, 0.22, 'random: 427', size=9.5, color='note')

# ================================================================= 6 · Interpretability
s = new_slide('Act I · Interpretability',
              'What little the models learn is mostly the driver\'s race, and only the white box explains it exactly',
              source='Source: FINDINGS.md (XPER, race-proxy AUC, permutation importance); MASTER §4, §6.1–6.3; '
                     'surrogate: reports/xgboost_officer (Julia).',
              note='XPER: the course\'s Shapley decomposition of AUC above 0.5, exact over 1,024 coalitions. Importances: '
                   'permutation, test set. The surrogate tree was fitted on the Act II XGBoost.',
              notes="What do the models learn? The same stop and driver features predict the driver's race at AUC 0.69 "
                    "to 0.73, far better than they predict contraband. In the white box, race is 73.5% of the "
                    "above-chance signal; its largest coefficient is 'driver Hispanic'. TabPFN's largest feature is "
                    "race, fifteen times its weight in XGBoost. And explanations: the white box is its own "
                    "explanation; XGBoost has exact SHAP; TabPFN only has approximations. XGBoost and TabPFN reach "
                    "the same AUC for opposite reasons: their importance rankings correlate at minus 0.32.")
chart_title(s, L, 1.7, 4.8, 'What the same stop and driver features predict (test AUC)', h=0.45)
lo, hi, px0, pw = 0.5, 0.8, 2.25, 2.8
for k, (lab, a, b, col) in enumerate([('Contraband found', 0.55, 0.57, 'ref'), ('The driver\'s race', 0.69, 0.73,
                                                                                'main')]):
    y = 2.4 + k * 0.72
    tb(s, L, y, 1.6, 0.42, lab, size=11.5, bold=True, anchor='m', color='main' if col == 'main' else 'ink')
    box(s, px0, y + 0.08, xs(a, lo, hi, px0, pw) - px0, 0.26, fill='t3' if col == 'main' else 'panel')
    box(s, xs(a, lo, hi, px0, pw), y, xs(b, lo, hi, px0, pw) - xs(a, lo, hi, px0, pw), 0.42, fill=col)
    tb(s, xs(b, lo, hi, px0, pw) + 0.06, y, 0.9, 0.42, f'{a:.2f}–{b:.2f}', size=11, bold=True, anchor='m')
axis(s, lo, hi, [0.5, 0.6, 0.7, 0.8], px0, pw, 3.7, lambda t: f'{t:.1f}', top=2.3, ref=0.5, ref_label='random')
tb(s, L, 4.35, 1.75, 0.7, '73.5%', size=34, bold=True, color='main')
tb(s, 2.4, 4.37, 3.0, 0.9, 'of the white box\'s above-chance AUC is the driver\'s race (XPER)', size=11.5,
   bold=True)
tb(s, L, 5.35, 4.8, 1.0, 'Dropping race does not remove it: the remaining features recover race at AUC 0.69–0.73. '
                         'The white box\'s largest coefficient is "driver Hispanic" (odds ratio 0.43, −11 pp).',
   size=11)
cards = [('White box', 'wb', 'Driver Hispanic is the largest coefficient; race is 73.5% of the signal',
          'Exact: coefficient × (x − mean)', 'The model is its own explanation'),
         ('XGBoost', 'xgb', 'Flat: precinct, zone, month, age, hour ≈ 0.01 each; race 0.0025',
          'Exact TreeSHAP', 'SHAP; a depth-3 surrogate tree reproduces only 39% of the score'),
         ('TabPFN', 'pfn', 'Race is its largest feature (0.038, 15× its weight in XGBoost)',
          'Occlusion only: approximate, error cannot be bounded', 'Permutation importance; no native attribution')]
for k, (name, col, drv, one, glob) in enumerate(cards):
    x0 = 5.75 + k * 2.36
    panel(s, x0, 1.7, 2.25, 3.95)
    dot(s, x0 + 0.27, 1.98, 0.09, col)
    tb(s, x0 + 0.45, 1.83, 1.7, 0.3, name, size=13, bold=True)
    tb(s, x0 + 0.18, 2.35, 1.93, 3.6, [
        {'runs': 'MAIN DRIVER', 'size': 9, 'bold': True, 'color': 'note', 'after': 2},
        {'runs': drv, 'after': 12},
        {'runs': 'ONE DECISION EXPLAINED BY', 'size': 9, 'bold': True, 'color': 'note', 'after': 2},
        {'runs': one, 'after': 12},
        {'runs': 'GLOBAL EXPLANATION', 'size': 9, 'bold': True, 'color': 'note', 'after': 2},
        {'runs': glob}], size=11.5)
tb(s, 5.75, 5.85, 6.98, 0.45, [[('Same accuracy, contradictory explanations: ', B),
                                'XGBoost\'s and TabPFN\'s importance rankings correlate at −0.32.']], size=11)

# ================================================================= 7 · Stability
s = new_slide('Act I · Stability',
              'Retrain the model on a slightly different sample of the same years, and its shortlist barely '
              'survives',
              source='Source: FINDINGS.md; MASTER §2, §4, §6.3.',
              notes="One more problem before we get to what actually matters: none of these models is stable. If we "
                    "retrain on a slightly different random sample of the same years, about 93% of the 200 "
                    "highest-scored searches change. A stop flagged as high-risk today might not be flagged tomorrow, "
                    "for the exact same driver and officer, just because the training data shifted a little. TabPFN "
                    "is the most sensitive: which historical searches happen to be in its memory moves its score by "
                    "more than the entire gap between all three models. A model whose shortlist changes every time "
                    "you retrain it is not something officers or a court can rely on.")
tb(s, L, 2.3, 4.4, 2.2, '93%', size=100, bold=True, color='main')
tb(s, L, 4.35, 4.6, 1.7, 'of the 200 highest-scored searches change identity when we retrain on a different random '
                         'sample of the same years', size=15, bold=True)
panel(s, 5.55, 2.0, 6.983, 4.4)
tb(s, 5.77, 2.15, 6.5, 4.1, [
    {'runs': 'Same driver, same officer, same years of data — just a different random split — and the list '
             'of "high-risk" stops is almost a different list.', 'after': 14},
    {'runs': 'Top-200 overlap is only 2.3× what pure chance would give: barely better than picking names out of '
             'a hat.', 'bullet': True, 'after': 8},
    {'runs': 'TabPFN is the most sensitive of the three: which past searches happen to be in its memory moves its '
             'own score by more than the entire gap between all three models.', 'bullet': True, 'after': 20},
    {'runs': 'A shortlist that changes every time you retrain it is not something officers, or a court, can rely '
             'on.', 'bold': True, 'color': 'main'}], size=12.5)

# ================================================================= 8 · Officer record: gain and limit
s = new_slide('Act II · Where search outcomes come from',
              'The officer\'s past record lifts every model by 0.07, yet inside one officer\'s own searches the '
              'scores fall back to chance',
              source='Source: data/officer_model_diagnostics.json §F, data/officer_model_results.json; TabPFN: src, '
                     'run interactively (release). MASTER §5.2–5.3.',
              note='Within-officer AUC: AUC inside each officer\'s own test searches, 209 officers with ≥ 10 searches '
                   'and both outcomes (7,005 searches), weighted by searches; CI resamples officers. The record uses '
                   'only events strictly before the stop (verified on 400 stops).',
              notes="Why does the task fail? Contraband is only seen when an officer chose to search, so we gave the "
                    "models the searching officer's own past record. Every family gains 0.07 AUC, on officers never "
                    "seen in training, in every year. But an officer decides among their own stops. Measured inside "
                    "each officer's own searches, the white box with the record is at 0.50, a coin flip, and XGBoost "
                    "at 0.555. The record tells you which officer, never which driver. So we use it to explain where "
                    "outcomes come from, not as an input to deploy.")
chart_title(s, L, 1.7, 7.7, 'Pooled AUC vs AUC inside each officer\'s own searches, test 2016–2018 (95% CI)')
legend(s, L, 2.12, [('pooled across all officers', 'ref', 'diamond'),
                    ('inside each officer\'s own searches', 'ink', 'dot')], size=10)
lo, hi, px0, pw = 0.40, 0.70, 3.55, 4.4
rows8 = [('White box, stop & driver only', F['full_time/scorecard'], 'wb', False),
         ('White box (PLTR, 19 rules) + officer record', F['full_time_officer/pltr_sparse'], 'wb', True),
         ('XGBoost, stop & driver only', F['full_time/xgboost'], 'xgb', False),
         ('XGBoost + officer record', F['full_time_officer/xgboost'], 'xgb', True),
         ('TabPFN + officer record (all 25 features)', {'pooled_auc': 0.653}, 'pfn', True),
         ('Officer record only (logistic)', F['officer_only/scorecard'], 'note', True)]
for k, (lab, r, col, filled) in enumerate(rows8):
    y = 2.78 + k * 0.58
    if k in (2, 4, 5):
        ln(s, L, y - 0.29, 7.95, y - 0.29, 'rule', 0.5)
    tb(s, L, y - 0.24, 2.85, 0.48, lab, size=11, bold=filled and k != 5, anchor='m')
    xp = xs(r['pooled_auc'], lo, hi, px0, pw)
    if 'within_officer_auc' in r:
        xw = xs(r['within_officer_auc'], lo, hi, px0, pw)
        c_lo, c_hi = r['ci95_resampling_officers']
        ln(s, xp, y, xw, y, 'ref', 1.0, DASH)
        ln(s, xs(c_lo, lo, hi, px0, pw), y, xs(c_hi, lo, hi, px0, pw), y, col, 2.25)
        dot(s, xw, y, 0.095, col, filled=filled)
        tb(s, xs(c_lo, lo, hi, px0, pw) - 0.62, y - 0.14, 0.56, 0.28, f'{r["within_officer_auc"]:.3f}', size=11,
           bold=True, align='r', anchor='m')
    else:
        tb(s, xs(0.51, lo, hi, px0, pw), y - 0.14, 2.0, 0.28, 'within officer: not computed', size=9.5,
           color='note', anchor='m', italic=True)
    dot(s, xp, y, 0.08, 'ref', shape=MSO_SHAPE.DIAMOND, lw=1)
    tb(s, xp + 0.12, y - 0.14, 0.6, 0.28, f'{r["pooled_auc"]:.3f}', size=10, color='note', anchor='m')
axis(s, lo, hi, [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], px0, pw, 6.1, lambda t: f'{t:.2f}', top=2.5,
     ref=0.5)
tb(s, xs(0.5, lo, hi, px0, pw) - 0.55, 2.33, 1.1, 0.2, 'random', size=9, color='note', align='c')
panel(s, 8.5, 1.7, 4.233, 4.85)
tb(s, 8.72, 1.85, 3.8, 4.6, [
    {'runs': 'The gain is real', 'bold': True, 'color': 'main', 'size': 12, 'after': 4},
    {'runs': '+0.070 [+0.056, +0.084] for XGBoost; DeLong p = 6.6 × 10⁻²³', 'bullet': True, 'after': 4},
    {'runs': 'It holds on officers never seen in training (XGBoost 0.643, white box 0.629), in every test year '
             '(+0.05 to +0.09) and over 5 other officer draws', 'bullet': True, 'after': 14},
    {'runs': 'But it lies between officers', 'bold': True, 'color': 'main', 'size': 12, 'after': 4},
    {'runs': f'With the record, {F["full_time_officer/xgboost"]["share_of_score_variance_between_officers"]:.0%}–'
             f'{F["full_time_officer/pltr_sparse"]["share_of_score_variance_between_officers"]:.0%} of the score\'s '
             f'variance is between officers ({F["full_time/scorecard"]["share_of_score_variance_between_officers"]:.0%} '
             'without)', 'bullet': True, 'after': 4},
    {'runs': 'The record tells you which officer, never which driver', 'bullet': True, 'after': 14},
    {'runs': 'So we use it to explain outcomes, never as an input to a deployed score.', 'bold': True,
     'color': 'main'}], size=11)

# ================================================================= 9 · White box
READ = {
    'num__off_consent_hit_rate_365d_shrunk': "Officer's past-year consent hit rate (per SD)",
    'off_hit_rate_same_race_past <= 0.179 AND off_hit_rate_365d_shrunk <= 0.129':
        "Officer's past hit rate low on this driver's race and overall",
    'cat__subject_race_hispanic': 'Driver is Hispanic',
    'cat__subject_race_white': 'Driver is white',
    'off_experience_days > 569 AND off_minutes_since_first_stop_today <= 5.5':
        'Experienced officer, first stop of the calendar day',
    'off_log_minutes_since_last_search <= 4.7': "Officer's previous search < ~110 min ago",
    'subject_sex in {male} AND off_consent_365d <= 9.5': 'Male driver, officer with ≤ 9 consent searches last year',
    'off_stops_365d <= 1.71e+03 AND hour_sin > 0.921': 'Lower-volume officer, stop at ~4:30–7:30',
    'off_hit_gap_hisp_white > -0.0345 AND off_log_search_ratio_hisp_white > 0.475':
        'Officer searches Hispanic drivers > 1.6× as often as white',
    'off_search_rate_365d <= 0.255 AND subject_age <= 19.5': 'Low search-rate officer, driver aged 19 or younger',
    'off_stops_365d <= 1.71e+03 AND hour > 21.5': 'Lower-volume officer, stop after 21:30',
    'off_stops_365d <= 1.71e+03 AND month > 8.5': 'Lower-volume officer, September to December',
    'cat__zone_125': 'Zone 125',
    'num__off_search_rate_365d': "Officer's past-year search rate (per SD)",
    'precinct in {2, 3, 4, 6, 8, U, ... (7 values)} AND hour_sin > -0.223': 'Seven precincts, stop at ~23:00–13:00',
    'cat__reason_for_stop_moving traffic violation': 'Stopped for a moving traffic violation',
    'off_stops_365d <= 1.71e+03 AND day_of_week > 3.5': 'Lower-volume officer, late in the week',
    'hour_cos > -0.847 AND reason_for_stop in {__missing__, child restraint, moving traffic violation, parking '
    'violation, safety violation, seatbelt violation}': 'Routine violation, stop outside ~10:00–14:00',
    'cat__zone_113': 'Zone 113'}
DRIVER_ONLY = {'cat__subject_race_hispanic', 'cat__subject_race_white', 'cat__zone_125', 'cat__zone_113',
               'cat__reason_for_stop_moving traffic violation',
               'precinct in {2, 3, 4, 6, 8, U, ... (7 values)} AND hour_sin > -0.223',
               'hour_cos > -0.847 AND reason_for_stop in {__missing__, child restraint, moving traffic violation, '
               'parking violation, safety violation, seatbelt violation}'}
officer_share = sum(t['importance_share'] for t in TERMS if t['term'] not in DRIVER_ONLY)
s = new_slide('Act II · Interpretability of the white box',
              f'The white box reads like a profile of the searcher: {officer_share:.0%} of its weight is about the '
              'officer',
              source='Source: data/pltr_terms.json, data/pltr_whitebox_report.json (scripts_claude/pltr_whitebox_report.py '
                     '--fit); XGBoost SHAP: reports/xgboost_officer (Julia). MASTER §6.1–6.2.',
              note='PLTR (Dumitrescu, Hué, Hurlin & Tokpavi 2022): a logistic regression on data-driven threshold '
                   'rules, adaptive lasso, 1-SE rule. The score is exact: log-odds = baseline + the rules that apply. '
                   'Linear terms: odds ratio per standard deviation.',
              notes="With the officer record, the white box is 19 readable rules. 81% of their weight involves the "
                    "officer: first the officer's past consent hit rate, then a poor record on this driver's race and "
                    "overall, an experienced officer at their first stop of the day, a search within two hours of the "
                    "previous one. Driver race is still there, at 8% each. XGBoost agrees: 62% of its SHAP attribution "
                    "is the officer record. On the right, one decision made exact: the two highest-scored test "
                    "searches. Same score, same reasons, almost all from the officer's record. One found contraband, "
                    "one found nothing. The score describes who searches.")
chart_title(s, L, 1.7, 6.8, 'The white box\'s 10 heaviest rules: share of importance (odds ratio)')
top10 = TERMS[:10]
mx = top10[0]['importance_share']
for k, t in enumerate(top10):
    y = 2.12 + k * 0.365
    off = t['term'] not in DRIVER_ONLY
    tb(s, L, y, 3.55, 0.3, READ[t['term']], size=10, anchor='m', color='ink')
    w = t['importance_share'] / mx * 2.1
    box(s, 4.25, y + 0.04, w, 0.24, fill='main' if off else 'ref')
    tb(s, 4.25 + w + 0.07, y, 1.2, 0.3, [[(f'{t["importance_share"]:.0%}', B),
                                         (f'  OR {t["odds_ratio"]:.2f}', {'color': 'note'})]], size=10, anchor='m')
legend(s, L, 5.85, [('involves the officer', 'main', 'square'), ('driver or stop only', 'ref', 'square')])
tb(s, L, 6.15, 6.8, 0.4, 'XGBoost agrees: 62% of its SHAP attribution is the officer record.', size=10.5, bold=True)
rx, rw = 7.7, 5.033
chart_title(s, rx, 1.7, rw, 'One decision, rule by rule: the two highest-scored test searches (log-odds)', h=0.45)
cases = list(LOC['cases'].values())
hit, miss = cases[0], cases[1]


def contrib(case, term):
    return sum(c['contribution'] for c in case['top_contributions_log_odds'] if c['term'] == term)


named = ['num__off_consent_hit_rate_365d_shrunk',
         'off_experience_days > 569 AND off_minutes_since_first_stop_today <= 5.5',
         'cat__subject_race_white', 'num__off_search_rate_365d']
labels = ["Officer's past consent hit rate", 'Experienced officer, first stop of the day', 'Driver is white',
          "Officer's past search rate"]


def other(case):
    return case['log_odds'] - LOC['intercept_log_odds'] - sum(contrib(case, t) for t in named)


def sg(v):
    return f'{v:+.2f}'.replace('-', '−')


rows = [['', 'Found contraband', 'Found nothing'],
        ['Predicted probability', (f'{hit["score"]:.2f}', B), (f'{miss["score"]:.2f}', B)],
        ['Driver', f'{hit["driver"]["race"]} {"woman" if hit["driver"]["sex"] == "female" else "man"}',
         f'{miss["driver"]["race"]} {"woman" if miss["driver"]["sex"] == "female" else "man"}']]
rows += [[lab, (sg(contrib(hit, t)), {'bold': k == 0, 'color': 'main' if k == 0 else 'ink'}),
          (sg(contrib(miss, t)), {'bold': k == 0, 'color': 'main' if k == 0 else 'ink'})]
         for k, (lab, t) in enumerate(zip(labels, named))]
rows += [['Other rules', sg(other(hit)), sg(other(miss))],
         ['Baseline', sg(LOC['intercept_log_odds']), sg(LOC['intercept_log_odds'])],
         [('Total log-odds', B), (sg(hit['log_odds']), B), (sg(miss['log_odds']), B)]]
table(s, rx, 2.25, [2.6, 1.22, 1.213], rows, row_h=0.31, size=10.5, aligns={1: 'c', 2: 'c'},
      fills={(3, 1): 'panel', (3, 2): 'panel', (3, 0): 'panel'})
panel(s, rx, 5.55, rw, 0.95)
tb(s, rx + 0.15, 5.62, rw - 0.3, 0.85, 'Near-identical officer profile, near-identical score, opposite outcomes. In '
                                       'both, the officer\'s past consent hit rate is above 50% (16.1% across the '
                                       'force in training): the score describes who searches.', size=10.5,
   anchor='m')

# ================================================================= 10 · Officers' records
g = CF['groups']
s = new_slide('Act II · What the officers\' records say',
              f'A fifth of consent searches, made by {g["1"]["officers"]} officers, find contraband 12% of the time '
              'and target Hispanic drivers twice as often',
              source='Source: data/officer_model_diagnostics.json §D; data/officer_disparity_findings.json; MASTER §5.5. '
                     'Groups of officers only; IDs are hashed.',
              note='Test searches 2016–2018 in fifths (≈ 1,823 each) of the officer\'s past consent hit rate, known '
                   'before the search. Skew: each officer\'s own consent-search rate by driver race, per stop.',
              notes="Now describe the records, no model. Group the 2016 to 2018 searches by the searching officer's "
                    "past consent hit rate. The lowest fifth, made by 140 officers, finds contraband 12% of the time, "
                    "against 21% for the median fifth, and searches Hispanic drivers twice as often. Compare each "
                    "officer with themselves: the median officer asks Black drivers for consent 1.4 times as often "
                    "as white drivers, Hispanic drivers 2.3 times. The more skewed the officer, the less they find, "
                    "on white drivers too. We say 'consistent with a lower evidentiary bar', never intent.")
chart_title(s, L, 1.7, 7.3, 'Hit rate of 2016–2018 consent searches, by fifth of the officer\'s past consent hit '
                            'rate', h=0.45)
base, top_, vmax = 4.85, 2.45, 0.35
ln(s, L, base, 7.9, base, 'rule', 0.75)
med = round(CF['median_fifth_hit_rate'][0], 3)
names = ['Lowest', 'Second', 'Median', 'Fourth', 'Highest']
for k in range(5):
    gg = g[str(k + 1)]
    v = round(gg['hit_rate'], 3)
    cx = 1.45 + k * 1.42
    h = v / vmax * (base - top_)
    col = 'main' if k == 0 else 't3'
    box(s, cx - 0.42, base - h, 0.84, h, fill=col)
    tb(s, cx - 0.6, base - h - 0.3, 1.2, 0.27, pct(v), size=12, bold=True, align='c',
       color='main' if k == 0 else 'ink')
    c2 = 'main' if k == 0 else 'ink'
    tb(s, cx - 0.7, base + 0.07, 1.4, 0.25, names[k], size=11, bold=True, align='c', color=c2)
    tb(s, cx - 0.7, base + 0.37, 1.4, 0.25, f'{gg["officers"]} officers', size=10.5, align='c', color=c2)
    tb(s, cx - 0.7, base + 0.64, 1.4, 0.25, pct(gg['share_of_searched_drivers']['hispanic']), size=10.5,
       align='c', color=c2, bold=(k == 0))
tb(s, L, base + 0.92, 7.3, 0.25, 'Share of searched drivers who are Hispanic, by fifth (row above)', size=9.5,
   color='note')
panel(s, 8.3, 1.7, 4.433, 4.85)
tb(s, 8.52, 1.85, 4.0, 4.6, [
    {'runs': 'Each officer compared with themselves', 'bold': True, 'color': 'main', 'size': 12, 'after': 8},
    {'runs': [('1.4× and 2.3×: ', B), 'the median officer asks Black and Hispanic drivers for consent that much '
                                      'more often per stop than white drivers.'], 'after': 8},
    {'runs': [('Skewed officers find less, on white drivers too: ', B), 'most vs least skewed fifth hit 14.3% vs '
                                                                        '21.7% (Black/white skew), 5.0% vs 21.8% '
                                                                        '(Hispanic/white).'], 'after': 8},
    {'runs': [('9 officers ', B), 'carry out a quarter of all consent searches of Hispanic drivers (1,146 of '
                                  '4,645); 98% of those find nothing.'], 'after': 8},
    {'runs': [('77% of officers ', B), 'find less on Black than on white drivers; 91% less on Hispanic drivers.'],
     'after': 12},
    {'runs': 'Consistent with a lower evidentiary bar. The outcome test cannot prove intent.', 'italic': True,
     'color': 'note', 'size': 10.5}], size=11)

# ================================================================= 11 · Fairness
aware, blind = FP['white_box']['fairness_top22'], FP['white_box_race_blind']['fairness_top22']
s = new_slide('Fairness',
              'No version of the model treats every group fairly, because the searches it learned from didn\'t '
              'either',
              source='Source: data/matched_arms_diagnostics.json, data/pltr_whitebox_report.json; MASTER §7.',
              note='"Race-blind" removes the driver\'s race, sex and same-race feature; the officer\'s own past '
                   'search-skew features remain (a fully blind rerun is pending). XGBoost shows the same pattern '
                   '(MASTER §6.2). Full numbers, with confidence intervals, in backup.',
              notes="One last problem, and this one no model can fix. Historically, searches of white drivers found "
                    "something more often than searches of Black or Hispanic drivers: 25.5% versus 19.5% versus "
                    "15.1%. So any model that just chases accuracy learns to copy that pattern, and mostly flags "
                    "white drivers. TabPFN does this worst: 56.5% of white drivers flagged against 1.7% of Black "
                    "drivers. So, obvious idea: hide the driver's race from the model. Does that fix it? Only "
                    "halfway. It does flag groups more evenly. But now, among the drivers it flags, the Black and "
                    "Hispanic ones are 5 to 7 points less likely to actually be carrying anything than the white "
                    "ones it flags. It swapped one unfairness for another. The reason is simple: you cannot force a "
                    "model to treat groups equally AND to be equally right about each of them, when the groups' true "
                    "rates already differ in the data. That is not a modelling mistake. It means the fix has to "
                    "happen upstream, in how searches are decided, not inside the model. That is exactly what our "
                    "recommendation targets, next.")
chart_title(s, L, 1.7, 7.4, 'Share of each driver group flagged, at the search budget used earlier', h=0.45)
legend(s, L, 2.25, [('White drivers', 'gW', 'square'), ('Black drivers', 'gB', 'square'),
                    ('Hispanic drivers', 'gH', 'square')])
base, top_, vmax = 5.15, 2.75, 0.6
ln(s, L, base, 8.0, base, 'rule', 0.75)
for k, (arm, name, col) in enumerate([('scorecard', 'White box', 'wb'), ('gbm', 'XGBoost', 'xgb'),
                                      ('tabpfn', 'TabPFN', 'pfn')]):
    cx = 1.9 + k * 2.45
    f = M[arm]['fairness_top_k']
    for g_i, g in enumerate(RACES):
        v = f[g]['share_flagged'][0]
        bx = cx - 0.93 + g_i * 0.63
        h = max(v / vmax * (base - top_), 0.02)
        box(s, bx, base - h, 0.55, h, fill=GCOL[g])
        tb(s, bx - 0.2, base - h - 0.28, 0.95, 0.25, pct(v), size=10.5, bold=True, align='c')
    dot(s, cx - 0.62, base + 0.22, 0.07, col)
    tb(s, cx - 0.48, base + 0.08, 1.6, 0.28, name, size=12, bold=True)
tb(s, L, 5.62, 7.4, 0.4, 'Why: white drivers\' searches found something more often in these years (25.5% vs 19.5% '
                         'Black, 15.1% Hispanic) — a model chasing accuracy copies that.', size=10.5)
panel(s, 8.4, 1.7, 4.333, 4.85)
tb(s, 8.62, 1.9, 3.9, 4.5, [
    {'runs': 'Hiding race from the model only trades one unfairness for another', 'bold': True, 'color': 'main',
     'size': 13, 'after': 12},
    {'runs': 'Blind it to race, and it flags each group at close to the same rate.', 'bullet': True, 'after': 8},
    {'runs': 'But then the Black and Hispanic drivers it flags are 5–7 points less likely to actually be '
             'carrying anything than the white drivers it flags.', 'bullet': True, 'after': 8},
    {'runs': 'In plain terms: it can be fair on paper, or accurate — not both, because the groups’ real '
             'rates already differ.', 'bullet': True, 'after': 16},
    {'runs': 'The fix is not a better model. It is the search practice behind the data — which is exactly what '
             'we recommend acting on next.', 'bold': True, 'color': 'main'}], size=12)

# ================================================================= 12 · Trade-offs
s = new_slide('Trade-offs',
              'Accuracy barely separates the three models; explainability, stability and governance do',
              source='Source: MASTER §6.1–6.4 and the slides above. Full numbers with citations: backup.',
              notes="Put the three side by side and accuracy is almost a wash: the two black boxes are statistically "
                    "tied, and the white box gives up only a small amount. What actually separates them: only the "
                    "white box explains itself exactly, its results hold up best when retrained, it runs on any "
                    "laptop, and no stop data ever leaves MNPD. Fairness fails for all three, for the reason we just "
                    "saw, so that row is not a point in anyone's favour.")
H = {'bold': True}
rows = [['Dimension', 'White box', 'XGBoost', 'TabPFN'],
        [('Accuracy', H), 'Slightly behind, barely above random either way', 'Statistically tied with TabPFN',
         'Statistically tied with XGBoost'],
        [('Explainability', H), 'Exact — the model is its own explanation', 'Exact, but a simple version only '
                                                                                 'captures 39% of it',
         'Approximate only; no real explanation'],
        [('Stability on retrain', H), 'Most stable core', 'Stable, but its reasons shift over time',
         'Most sensitive of the three'],
        [('Fairness', H), 'Fails: too few minority drivers flagged, or wrong about them if blinded',
         'Fails the same way', 'Fails worst: 56.5% of white drivers flagged vs 1.7% of Black drivers'],
        [('Governance', H), 'Runs anywhere, easy to audit', 'Needs standard ML infrastructure',
         'Needs a GPU, or sends data to a third party']]
table(s, L, 1.9, [1.85, 3.428, 3.428, 3.428], rows, row_h=0.72, size=11.5,
      header_fills={1: 'wb', 2: 'xgb', 3: 'pfn'}, header_colors={1: 'white', 2: 'white'})
tb(s, L, 6.15, CW, 0.55, [[('The white box wins on everything except a small amount of accuracy', {'color': 'main'}),
                           ' — and that small gap buys nothing once we get to how it would actually be used.']],
   size=13, bold=True)

# ================================================================= 13 · Recommendation
v764, (c380, c1206) = CF['if_bottom_fifth_had_median_hit_rate']['same_finds_with_fewer_searches__fruitless_avoided']
s = new_slide('Recommendation',
              'MNPD should use the white box to review consent-search practice, not to score drivers at the '
              'roadside',
              source='Source: data/officer_model_diagnostics.json §D; MASTER §8–9.',
              note='Illustrative counterfactual: the least successful fifth keeps its finds and drops searches until it '
                   'reaches the median fifth\'s hit rate; 95% CI by resampling officers. "About two thirds" assumes '
                   'the avoided searches are spread like that fifth\'s searches.',
              notes="So what should MNPD do? Not put a score in officers' hands: at the roadside the best model is at "
                    "chance inside an officer's own stops, and without the officer record its signal is race. What "
                    "a model can do is support a review of consent-search practice. Each quarter, supervisors see, "
                    "per officer, the consent-search rate, the hit rate with an interval, the ratios and hit-rate gaps "
                    "by race, and a white-box benchmark for the stops they made. The value: had the least successful "
                    "fifth of searches reached the median hit rate, about 764 fewer fruitless searches, two thirds of "
                    "them of Black or Hispanic drivers. Why the white box: a review of people's conduct must be exact "
                    "and contestable.")
chart_title(s, L, 1.7, 6.6, 'Where a model can legitimately help — and where it can’t')
X = ('✗', {'bold': True, 'color': 'note', 'size': 14})
V = ('✓', {'bold': True, 'color': 'main', 'size': 14})
rows = [['Use', '', 'Why'],
        ['Tell an officer whom to search', X, 'At chance inside one officer’s stops; otherwise mostly race'],
        ['Justify a search after the fact', X, 'Would launder a discretionary decision'],
        ['Decide where to patrol', X, 'Barely predicts anything; risks a feedback loop'],
        [('Review search practice, by officer', H), V, ('This is our recommendation', H)],
        ['Train officers on real patterns', V, 'Patterns are readable and repeat'],
        ['Report to oversight, year on year', V, 'Tracks the disparity honestly']]
table(s, L, 2.05, [2.55, 0.45, 3.65], rows, row_h=0.6, size=11.5, aligns={1: 'c'},
      fills={(4, 0): 'panel', (4, 1): 'panel', (4, 2): 'panel'})
rx, rw = 7.55, 5.183
chart_title(s, rx, 1.7, rw, 'What the review looks like, per officer, each quarter')
tb(s, rx, 2.05, rw, 1.35, [
    {'runs': [('1  ', B), 'How often they ask for consent to search'], 'after': 4},
    {'runs': [('2  ', B), 'How often that search finds anything'], 'after': 4},
    {'runs': [('3  ', B), 'Whether that differs by the driver’s race'], 'after': 4},
    {'runs': [('4  ', B), 'How they compare with similar officers']}], size=12)
panel(s, rx, 3.6, rw, 1.55)
tb(s, rx + 0.18, 3.68, 1.7, 0.85, f'{v764:,}', size=42, bold=True, color='accent')
tb(s, rx + 1.75, 3.78, rw - 1.9, 0.8, 'fewer fruitless searches over 2016–2018, mostly of Black or Hispanic '
                                      'drivers, for the same number of finds', size=12, bold=True)
tb(s, rx + 0.18, 4.55, rw - 0.36, 0.5, f'[{c380:,.0f}–{c1206:,.0f}] — if the least successful fifth of '
                                       'officers matched the typical officer’s hit rate', size=10, color='note')
tb(s, rx, 5.35, rw, 1.1, [
    {'runs': 'Why the white box, not a black box', 'bold': True, 'color': 'main', 'size': 12, 'after': 4},
    {'runs': 'Reviewing someone’s conduct has to be explainable and contestable', 'bullet': True, 'after': 3},
    {'runs': 'The black boxes’ small accuracy edge buys nothing here', 'bullet': True, 'after': 3},
    {'runs': 'Runs anywhere; no data leaves MNPD', 'bullet': True}], size=11.5)

# ================================================================= 14 · Closing decision
s = new_slide('Decision',
              'We ask MNPD to approve a six-month pilot of the officer-level review',
              source='Source: MASTER §9.4, §10. Full trustworthy-AI checklist and pilot detail: backup.',
              note='Limits: selective labels (only searches made are observed); the outcome test cannot prove intent '
                   '(infra-marginality); blank contraband fields were recorded as "nothing found".',
              notes="To close, our five answers. No model predicts a successful consent search usefully. What they "
                    "learn is mostly race, and only the white box is exact. Their shortlists are unstable, and no "
                    "version treats every group fairly. Outcomes come from search practice. So: no roadside score, "
                    "and the white box in an officer-level review. This also happens to satisfy the trustworthy-AI "
                    "checklist the course asks for: a human stays in charge, the method is published, and the "
                    "target is the practice that produces the disparity, not the drivers. We ask MNPD to approve a "
                    "six-month pilot: human review only, retrained every year, published to the oversight board. "
                    "Thank you.")
chart_title(s, L, 1.7, 6.6, 'Our answers to the five questions')
rows = [['#', 'Question', 'Answer'],
        ['1', 'Can a model predict a successful search?', 'No: AUC 0.53–0.55, near random economically'],
        ['2', 'What do the models learn?', 'Mostly race; only the white box is exact'],
        ['3', 'Are they stable and fair?', 'No: unstable shortlists, no version is fair'],
        ['4', 'Where do outcomes come from?', 'The searching officer\'s practice'],
        ['5', 'Which model, for what?', ('The white box, for officer-level review; no roadside score', H)]]
table(s, L, 2.05, [0.4, 2.85, 3.4], rows, row_h=0.52, size=11, aligns={0: 'c'},
      fills={(5, 0): 'panel', (5, 1): 'panel', (5, 2): 'panel'})
rx, rw = 7.6, 5.133
chart_title(s, rx, 1.7, rw, 'What we’re asking MNPD to approve')
tb(s, rx, 2.05, rw, 2.2, [
    {'runs': 'A six-month pilot', 'bold': True, 'color': 'main', 'size': 12, 'after': 6},
    {'runs': 'Human review only — no automatic sanctions, and officers can see and contest their report',
     'bullet': True, 'after': 6},
    {'runs': 'Retrained every year, and monitored for drift', 'bullet': True, 'after': 6},
    {'runs': 'Method published to the oversight board, plus the threshold test as the next check', 'bullet': True}],
   size=12)
panel(s, rx, 4.5, rw, 0.75)
tb(s, rx + 0.18, 4.5, rw - 0.36, 0.75, 'This also satisfies the trustworthy-AI requirements the course asks for '
                                       '(human oversight, transparency, fairness) that a roadside score fails — '
                                       'full checklist in backup.', size=11, anchor='m')
panel(s, L, 5.7, CW, 0.62)
tb(s, L + 0.2, 5.7, CW - 0.4, 0.62, [[('Issue    ', {'bold': True, 'color': 'main'}),
                                      'whether to launch the supervisor-review pilot and publish the method to the '
                                      'oversight board']], size=12.5, anchor='m')

# ================================================================= Backups
BK = 'Backup'
# A1 features
s = new_slide(BK, 'Only the officer record adds signal, and it uses strictly past events',
              source='Source: MASTER §3; scripts_claude/features_and_augmentation.md, build_officer_features.py.')
rows = [['Block', 'Content', 'Predicts contraband?', 'Note'],
        [('Stop and driver', H), 'Age, sex, race, precinct, zone, reason for stop, plate state, hour, weekday, month',
         'AUC 0.55–0.57', 'The task as posed (Act I)'],
        [('Time', H), '7 cyclical and calendar features', '+0.007', 'The only block that helps in Act I'],
        [('Neighbourhood (ACS)', H), '800 m / 1,200 m population-weighted circles, release before the stop year',
         'Adds nothing once fixed', 'As first shipped, a year fingerprint (AUC 0.998 for the 2016 split); predicts '
                                    'race at 0.69–0.73'],
        [('Plate', H), 'State, region, distance', 'No', 'Re-encodes plate state; plate_missing is a 2017 recording '
                                                         'change'],
        [('Officer record', H), '25 features from the searching officer\'s own earlier stops', '+0.07 pooled AUC',
         'A measuring instrument, not a deployable input']]
table(s, L, 1.75, [2.0, 4.6, 2.0, 3.533], rows, row_h=0.55, size=10.5)
tb(s, L, 5.35, CW, 1.2, [
    {'runs': 'Leakage controls', 'bold': True, 'color': 'main', 'size': 12, 'after': 3},
    {'runs': 'The 11 post-decision columns carry a post_ prefix and are blocked.', 'bullet': True, 'after': 2},
    {'runs': 'Officer features use only events strictly before the stop: same-minute stops never see each other, the '
             'stop\'s own outcome is never counted; checked by brute force on 400 random stops.', 'bullet': True,
     'after': 2},
    {'runs': 'PLTR re-learns its rules inside each cross-validation fold; on pure noise it gives CV AUC 0.505.',
     'bullet': True}], size=10.5)

# A2 Act II stability
A = D['A_auc_by_test_year']
yr = lambda key: ' / '.join(f'{A[key][y]["auc"]:.3f}' for y in ['2016', '2017', '2018'])
bs = WB['coefficient_bootstrap']
s = new_slide(BK, 'The officer-record result holds across years, unseen officers and resamples',
              source='Source: data/officer_model_diagnostics.json §A, data/officer_holdout_stability.json, '
                     'data/pltr_whitebox_report.json; XGBoost: reports/xgboost_officer (Julia). MASTER §6.1–6.2.')
rows = [['Check', 'White box (PLTR, 19 rules)', 'XGBoost + officer record'],
        [('Test AUC 2016 / 2017 / 2018', H), yr('full_time_officer/pltr_sparse'), yr('full_time_officer/xgboost')],
        [('Officers never seen in training', H), '0.629', '0.643'],
        [('5 other draws of held-out officers', H), '0.618 ± 0.026', '0.635 (0.595–0.695)'],
        [('Resampling', H), f'{bs["resamples"]} bootstraps: AUC {bs["test_auc_mean_sd_min_max"][0]:.3f} ± '
                            f'{bs["test_auc_mean_sd_min_max"][1]:.3f}; {bs["selected_terms_with_any_sign_change"]} '
                            'sign changes among selected rules',
         '10 seeds 0.640 ± 0.002; 20 bootstraps 0.635 [0.628, 0.641]'],
        [('What stays', H), 'Officer consent hit rate selected in every resample (OR 1.25–1.48); rules sharing '
                            'variables carry 62% of importance',
         'Importance ranking correlates 0.94 across bootstraps, 0.85 across eras'],
        [('Drift', H), 'AUC drops in 2018', 'Top officer features drift strongly (PSI 0.75–0.95)']]
table(s, L, 1.75, [3.0, 4.6, 4.533], rows, row_h=0.6, size=11)
tb(s, L, 6.1, CW, 0.4, 'Consequence for deployment: retrain yearly and monitor drift (slide 13).', size=11,
   bold=True)

# A3 the 19 rules
s = new_slide(BK, 'The white box in full: 19 rules, read as a sum of log-odds',
              source='Source: data/pltr_terms.json (temporal split, officer record, 1-SE rule then at most 30 terms).')
rows = [['Rule', 'Odds ratio', 'Share of searches', 'Share of importance']]
rows += [[READ[t['term']], f'{t["odds_ratio"]:.2f}', pct(t['support_share'], 0), pct(t['importance_share'], 1)]
         for t in TERMS]
table(s, L, 1.62, [6.7, 1.7, 1.8, 1.933], rows, row_h=0.245, size=9.5, aligns={1: 'r', 2: 'r', 3: 'r'})

# A4 first stop (native chart)
s = new_slide(BK, 'Searches at the officer\'s first stop of the calendar day hit more often, in every time band',
              source='Source: data/officer_model_diagnostics.json §G (consent searches 2011–2018; 2010 is the warm-up '
                     'year).',
              note='First = no earlier stop by this officer on the same calendar day. Reading: later, repeated searches '
                   'look more speculative; a hypothesis for training, not a causal claim.')
chart_title(s, L, 1.7, 7.5, 'Hit rate of consent searches, first stop of the day vs later stops, by hour band')
cd = CategoryChartData()
bands = list(G['by_hour_band'].keys())
cd.categories = [b.replace('-', '–') for b in bands]
cd.add_series('First stop of the day', [G['by_hour_band'][b]['hit_first'] for b in bands])
cd.add_series('Later stops', [G['by_hour_band'][b]['hit_later'] for b in bands])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, IN(L), IN(2.05), IN(7.6), IN(4.3), cd).chart
ch.font.name, ch.font.size = FONT, Pt(10.5)
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.minimum_scale = 0
ch.category_axis.format.line.color.rgb = C['rule']
pl = ch.plots[0]
pl.gap_width, pl.overlap = 60, 0
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0%', False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(10)
for ser, col in zip(pl.series, ['main', 'ref']):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = C[col]
vol = G['by_officer_daily_stop_volume']
panel(s, 8.6, 1.7, 4.133, 4.8)
tb(s, 8.82, 1.85, 3.7, 4.6, [
    {'runs': f'{G["share_at_first_stop"]:.0%}', 'bold': True, 'color': 'main', 'size': 30},
    {'runs': 'of consent searches happen at the officer\'s first stop of the calendar day', 'bold': True, 'after': 10},
    {'runs': f'Hit rate {pct(G["hit_rate_first_vs_later"][0])} at the first stop vs '
             f'{pct(G["hit_rate_first_vs_later"][1])} later', 'bullet': True, 'after': 4},
    {'runs': f'{G["share_of_first_stop_searches_between_00_and_04"]:.0%} of first-stop searches fall between 00:00 '
             'and 03:59: this is not "start of shift"', 'bullet': True, 'after': 4},
    {'runs': 'Holds for low-, mid- and high-volume officers: ' + ', '.join(
        f'{pct(vol[k]["hit_first"], 0)} vs {pct(vol[k]["hit_later"], 0)}' for k in ['low', 'mid', 'high']),
     'bullet': True}], size=11)

# A5 Act I fairness detail
s = new_slide(BK, 'Act I in detail: every model flags mostly white drivers, and TabPFN is the most skewed',
              source='Source: data/matched_arms_diagnostics.json (matched run, K = 2,000 of 9,113; CIs: 500 bootstraps).',
              note='Predicted − observed is negative everywhere: models trained on 2010–2015 (16.1% hit) under-predict '
                   '2016–2018 (21.4%); compare groups, not levels.')


def ci_txt(m, d=1):
    v, (a, b) = m
    if v is None:
        return 'n/a'
    return f'{v * 100:.{d}f} [{a * 100:.{d}f}, {b * 100:.{d}f}]'.replace('-', '−')


rows = [['Model', 'Drivers', 'Flagged %', 'False-positive rate %', 'Hit rate of flagged %', 'Predicted − observed pp']]
for arm, name in [('scorecard', 'White box'), ('gbm', 'XGBoost'), ('tabpfn', 'TabPFN')]:
    f = M[arm]['fairness_top_k']
    for gi, gname in enumerate(RACES):
        rows.append([(name, H) if gi == 0 else '', gname.capitalize(), ci_txt(f[gname]['share_flagged']),
                     ci_txt(f[gname]['false_positive_rate']), ci_txt(f[gname]['hit_rate_flagged']),
                     ci_txt(f[gname]['calibration_pred_minus_obs'])])
table(s, L, 1.75, [1.5, 1.2, 2.25, 2.4, 2.4, 2.383], rows, row_h=0.36, size=10.5,
      aligns={2: 'r', 3: 'r', 4: 'r', 5: 'r'})
tb(s, L, 5.55, CW, 0.9, [[('χ² (selection, Black vs white): ', B),
                          ' · '.join(f'{n} {M[a]["fairness_top_k"]["chi2_selection_black_vs_white"][0]:,.0f}'
                                     for a, n in [('scorecard', 'white box'), ('gbm', 'XGBoost'),
                                                  ('tabpfn', 'TabPFN')])],
                         [('Hit-rate gap at equal score, Black − white (pp): ', B),
                          ' · '.join(f'{n} {M[a]["fairness_top_k"]["hit_rate_gap_at_equal_score_black_minus_white"][0] * 100:+.1f}'
                                     .replace('-', '−') for a, n in [('scorecard', 'white box'), ('gbm', 'XGBoost'),
                                                                     ('tabpfn', 'TabPFN')])]], size=11)

# A5b fairness in detail: race-aware vs race-blind, with confidence intervals
aware, blind = FP['white_box']['fairness_top22'], FP['white_box_race_blind']['fairness_top22']
E = D['E_calibration_by_race']
s = new_slide(BK, 'Removing race: selection evens out, but the model becomes wrong about minority drivers',
              source='Source: data/pltr_whitebox_report.json, data/officer_model_diagnostics.json §E; fairness partial '
                     'dependence: FINDINGS.md; MASTER §7.',
              note='White box = PLTR, 19 rules, with the officer record; top 22% ≈ 2,005 of 9,113. Race-blind = no '
                   'driver race, sex or same-race feature; the officer\'s past search-skew features remain (a fully '
                   'blind rerun is pending). No single feature restores parity (fairness partial dependence: 0 of '
                   '32).')
chart_title(s, L, 1.7, 5.8, 'Share of each group flagged at the top 22%, white box with officer record', h=0.45)
legend(s, L, 2.25, [('White', 'gW', 'square'), ('Black', 'gB', 'square'), ('Hispanic', 'gH', 'square')])
base, top_, vmax = 5.0, 2.8, 0.4
ln(s, L, base, 6.4, base, 'rule', 0.75)
for k, (lab, fr, auc) in enumerate([('Race-aware', aware, FP['white_box']['performance']['auc']),
                                    ('Race-blind', blind, FP['white_box_race_blind']['performance']['auc'])]):
    cx = 2.05 + k * 2.7
    for g_i, gname in enumerate(RACES):
        v = fr[gname]['share_flagged'][0]
        bx = cx - 0.99 + g_i * 0.67
        h = v / vmax * (base - top_)
        box(s, bx, base - h, 0.6, h, fill=GCOL[gname])
        tb(s, bx - 0.2, base - h - 0.28, 1.0, 0.25, pct(v), size=10.5, bold=True, align='c')
    tb(s, cx - 1.1, base + 0.08, 2.2, 0.28, lab, size=12, bold=True, align='c')
    tb(s, cx - 1.1, base + 0.38, 2.2, 0.25, f'AUC {auc:.3f}', size=10, color='note', align='c')
rx = 6.9
chart_title(s, rx, 1.7, 5.8, 'Hit-rate gap vs white drivers at equal score (pp, 95% CI)', h=0.45)
lo, hi, px0, pw = -12, 8, 9.2, 3.3
gaps = [('Race-aware · Black', E['full_time_officer/pltr_sparse']['hit_rate_gap_at_equal_score']['black_minus_white'], 'gB'),
        ('Race-aware · Hispanic', E['full_time_officer/pltr_sparse']['hit_rate_gap_at_equal_score']['hispanic_minus_white'], 'gH'),
        ('Race-blind · Black', E['full_time_officer_blind/pltr_sparse']['hit_rate_gap_at_equal_score']['black_minus_white'], 'gB'),
        ('Race-blind · Hispanic', E['full_time_officer_blind/pltr_sparse']['hit_rate_gap_at_equal_score']['hispanic_minus_white'], 'gH')]
for k, (lab, (v, (c_lo, c_hi)), col) in enumerate(gaps):
    y = 2.5 + k * 0.52 + (0.25 if k >= 2 else 0)
    tb(s, rx, y - 0.15, 2.2, 0.3, lab, size=11, anchor='m', bold=True)
    ln(s, xs(c_lo * 100, lo, hi, px0, pw), y, xs(c_hi * 100, lo, hi, px0, pw), y, col, 2.25)
    dot(s, xs(v * 100, lo, hi, px0, pw), y, 0.09, col)
    tb(s, xs(c_hi * 100, lo, hi, px0, pw) + 0.08, y - 0.14, 0.7, 0.28, f'{v * 100:+.1f}'.replace('-', '−'), size=11,
       bold=True, anchor='m')
ln(s, xs(0, lo, hi, px0, pw), 2.3, xs(0, lo, hi, px0, pw), 4.55, 'ink', 1.0)
axis(s, lo, hi, [-10, -5, 0, 5], px0, pw, 4.6, lambda t: f'{t:+d}'.replace('-', '−') if t else '0')
tb(s, rx, 4.95, 5.8, 0.5, [[('≈ 0: calibrated. ', B), 'Below 0: at the same score, minority drivers carry contraband '
                                                        'less often, so they are over-scored.']], size=10.5)
panel(s, L, 5.7, CW, 0.82)
tb(s, L + 0.2, 5.72, CW - 0.4, 0.78, [
    [('Test hit rates differ: ', B), 'white 25.5%, Black 19.5%, Hispanic 15.1%. No single feature restores parity '
                                     '(fairness partial dependence: 0 of 32). '],
    [('The disparity is in the labels, ', B), 'produced by search decisions, so the fix is upstream, in search '
                                              'practice.']], size=11, anchor='m')

# A5c trustworthy AI checklist
s = new_slide(BK, 'The trustworthy-AI checklist, in full (EU High-Level Expert Group, 2019)',
              source='Source: EU High-Level Expert Group on AI, Ethics Guidelines for Trustworthy AI (2019); MASTER §9.3.',
              note='Officers are subjects too: intervals and shrinkage on every rate, false-discovery control across '
                   '~1,500 officers, human review only, the right to see and contest one\'s report, no automatic '
                   'sanctions.')
XX = ('✗  ', {'bold': True, 'color': 'note'})
VV = ('✓  ', {'bold': True, 'color': 'main'})
req = [('Human agency and oversight', 'Pre-empts a discretionary legal decision',
        'A supervisor decides; the model only flags'),
       ('Technical robustness', 'Near-chance, unstable shortlist',
        'Officer records persist, transfer and hold every year; retrained yearly for drift'),
       ('Privacy and data governance', 'Scores every driver',
        'Aggregates per officer; hashed IDs in the analysis; no third-party API'),
       ('Transparency', 'Race drives the signal; TabPFN cannot be explained', '19 readable rules; published method'),
       ('Non-discrimination and fairness', 'Reproduces or inverts the disparity',
        'Targets the practice that produces it'),
       ('Societal well-being', 'More searches of the wrong people',
        'Fewer fruitless searches, most of them of minority drivers'),
       ('Accountability', 'Who answers for a score?', 'Review protocol, appeal for officers, oversight reporting')]
rows = [['Requirement (EU HLEG 2019)', 'Roadside search score', 'Officer-level review (white box)']]
rows += [[(r, H), [XX, a], [VV, b]] for r, a, b in req]
table(s, L, 1.75, [3.0, 4.3, 4.833], rows, row_h=0.52, size=11, header_fills={2: 'main'},
      header_colors={2: 'white'}, fills={(None, 2): 'panel'})

# A6 TabPFN
s = new_slide(BK, 'TabPFN: what we can claim, and what we cannot',
              source='Source: MASTER §6.3 and register #4; FINDINGS.md §10, §17–19 and limitation 7; '
                     'data/matched_arms_diagnostics.json.')
colsA6 = [('Reproducible from src (Act I)', [
    f'AUC {M["tabpfn"]["auc"]:.3f} matched, 0.573–0.580 with all training rows',
    'Brier 0.169, still worse than a constant (0.168)',
    f'Within officer {M["tabpfn"]["within_officer"]["within_officer_auc"]:.3f} '
    f'[{M["tabpfn"]["within_officer"]["ci95_resampling_officers"][0]:.3f}, '
    f'{M["tabpfn"]["within_officer"]["ci95_resampling_officers"][1]:.3f}]',
    'Permutation importance: race first (0.038)',
    'Flags 56.5% / 1.7% / 0% of white / Black / Hispanic drivers (χ² 3,314)',
    'Resampling its training context moves AUC by 0.024']),
    ('Run interactively (Act II), reported as is', [
        'AUC 0.653 with all 25 officer features (XGBoost on the same 25: 0.646, vs 0.640 with 16)',
        '0.650 without driver race but with 8 race-derived officer features: not race-blind',
        'Within-officer AUC, economics and fairness not computed (no predictions saved)']),
    ('Governance', [
        '209 s on CPU to score 9,113 searches',
        'The hosted API sends stop records to a third party',
        'No native attribution; occlusion error cannot be bounded',
        'No global surrogate, by choice: on XGBoost a depth-3 tree reproduces only 39% of the score (course slide 86: '
        '"illusion of interpretability")'])]
for k, (head, items) in enumerate(colsA6):
    x0 = L + k * 4.1
    panel(s, x0, 1.75, 3.93, 3.3)
    tb(s, x0 + 0.18, 1.9, 3.6, 4.5, [{'runs': head, 'bold': True, 'color': 'main', 'size': 12, 'after': 6}] +
       [{'runs': it, 'bullet': True, 'after': 5} for it in items], size=11)

# A7 glossary
s = new_slide(BK, 'Glossary',
              source='Source: MASTER §2–§7; course material (XPER, PLTR, global surrogates).')
rows = [['Term', 'Meaning'],
        ['AUC', 'Probability that the model ranks a random successful search above a random fruitless one; 0.5 = '
                'random'],
        ['Within-officer AUC', 'AUC inside each officer\'s own searches, averaged (weighted by searches): the value of '
                               'a score for the decision an officer actually faces'],
        ['Break-even cost ratio', 'Hit rate / (1 − hit rate) of the searches kept: above this ratio of (cost of a '
                                  'fruitless search / value of a find), searching destroys value'],
        ['Outcome test', 'Compare hit rates by group: a lower hit rate is consistent with a lower bar for searching '
                         'that group'],
        ['Infra-marginality', 'Why the outcome test cannot prove intent when groups\' risk distributions differ; the '
                              'threshold test addresses it'],
        ['Selective labels', 'The outcome is observed only for the searches officers chose to make'],
        ['Calibration at equal score', 'Among drivers with the same score, do groups carry contraband equally '
                                       'often?'],
        ['XPER', 'Shapley decomposition of the AUC above 0.5 across features (course method)'],
        ['PLTR', 'Penalised logistic tree regression: a logistic regression on data-driven threshold rules'],
        ['χ² (selection)', 'Test that the share flagged is independent of the driver\'s race']]
table(s, L, 1.7, [2.6, 9.533], rows, row_h=0.42, size=11, bold_first_col=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f'wrote {OUT}: {len(prs.slides)} slides')
