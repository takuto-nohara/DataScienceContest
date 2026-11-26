# プレビューから書き出した配置（layoutX.json）を、編集できる PowerPoint にする
# 使い方: python -I -B build_pptx.py <layout.json> <deck folder> <image folder> <out.pptx>
import json
import os
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

EMU = 6350  # 1920px = 13.333in
PT = 0.5  # 1px = 1/144in = 0.5pt
FONT = 'BIZ UDPGothic'

layout_path, deck_dir, img_dir, out_path = sys.argv[1:5]
layout = json.load(open(layout_path, encoding='utf-8'))


def rgba(s):
    m = re.match(r'rgba?\(([^)]*)\)', s or '')
    if not m:
        return None
    v = [float(x) for x in m.group(1).split(',')]
    if len(v) == 4 and v[3] == 0:
        return None
    return RGBColor(int(v[0]), int(v[1]), int(v[2]))


def e(v):
    return Emu(int(round(v * EMU)))


def set_font(run, size, color, bold):
    f = run.font
    f.name = FONT
    f.size = Pt(size * PT)
    f.bold = bool(bold)
    c = rgba(color)
    if c is not None:
        f.color.rgb = c
    rpr = run._r.get_or_add_rPr()
    for tag in ('a:ea', 'a:cs'):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set('typeface', FONT)


def background(slide, cover):
    fill = slide.background.fill
    fill.gradient()
    fill.gradient_angle = 20 if cover else 70
    gs = fill._xPr.find('.//' + qn('a:gsLst'))
    for g in list(gs):
        gs.remove(g)
    stops = [(0, 'F4F6FA'), (45000, 'E3EAF4'), (100000, '3F6DB2')] if cover else [(0, 'F7F9FC'), (100000, 'E8EDF5')]
    for pos, col in stops:
        g = etree.SubElement(gs, qn('a:gs'), pos=str(pos))
        etree.SubElement(g, qn('a:srgbClr'), val=col)


def add_box(slide, it):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if it['radius'] > 0 else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, e(it['x']), e(it['y']), e(it['w']), e(it['h']))
    if it['radius'] > 0:
        sh.adjustments[0] = min(0.5, it['radius'] / min(it['w'], it['h']))
    bg = rgba(it['bg'])
    if bg is not None:
        sh.fill.solid()
        sh.fill.fore_color.rgb = bg
    else:
        sh.fill.background()
    if it['bw'] > 0 and it['bw'] >= it['lw']:
        sh.line.color.rgb = rgba(it['border'])
        sh.line.width = e(it['bw'])
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    if it['lw'] > it['bw']:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, e(it['x']), e(it['y']), e(it['lw']), e(it['h']))
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgba(it['lc'])
        bar.line.fill.background()
        bar.shadow.inherit = False


def add_text(slide, it):
    tb = slide.shapes.add_textbox(e(it['x']), e(it['y']), e(it['w'] + 10), e(it['h']))
    tf = tb.text_frame
    tf.word_wrap = not it.get('nowrap')
    tf.auto_size = None
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.line_spacing = Pt(it['lh'] * PT) if it['lh'] else None
    if it['align'] == 'center':
        p.alignment = PP_ALIGN.CENTER
    runs = list(it['runs'])
    if it['bullet'] and runs:
        runs.insert(0, {'t': '• ', 'color': runs[0]['color'], 'bold': False})
    for r in runs:
        if r['t'] == '\n':
            p = tf.add_paragraph()
            p.line_spacing = Pt(it['lh'] * PT) if it['lh'] else None
            if it['align'] == 'center':
                p.alignment = PP_ALIGN.CENTER
            continue
        run = p.add_run()
        run.text = r['t']
        set_font(run, r.get('size') or it['size'], r['color'], r['bold'])


def add_table(slide, it):
    rows, cols = it['rows'], it['cols']
    gf = slide.shapes.add_table(len(rows), len(cols), e(it['x']), e(it['y']), e(it['w']), e(it['h']))
    tbl = gf.table
    tblpr = tbl._tbl.tblPr
    for k in ('firstRow', 'bandRow'):
        tblpr.set(k, '0')
    style = tblpr.find(qn('a:tableStyleId'))
    if style is not None:
        style.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'  # No Style, No Grid
    for j, w in enumerate(cols):
        tbl.columns[j].width = e(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = e(row['h'])
        bg = rgba(row['bg'])
        for j, cell in enumerate(row['cells'][:len(cols)]):
            c = tbl.cell(i, j)
            if bg is not None:
                c.fill.solid()
                c.fill.fore_color.rgb = bg
            else:
                c.fill.background()
            c.margin_left = c.margin_right = e(14)
            c.margin_top = c.margin_bottom = e(8)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            run = tf.paragraphs[0].add_run()
            run.text = cell['text']
            set_font(run, cell.get('size') or it['size'], cell.get('color') or it['color'], cell['bold'])
            # 罫線は下だけ細く引く
            tcpr = c._tc.get_or_add_tcPr()
            for k, side in enumerate(('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB')):
                ln = etree.Element(qn(side), w=str(2 * EMU) if side == 'a:lnB' else '0')
                tcpr.insert(k, ln)  # 罫線は塗りより前に置く（順番が違うと無視される）
                if side == 'a:lnB':
                    sf = etree.SubElement(ln, qn('a:solidFill'))
                    etree.SubElement(sf, qn('a:srgbClr'), val='D3DCE8')
                else:
                    etree.SubElement(ln, qn('a:noFill'))


prs = Presentation()
prs.slide_width = Emu(1920 * EMU)
prs.slide_height = Emu(1080 * EMU)
blank = prs.slide_layouts[6]
for s in layout:
    slide = prs.slides.add_slide(blank)
    background(slide, 'cover' in s['bg'] or s['id'] == 'cover')
    for it in s['items']:
        if it['kind'] == 'box':
            add_box(slide, it)
        elif it['kind'] == 'img':
            path = os.path.join(img_dir, os.path.basename(it['src']))
            slide.shapes.add_picture(path, e(it['x']), e(it['y']), e(it['w']), e(it['h']))
        elif it['kind'] == 'table':
            add_table(slide, it)
        elif it['kind'] == 'text':
            add_text(slide, it)
    html = open(os.path.join(deck_dir, 'project', 'slides', s['id'] + '.html'), encoding='utf-8').read()
    m = re.search(r'<aside>(.*?)</aside>', html, re.S)
    if m:
        slide.notes_slide.notes_text_frame.text = m.group(1).strip()
prs.save(out_path)
print('saved', out_path, len(layout), 'slides')
