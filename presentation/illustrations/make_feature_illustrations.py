"""特徴量の6つの観点を、人の動きのイラスト（SVG）で描く。出力: illus/*.svg（480x200）"""
import math, os, sys

OUT = sys.argv[1]
INK, ACC, ACC2, WARN, MUTED, LINE = '#1C2B4A', '#2B5EA7', '#5B86C4', '#B8501A', '#6A7890', '#B9C6D8'
FONT = "font-family=\"'BIZ UDPGothic', sans-serif\""

POSES = {
    'stand': dict(head=(0, -76), neck=(0, -62), sh=(0, -50), hL=(-14, -8), hR=(14, -8), kL=(-5, 37), kR=(5, 37), fL=(-8, 72), fR=(8, 72)),
    'walk': dict(head=(4, -76), neck=(3, -62), sh=(2, -50), hL=(-18, -12), hR=(20, -10), kL=(-8, 37), kR=(16, 34), fL=(-24, 70), fR=(26, 72)),
    'sit': dict(head=(-8, -78), neck=(-6, -64), sh=(-5, -52), hL=(26, -10), hR=(26, -10), kL=(46, 0), kR=(46, 0), fL=(48, 58), fR=(48, 58)),
    'down': dict(head=(6, -76), neck=(5, -62), sh=(4, -50), hL=(-14, -14), hR=(22, -12), kL=(-6, 36), kR=(22, 40), fL=(-16, 70), fR=(28, 104)),
    'climb': dict(head=(8, -75), neck=(7, -61), sh=(6, -49), hL=(-12, -14), hR=(24, -18), kL=(-8, 38), kR=(24, 8), fL=(-18, 72), fR=(26, 38)),
}


def person(pose, hx, hy, s=0.9, rot=0, phone=True):
    p = POSES[pose]
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def T(v):
        x, y = v
        return (hx + s * (x * c - y * si), hy + s * (x * si + y * c))
    w = 7 * s
    seg = lambda a, b: f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>'
    hip = (hx, hy)
    out = [f'<g stroke="{INK}" stroke-width="{w:.1f}" stroke-linecap="round" fill="none">']
    out += [seg(T(p['neck']), hip), seg(T(p['sh']), T(p['hL'])), seg(T(p['sh']), T(p['hR'])),
            seg(hip, T(p['kL'])), seg(T(p['kL']), T(p['fL'])), seg(hip, T(p['kR'])), seg(T(p['kR']), T(p['fR']))]
    out.append('</g>')
    hd = T(p['head'])
    out.append(f'<circle cx="{hd[0]:.1f}" cy="{hd[1]:.1f}" r="{12 * s:.1f}" fill="{INK}"/>')
    if phone:
        out.append(f'<rect x="{-4.5 * s:.1f}" y="{-8 * s:.1f}" width="{9 * s:.1f}" height="{16 * s:.1f}" rx="2" fill="{ACC}" '
                   f'transform="translate({T((9, -6))[0]:.1f} {T((9, -6))[1]:.1f}) rotate({rot})"/>')
    return '\n'.join(out)


def arrow(x1, y1, x2, y2, color, w=4):
    a = math.atan2(y2 - y1, x2 - x1)
    L = 11
    p1 = (x2 - L * math.cos(a - 0.45), y2 - L * math.sin(a - 0.45))
    p2 = (x2 - L * math.cos(a + 0.45), y2 - L * math.sin(a + 0.45))
    bx, by = x2 - 0.8 * L * math.cos(a), y2 - 0.8 * L * math.sin(a)
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


def text(x, y, s, size=20, color=MUTED, anchor='middle', bold=True):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{700 if bold else 400}" {FONT}>{s}</text>'


def poly(pts, color, w=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{d}/>'


def wave(x0, x1, y, amp, period, color, w=3, phase=0.0):
    pts = [(x, y - amp * math.sin(2 * math.pi * (x - x0) / period + phase)) for x in range(int(x0), int(x1) + 1, 2)]
    return poly(pts, color, w)


def svg(body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="400" viewBox="0 0 480 200">{body}</svg>'


def tilt():
    b = [person('stand', 70, 104)]
    b += [arrow(96, 100, 96, 56, ACC), arrow(96, 100, 96, 146, WARN)]
    # 椅子と座位（体幹を少し後ろへ倒す）
    b += [poly([(196, 70), (202, 123), (276, 123)], LINE, 6), poly([(206, 123), (206, 182)], LINE, 6), poly([(270, 123), (270, 182)], LINE, 6)]
    b += [person('sit', 226, 116)]
    dx, dy = -6 / 64.3, -64 / 64.3
    b += [arrow(262, 100, 262 + 44 * dx, 100 + 44 * dy, ACC), arrow(262, 100, 262, 146, WARN)]
    # 臥位
    b += [poly([(312, 156), (476, 156)], LINE, 6), person('stand', 404, 140, rot=-90)]
    b += [arrow(420, 112, 378, 112, ACC), arrow(420, 112, 420, 146, WARN)]
    b += [text(70, 196, '立位'), text(238, 196, '座位'), text(400, 196, '臥位')]
    b += [arrow(330, 22, 360, 22, ACC, 3), text(366, 29, '体幹の軸', 17, anchor='start'),
          arrow(330, 50, 360, 50, WARN, 3), text(366, 57, '重力', 17, anchor='start')]
    return svg(''.join(b))


def sway():
    b = [person('stand', 60, 104)]
    jit = [(x, 100 + 2.5 * math.sin(x * 0.9) * math.sin(x * 0.13)) for x in range(110, 211, 2)]
    b += [poly(jit, ACC, 3), text(160, 70, '小さい', 18)]
    b += [person('walk', 290, 104)]
    b += ['<path d="M 266 50 q -10 16 0 32" fill="none" stroke="%s" stroke-width="3" stroke-linecap="round"/>' % ACC2,
          '<path d="M 318 50 q 10 16 0 32" fill="none" stroke="%s" stroke-width="3" stroke-linecap="round"/>' % ACC2,
          f'<path d="M 266 112 A 26 8 0 1 0 316 108" fill="none" stroke="{WARN}" stroke-width="3.5" stroke-linecap="round"/>',
          arrow(312, 103, 318, 108, WARN, 3.5)]
    b += [wave(346, 470, 100, 32, 62, ACC), text(408, 52, '大きい', 18)]
    b += [text(60, 196, '静止'), text(290, 196, '歩行')]
    return svg(''.join(b))


def impact():
    b = [poly([(0, 118), (22, 118), (22, 150), (74, 150), (74, 182), (160, 182)], LINE, 6)]
    b += [person('down', 62, 86, 0.9)]
    fx, fy = 62 + 28 * 0.9, 86 + 104 * 0.9
    for a in (-160, -125, -55, -20):
        r = math.radians(a)
        b.append(f'<line x1="{fx + 12 * math.cos(r):.1f}" y1="{fy + 12 * math.sin(r):.1f}" x2="{fx + 24 * math.cos(r):.1f}" '
                 f'y2="{fy + 24 * math.sin(r):.1f}" stroke="{WARN}" stroke-width="3" stroke-linecap="round"/>')
    # 下り：鋭いとがり、上り：なだらかな山
    pts = []
    for k in range(4):
        x = 250 + k * 56
        pts += [(x, 78), (x + 20, 78), (x + 26, 24), (x + 32, 78), (x + 56, 78)]
    b += [text(196, 62, '下り', 20, INK, 'start'), poly(pts, WARN, 3)]
    up = 'M 250 168 ' + ' '.join(f'q 28 -44 56 0' for _ in range(4))
    b += [text(196, 160, '上り', 20, INK, 'start'), f'<path d="{up}" fill="none" stroke="{ACC}" stroke-width="3" stroke-linecap="round"/>']
    b += [text(368, 112, '加速度の変化率', 16, MUTED)]
    return svg(''.join(b))


def rhythm():
    b = [person('walk', 62, 104)]
    b += [wave(130, 470, 104, 34, 68, ACC)]
    x0 = 130 + 68 * 0.25
    b += [poly([(x0, 52), (x0, 44), (x0 + 68, 44), (x0 + 68, 52)], INK, 2.5), text(x0 + 34, 34, '1歩', 18, INK)]
    for k in range(5):
        x = x0 + 68 * k
        b.append(f'<circle cx="{x:.1f}" cy="70" r="4.5" fill="{WARN}"/>')
    b += [text(300, 178, '同じ間隔でくり返す', 18)]
    return svg(''.join(b))


def coupling():
    b = [poly([(0, 182), (62, 182), (62, 152), (132, 152), (132, 122), (176, 122)], LINE, 6)]
    hx, hy = 72, 182 - 72 * 0.85
    b += [person('climb', hx, hy, 0.85)]
    px, py = hx + 12, hy - 6
    b += [arrow(px + 8, py, px + 8, py - 44, ACC), arrow(px + 8, py, px + 52, py, WARN)]
    b += [text(px + 16, py - 46, '上下', 17, ACC, 'start'), text(px + 54, py + 22, '水平', 17, WARN, 'start')]
    b += [wave(214, 470, 70, 24, 64, ACC), wave(214, 470, 140, 24, 64, WARN)]
    for k in range(4):
        x = 214 + 64 * 0.25 + 64 * k
        b.append(poly([(x, 46), (x, 116)], MUTED, 2, '4 4'))
    b += [text(470, 34, '上下', 17, ACC, 'end'), text(470, 186, '水平', 17, WARN, 'end')]
    return svg(''.join(b))


def pivot():
    b = [person('stand', 90, 104, phone=False)]
    b += [f'<rect x="99" y="91" width="9" height="16" rx="2" fill="{ACC}"/>',
          f'<path d="M 66 110 A 26 8 0 1 0 116 106" fill="none" stroke="{WARN}" stroke-width="3.5" stroke-linecap="round"/>',
          arrow(112, 101, 118, 106, WARN, 3.5), f'<circle cx="90" cy="104" r="5" fill="{WARN}"/>',
          text(150, 70, '支点が近い', 17, INK, 'start'), text(150, 94, '約2cm', 20, WARN, 'start')]
    b += [poly([(274, 70), (280, 127), (354, 127), ], LINE, 6), poly([(284, 127), (284, 182)], LINE, 6), poly([(348, 127), (348, 182)], LINE, 6)]
    b += [person('sit', 304, 120, phone=False)]
    phx, phy = 312, 102
    b += [f'<rect x="{phx - 4.5}" y="{phy - 8}" width="9" height="16" rx="2" fill="{ACC}"/>']
    pvx, pvy = 306, 130
    b += [poly([(pvx, pvy), (phx, phy)], WARN, 2.5, '5 4'), f'<circle cx="{pvx}" cy="{pvy}" r="5" fill="{WARN}"/>']
    r = 54
    a0, a1 = math.radians(-140), math.radians(-60)
    b += [f'<path d="M {pvx + r * math.cos(a0):.1f} {pvy + r * math.sin(a0):.1f} A {r} {r} 0 0 1 {pvx + r * math.cos(a1):.1f} {pvy + r * math.sin(a1):.1f}" '
          f'fill="none" stroke="{WARN}" stroke-width="3.5" stroke-linecap="round"/>',
          arrow(pvx + r * math.cos(a1 - 0.25), pvy + r * math.sin(a1 - 0.25), pvx + r * math.cos(a1 + 0.05), pvy + r * math.sin(a1 + 0.05), WARN, 3.5)]
    b += [text(380, 70, '座面が支点', 17, INK, 'start'), text(380, 94, '約9cm', 20, WARN, 'start')]
    b += [text(90, 196, '立位'), text(320, 196, '座位')]
    return svg(''.join(b))


os.makedirs(OUT, exist_ok=True)
for name, f in [('feat_tilt', tilt), ('feat_sway', sway), ('feat_impact', impact), ('feat_rhythm', rhythm),
                ('feat_coupling', coupling), ('feat_pivot', pivot)]:
    open(os.path.join(OUT, name + '.svg'), 'w', encoding='utf-8').write(f())
print('ok')
