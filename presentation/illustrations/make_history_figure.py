"""試行錯誤のスライド用の図（正解率の階段グラフ＋各段階の根拠の小図）。出力: illus/history.svg"""
import math, os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_feature_illustrations.py'), encoding='utf-8').read().split('os.makedirs(OUT')[0]
exec(src)
GRAY = '#C9D2DE'
W, H = 1800, 760

stages = [  # 名前, 正解率, 伸びの色（何の取り違えが減ったか）
    ('主催者の12列', 91.94, None),
    ('観点で30列', 96.80, ACC),
    ('揺れを対数に', 98.06, WARN),
    ('ねじれの細かさ', 98.20, ACC),
    ('回転の支点', 98.57, WARN),
]
cx = [180 + 360 * k for k in range(5)]
base, bw = 318, 150
y100 = 84
sc = (base - y100) / 10.0          # 90%〜100%
Y = lambda acc: base - (acc - 90) * sc
STEP = '#AEBBCD'
b = []
# 凡例
b += [f'<rect x="0" y="6" width="22" height="22" rx="4" fill="{ACC}"/>', text(32, 26, '歩行と階段の取り違えが減った分', 22, INK, 'start', False),
      f'<rect x="420" y="6" width="22" height="22" rx="4" fill="{WARN}"/>', text(452, 26, '座位と立位の取り違えが減った分', 22, INK, 'start', False)]
# 目盛り：90%と100%
b.append(f'<line x1="10" y1="{base}" x2="{W - 10}" y2="{base}" stroke="{LINE}" stroke-width="2"/>')
b.append(text(W - 10, base - 10, '90%', 22, MUTED, 'end'))
prev = None
for k, (nm, acc, col) in enumerate(stages):
    x = cx[k] - bw / 2
    top_prev = Y(prev) if prev is not None else Y(acc)
    # これまでに積み上げた分
    b.append(f'<rect x="{x}" y="{top_prev:.1f}" width="{bw}" height="{base - top_prev:.1f}" fill="{STEP}"/>')
    if prev is not None:
        h = Y(prev) - Y(acc)
        b.append(f'<rect x="{x}" y="{Y(acc):.1f}" width="{bw}" height="{max(h, 3):.1f}" fill="{col}"/>')
        b.append(text(cx[k], Y(acc) - 12, f'+{acc - prev:.1f}', 30, col))
        # 前の段からの段差を点線でつなぐ
        b.append(f'<line x1="{cx[k - 1] + bw / 2}" y1="{Y(prev):.1f}" x2="{x}" y2="{Y(prev):.1f}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="4 5"/>')
    b.append(text(cx[k], base + 44, nm, 28, INK))
    b.append(text(cx[k], base + 92, f'{acc:.1f}%', 40, ACC))
    prev = acc

# 下段：各段階の根拠の小図
top = 472


def cap(x, s):
    return text(x, 740, s, 23, INK, bold=False)


# 1 平均と標準偏差だけ
x0 = cx[0] - 150
b.append(f'<rect x="{x0}" y="{top + 70}" width="300" height="80" fill="{ACC}" opacity="0.13"/>')
b.append(wave(x0, x0 + 300, top + 110, 46, 75, GRAY, 3))
b.append(poly([(x0, top + 110), (x0 + 300, top + 110)], ACC, 5))
b.append(cap(cx[0], '平均と標準偏差だけ'))
# 2 着地と周期
x0 = cx[1] - 150
pts = []
for x in range(int(x0), int(x0 + 301), 2):
    t = (x - x0) / 75
    pts.append((x, top + 120 - 30 * math.sin(2 * math.pi * t) - 34 * math.exp(-((t % 1 - 0.25) * 9) ** 2)))
b.append(poly(pts, ACC, 3))
for k in range(4):
    b.append(f'<circle cx="{x0 + 75 * (k + 0.25):.1f}" cy="{top + 50}" r="7" fill="{WARN}"/>')
b.append(poly([(x0 + 18.75, top + 192), (x0 + 18.75, top + 200), (x0 + 93.75, top + 200), (x0 + 93.75, top + 192)], INK, 2.5))
b.append(text(x0 + 56, top + 186, '1歩', 20, INK))
b.append(cap(cx[1], '着地の衝撃と周期を足す'))
# 3 対数
vals = [('歩行', 0.5, GRAY), ('立位', 0.022, ACC), ('座位', 0.007, WARN)]
for g, (gx, title, f) in enumerate([(cx[2] - 160, 'そのまま', lambda v: v * 300), (cx[2] + 20, '対数', lambda v: (math.log10(v) + 2.5) * 62)]):
    gb = top + 190
    for j, (nm, v, col) in enumerate(vals):
        bx = gx + j * 48
        h = max(f(v), 1.5)
        b.append(f'<rect x="{bx}" y="{gb - h:.1f}" width="38" height="{h:.1f}" fill="{col}"/>')
        b.append(text(bx + 19, gb + 24, nm, 19, MUTED, bold=False))
    b.append(text(gx + 67, top + 30, title, 22, INK))
b.append(cap(cx[2], 'ねじれの大きさを対数にする'))
# 4 差分
x0 = cx[3] - 150
sig = [(x, top + 70 - 34 * math.sin(2 * math.pi * (x - x0) / 150) - 6 * math.sin(2 * math.pi * (x - x0) / 14)) for x in range(int(x0), int(x0 + 301), 1)]
b.append(poly(sig, MUTED, 2.5))
dif = [(x, top + 160 - 26 * math.sin(2 * math.pi * (x - x0) / 14 + 1.6) - 4 * math.cos(2 * math.pi * (x - x0) / 150)) for x in range(int(x0), int(x0 + 301), 1)]
b.append(poly(dif, ACC, 2.5))
b.append(text(x0 - 6, top + 76, 'ω', 24, MUTED, 'end'))
b.append(text(x0 - 6, top + 166, 'Δω', 24, ACC, 'end'))
b.append(cap(cx[3], '隣の点との差で細かい揺れを拾う'))
# 5 回転の支点
px, py = cx[4] - 100, top + 100
b.append(person('stand', px, py, 1.0, phone=False))
b.append(f'<circle cx="{px}" cy="{py}" r="7" fill="{WARN}"/>')
b.append(f'<rect x="{px + 6}" y="{py - 10}" width="10" height="18" rx="2" fill="{ACC}"/>')
b.append(text(px, top + 224, '立位 1.7cm', 22, INK))
sx, sy = cx[4] + 50, top + 116
b += [poly([(sx - 26, sy - 50), (sx - 20, sy + 8), (sx + 56, sy + 8)], LINE, 6), poly([(sx - 16, sy + 8), (sx - 16, sy + 66)], LINE, 6),
      poly([(sx + 50, sy + 8), (sx + 50, sy + 66)], LINE, 6)]
b.append(person('sit', sx, sy, 1.0, phone=False))
pvx, pvy = sx + 2, sy + 8
b.append(f'<circle cx="{pvx}" cy="{pvy}" r="7" fill="{WARN}"/>')
b.append(f'<rect x="{sx - 2}" y="{sy - 34}" width="10" height="18" rx="2" fill="{ACC}"/>')
b.append(poly([(pvx, pvy), (sx + 3, sy - 16)], WARN, 3, '5 4'))
b.append(text(sx + 10, top + 224, '座位 9.3cm', 22, INK))
b.append(cap(cx[4], '骨盤が回る支点までの距離'))

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'history.svg'), 'w', encoding='utf-8').write(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * 2}" height="{H * 2}" viewBox="0 0 {W} {H}">{"".join(b)}</svg>')
print('ok')
