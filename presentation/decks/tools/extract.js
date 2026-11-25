// プレビューの各スライドから、図形・文字・表・画像の位置と見た目を JSON にする
(async () => {
  await document.fonts.ready;
  await new Promise(r => setTimeout(r, 1500));
  const px = v => parseFloat(v) || 0;
  const isPainted = c => c && c !== 'transparent' && !/rgba\(\s*0,\s*0,\s*0,\s*0\s*\)/.test(c);
  const out = [];
  for (const sec of document.querySelectorAll('section')) {
    const R = sec.getBoundingClientRect();
    const rel = b => ({ x: b.left - R.left, y: b.top - R.top, w: b.width, h: b.height });
    const items = [];
    const walk = el => {
      const cs = getComputedStyle(el);
      if (el.tagName === 'ASIDE') return;
      const b = el.getBoundingClientRect();
      if (el.tagName === 'TABLE') {
        const rows = [...el.querySelectorAll('tr')].map(tr => ({
          bg: getComputedStyle(tr).backgroundColor,
          h: tr.getBoundingClientRect().height,
          cells: [...tr.children].map(td => ({ text: td.innerText.trim(), bold: td.tagName === 'TH' || !!td.querySelector('b') }))
        }));
        const cols = [...el.querySelector('tr').children].map(c => c.getBoundingClientRect().width);
        items.push({ kind: 'table', ...rel(b), rows, cols, size: px(cs.fontSize), color: cs.color });
        return;
      }
      if (el !== sec && (isPainted(cs.backgroundColor) || px(cs.borderTopWidth) > 0 || px(cs.borderLeftWidth) > 0)) {
        items.push({ kind: 'box', ...rel(b), bg: cs.backgroundColor, border: cs.borderTopColor, bw: px(cs.borderTopWidth),
          lw: px(cs.borderLeftWidth), lc: cs.borderLeftColor, radius: px(cs.borderTopLeftRadius) });
      }
      if (el.tagName === 'IMG') {
        items.push({ kind: 'img', ...rel(b), src: el.getAttribute('src') });
        return;
      }
      if (/^(P|H1|H2|H3|LI)$/.test(el.tagName)) {
        const runs = [];
        const rec = n => {
          if (n.nodeType === 3) {
            const t = n.textContent.replace(/\s*\n\s*/g, '');
            if (t) { const s = getComputedStyle(n.parentElement); runs.push({ t, color: s.color, bold: parseInt(s.fontWeight) >= 600 }); }
          } else if (n.tagName === 'BR') runs.push({ t: '\n' });
          else n.childNodes.forEach(rec);
        };
        el.childNodes.forEach(rec);
        const cb = { x: b.left - R.left + px(cs.paddingLeft), y: b.top - R.top + px(cs.paddingTop),
          w: b.width - px(cs.paddingLeft) - px(cs.paddingRight), h: b.height - px(cs.paddingTop) - px(cs.paddingBottom) };
        items.push({ kind: 'text', ...cb, runs, size: px(cs.fontSize), lh: px(cs.lineHeight), bullet: el.tagName === 'LI', align: cs.textAlign });
        return;
      }
      [...el.children].forEach(walk);
    };
    [...sec.children].forEach(walk);
    out.push({ id: sec.id, bg: getComputedStyle(sec).backgroundImage, items });
  }
  const name = location.pathname.replace(/^.*\//, '').replace('.html', '.json');
  const res = await fetch('/save/' + name, { method: 'POST', body: JSON.stringify(out) });
  document.title = 'saved:' + res.status;
})();
