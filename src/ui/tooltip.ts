// One floating tooltip shared by all plots. Lives on <body> so it is never clipped by a plot or
// covered by a neighbouring one, and sizes itself to its text.

let el: HTMLDivElement | null = null;

function tipEl(): HTMLDivElement {
  if (!el) {
    el = document.createElement('div');
    el.className = 'plot-tip';
    document.body.appendChild(el);
  }
  return el;
}

export interface TipLine {
  text: string;
  color?: string;
}

/** Show the tooltip with its top-left corner just below-right of the mouse at (clientX, clientY). */
export function showTip(clientX: number, clientY: number, lines: TipLine[]) {
  const t = tipEl();
  t.replaceChildren(
    ...lines.map((l) => {
      const row = document.createElement('div');
      if (l.color) {
        const sw = document.createElement('span');
        sw.className = 'plot-tip-swatch';
        sw.style.background = l.color;
        row.appendChild(sw);
      }
      row.append(l.text);
      return row;
    }),
  );
  t.style.display = 'block';
  const off = 12;
  let x = clientX + off;
  let y = clientY + off;
  const { width, height } = t.getBoundingClientRect();
  if (x + width > window.innerWidth - 4) x = clientX - off - width;
  if (y + height > window.innerHeight - 4) y = clientY - off - height;
  t.style.left = `${Math.max(4, x)}px`;
  t.style.top = `${Math.max(4, y)}px`;
}

export function hideTip() {
  if (el) el.style.display = 'none';
}
