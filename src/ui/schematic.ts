import { MAX_DEPTH, WIDTHS, branchShape, dropAt, postNormKind, type Act, type ModelConfig, type ModelLayout, type Norm } from '../model';

const NORM_CYCLE: Norm[] = ['layernorm', 'rmsnorm', 'batchnorm'];
const ACT_CYCLE: Act[] = ['relu', 'gelu', 'swiglu'];
const NORM_LABEL: Record<Norm, string> = { layernorm: 'LayerNorm', rmsnorm: 'RMSNorm', batchnorm: 'BatchNorm' };
const ACT_LABEL: Record<Act, string> = { relu: 'ReLU', gelu: 'GELU', swiglu: 'SwiGLU  silu(a)·b' };

const C = {
  norm: '#b48cff',
  lin: '#5aa9ff',
  act: '#4fd18b',
  res: '#ffaa3c',
  text: '#dfe3ea',
  dim: '#8a8f98',
  box: '#1c1f25',
};

function sizeToPx(n: number, lo = 8, hi = 4096, min = 6, max = 70) {
  const t = (Math.log2(Math.max(lo, n)) - Math.log2(lo)) / (Math.log2(hi) - Math.log2(lo));
  return min + Math.max(0, Math.min(1, t)) * (max - min);
}

const LIN_MIN = 120;
const LIN_MAX = 236;
const linWidthPx = (d: number) => LIN_MIN + ((Math.log2(d) - Math.log2(WIDTHS[0])) / (Math.log2(WIDTHS[WIDTHS.length - 1]) - Math.log2(WIDTHS[0]))) * (LIN_MAX - LIN_MIN);

function nearestWidth(log2d: number): number {
  let best = WIDTHS[0];
  for (const w of WIDTHS) if (Math.abs(Math.log2(w) - log2d) < Math.abs(Math.log2(best) - log2d)) best = w;
  return best;
}

interface DragCtx {
  m: ModelConfig;
  per: number;
  cx: number;
  W: number;
  onChange: (patch: Partial<ModelConfig>) => void;
}

// The SVG is rebuilt on every change, so drag state and listeners live on the persistent container.
const dragCtx = new WeakMap<HTMLElement, DragCtx>();

function installDrag(el: HTMLElement, ctx: DragCtx) {
  const first = !dragCtx.has(el);
  dragCtx.set(el, ctx);
  if (!first) return;
  let drag: { kind: 'ov' | 'lin'; lin?: 'fc1' | 'fc2'; x0: number; width: number; depth: number; per: number; sx: number; moved: boolean } | null = null;
  const toSvg = (e: PointerEvent) => {
    const svg = el.querySelector('svg')!;
    const r = svg.getBoundingClientRect();
    const k = dragCtx.get(el)!.W / r.width;
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
  };
  el.addEventListener('pointerdown', (e) => {
    const t = (e.target as Element).closest('[data-drag]') as SVGElement | null;
    if (!t) return;
    const c = dragCtx.get(el)!;
    const p = toSvg(e);
    drag = {
      kind: t.dataset.drag as 'ov' | 'lin',
      lin: t.dataset.lin as 'fc1' | 'fc2' | undefined,
      x0: p.x,
      width: c.m.width,
      depth: c.m.depth,
      per: c.per,
      sx: Math.sign(p.x - c.cx) || 1,
      moved: false,
    };
    el.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  el.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const c = dragCtx.get(el)!;
    const p = toSvg(e);
    const dx = p.x - drag.x0;
    if (Math.abs(dx) > 3) drag.moved = true;
    const patch: Partial<ModelConfig> = {};
    if (drag.kind === 'lin') {
      const target = linWidthPx(drag.width) + dx * drag.sx;
      const t = (target - LIN_MIN) / (LIN_MAX - LIN_MIN);
      patch.width = nearestWidth(Math.log2(WIDTHS[0]) + t * (Math.log2(WIDTHS[WIDTHS.length - 1]) - Math.log2(WIDTHS[0])));
    } else {
      patch.depth = Math.max(0, Math.min(MAX_DEPTH, Math.round(drag.depth + dx / drag.per)));
    }
    if ((patch.width !== undefined && patch.width !== c.m.width) || (patch.depth !== undefined && patch.depth !== c.m.depth)) c.onChange(patch);
  });
  // A press on a Linear box that never moved is a click: remove that layer.
  el.addEventListener('pointerup', () => {
    if (drag?.lin && !drag.moved) dragCtx.get(el)!.onChange(drag.lin === 'fc1' ? { fc1: false } : { fc2: false });
    drag = null;
  });
  el.addEventListener('pointercancel', () => (drag = null));
}

export function renderSchematic(el: HTMLElement, m: ModelConfig, L: ModelLayout, onChange: (patch: Partial<ModelConfig>) => void) {
  const W = 340;
  const parts: string[] = [];

  // ---- Overview: bar heights ~ log width; one bar per block -> aspect ratio at a glance.
  const ovH = 116;
  const midY = 60;
  const inX = 6;
  const stemX = 36;
  const headX = W - 36;
  const bx0 = stemX + 22;
  const bx1 = headX - 12;
  const hIn = sizeToPx(L.inDim);
  const hd = sizeToPx(L.width);
  const hOut = sizeToPx(10);
  // Main signal path, under everything else.
  parts.push(`<line x1="${inX + 18}" y1="${midY}" x2="${headX + 10}" y2="${midY}" stroke="#4a4f58" stroke-width="1.2"/>`);
  parts.push(`<rect x="${inX}" y="${midY - hIn / 2}" width="18" height="${hIn}" rx="2" fill="#666b75"><title>input image, ${L.inDim} values</title></rect>`);
  parts.push(`<rect x="${stemX}" y="${midY - hd / 2}" width="10" height="${hd}" rx="2" fill="${C.lin}"><title>stem Linear ${L.inDim}→${L.width}</title></rect>`);
  parts.push(`<rect x="${headX}" y="${midY - hOut / 2}" width="10" height="${hOut}" rx="2" fill="${C.lin}"><title>head Linear ${L.width}→10</title></rect>`);
  const label = (x: number, y: number, s: string, color = C.dim) =>
    `<text x="${x}" y="${y}" fill="${color}" font-size="9" text-anchor="middle">${s}</text>`;
  parts.push(label(inX + 9, midY - hIn / 2 - 5, 'input'));
  parts.push(label(stemX + 5, midY - hd / 2 - 5, 'stem', C.lin));
  parts.push(label(headX + 5, midY - hOut / 2 - 5, 'head', C.lin));
  const per = m.depth > 0 ? Math.min(36, (bx1 - bx0) / m.depth) : 36;
  let focus: { x0: number; x1: number; y0: number; y1: number } | null = null;
  if (m.depth > 0) {
    const span = per * m.depth;
    const start = bx0 + (bx1 - bx0 - span) / 2;
    const top = midY - hd / 2;
    // Each skip connection leaves the stream just before a block, runs over it, and drops back in
    // just after it. Inset so neighbouring skips keep separate vertical legs.
    const gap = Math.min(8, per * 0.35);
    const peak = top - 6;
    const inset = Math.min(1.5, gap / 4);
    for (let l = 0; l < m.depth; l++) {
      const x = start + l * per + gap / 2;
      const w = Math.max(2, per - gap);
      parts.push(`<rect x="${x}" y="${top}" width="${w}" height="${hd}" rx="2" fill="#3a4250"/>`);
      if (m.residual) {
        const a = x - gap / 2 + inset;
        const b = x + w + gap / 2 - inset;
        parts.push(`<path d="M ${a} ${midY} V ${peak} H ${b} V ${midY}" fill="none" stroke="${C.res}" stroke-width="1.3" stroke-linejoin="round"/>`);
      }
    }
    const mid = Math.floor((m.depth - 1) / 2);
    const hx = start + mid * per;
    focus = { x0: hx - 2, x1: hx + per + 2, y0: (m.residual ? peak : top) - 5, y1: midY + hd / 2 + 4 };
    parts.push(`<text x="${(bx0 + bx1) / 2}" y="${ovH - 4}" fill="${C.dim}" font-size="9.5" text-anchor="middle">${m.depth} × block · d=${L.width} · hidden=${L.hidden}</text>`);
  } else {
    parts.push(`<text x="${(bx0 + bx1) / 2}" y="${ovH - 4}" fill="${C.dim}" font-size="9.5" text-anchor="middle">depth 0 → linear model</text>`);
  }
  parts.push(label(inX + 9, midY + hIn / 2 + 11, String(L.inDim)));
  parts.push(label(stemX + 5, midY + hd / 2 + 11, String(L.width)));
  parts.push(label(headX + 5, midY + hOut / 2 + 11, '10'));
  parts.push(`<rect data-drag="ov" x="${bx0 - 6}" y="8" width="${bx1 - bx0 + 12}" height="${ovH - 24}" fill="transparent" style="cursor:ew-resize"><title>drag left/right to change depth</title></rect>`);

  // ---- Block detail (data flows bottom -> top)
  const top = ovH + 26;
  const cx = 130;
  const bw = 170;
  const rx = 262;
  const yOut = top + 22;
  type Kind = 'postNorm' | 'add' | 'lin2' | 'act' | 'lin1' | 'preNorm';
  const rows: { kind: Kind; y: number }[] = [];
  let y = yOut + 26;
  for (const [kind, gap] of [
    ['postNorm', 38],
    ['add', 38],
    ['lin2', 36],
    ['act', 36],
    ['lin1', 38],
    ['preNorm', 32],
  ] as [Kind, number][]) {
    rows.push({ kind, y });
    y += gap;
  }
  const yIn = y;
  const H = yIn + (L.finalNorm ? 30 : 16);
  const yAt = (k: Kind) => rows.find((r) => r.kind === k)!.y;

  const box = (yy: number, color: string, label: string, sub: string, action?: string, faded = false, w = bw) => {
    const a = action ? `data-act="${action}" class="clickable"` : '';
    return `<g ${a}>
      <rect x="${cx - w / 2}" y="${yy - 13}" width="${w}" height="26" rx="5" fill="${C.box}" stroke="${color}" stroke-width="1.5" ${faded ? 'stroke-dasharray="4 3" opacity="0.55"' : ''}/>
      <text x="${cx - w / 2 + 10}" y="${yy + 4}" fill="${faded ? C.dim : C.text}" font-size="11" font-weight="${faded ? 400 : 600}">${label}</text>
      <text x="${cx + w / 2 - 8}" y="${yy + 4}" fill="${C.dim}" font-size="9.5" text-anchor="end">${sub}</text>
    </g>`;
  };
  // Linear boxes are drawn wider for larger d; dragging their edges resizes d.
  const lw = linWidthPx(L.width);
  const linBox = (yy: number, which: 'fc1' | 'fc2', on: boolean, sub: string) =>
    on
      ? `<g data-drag="lin" data-lin="${which}" style="cursor:ew-resize"><title>drag to resize · click to remove</title>${box(yy, C.lin, 'Linear', sub, undefined, false, lw)}
      <rect x="${cx - lw / 2 - 2}" y="${yy - 6}" width="3" height="12" rx="1.5" fill="${C.lin}"/>
      <rect x="${cx + lw / 2 - 1}" y="${yy - 6}" width="3" height="12" rx="1.5" fill="${C.lin}"/></g>`
      : box(yy, C.lin, 'no Linear', 'click to add', `${which}Slot`, true);

  parts.push(`<text x="10" y="${top}" fill="${C.text}" font-size="11" font-weight="600">Residual block</text>`);
  // spine (input at the bottom, output arrow at the top)
  parts.push(`<line x1="${cx}" y1="${yIn}" x2="${cx}" y2="${yOut + 6}" stroke="#4a4f58" stroke-width="1.5"/>`);
  parts.push(`<polygon points="${cx - 4},${yOut + 6} ${cx + 4},${yOut + 6} ${cx},${yOut}" fill="#4a4f58"/>`);
  parts.push(`<text x="${cx - 8}" y="${yOut + 6}" fill="${C.dim}" font-size="9.5" text-anchor="end">x_l+1</text>`);
  parts.push(`<text x="${cx - 8}" y="${yIn + 2}" fill="${C.dim}" font-size="9.5" text-anchor="end">x_l</text>`);

  // residual path: from the block input up to the add
  const resColor = m.residual ? C.res : '#555';
  const yAdd = yAt('add');
  const resPath = `M ${cx} ${yIn - 6} H ${rx} V ${yAdd} H ${cx + 11}`;
  parts.push(`<g data-act="residual" class="clickable">
    <path d="${resPath}" fill="none" stroke="${resColor}" stroke-width="${m.residual ? 2 : 1.5}" ${m.residual ? '' : 'stroke-dasharray="5 4"'}/>
    <path d="${resPath}" fill="none" stroke="transparent" stroke-width="14"/>
    <text x="${rx + 6}" y="${(yIn + yAdd) / 2}" fill="${resColor}" font-size="10" transform="rotate(90 ${rx + 6} ${(yIn + yAdd) / 2})" text-anchor="middle">${m.residual ? 'residual' : 'no residual'}</text>
  </g>`);

  const h1 = L.hidden1;
  const bs = branchShape(m);
  const normSlot = (yy: number, on: boolean, which: 'pre' | 'post') =>
    on
      ? box(yy, C.norm, `${NORM_LABEL[which === 'pre' ? m.norm : postNormKind(m)]}`, `${which} · d=${L.width}`, `${which}Slot`)
      : box(yy, C.norm, `no ${which}-norm`, 'click to add', `${which}Slot`, true);
  for (const r of rows) {
    if (r.kind === 'postNorm') parts.push(normSlot(r.y, m.postNorm, 'post'));
    if (r.kind === 'preNorm') parts.push(normSlot(r.y, m.preNorm, 'pre'));
    if (r.kind === 'lin1') parts.push(linBox(r.y, 'fc1', bs.fc1, `${L.width} → ${h1}`));
    if (r.kind === 'act') {
      if (!bs.act) parts.push(box(r.y, C.act, 'no activation', 'click to add', 'act', true));
      else {
        const drop = dropAt(m, 'act') ? ` · drop ${m.dropout}` : '';
        if (bs.actKind === 3) parts.push(box(r.y, C.act, 'SwiGLU  silu(u)·u', `${L.hidden}${drop}`, 'act'));
        else parts.push(box(r.y, C.act, ACT_LABEL[m.act], (bs.actKind === 2 ? `${h1} → ${L.hidden}` : `${L.hidden}`) + drop, 'act'));
      }
    }
    if (r.kind === 'lin2') parts.push(linBox(r.y, 'fc2', bs.fc2, `${L.hidden} → ${L.width}${m.zeroInitLast ? ' (0-init)' : ''}`));
    if (r.kind === 'add') {
      parts.push(
        `<circle cx="${cx}" cy="${r.y}" r="10" fill="${C.box}" stroke="${resColor}" stroke-width="1.5"/><text x="${cx}" y="${r.y + 4}" fill="${C.text}" font-size="13" text-anchor="middle">+</text>`,
      );
      if (dropAt(m, 'branch')) parts.push(`<text x="${cx - 16}" y="${r.y + 14}" fill="${C.dim}" font-size="9" text-anchor="end">branch drop ${m.dropout}</text>`);
    }
  }  if (L.finalNorm) {
    parts.push(`<text x="6" y="${H - 2}" fill="${C.dim}" font-size="9">+ final ${NORM_LABEL[m.norm]} before the head</text>`);
  }

  // Zoom indicator: the highlighted middle block expands into the detail frame below.
  const frameY = ovH + 12;
  const frameH = (L.finalNorm ? H - 14 : H - 2) - frameY;
  const zoom = '#8fa3c8';
  parts.push(`<g pointer-events="none">`);
  parts.push(`<rect x="2" y="${frameY}" width="${W - 4}" height="${frameH}" rx="8" fill="none" stroke="${zoom}" stroke-opacity="0.45" stroke-width="1.2"/>`);
  if (focus) {
    parts.push(`<rect x="${focus.x0}" y="${focus.y0}" width="${focus.x1 - focus.x0}" height="${focus.y1 - focus.y0}" rx="3" fill="none" stroke="${zoom}" stroke-width="1.4"/>`);
    parts.push(`<line x1="${focus.x0}" y1="${focus.y1}" x2="10" y2="${frameY}" stroke="${zoom}" stroke-opacity="0.45" stroke-dasharray="3 3"/>`);
    parts.push(`<line x1="${focus.x1}" y1="${focus.y1}" x2="${W - 10}" y2="${frameY}" stroke="${zoom}" stroke-opacity="0.45" stroke-dasharray="3 3"/>`);
  }
  parts.push(`</g>`);

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;user-select:none;touch-action:none">${parts.join('')}</svg>`;
  installDrag(el, { m, per, cx, W, onChange });
  // Clicking an active norm cycles its type, and cycling past the last type turns that slot off.
  // Each slot has its own type; the post slot's type is pinned so changing one never moves the other.
  const cycleSlot = (on: boolean, key: 'preNorm' | 'postNorm') => {
    const pre = key === 'preNorm';
    const cur = pre ? m.norm : postNormKind(m);
    const typed = (n: Norm): Partial<ModelConfig> => (pre ? { norm: n, normPost: postNormKind(m) } : { normPost: n });
    if (!on) return onChange({ [key]: true });
    const i = NORM_CYCLE.indexOf(cur);
    if (i === NORM_CYCLE.length - 1) onChange({ [key]: false, ...typed(NORM_CYCLE[0]) });
    else onChange(typed(NORM_CYCLE[i + 1]));
  };
  el.querySelectorAll<SVGGElement>('[data-act]').forEach((g) => {
    g.addEventListener('click', () => {
      const a = g.dataset.act;
      if (a === 'preSlot') cycleSlot(m.preNorm, 'preNorm');
      if (a === 'postSlot') cycleSlot(m.postNorm, 'postNorm');
      // Same pattern as the norms: cycle the type, and cycling past the last one removes it.
      if (a === 'act') {
        const i = ACT_CYCLE.indexOf(m.act);
        if (!bs.act) onChange({ actOn: true });
        else if (i === ACT_CYCLE.length - 1) onChange({ actOn: false, act: ACT_CYCLE[0] });
        else onChange({ act: ACT_CYCLE[i + 1] });
      }
      if (a === 'fc1Slot') onChange({ fc1: true });
      if (a === 'fc2Slot') onChange({ fc2: true });
      if (a === 'residual') onChange({ residual: !m.residual });
    });
  });
}

