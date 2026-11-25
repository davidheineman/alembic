// Minimal canvas line/scatter plot with log axes, bands, and hover readout.

import { hideTip, showTip } from './tooltip';

export interface Series {
  label: string;
  color: string;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  lo?: ArrayLike<number>;
  hi?: ArrayLike<number>;
  width?: number;
  dash?: number[];
  points?: boolean;
  line?: boolean;
  pointRadius?: number;
  alpha?: number;
}

export interface PlotOpts {
  title: string;
  xLabel?: string;
  yLabel?: string;
  logX?: boolean;
  logY?: boolean;
  height?: number;
  xTickFormat?: (v: number) => string;
  xCategories?: string[];
  yMin?: number;
  yMax?: number;
  legend?: boolean;
  padX?: boolean;
}

export interface Transform {
  x: (v: number) => number;
  y: (v: number) => number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  dom: { xmin: number; xmax: number; ymin: number; ymax: number };
}

const PAD = { l: 52, r: 12, t: 26, b: 34 };

export function fmtNum(v: number): string {
  if (!isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    const m = v / Math.pow(10, e);
    return (Math.abs(m - Math.round(m)) < 1e-6 ? String(Math.round(m)) : m.toFixed(1)) + 'e' + e;
  }
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1).replace(/\.0$/, '');
  return Number(v.toPrecision(3)).toString();
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function logTicks(lo: number, hi: number): { major: number[]; minor: number[] } {
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const major: number[] = [];
  const minor: number[] = [];
  const span = b - a;
  const every = Math.max(1, Math.ceil(span / 7));
  for (let e = a; e <= b; e++) {
    for (let m = 1; m < 10; m++) {
      const v = m * Math.pow(10, e);
      if (v < lo * 0.999 || v > hi * 1.001) continue;
      if (m === 1 && (e - a) % every === 0) major.push(v);
      else if (span <= 1 && (m === 2 || m === 5)) major.push(v);
      else if (span <= 4) minor.push(v);
    }
  }
  if (major.length < 2) {
    return { major: niceTicks(lo, hi, 4).filter((v) => v > 0), minor: [] };
  }
  return { major, minor };
}

export class Plot {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private series: Series[] = [];
  private hover: { x: number; y: number } | null = null;
  overlay?: (ctx: CanvasRenderingContext2D, tf: Transform) => void;
  private dirty = false;

  constructor(
    parent: HTMLElement,
    public opts: PlotOpts,
  ) {
    this.el = document.createElement('div');
    this.el.className = 'plot';
    this.canvas = document.createElement('canvas');
    this.el.appendChild(this.canvas);
    parent.appendChild(this.el);
    this.ctx = this.canvas.getContext('2d')!;
    new ResizeObserver(() => this.requestDraw()).observe(this.el);
    this.canvas.addEventListener('mousemove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hover = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.requestDraw();
    });
    this.canvas.addEventListener('mouseleave', () => {
      this.hover = null;
      hideTip();
      this.requestDraw();
    });
  }

  setSeries(s: Series[]) {
    this.series = s;
    this.requestDraw();
  }

  setOpts(o: Partial<PlotOpts>) {
    Object.assign(this.opts, o);
    this.requestDraw();
  }

  requestDraw() {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => {
      this.dirty = false;
      this.draw();
    });
  }

  private domain() {
    const { logX, logY } = this.opts;
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const s of this.series) {
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i];
        if (!isFinite(x) || (logX && x <= 0)) continue;
        for (const y of [s.y[i], s.lo?.[i], s.hi?.[i]]) {
          if (y === undefined || !isFinite(y) || (logY && y <= 0)) continue;
          if (y < ymin) ymin = y;
          if (y > ymax) ymax = y;
        }
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
      }
    }
    if (this.opts.yMin !== undefined) ymin = this.opts.yMin;
    if (this.opts.yMax !== undefined) ymax = this.opts.yMax;
    if (!isFinite(xmin)) {
      xmin = logX ? 1 : 0;
      xmax = logX ? 10 : 1;
    }
    if (!isFinite(ymin)) {
      ymin = logY ? 0.1 : 0;
      ymax = logY ? 10 : 1;
    }
    if (xmax === xmin) {
      xmax = logX ? xmin * 10 : xmin + 1;
      if (!logX) xmin -= 1;
    }
    if (ymax === ymin) {
      if (logY) {
        ymax *= 1.5;
        ymin /= 1.5;
      } else {
        ymax += Math.abs(ymax) * 0.1 + 0.1;
        ymin -= Math.abs(ymin) * 0.1 + 0.1;
      }
    }
    if (logY) {
      const r = Math.log10(ymax / ymin);
      ymax *= Math.pow(10, r * 0.04);
      ymin /= Math.pow(10, r * 0.04);
    } else {
      const r = ymax - ymin;
      ymax += r * 0.04;
      if (this.opts.yMin === undefined) ymin -= r * 0.04;
    }
    if (this.opts.padX) {
      if (logX) {
        const r = Math.log10(xmax / xmin);
        xmax *= Math.pow(10, r * 0.05);
        xmin /= Math.pow(10, r * 0.05);
      } else {
        const r = xmax - xmin;
        xmax += r * 0.05;
        xmin -= r * 0.05;
      }
    }
    if (this.opts.xCategories) {
      xmin = -0.5;
      xmax = this.opts.xCategories.length - 0.5;
    }
    return { xmin, xmax, ymin, ymax };
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth;
    const H = this.opts.height ?? 200;
    if (W === 0) return;
    if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(H * dpr)) {
      this.canvas.width = Math.round(W * dpr);
      this.canvas.height = Math.round(H * dpr);
      this.canvas.style.height = H + 'px';
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const { logX, logY } = this.opts;
    const dom = this.domain();
    const x0 = PAD.l;
    const x1 = W - PAD.r;
    const y0 = H - PAD.b;
    const y1 = PAD.t;
    const fx = logX
      ? (v: number) => x0 + ((Math.log10(v) - Math.log10(dom.xmin)) / (Math.log10(dom.xmax) - Math.log10(dom.xmin))) * (x1 - x0)
      : (v: number) => x0 + ((v - dom.xmin) / (dom.xmax - dom.xmin)) * (x1 - x0);
    const fy = logY
      ? (v: number) => y0 - ((Math.log10(v) - Math.log10(dom.ymin)) / (Math.log10(dom.ymax) - Math.log10(dom.ymin))) * (y0 - y1)
      : (v: number) => y0 - ((v - dom.ymin) / (dom.ymax - dom.ymin)) * (y0 - y1);
    const tf: Transform = { x: fx, y: fy, x0, x1, y0, y1, dom };

    // Title
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillStyle = '#e6e6e6';
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillText(this.opts.title, x0, 16);

    // Grid + ticks
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = '#8a8f98';
    ctx.strokeStyle = '#2a2e35';
    ctx.lineWidth = 1;
    const yt = logY ? logTicks(dom.ymin, dom.ymax) : { major: niceTicks(dom.ymin, dom.ymax, 4), minor: [] };
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt.minor) {
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(x0, fy(v) + 0.5);
      ctx.lineTo(x1, fy(v) + 0.5);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    for (const v of yt.major) {
      const y = Math.round(fy(v)) + 0.5;
      if (y < y1 - 1 || y > y0 + 1) continue;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.fillText(fmtNum(v), x0 - 5, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    if (this.opts.xCategories) {
      this.opts.xCategories.forEach((c, i) => {
        const x = Math.round(fx(i)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y0);
        ctx.stroke();
        ctx.fillText(c, x, y0 + 5);
      });
    } else {
      const xt = logX ? logTicks(dom.xmin, dom.xmax) : { major: niceTicks(dom.xmin, dom.xmax, 5), minor: [] };
      for (const v of xt.minor) {
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        ctx.moveTo(fx(v) + 0.5, y1);
        ctx.lineTo(fx(v) + 0.5, y0);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      for (const v of xt.major) {
        const x = Math.round(fx(v)) + 0.5;
        if (x < x0 - 1 || x > x1 + 1) continue;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y0);
        ctx.stroke();
        ctx.fillText((this.opts.xTickFormat ?? fmtNum)(v), x, y0 + 5);
      }
    }
    if (this.opts.xLabel) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(this.opts.xLabel, x1, 16);
    }
    ctx.strokeStyle = '#4a4f58';
    ctx.strokeRect(x0 + 0.5, y1 + 0.5, x1 - x0, y0 - y1);

    // Series
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y1, x1 - x0, y0 - y1);
    ctx.clip();
    const okX = (v: number) => isFinite(v) && (!logX || v > 0);
    const okY = (v: number) => isFinite(v) && (!logY || v > 0);
    for (const s of this.series) {
      const n = s.x.length;
      if (s.lo && s.hi) {
        ctx.fillStyle = s.color;
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          if (!okX(s.x[i]) || !okY(s.hi[i])) continue;
          const px = fx(s.x[i]);
          const py = fy(s.hi[i]);
          if (!started) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
          started = true;
        }
        for (let i = n - 1; i >= 0; i--) {
          if (!okX(s.x[i]) || !okY(s.lo[i])) continue;
          ctx.lineTo(fx(s.x[i]), fy(s.lo[i]));
        }
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (s.line !== false) {
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = s.alpha ?? 1;
        ctx.lineWidth = s.width ?? 1.5;
        ctx.setLineDash(s.dash ?? []);
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < n; i++) {
          if (!okX(s.x[i]) || !okY(s.y[i])) {
            pen = false;
            continue;
          }
          const px = fx(s.x[i]);
          const py = fy(s.y[i]);
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      if (s.points) {
        ctx.fillStyle = s.color;
        const r = s.pointRadius ?? 3;
        for (let i = 0; i < n; i++) {
          if (!okX(s.x[i]) || !okY(s.y[i])) continue;
          ctx.beginPath();
          ctx.arc(fx(s.x[i]), fy(s.y[i]), r, 0, Math.PI * 2);
          ctx.fill();
          if (s.lo && s.hi && okY(s.lo[i]) && okY(s.hi[i])) {
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(fx(s.x[i]), fy(s.lo[i]));
            ctx.lineTo(fx(s.x[i]), fy(s.hi[i]));
            ctx.stroke();
          }
        }
      }
    }
    this.overlay?.(ctx, tf);
    ctx.restore();

    if (this.opts.legend) this.drawLegend(ctx, tf);
    if (this.hover && !this.drawHover(ctx, tf)) hideTip();
  }

  private drawLegend(ctx: CanvasRenderingContext2D, tf: Transform) {
    const items = this.series.filter((s) => s.label);
    if (!items.length || items.length > 12) return;
    ctx.font = '10px system-ui, sans-serif';
    const w = Math.max(...items.map((s) => ctx.measureText(s.label).width)) + 22;
    const x = tf.x1 - w - 6;
    let y = tf.y1 + 6;
    ctx.fillStyle = 'rgba(20,22,26,0.85)';
    ctx.fillRect(x, y, w, items.length * 14 + 6);
    y += 10;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const s of items) {
      ctx.fillStyle = s.color;
      ctx.fillRect(x + 5, y - 1, 10, 3);
      ctx.fillStyle = '#ccc';
      ctx.fillText(s.label, x + 19, y);
      y += 14;
    }
  }

  /** Draws the crosshair and shows the tooltip; false if there is nothing to show. */
  private drawHover(ctx: CanvasRenderingContext2D, tf: Transform): boolean {
    const h = this.hover!;
    if (h.x < tf.x0 || h.x > tf.x1 || h.y < tf.y1 || h.y > tf.y0) return false;
    const { logX } = this.opts;
    const frac = (h.x - tf.x0) / (tf.x1 - tf.x0);
    const xv = logX
      ? Math.pow(10, Math.log10(tf.dom.xmin) + frac * (Math.log10(tf.dom.xmax) - Math.log10(tf.dom.xmin)))
      : tf.dom.xmin + frac * (tf.dom.xmax - tf.dom.xmin);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath();
    ctx.moveTo(h.x + 0.5, tf.y1);
    ctx.lineTo(h.x + 0.5, tf.y0);
    ctx.stroke();
    const rows: [string, string, number][] = [];
    for (const s of this.series) {
      if (!s.label || s.x.length === 0) continue;
      let best = -1;
      let bd = Infinity;
      for (let i = 0; i < s.x.length; i++) {
        const dd = Math.abs(tf.x(s.x[i]) - h.x);
        if (dd < bd && isFinite(s.y[i])) {
          bd = dd;
          best = i;
        }
      }
      if (best >= 0 && bd < 40) {
        rows.push([s.color, s.label, s.y[best]]);
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(tf.x(s.x[best]), tf.y(s.y[best]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (!rows.length) return false;
    const r = this.canvas.getBoundingClientRect();
    showTip(r.left + h.x, r.top + h.y, [
      { text: `x=${this.opts.xCategories ? this.opts.xCategories[Math.round(xv)] ?? '' : (this.opts.xTickFormat ?? fmtNum)(xv)}` },
      ...rows.slice(0, 10).map(([color, l, v]) => ({ color, text: `${fmtNum(v)}  ${l}` })),
    ]);
    return true;
  }
}
