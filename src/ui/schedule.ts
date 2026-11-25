// Mouse-drawable LR schedule. Values are multipliers in [0, 1] on the peak LR,
// sampled uniformly over training progress.

export const SCHED_POINTS = 200;
const AXIS_W = 40; // left margin for the LR axis labels

const fmtLr = (x: number) => (x === 0 ? '0' : x.toExponential(1).replace('.0e', 'e'));

export type Preset = 'constant' | 'cosine' | 'warmup-cosine' | 'linear' | 'wsd' | 'step' | 'inv-sqrt' | 'cyclic';

export function presetSchedule(p: Preset, warmup = 0.05): number[] {
  const n = SCHED_POINTS;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let v = 1;
    switch (p) {
      case 'constant':
        v = 1;
        break;
      case 'cosine':
        v = 0.5 * (1 + Math.cos(Math.PI * t));
        break;
      case 'warmup-cosine':
        v = t < warmup ? t / warmup : 0.5 * (1 + Math.cos((Math.PI * (t - warmup)) / (1 - warmup)));
        break;
      case 'linear':
        v = 1 - t;
        break;
      case 'wsd':
        v = t < warmup ? t / warmup : t < 0.8 ? 1 : 1 - (t - 0.8) / 0.2;
        break;
      case 'step':
        v = t < 0.5 ? 1 : t < 0.75 ? 0.1 : 0.01;
        break;
      case 'inv-sqrt': {
        const w = Math.max(warmup, 0.01);
        v = t < w ? t / w : Math.sqrt(w / t);
        break;
      }
      case 'cyclic':
        v = 0.5 * (1 + Math.cos(Math.PI * ((t * 4) % 1)));
        break;
    }
    out.push(Math.max(0, Math.min(1, v)));
  }
  return out;
}

export class ScheduleEditor {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private last: { i: number; v: number } | null = null;
  progress = -1;
  peakLr = 1e-3;
  onChange?: () => void;

  constructor(
    parent: HTMLElement,
    public values: number[],
  ) {
    this.el = document.createElement('div');
    this.el.className = 'sched';
    this.canvas = document.createElement('canvas');
    this.canvas.title = 'Drag to draw the LR multiplier over training';
    this.el.appendChild(this.canvas);
    parent.appendChild(this.el);
    this.ctx = this.canvas.getContext('2d')!;
    new ResizeObserver(() => this.draw()).observe(this.el);
    const pos = (e: PointerEvent) => {
      const r = this.canvas.getBoundingClientRect();
      const fx = (e.clientX - r.left - AXIS_W) / (r.width - AXIS_W - 6);
      const fy = 1 - (e.clientY - r.top - 8) / (r.height - 22);
      return {
        i: Math.round(Math.max(0, Math.min(1, fx)) * (SCHED_POINTS - 1)),
        v: Math.max(0, Math.min(1, fy)),
      };
    };
    this.canvas.addEventListener('pointerdown', (e) => {
      try {
        this.canvas.setPointerCapture(e.pointerId);
      } catch {
        // synthetic events have no capturable pointer
      }
      const p = pos(e);
      this.values[p.i] = p.v;
      this.last = p;
      this.draw();
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.last) return;
      const p = pos(e);
      const a = this.last;
      const lo = Math.min(a.i, p.i);
      const hi = Math.max(a.i, p.i);
      for (let i = lo; i <= hi; i++) {
        const t = hi === lo ? 1 : (i - a.i) / (p.i - a.i);
        this.values[i] = a.v + (p.v - a.v) * t;
      }
      this.last = p;
      this.draw();
    });
    const end = () => {
      if (this.last) {
        this.last = null;
        this.onChange?.();
      }
    };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
  }

  /** Replace values in place so a running trainer that holds this array sees the change. */
  set(v: number[]) {
    for (let i = 0; i < SCHED_POINTS; i++) this.values[i] = v[i];
    this.draw();
    this.onChange?.();
  }

  smooth() {
    const v = this.values.slice();
    for (let i = 0; i < SCHED_POINTS; i++) {
      let s = 0;
      let c = 0;
      for (let k = -4; k <= 4; k++) {
        const j = i + k;
        if (j < 0 || j >= SCHED_POINTS) continue;
        s += v[j];
        c++;
      }
      this.values[i] = s / c;
    }
    this.draw();
    this.onChange?.();
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth;
    const H = 120;
    if (!W) return;
    this.canvas.width = W * dpr;
    this.canvas.height = H * dpr;
    this.canvas.style.height = H + 'px';
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const x0 = AXIS_W;
    const x1 = W - 6;
    const y0 = H - 14;
    const y1 = 8;
    ctx.fillStyle = '#16181c';
    ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
    ctx.strokeStyle = '#2a2e35';
    for (let k = 1; k < 4; k++) {
      const y = y1 + ((y0 - y1) * k) / 4 + 0.5;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      const x = x0 + ((x1 - x0) * k) / 4 + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y0);
      ctx.stroke();
    }
    const px = (i: number) => x0 + (i / (SCHED_POINTS - 1)) * (x1 - x0);
    const py = (v: number) => y0 - v * (y0 - y1);
    ctx.beginPath();
    ctx.moveTo(px(0), y0);
    this.values.forEach((v, i) => ctx.lineTo(px(i), py(v)));
    ctx.lineTo(px(SCHED_POINTS - 1), y0);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 170, 60, 0.18)';
    ctx.fill();
    ctx.beginPath();
    this.values.forEach((v, i) => (i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v))));
    ctx.strokeStyle = '#ffaa3c';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    if (this.progress >= 0) {
      const x = x0 + this.progress * (x1 - x0);
      ctx.strokeStyle = '#5ad1ff';
      ctx.beginPath();
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y0);
      ctx.stroke();
    }
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = '#8a8f98';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const v of [1, 0.5, 0]) ctx.fillText(fmtLr(v * this.peakLr), x0 - 4, py(v));
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText('start', x0, y0 + 2);
    ctx.textAlign = 'right';
    ctx.fillText('end of training', x1, y0 + 2);
  }
}
