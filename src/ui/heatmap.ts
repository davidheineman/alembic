// 2D sweep heatmap: interpolated color field over the (param1, param2) grid with contour lines,
// per-cell values, and the best cell highlighted. Cells are evenly spaced by index.

import { fmtNum } from './plot';
import { hideTip, showTip } from './tooltip';

export interface HeatmapData {
  title: string;
  xLabel: string;
  yLabel: string;
  xTicks: string[];
  yTicks: string[];
  /** z[j][i]: row j = yTicks[j] (drawn bottom to top), column i = xTicks[i]. NaN = missing/diverged. */
  z: number[][];
  /** Status per cell for NaN values: 'pending' cells are blank, 'diverged' get an ×. */
  missing?: ('pending' | 'diverged' | null)[][];
  lowerIsBetter: boolean;
  logScale: boolean;
}

const PAD = { l: 64, r: 70, t: 26, b: 38 };
const SAMPLES = 12; // field samples per cell side
const LEVELS = 9;
const VIRIDIS = ['#440154', '#482878', '#3e4a89', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'].map((c) => [
  parseInt(c.slice(1, 3), 16),
  parseInt(c.slice(3, 5), 16),
  parseInt(c.slice(5, 7), 16),
]);

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export class Heatmap {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private data: HeatmapData | null = null;
  private hover: { x: number; y: number } | null = null;
  private dirty = false;

  constructor(
    parent: HTMLElement,
    private height = 300,
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

  set(d: HeatmapData | null) {
    this.data = d;
    this.el.style.display = d ? '' : 'none';
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

  draw() {
    const d = this.data;
    const dpr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth;
    const H = this.height;
    if (!d || W === 0) return;
    if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(H * dpr)) {
      this.canvas.width = Math.round(W * dpr);
      this.canvas.height = Math.round(H * dpr);
      this.canvas.style.height = H + 'px';
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillStyle = '#e6e6e6';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(d.title, PAD.l, 16);

    const nx = d.xTicks.length;
    const ny = d.yTicks.length;
    if (!nx || !ny) return;
    const x0 = PAD.l;
    const x1 = W - PAD.r;
    const y0 = H - PAD.b; // bottom
    const y1 = PAD.t; // top
    const cw = (x1 - x0) / nx;
    const ch = (y0 - y1) / ny;

    // Value transform: color + contours live in f-space; "good" end of the scale is bright.
    const f = (v: number) => (d.logScale && v > 0 ? Math.log(v) : v);
    let lo = Infinity;
    let hi = -Infinity;
    let best: [number, number] | null = null;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const v = d.z[j][i];
        if (!isFinite(v) || (d.logScale && v <= 0)) continue;
        const fv = f(v);
        lo = Math.min(lo, fv);
        hi = Math.max(hi, fv);
        if (!best || (d.lowerIsBetter ? v < d.z[best[0]][best[1]] : v > d.z[best[0]][best[1]])) best = [j, i];
      }
    if (!isFinite(lo)) {
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = '#8a8f98';
      ctx.fillText('waiting for runs…', x0, (y0 + y1) / 2);
      return;
    }
    if (hi === lo) hi = lo + 1e-6;
    const good = (fv: number) => {
      const t = (fv - lo) / (hi - lo);
      return d.lowerIsBetter ? 1 - t : t;
    };
    const cell = (j: number, i: number) => {
      const v = d.z[j]?.[i];
      return v !== undefined && isFinite(v) && !(d.logScale && v <= 0) ? f(v) : NaN;
    };

    // Interpolated field sampled on a (nx·S) × (ny·S) grid; sample (a, b) sits at index-space (u, v).
    const SX = nx * SAMPLES;
    const SY = ny * SAMPLES;
    const field = new Float32Array(SX * SY);
    for (let b = 0; b < SY; b++) {
      const v = (b + 0.5) / SAMPLES - 0.5;
      for (let a = 0; a < SX; a++) {
        const u = (a + 0.5) / SAMPLES - 0.5;
        const ni = Math.min(nx - 1, Math.max(0, Math.round(u)));
        const nj = Math.min(ny - 1, Math.max(0, Math.round(v)));
        if (!isFinite(cell(nj, ni))) {
          field[b * SX + a] = NaN;
          continue;
        }
        const i0 = Math.max(0, Math.min(nx - 1, Math.floor(u)));
        const j0 = Math.max(0, Math.min(ny - 1, Math.floor(v)));
        const i1 = Math.min(nx - 1, i0 + 1);
        const j1 = Math.min(ny - 1, j0 + 1);
        const fu = Math.max(0, Math.min(1, u - i0));
        const fv = Math.max(0, Math.min(1, v - j0));
        let acc = 0;
        let wsum = 0;
        for (const [jj, ii, w] of [
          [j0, i0, (1 - fu) * (1 - fv)],
          [j0, i1, fu * (1 - fv)],
          [j1, i0, (1 - fu) * fv],
          [j1, i1, fu * fv],
        ] as [number, number, number][]) {
          const c = cell(jj, ii);
          if (w > 0 && isFinite(c)) {
            acc += w * c;
            wsum += w;
          }
        }
        field[b * SX + a] = wsum > 0 ? acc / wsum : cell(nj, ni);
      }
    }

    // Color field -> offscreen image (row b=0 is the bottom row, so flip when writing).
    const img = new ImageData(SX, SY);
    for (let b = 0; b < SY; b++)
      for (let a = 0; a < SX; a++) {
        const val = field[b * SX + a];
        const k = ((SY - 1 - b) * SX + a) * 4;
        if (!isFinite(val)) {
          img.data.set([40, 42, 48, 255], k);
          continue;
        }
        const [r, g, bl] = viridis(good(val));
        img.data.set([r, g, bl, 255], k);
      }
    const off = document.createElement('canvas');
    off.width = SX;
    off.height = SY;
    off.getContext('2d')!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, x0, y1, x1 - x0, y0 - y1);

    // Contours via marching squares on the sample grid.
    const px = (a: number) => x0 + ((a + 0.5) / SX) * (x1 - x0);
    const py = (b: number) => y0 - ((b + 0.5) / SY) * (y0 - y1);
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let l = 1; l < LEVELS; l++) {
      const lv = lo + ((hi - lo) * l) / LEVELS;
      for (let b = 0; b < SY - 1; b++)
        for (let a = 0; a < SX - 1; a++) {
          const v00 = field[b * SX + a];
          const v10 = field[b * SX + a + 1];
          const v01 = field[(b + 1) * SX + a];
          const v11 = field[(b + 1) * SX + a + 1];
          if (!(isFinite(v00) && isFinite(v10) && isFinite(v01) && isFinite(v11))) continue;
          const code = (v00 > lv ? 1 : 0) | (v10 > lv ? 2 : 0) | (v11 > lv ? 4 : 0) | (v01 > lv ? 8 : 0);
          if (code === 0 || code === 15) continue;
          const lerp = (p: number, q: number) => (lv - p) / (q - p);
          // edge crossing points: bottom (00-10), right (10-11), top (01-11), left (00-01)
          const E = {
            B: [px(a + lerp(v00, v10)), py(b)],
            R: [px(a + 1), py(b + lerp(v10, v11))],
            T: [px(a + lerp(v01, v11)), py(b + 1)],
            L: [px(a), py(b + lerp(v00, v01))],
          };
          const seg = (p: number[], q: number[]) => {
            ctx.moveTo(p[0], p[1]);
            ctx.lineTo(q[0], q[1]);
          };
          switch (code) {
            case 1: case 14: seg(E.L, E.B); break;
            case 2: case 13: seg(E.B, E.R); break;
            case 3: case 12: seg(E.L, E.R); break;
            case 4: case 11: seg(E.R, E.T); break;
            case 6: case 9: seg(E.B, E.T); break;
            case 7: case 8: seg(E.L, E.T); break;
            case 5: seg(E.L, E.T); seg(E.B, E.R); break;
            case 10: seg(E.L, E.B); seg(E.R, E.T); break;
          }
        }
    }
    ctx.stroke();

    // Cell values, missing markers, best cell.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const showVals = cw > 26 && ch > 14;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const cx = x0 + (i + 0.5) * cw;
        const cy = y0 - (j + 0.5) * ch;
        const v = d.z[j][i];
        const fv = cell(j, i);
        if (!isFinite(fv)) {
          const m = d.missing?.[j]?.[i];
          if (m === 'diverged') {
            ctx.fillStyle = '#ff6b8b';
            ctx.font = '600 11px system-ui, sans-serif';
            ctx.fillText('×', cx, cy);
          }
          continue;
        }
        if (!showVals) continue;
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillStyle = good(fv) > 0.6 ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)';
        ctx.fillText(fmtNum(v), cx, cy);
      }
    if (best) {
      const [j, i] = best;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + i * cw + 1, y0 - (j + 1) * ch + 1, cw - 2, ch - 2);
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('★', x0 + (i + 1) * cw - 3, y0 - (j + 1) * ch + 3);
    }

    // Axes: one tick label per row/column.
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = '#8a8f98';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xEvery = Math.max(1, Math.ceil(nx / Math.max(1, Math.floor((x1 - x0) / 44))));
    d.xTicks.forEach((t, i) => i % xEvery === 0 && ctx.fillText(t, x0 + (i + 0.5) * cw, y0 + 4));
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yEvery = Math.max(1, Math.ceil(ny / Math.max(1, Math.floor((y0 - y1) / 14))));
    d.yTicks.forEach((t, j) => j % yEvery === 0 && ctx.fillText(t, x0 - 5, y0 - (j + 0.5) * ch));
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(d.xLabel, (x0 + x1) / 2, H - 6);
    ctx.save();
    ctx.translate(12, (y0 + y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(d.yLabel, 0, 0);
    ctx.restore();

    // Colorbar (bright = better).
    const bx = x1 + 14;
    const bw = 10;
    for (let k = 0; k < y0 - y1; k++) {
      const fv = lo + ((hi - lo) * k) / (y0 - y1);
      const [r, g, b] = viridis(good(fv));
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect(bx, y0 - k - 1, bw, 1);
    }
    const inv = (fv: number) => (d.logScale ? Math.exp(fv) : fv);
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = '#8a8f98';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const fv = lo + ((hi - lo) * k) / 4;
      ctx.fillText(fmtNum(inv(fv)), bx + bw + 4, y0 - ((y0 - y1) * k) / 4);
    }

    // Hover readout.
    const hv = this.hover;
    if (hv && hv.x >= x0 && hv.x < x1 && hv.y >= y1 && hv.y < y0) {
      const i = Math.floor((hv.x - x0) / cw);
      const j = Math.floor((y0 - hv.y) / ch);
      const v = d.z[j]?.[i];
      const m = d.missing?.[j]?.[i];
      const r = this.canvas.getBoundingClientRect();
      showTip(
        r.left + hv.x,
        r.top + hv.y,
        [`${d.xLabel} = ${d.xTicks[i]}`, `${d.yLabel} = ${d.yTicks[j]}`, isFinite(v) ? `value = ${fmtNum(v)}` : m === 'diverged' ? 'diverged' : 'pending'].map((text) => ({ text })),
      );
    } else if (hv) {
      hideTip();
    }
  }
}
