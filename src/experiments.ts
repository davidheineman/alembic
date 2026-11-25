import type { TrainConfig } from './model';

export interface SweepParam {
  label: string;
  numeric: boolean;
  defaults: string;
  apply: (c: TrainConfig, v: number | string) => void;
  get: (c: TrainConfig) => number | string;
  options?: string[];
}

export const SWEEP_PARAMS: Record<string, SweepParam> = {
  lr: { label: 'Learning rate', numeric: true, defaults: '1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1', apply: (c, v) => (c.opt.lr = +v), get: (c) => c.opt.lr },
  batchSize: {
    label: 'Batch size',
    numeric: true,
    defaults: '4, 16, 64, 256, 1024',
    apply: (c, v) => {
      c.batchMode = 'minibatch';
      c.batchSize = +v;
    },
    get: (c) => c.batchSize,
  },
  width: { label: 'Width', numeric: true, defaults: '32, 64, 128, 256, 512', apply: (c, v) => (c.model.width = +v), get: (c) => c.model.width },
  depth: { label: 'Depth', numeric: true, defaults: '0, 1, 2, 4, 8', apply: (c, v) => (c.model.depth = +v), get: (c) => c.model.depth },
  mlpRatio: { label: 'MLP ratio', numeric: true, defaults: '1, 2, 4, 8', apply: (c, v) => (c.model.mlpRatio = +v), get: (c) => c.model.mlpRatio },
  wd: { label: 'Weight decay', numeric: true, defaults: '0, 0.01, 0.1, 1, 10', apply: (c, v) => (c.opt.wd = +v), get: (c) => c.opt.wd },
  momentum: { label: 'Momentum', numeric: true, defaults: '0, 0.5, 0.9, 0.99', apply: (c, v) => (c.opt.momentum = +v), get: (c) => c.opt.momentum },
  beta2: { label: 'Adam β2', numeric: true, defaults: '0.9, 0.95, 0.99, 0.999', apply: (c, v) => (c.opt.beta2 = +v), get: (c) => c.opt.beta2 },
  clip: { label: 'Grad clip', numeric: true, defaults: '0, 0.1, 1, 10', apply: (c, v) => (c.opt.clip = +v), get: (c) => c.opt.clip },
  dropout: { label: 'Dropout', numeric: true, defaults: '0, 0.1, 0.2, 0.3, 0.5', apply: (c, v) => (c.model.dropout = +v), get: (c) => c.model.dropout ?? 0 },
  precision: {
    label: 'Precision',
    numeric: false,
    defaults: 'fp32, mp, bf16',
    options: ['fp32', 'mp', 'bf16'],
    apply: (c, v) => (c.precision = v as TrainConfig['precision']),
    get: (c) => c.precision ?? 'fp32',
  },
  trainSize: { label: 'Train set size', numeric: true, defaults: '1000, 3000, 10000, 30000', apply: (c, v) => (c.data.trainSize = +v), get: (c) => c.data.trainSize },
  budget: { label: 'Training budget', numeric: true, defaults: '1, 2, 4, 8', apply: (c, v) => (c.budget = +v), get: (c) => c.budget },
  initSeed: { label: 'Init seed', numeric: true, defaults: '0, 1, 2, 3, 4', apply: (c, v) => (c.initSeed = +v), get: (c) => c.initSeed },
  dataSeed: { label: 'Data-order seed', numeric: true, defaults: '0, 1, 2, 3, 4', apply: (c, v) => (c.dataSeed = +v), get: (c) => c.dataSeed },
  init: {
    label: 'Init scheme',
    numeric: false,
    defaults: 'zero, uniform, kaiming, xavier',
    options: ['zero', 'uniform', 'kaiming', 'xavier'],
    apply: (c, v) => (c.model.init = v as TrainConfig['model']['init']),
    get: (c) => c.model.init,
  },
  act: {
    label: 'Activation',
    numeric: false,
    defaults: 'relu, gelu, swiglu',
    options: ['relu', 'gelu', 'swiglu'],
    apply: (c, v) => (c.model.act = v as TrainConfig['model']['act']),
    get: (c) => c.model.act,
  },
  norm: {
    label: 'Normalization',
    numeric: false,
    defaults: 'none, layernorm, rmsnorm, batchnorm',
    options: ['none', 'layernorm', 'rmsnorm', 'batchnorm'],
    apply: (c, v) => {
      if (v === 'none') {
        c.model.preNorm = false;
        c.model.postNorm = false;
        return;
      }
      c.model.norm = v as TrainConfig['model']['norm'];
      c.model.normPost = c.model.norm;
      if (!c.model.preNorm && !c.model.postNorm) c.model.preNorm = true;
    },
    get: (c) => (c.model.preNorm || c.model.postNorm ? c.model.norm : 'none'),
  },
  placement: {
    label: 'Norm placement',
    numeric: false,
    defaults: 'none, pre, post, both',
    options: ['none', 'pre', 'post', 'both'],
    apply: (c, v) => {
      c.model.preNorm = v === 'pre' || v === 'both';
      c.model.postNorm = v === 'post' || v === 'both';
    },
    get: (c) => (c.model.preNorm ? (c.model.postNorm ? 'both' : 'pre') : c.model.postNorm ? 'post' : 'none'),
  },
  optimizer: {
    label: 'Optimizer',
    numeric: false,
    defaults: 'sgd, momentum, adam',
    options: ['sgd', 'momentum', 'adam'],
    apply: (c, v) => (c.opt.kind = v as TrainConfig['opt']['kind']),
    get: (c) => c.opt.kind,
  },
  batchMode: {
    label: 'Batch mode',
    numeric: false,
    defaults: 'full, minibatch, sgd',
    options: ['full', 'minibatch', 'sgd'],
    apply: (c, v) => (c.batchMode = v as TrainConfig['batchMode']),
    get: (c) => c.batchMode,
  },
  resolution: {
    label: 'Resolution',
    numeric: true,
    defaults: '8, 16, 32',
    apply: (c, v) => (c.data.resolution = +v as 8 | 16 | 32),
    get: (c) => c.data.resolution,
  },
};

export function parseValues(p: SweepParam, text: string): (number | string)[] {
  const parts = text
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (!p.numeric) return parts.filter((s) => !p.options || p.options.includes(s));
  return parts.map(Number).filter((v) => isFinite(v));
}

export function meanStd(xs: number[]): { mean: number; std: number; n: number } {
  const v = xs.filter((x) => isFinite(x));
  if (!v.length) return { mean: NaN, std: NaN, n: 0 };
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const std = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : 0;
  return { mean, std, n: v.length };
}

export function ema(ys: ArrayLike<number>, alpha: number): number[] {
  const out: number[] = new Array(ys.length);
  let m = 0;
  let w = 0;
  for (let i = 0; i < ys.length; i++) {
    const y = ys[i];
    if (!isFinite(y)) {
      out[i] = y;
      continue;
    }
    m = alpha * m + (1 - alpha) * y;
    w = alpha * w + (1 - alpha);
    out[i] = m / w;
  }
  return out;
}

export interface PowerFit {
  alpha: number;
  beta: number;
  r2: number;
  n: number;
}

/** Least-squares fit of L = alpha * C^beta in log-log space. */
export function fitPowerLaw(C: number[], L: number[]): PowerFit | null {
  const xs: number[] = [];
  const ys: number[] = [];
  C.forEach((c, i) => {
    if (c > 0 && L[i] > 0 && isFinite(c) && isFinite(L[i])) {
      xs.push(Math.log(c));
      ys.push(Math.log(L[i]));
    }
  });
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  if (sxx === 0) return null;
  const beta = sxy / sxx;
  const loga = my - beta * mx;
  const r2 = syy > 0 ? (sxy * sxy) / (sxx * syy) : 1;
  return { alpha: Math.exp(loga), beta, r2, n };
}

/**
 * Compute-optimal frontier: running minimum of loss over all points sorted by compute,
 * then thinned to at most one point per log-compute bin.
 */
export function frontier(points: { c: number; l: number }[], bins = 24): { c: number; l: number }[] {
  const pts = points.filter((p) => p.c > 0 && isFinite(p.l) && p.l > 0).sort((a, b) => a.c - b.c);
  if (!pts.length) return [];
  const env: { c: number; l: number }[] = [];
  let best = Infinity;
  for (const p of pts) {
    if (p.l < best) {
      best = p.l;
      env.push(p);
    }
  }
  const lo = Math.log(env[0].c);
  const hi = Math.log(env[env.length - 1].c);
  if (hi - lo < 1e-9) return env;
  const out: { c: number; l: number }[] = [];
  let lastBin = -1;
  for (const p of env) {
    const b = Math.min(bins - 1, Math.floor(((Math.log(p.c) - lo) / (hi - lo)) * bins));
    if (b !== lastBin) {
      out.push(p);
      lastBin = b;
    } else {
      out[out.length - 1] = p;
    }
  }
  return out;
}
