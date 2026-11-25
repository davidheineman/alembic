import { Rng } from './rng';

export type Norm = 'layernorm' | 'rmsnorm' | 'batchnorm';
export type Act = 'relu' | 'gelu' | 'swiglu';
export type Init = 'zero' | 'uniform' | 'kaiming' | 'xavier';
export type OptKind = 'sgd' | 'momentum' | 'adam';
export type BatchMode = 'full' | 'sgd' | 'minibatch';

export interface ModelConfig {
  width: number;
  depth: number;
  mlpRatio: number;
  act: Act;
  norm: Norm; // pre-norm (and final norm) type
  normPost?: Norm; // post-norm type; defaults to `norm`
  preNorm: boolean; // normalize the branch input
  postNorm: boolean; // normalize after the residual add
  residual: boolean;
  // Branch layers that can be removed (default on).
  fc1?: boolean;
  actOn?: boolean;
  fc2?: boolean;
  dropout?: number; // probability, training only
  dropoutAt?: Partial<Record<DropSite, boolean>>; // sites default to on
  init: Init;
  zeroInitLast: boolean;
}

export interface DataConfig {
  resolution: 8 | 16 | 32;
  trainSize: number;
  testSize: number;
  flip: boolean;
  crop: boolean; // random shift by up to res/8 px, zero-padded (= pad-and-crop)
  cutout: boolean; // zero a random (res/2)-sized square
}

export interface OptConfig {
  kind: OptKind;
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  wd: number;
  clip: number; // 0 = off
}

export interface TrainConfig {
  model: ModelConfig;
  data: DataConfig;
  opt: OptConfig;
  batchMode: BatchMode;
  batchSize: number;
  budgetUnit: 'steps' | 'epochs';
  budget: number;
  schedule: number[]; // LR multiplier sampled uniformly over [0, 1] of training
  initSeed: number;
  dataSeed: number;
  numEvals: number;
  logEvals?: boolean;
  // fp32; mp = bf16 matmul inputs with fp32 accumulation and weights (autocast);
  // bf16 = additionally bf16 matmul outputs and weights. Emulated by rounding, so not faster.
  precision?: Precision;
  microBatches?: number; // minimum gradient-accumulation chunks per step (0/undefined = as memory requires)
}

export type Precision = 'fp32' | 'mp' | 'bf16';

/** Round to the nearest bfloat16, matching bf16() in kernels.ts. */
export function roundBf16(a: Float32Array): Float32Array {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  for (let i = 0; i < u.length; i++) u[i] = ((u[i] + 0x7fff + ((u[i] >>> 16) & 1)) & 0xffff0000) >>> 0;
  return a;
}

export const NUM_CLASSES = 10;
export const WIDTHS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];
export const MAX_DEPTH = 16;

export type ParamKind = 'matrix' | 'bias' | 'norm';

export interface ParamSpec {
  name: string;
  kind: ParamKind;
  offset: number; // in floats, multiple of 64 (256-byte aligned)
  size: number;
  rows: number; // for matrices: fan-in
  cols: number; // for matrices: fan-out; for norm: D (segment holds [gamma(D), beta(D)])
  isLast?: boolean; // output projection of a residual branch, or the head
  norm?: Norm; // for norm segments
}

/** stem: stem output; act: activation output; branch: residual-branch output, before the add. */
export type DropSite = 'stem' | 'act' | 'branch';
export const DROP_SITES: DropSite[] = ['stem', 'act', 'branch'];

/** Dropout probability at a site (0 when dropout is off there). */
export const dropAt = (m: ModelConfig, site: DropSite): number => (m.dropout && m.dropoutAt?.[site] !== false ? m.dropout : 0);

export const postNormKind = (m: ModelConfig): Norm => m.normPost ?? m.norm;

export interface LinearRef {
  W: ParamSpec;
  b: ParamSpec;
}

export interface BlockLayout {
  normPre?: ParamSpec;
  normPost?: ParamSpec;
  fc1?: LinearRef;
  fc2?: LinearRef;
}

export interface ModelLayout {
  inDim: number;
  width: number;
  hidden: number; // post-activation hidden width
  hidden1: number; // fc1 output width (2*hidden for SwiGLU)
  stem: LinearRef;
  blocks: BlockLayout[];
  finalNorm?: ParamSpec;
  head: LinearRef;
  specs: ParamSpec[];
  totalFloats: number; // padded
  numParams: number; // actual trainable params
  matmulParams: number;
}

export function hiddenWidth(m: ModelConfig): number {
  const h = m.width * m.mlpRatio;
  // SwiGLU uses 2/3 of the hidden width so the parameter count matches a plain MLP.
  if (m.act === 'swiglu') return Math.max(8, Math.round((h * 2) / 3 / 8) * 8);
  return Math.max(1, Math.round(h));
}

export interface BranchShape {
  fc1: boolean;
  act: boolean;
  fc2: boolean;
  hidden: number; // activation output width (fc2 input)
  hidden1: number; // activation input width (fc1 output, or d without fc1)
  actKind: number; // see ACT_COMMON in kernels.ts
}

/** Widths inside the residual branch, accounting for removed layers. */
export function branchShape(m: ModelConfig): BranchShape {
  const fc1 = m.fc1 !== false;
  const act = m.actOn !== false;
  const fc2 = m.fc2 !== false;
  const glu = act && m.act === 'swiglu';
  // Without fc1 the activation sees u (width d); without fc2 its output is the branch output (width d).
  const hidden = !fc1 || !fc2 ? m.width : glu ? hiddenWidth(m) : Math.max(1, Math.round(m.width * m.mlpRatio));
  const hidden1 = glu && fc1 ? 2 * hidden : hidden;
  const actKind = !act ? 4 : m.act === 'relu' ? 0 : m.act === 'gelu' ? 1 : fc1 ? 2 : 3;
  return { fc1, act, fc2, hidden, hidden1, actKind };
}

export function buildLayout(m: ModelConfig, inDim: number): ModelLayout {
  const specs: ParamSpec[] = [];
  let off = 0;
  const add = (name: string, kind: ParamKind, rows: number, cols: number, size: number, isLast = false) => {
    const s: ParamSpec = { name, kind, offset: off, size, rows, cols, isLast };
    specs.push(s);
    off += Math.ceil(size / 64) * 64;
    return s;
  };
  const linear = (name: string, fin: number, fout: number, isLast = false): LinearRef => ({
    W: add(`${name}.W`, 'matrix', fin, fout, fin * fout, isLast),
    b: add(`${name}.b`, 'bias', 1, fout, fout, isLast),
  });
  const d = m.width;
  const norm = (name: string, kind: Norm) => Object.assign(add(name, 'norm', 1, d, 2 * d), { norm: kind });
  const { hidden, hidden1, fc1: hasFc1, fc2: hasFc2 } = branchShape(m);
  const stem = linear('stem', inDim, d);
  const blocks: BlockLayout[] = [];
  for (let l = 0; l < m.depth; l++) {
    const normPre = m.preNorm ? norm(`block${l}.preNorm`, m.norm) : undefined;
    const fc1 = hasFc1 ? linear(`block${l}.fc1`, d, hidden1) : undefined;
    const fc2 = hasFc2 ? linear(`block${l}.fc2`, hidden, d, true) : undefined;
    const normPost = m.postNorm ? norm(`block${l}.postNorm`, postNormKind(m)) : undefined;
    blocks.push({ normPre, normPost, fc1, fc2 });
  }
  // Pre-norm leaves the residual stream un-normalized, so it needs a final norm before the head.
  const finalNorm = m.preNorm && !m.postNorm ? norm('final.norm', m.norm) : undefined;
  const head = linear('head', d, NUM_CLASSES, true);
  let numParams = 0;
  let matmulParams = 0;
  for (const s of specs) {
    if (s.kind === 'norm') numParams += s.norm === 'rmsnorm' ? s.cols : 2 * s.cols;
    else numParams += s.size;
    if (s.kind === 'matrix') matmulParams += s.size;
  }
  return { inDim, width: d, hidden, hidden1, stem, blocks, finalNorm, head, specs, totalFloats: off, numParams, matmulParams };
}

/** Training FLOPs per example (forward + backward ~= 6 * matmul params). */
export function flopsPerExample(layout: ModelLayout): number {
  return 6 * layout.matmulParams;
}

export function initParams(layout: ModelLayout, m: ModelConfig, seed: number): Float32Array {
  const p = new Float32Array(layout.totalFloats);
  const rng = new Rng(seed, 1);
  for (const s of layout.specs) {
    if (s.kind === 'norm') {
      p.fill(1, s.offset, s.offset + s.cols);
      continue;
    }
    if (s.kind === 'bias') continue;
    if (m.init === 'zero') continue;
    if (m.zeroInitLast && s.isLast) continue;
    const fanIn = s.rows;
    const fanOut = s.cols;
    if (m.init === 'uniform') {
      // PyTorch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
      const a = 1 / Math.sqrt(fanIn);
      for (let i = 0; i < s.size; i++) p[s.offset + i] = (2 * rng.uniform() - 1) * a;
      continue;
    }
    const std = m.init === 'kaiming' ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));
    for (let i = 0; i < s.size; i++) p[s.offset + i] = rng.normal() * std;
  }
  return p;
}

/** 1 for parameters that get weight decay (matrices), 0 otherwise. */
export function decayMask(layout: ModelLayout): Float32Array {
  const mask = new Float32Array(layout.totalFloats);
  for (const s of layout.specs) if (s.kind === 'matrix') mask.fill(1, s.offset, s.offset + s.size);
  return mask;
}

export function effectiveBatch(cfg: TrainConfig): number {
  if (cfg.batchMode === 'full') return cfg.data.trainSize;
  if (cfg.batchMode === 'sgd') return 1;
  return Math.max(1, Math.min(cfg.batchSize, cfg.data.trainSize));
}

export function totalSteps(cfg: TrainConfig): number {
  const B = effectiveBatch(cfg);
  if (cfg.budgetUnit === 'steps') return Math.max(1, Math.round(cfg.budget));
  return Math.max(1, Math.round((cfg.budget * cfg.data.trainSize) / B));
}

export function lrMultiplier(schedule: number[], frac: number): number {
  const n = schedule.length;
  const x = Math.min(1, Math.max(0, frac)) * (n - 1);
  const i = Math.min(n - 2, Math.floor(x));
  const t = x - i;
  return schedule[i] * (1 - t) + schedule[i + 1] * t;
}

export function formatCount(n: number): string {
  if (n >= 1e9) return +(n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return +(n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return +(n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}

export function formatSci(x: number, digits = 2): string {
  if (!isFinite(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e-2 && a < 1e4) return Number(x.toPrecision(digits + 1)).toString();
  return x.toExponential(digits).replace('e+', 'e');
}
