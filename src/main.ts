import './style.css';
import { Engine, Trainer, type BatchSample, type ChunkUpdate } from './gpu/engine';
import { loadCifar, getGpuDataset, type Cifar, type GpuDataset } from './data';
import {
  buildLayout,
  effectiveBatch,
  flopsPerExample,
  formatCount,
  formatSci,
  totalSteps,
  postNormKind,
  MAX_DEPTH,
  WIDTHS,
  type Act,
  type ModelConfig,
  type Precision,
  type DropSite,
  DROP_SITES,
  dropAt,
  type TrainConfig,
} from './model';
import { Plot, fmtNum, type Series } from './ui/plot';
import { ScheduleEditor, presetSchedule, type Preset } from './ui/schedule';
import { renderSchematic } from './ui/schematic';
import { Heatmap } from './ui/heatmap';
import { hideTip, showTip } from './ui/tooltip';
import { GPU_PRICES, GPU_TYPES, RemotePool, RemoteRun, remoteAvailable, remoteGpu } from './remote';
import { SWEEP_PARAMS, parseValues, meanStd, ema, fitPowerLaw, frontier, type PowerFit } from './experiments';

// ============================================================================
// State
// ============================================================================

const schedule = presetSchedule('warmup-cosine');

const cfg: TrainConfig = {
  model: { width: 256, depth: 3, mlpRatio: 4, act: 'gelu', norm: 'layernorm', preNorm: true, postNorm: false, residual: true, init: 'kaiming', zeroInitLast: false },
  data: { resolution: 16, trainSize: 10000, testSize: 10000, flip: false, crop: false, cutout: false, noise: 0, labelNoise: 0 },
  opt: { kind: 'adam', lr: 1e-3, momentum: 0.9, beta1: 0.9, beta2: 0.99, wd: 0, clip: 0 },
  batchMode: 'minibatch',
  batchSize: 128,
  budgetUnit: 'epochs',
  budget: 3,
  schedule,
  initSeed: 0,
  dataSeed: 42,
  numEvals: 101, // every 1% of training
};
const DEFAULT_CFG: TrainConfig = structuredClone({ ...cfg, schedule: [] });

function resetConfig() {
  for (const k of Object.keys(cfg)) delete (cfg as unknown as Record<string, unknown>)[k];
  Object.assign(cfg, structuredClone(DEFAULT_CFG), { schedule });
  schedEditor.set(presetSchedule('warmup-cosine'));
  rerollSeeds = true;
  sync();
}
type Status = 'queued' | 'running' | 'done' | 'diverged' | 'stopped' | 'error';

interface Run {
  id: number;
  label: string;
  baseLabel: string;
  color: string;
  group: string | null;
  cfg: TrainConfig;
  numParams: number;
  flopsPerEx: number;
  B: number;
  T: number;
  step: number[];
  loss: number[];
  acc: number[];
  gnorm: number[];
  pnorm: number[];
  logitMax: number[];
  actMax: number[];
  lr: number[];
  evStep: number[];
  evLoss: number[];
  evAcc: number[];
  status: Status;
  visible: boolean;
  sps: number;
  error?: string;
  sweep?: { v1?: number | string; v2?: number | string; rep: number };
}

const runs: Run[] = [];
const queue: Run[] = [];
let current: { run: Run; trainer: Trainer } | null = null;
// Remote runs execute concurrently on Modal GPUs instead of going through the local queue.
let backend: 'local' | 'remote' = 'local';
const remote = new Map<Run, RemoteRun>();
let engine: Engine | null = null;
let cifar: Cifar | null = null;
let nextId = 1;
let lastStem: { w: Float32Array; run: Run } | null = null;
let lastBatch: { b: BatchSample; run: Run; mean: number[]; std: number[] } | null = null;

const PALETTE = ['#5ad1ff', '#ffaa3c', '#4fd18b', '#ff6b8b', '#b48cff', '#ffd84a', '#7af0d0', '#ff8f4a', '#8fb3ff', '#e07aff', '#c3e88d', '#f78c6c'];
const VIRIDIS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];
const PLASMA = ['#5b02a3', '#9c179e', '#cc4778', '#ed7953', '#fdb42f', '#f0f921'];

function cmap(stops: string[], t: number): string {
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = parseInt(stops[i].slice(1), 16);
  const b = parseInt(stops[i + 1].slice(1), 16);
  const ch = (s: number) => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}

// ============================================================================
// DOM helpers
// ============================================================================

type Child = Node | string | null | undefined | false;

function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, unknown> = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'class') el.className = String(v);
    else if (k === 'style') el.setAttribute('style', String(v));
    else if (v !== undefined && v !== false) (el as unknown as Record<string, unknown>)[k] = v;
  }
  for (const c of children) if (c) el.append(c);
  return el;
}

const updaters: (() => void)[] = [];

function sync() {
  for (const u of updaters) u();
  onConfigChanged();
}

function seg<T extends string | number>(opts: [T, string][], get: () => T, set: (v: T) => void, title?: Record<string, string>): HTMLElement {
  const el = h('div', { class: 'seg' });
  const btns = opts.map(([v, label]) => {
    const b = h('button', { title: title?.[String(v)] ?? '', onclick: () => (set(v), sync()) }, label);
    el.append(b);
    return [v, b] as const;
  });
  updaters.push(() => btns.forEach(([v, b]) => b.classList.toggle('on', v === get())));
  return el;
}

/** Without Modal credentials in the build (e.g. GitHub Pages), Remote GPU is greyed out and explains why on hover. */
function backendSeg(el: HTMLElement): HTMLElement {
  if (remoteAvailable) return el;
  const b = el.lastElementChild as HTMLButtonElement;
  b.classList.add('unavailable');
  b.title = '';
  const tip = (e: MouseEvent) => showTip(e.clientX, e.clientY, [{ text: 'Run this repo locally to use Modal for GPUs' }]);
  b.addEventListener('mouseenter', tip);
  b.addEventListener('mousemove', tip);
  b.addEventListener('mouseleave', hideTip);
  return el;
}

function sel<T extends string | number>(opts: [T, string][], get: () => T, set: (v: T) => void): HTMLSelectElement {
  const el = h('select', {});
  opts.forEach(([v, label], i) => el.append(h('option', { value: String(i) }, label)));
  el.addEventListener('change', () => {
    set(opts[+el.value][0]);
    sync();
  });
  updaters.push(() => {
    const i = opts.findIndex(([v]) => v === get());
    if (i >= 0) el.value = String(i);
  });
  return el;
}

function num(get: () => number, set: (v: number) => void, o: { min?: number; max?: number; int?: boolean; title?: string } = {}): HTMLInputElement {
  const el = h('input', { class: 'num', type: 'text', title: o.title ?? '' });
  el.addEventListener('change', () => {
    let v = Number(el.value);
    if (!isFinite(v)) return void (el.value = String(get()));
    if (o.int) v = Math.round(v);
    if (o.min !== undefined) v = Math.max(o.min, v);
    if (o.max !== undefined) v = Math.min(o.max, v);
    set(v);
    sync();
  });
  updaters.push(() => {
    if (document.activeElement !== el) el.value = o.int ? String(Math.round(get())) : fmtNum(get());
  });
  return el;
}

/** Range slider that snaps to `values` (nearest under `scale`), paired with a free-form number box. */
function snap(
  values: number[] | (() => number[]),
  get: () => number,
  set: (v: number) => void,
  o: { min?: number; max?: number; int?: boolean; title?: string; scale?: (v: number) => number; disabled?: () => boolean } = {},
): HTMLElement[] {
  const vals = typeof values === 'function' ? values : () => values;
  const f = o.scale ?? ((v: number) => v);
  const s = h('input', { type: 'range', min: 0, step: 1, class: 'grow', title: o.title ?? '' });
  s.addEventListener('input', () => (set(vals()[+s.value]), sync()));
  updaters.push(() => {
    const vs = vals();
    const v = f(get());
    let best = 0;
    for (let i = 1; i < vs.length; i++) if (Math.abs(f(vs[i]) - v) < Math.abs(f(vs[best]) - v)) best = i;
    s.max = String(vs.length - 1);
    s.value = String(best);
    s.disabled = o.disabled?.() ?? false;
  });
  return [s, num(get, set, o)];
}

function toggles(opts: [string, () => boolean, (v: boolean) => void, string?][]): HTMLElement {
  const el = h('div', { class: 'seg' });
  for (const [label, get, set, title] of opts) {
    const b = h('button', { title: title ?? '', onclick: () => (set(!get()), sync()) }, label);
    updaters.push(() => b.classList.toggle('on', get()));
    el.append(b);
  }
  return el;
}

const logScale = (v: number) => Math.log(Math.max(v, 1e-12));
const betaScale = (v: number) => -Math.log(Math.max(1 - v, 1e-6));

function check(label: string, get: () => boolean, set: (v: boolean) => void, title = ''): HTMLLabelElement {
  const cb = h('input', { type: 'checkbox' });
  cb.addEventListener('change', () => (set(cb.checked), sync()));
  updaters.push(() => (cb.checked = get()));
  return h('label', { style: 'display:inline-flex;gap:5px;align-items:center', title }, cb, label);
}

function row(label: string, ...els: Child[]): HTMLDivElement {
  return h('div', { class: 'row' }, h('span', { class: 'lbl' }, label), ...els);
}

function section(title: string | Node, ...els: Child[]): HTMLDivElement {
  return h('div', { class: 'section' }, h('h3', {}, title), ...els);
}

function info(fn: () => string, cls = 'info'): HTMLDivElement {
  const el = h('div', { class: cls });
  updaters.push(() => (el.innerHTML = fn()));
  return el;
}

// ============================================================================
// Sidebar
// ============================================================================

const side = document.getElementById('side')!;

let rerollSeeds = true;
const trainBtn = h('button', { class: 'primary', onclick: () => trainSingle() }, '▶ Train');
const stopBtn = h('button', { class: 'danger', onclick: () => stopAll() }, '■ Stop');
const progressBar = h('div');
const liveInfo = h('div', { class: 'info' }, 'idle');
// Loading progress and errors; hidden when empty.
const statusInfo = h('div', { class: 'info warn', style: 'display:none' });
const poolDot = h('span', { class: 'pool-dot', title: 'remote GPU off' });
const poolInfo = h('div', { class: 'info' });
const pool = new RemotePool(() => {
  const s = pool.state;
  poolDot.className = `pool-dot ${s}`;
  const text = {
    off: pool.released && backend === 'remote' ? 'GPU auto-off' : '',
    warming: 'starting GPU…',
    ready: `1x${pool.gpuLabel} ($${pool.costPerHour.toFixed(2)}/hr) · ${pool.util}% util`,
    stopping: 'shutting down GPU…',
    error: `remote backend unreachable: ${pool.error}`,
  }[s];
  poolDot.title = text || 'remote GPU off';
  poolInfo.textContent = text;
  poolInfo.style.display = text ? '' : 'none';
});
poolInfo.style.display = 'none';
addEventListener('pagehide', () => pool.leave());
if (remoteAvailable && localStorage.getItem('backend') === 'remote') {
  backend = 'remote';
  pool.start(false);
}
const gpuSeg = seg(
  GPU_TYPES.map((g) => [g, g] as [string, string]),
  () => remoteGpu,
  (v) => {
    stopRemote();
    pool.setGpu(v);
    pool.prepare(upcomingConfigs());
  },
  Object.fromEntries(GPU_TYPES.map((g) => [g, `~$${GPU_PRICES[g].toFixed(2)}/hr. Switching stops runs on the current GPU.`])),
);
gpuSeg.classList.add('compact');
const runSection = section(
    'Run',
    h('div', { class: 'row' }, trainBtn, stopBtn, h('button', { onclick: () => clearRuns(), title: 'Stop queued and remote runs, then remove all runs' }, 'Clear runs'),
      h('button', { onclick: () => resetConfig(), title: 'Restore every setting in the sidebar to its default' }, 'Reset config'),
    ),
    row(
      'Backend',
      backendSeg(seg<'local' | 'remote'>(
        [
          ['local', 'Local GPU'],
          ['remote', 'Remote GPU'],
        ],
        () => backend,
        (v) => {
          if (!remoteAvailable) return;
          backend = v;
          localStorage.setItem('backend', v);
          if (v === 'remote') {
            pool.start();
            pool.prepare(upcomingConfigs());
          } else {
            stopRemote();
            pool.stop();
          }
        },
        { remote: 'Train on one Modal cloud GPU (remote/modal_app.py), kept warm while selected so runs start instantly. Up to 8 runs train on it at once.' },
      )),
      remoteAvailable && poolDot,
    ),
    remoteAvailable &&
      hideUnless(
        row('GPU', gpuSeg),
        () => backend === 'remote',
      ),
    remoteAvailable && poolInfo,
    h('div', { class: 'progress' }, progressBar),
    liveInfo,
    statusInfo,
);
runSection.classList.add('pinned');
side.append(runSection);

// --- Seeds
side.append(
  section(
    'Random Seed',
    row(
      'Init seed',
      num(() => cfg.initSeed, (v) => (cfg.initSeed = v), { int: true, min: 0 }),
      h('button', { onclick: () => ((cfg.initSeed = Math.floor(Math.random() * 10000)), sync()), title: 'Random seed' }, '🎲'),
    ),
    row(
      'Data seed',
      num(() => cfg.dataSeed, (v) => (cfg.dataSeed = v), { int: true, min: 0 }),
      h('button', { onclick: () => ((cfg.dataSeed = Math.floor(Math.random() * 10000)), sync()), title: 'Random seed' }, '🎲'),
    ),
    row('', check('new seeds every run', () => rerollSeeds, (v) => (rerollSeeds = v), 'Roll both seeds each time you press Train')),
  ),
);

// --- Data
const EPOCH_STOPS = [0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100];
const NOISE_STOPS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1];
const LABEL_NOISE_STOPS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const epochRange = h('input', { type: 'range', min: 0, max: EPOCH_STOPS.length - 1, step: 1, class: 'grow' });
epochRange.addEventListener('input', () => ((cfg.budgetUnit = 'epochs'), (cfg.budget = EPOCH_STOPS[+epochRange.value]), sync()));
const epochLbl = h('span', { class: 'info', style: 'width:38px;text-align:right' });
updaters.push(() => {
  const v = Math.log(cfg.budget);
  let best = 0;
  EPOCH_STOPS.forEach((s, i) => Math.abs(Math.log(s) - v) < Math.abs(Math.log(EPOCH_STOPS[best]) - v) && (best = i));
  epochRange.value = String(best);
  epochLbl.textContent = fmtNum(cfg.budget);
});
side.append(
  section(
    'Data',
    row('Resolution', seg<number>([[8, '8×8'], [16, '16×16'], [32, '32×32']], () => cfg.data.resolution, (v) => (cfg.data.resolution = v as 8 | 16 | 32))),
    row(
      'Train set',
      sel<number>(
        [500, 1000, 2000, 5000, 10000, 20000, 50000].map((n) => [n, formatCount(n)]),
        () => cfg.data.trainSize,
        (v) => (cfg.data.trainSize = v),
      ),
      h('span', { class: 'lbl', style: 'width:auto' }, 'test'),
      sel<number>(
        [500, 1000, 2000, 5000, 10000].map((n) => [n, formatCount(n)]),
        () => cfg.data.testSize,
        (v) => (cfg.data.testSize = v),
      ),
    ),
    row('Epochs', epochRange, epochLbl),
    row(
      'Augment',
      check('flip', () => cfg.data.flip, (v) => (cfg.data.flip = v), 'Random horizontal flip (p = 0.5)'),
      check('crop', () => cfg.data.crop, (v) => (cfg.data.crop = v), 'Random crop: pad by res/8 px (4 px at 32×32), then crop back, i.e. a random shift'),
      check('cutout', () => cfg.data.cutout, (v) => (cfg.data.cutout = v), 'Cutout: blank a random (res/2)×(res/2) square (16×16 at 32×32)'),
    ),
    row(
      'Random noise',
      ...snap(NOISE_STOPS, () => cfg.data.noise ?? 0, (v) => (cfg.data.noise = v), {
        min: 0,
        max: 1,
        title: 'Std of Gaussian noise added once to each training pixel (pixels in [0, 1]); fixed per data seed, test set stays clean',
      }),
    ),
    row(
      'Label error',
      ...snap(LABEL_NOISE_STOPS, () => cfg.data.labelNoise ?? 0, (v) => (cfg.data.labelNoise = v), {
        min: 0,
        max: 1,
        title: 'Fraction of training labels replaced by a random wrong class; fixed per data seed, test set stays clean',
      }),
    ),
    info(() => `data ${formatCount(cfg.data.trainSize)} · ${formatCount(totalSteps(cfg) * effectiveBatch(cfg))} instances seen`),
    info(() => {
      if (backend !== 'remote') return '';
      const off = [cfg.data.crop && 'crop', cfg.data.cutout && 'cutout', cfg.data.noise && 'random noise', cfg.data.labelNoise && 'label error'].filter(Boolean);
      return off.length ? `The remote GPU backend only supports flips; ${off.join(', ')} ${off.length > 1 ? 'are' : 'is'} ignored there.` : '';
    }, 'info warn'),
  ),
);

// --- Batching
const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
side.append(
  section(
    'Batching',
    row(
      'Mode',
      seg(
        [
          ['full', 'Full batch'],
          ['minibatch', 'Mini-batch'],
          ['sgd', 'SGD (B=1)'],
        ],
        () => cfg.batchMode,
        (v) => (cfg.batchMode = v),
      ),
    ),
    row(
      'Batch size',
      ...snap(BATCHES, () => effectiveBatch(cfg), (v) => ((cfg.batchSize = v), (cfg.batchMode = 'minibatch')), {
        min: 1,
        max: 50000,
        int: true,
        scale: logScale,
      }),
    ),
    row(
      'Eval interval (% of train)',
      num(
        () => 100 / (cfg.numEvals - 1),
        (pct) => (cfg.numEvals = Math.round(100 / pct) + 1),
        { min: 0.25, max: 100, title: 'Evaluate on the test set every this % of training (plus at step 0)' },
      ),
    ),
  ),
);

// --- Model
const schematicEl = h('div');
const DROP_STOPS = [0, 0.05, 0.1, 0.2, 0.3, 0.5];
const setModel = (p: Partial<ModelConfig>) => {
  Object.assign(cfg.model, p);
  sync();
};
const widthSlider = h('input', { type: 'range', min: 0, max: WIDTHS.length - 1, step: 1, class: 'grow' });
widthSlider.addEventListener('input', () => setModel({ width: WIDTHS[+widthSlider.value] }));
const widthLbl = h('span', { class: 'info', style: 'width:38px;text-align:right' });
updaters.push(() => {
  const i = WIDTHS.indexOf(cfg.model.width);
  widthSlider.value = String(i >= 0 ? i : WIDTHS.findIndex((w) => w >= cfg.model.width));
  widthLbl.textContent = String(cfg.model.width);
});
const depthSlider = h('input', { type: 'range', min: 0, max: MAX_DEPTH, step: 1, class: 'grow' });
depthSlider.addEventListener('input', () => setModel({ depth: +depthSlider.value }));
const depthLbl = h('span', { class: 'info', style: 'width:38px;text-align:right' });
updaters.push(() => {
  depthSlider.value = String(cfg.model.depth);
  depthLbl.textContent = String(cfg.model.depth);
});

side.append(
  section(
    'Model',
    schematicEl,
    row('Width d', widthSlider, widthLbl),
    row('Depth L', depthSlider, depthLbl),
    row('MLP ratio', seg<number>([[1, '1×'], [2, '2×'], [4, '4×'], [8, '8×']], () => cfg.model.mlpRatio, (v) => (cfg.model.mlpRatio = v))),
    row(
      'Activation',
      seg<Act | 'none'>(
        [
          ['relu', 'ReLU'],
          ['gelu', 'GELU'],
          ['swiglu', 'SwiGLU'],
          ['none', 'off'],
        ],
        () => (cfg.model.actOn === false ? 'none' : cfg.model.act),
        (v) => {
          cfg.model.actOn = v !== 'none';
          if (v !== 'none') cfg.model.act = v;
        },
      ),
    ),
    row(
      'Norm',
      seg(
        [
          ['layernorm', 'LN'],
          ['rmsnorm', 'RMS'],
          ['batchnorm', 'BN'],
        ],
        () => cfg.model.norm,
        (v) => {
          cfg.model.norm = v;
          cfg.model.normPost = v;
          if (!cfg.model.preNorm && !cfg.model.postNorm) cfg.model.preNorm = true;
        },
      ),
      toggles([
        ['pre', () => cfg.model.preNorm, (v) => (cfg.model.preNorm = v), 'Normalize the input of each residual branch'],
        ['post', () => cfg.model.postNorm, (v) => (cfg.model.postNorm = v), 'Normalize the residual stream after each add'],
      ]),
    ),
    row(
      'Init',
      seg(
        [
          ['zero', '0-init'],
          ['uniform', 'Uniform'],
          ['kaiming', 'Kaiming'],
          ['xavier', 'Xavier'],
        ],
        () => cfg.model.init,
        (v) => (cfg.model.init = v),
        {
          zero: 'All weights zero',
          uniform: 'U(−1/√fan_in, 1/√fan_in), the PyTorch nn.Linear default',
          kaiming: 'Kaiming He: N(0, 2/fan_in)',
          xavier: 'Xavier: N(0, 2/(fan_in+fan_out))',
        },
      ),
    ),
    row(
      'Dropout',
      ...snap(DROP_STOPS, () => cfg.model.dropout ?? 0, (v) => (cfg.model.dropout = v), {
        min: 0,
        max: 0.9,
        title: 'Dropout probability at the sites selected below (training only)',
      }),
    ),
    hideUnless(
      row(
        'Dropout at',
        toggles(
          (
            [
              ['stem', 'stem', 'Stem output (the start of the residual stream)'],
              ['act', 'activation', 'Hidden activations inside each residual branch'],
              ['branch', 'branch out', 'Output of each residual branch, before it is added to the stream'],
            ] as [DropSite, string, string][]
          ).map(([site, label, title]) => [
            label,
            () => cfg.model.dropoutAt?.[site] !== false,
            (v: boolean) => (cfg.model.dropoutAt = { ...cfg.model.dropoutAt, [site]: v }),
            title,
          ]),
        ),
      ),
      () => (cfg.model.dropout ?? 0) > 0,
    ),
    row('', check('zero-init last layers', () => cfg.model.zeroInitLast, (v) => (cfg.model.zeroInitLast = v), 'Zero the output projection of each residual branch and the head')),
    info(() => {
      const L = buildLayout(cfg.model, 3 * cfg.data.resolution ** 2);
      return `params ${formatCount(L.numParams)} · ${formatSci(flopsPerExample(L))} train FLOPs/example`;
    }),
    info(() => modelWarnings(), 'info warn'),
  ),
);

function modelWarnings(): string {
  const w: string[] = [];
  const m = cfg.model;
  const bn = ((m.preNorm && m.norm === 'batchnorm') || (m.postNorm && postNormKind(m) === 'batchnorm')) && m.depth > 0;
  if (bn && (effectiveBatch(cfg) > 1024 || (cfg.microBatches ?? 0) > 1)) w.push('Batches > 1024 are split into micro-batches; BatchNorm uses per-micro-batch ("ghost") statistics.');
  return w.join('\n');
}

// --- Optimization
const LR_STOPS = [-5, -4, -3, -2, -1].flatMap((e) => [1, 2, 3, 5].map((m) => Number((m * 10 ** e).toPrecision(1)))).concat([1, 2, 3]);
const BETA1_STOPS = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];
const BETA2_STOPS = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999];
const CLIP_STOPS = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10];
const WD_STOPS = [0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

function hideUnless(el: HTMLElement, show: () => boolean) {
  updaters.push(() => (el.style.display = show() ? '' : 'none'));
  return el;
}
const beta1Lbl = h('span', { class: 'lbl' });
updaters.push(() => (beta1Lbl.textContent = cfg.opt.kind === 'adam' ? 'β1' : 'Momentum μ'));

side.append(
  section(
    'Optimizer',
    row(
      'Algorithm',
      seg(
        [
          ['sgd', 'SGD'],
          ['momentum', 'SGD+mom'],
          ['adam', 'AdamW'],
        ],
        () => cfg.opt.kind,
        (v) => (cfg.opt.kind = v),
      ),
    ),
    row('Peak LR', ...snap(LR_STOPS, () => cfg.opt.lr, (v) => (cfg.opt.lr = v), { min: 0, scale: logScale })),
    hideUnless(
      h(
        'div',
        { class: 'row' },
        beta1Lbl,
        ...snap(
          BETA1_STOPS,
          () => (cfg.opt.kind === 'adam' ? cfg.opt.beta1 : cfg.opt.momentum),
          (v) => (cfg.opt.kind === 'adam' ? (cfg.opt.beta1 = v) : (cfg.opt.momentum = v)),
          { min: 0, max: 0.9999, scale: betaScale },
        ),
      ),
      () => cfg.opt.kind !== 'sgd',
    ),
    hideUnless(row('β2', ...snap(BETA2_STOPS, () => cfg.opt.beta2, (v) => (cfg.opt.beta2 = v), { min: 0, max: 0.99999, scale: betaScale })), () => cfg.opt.kind === 'adam'),
    row('Grad clip', ...snap(CLIP_STOPS, () => cfg.opt.clip, (v) => (cfg.opt.clip = v), { min: 0, title: 'Global grad-norm clip (0 = off)', scale: (v) => Math.log1p(10 * v) })),
    row(
      'Weight decay',
      ...snap(WD_STOPS, () => cfg.opt.wd, (v) => (cfg.opt.wd = v), {
        min: 0,
        title: 'Decoupled (AdamW-style) weight decay on weight matrices: each step multiplies them by 1 − lr·wd',
        scale: (v) => Math.log1p(1e3 * v),
      }),
    ),
  ),
);

// --- Schedule
const schedEditor = new ScheduleEditor(h('div'), schedule);
updaters.push(() => {
  schedEditor.peakLr = cfg.opt.lr;
  schedEditor.draw();
});
const peakRange = h('input', { type: 'range', min: 0, max: LR_STOPS.length - 1, step: 1, class: 'vslider', title: 'Peak LR' });
peakRange.addEventListener('input', () => ((cfg.opt.lr = LR_STOPS[+peakRange.value]), sync()));
const peakLbl = h('span', { class: 'vslider-lbl' });
updaters.push(() => {
  const v = Math.log(cfg.opt.lr);
  let best = 0;
  LR_STOPS.forEach((s, i) => Math.abs(Math.log(s) - v) < Math.abs(Math.log(LR_STOPS[best]) - v) && (best = i));
  peakRange.value = String(best);
  peakLbl.textContent = fmtNum(cfg.opt.lr);
});
const peakSlider = h('div', { class: 'vslider-wrap', title: 'Peak LR (same as the Optimizer panel)' }, peakLbl, peakRange);
const presets: [Preset, string][] = [
  ['constant', 'const'],
  ['warmup-cosine', 'warmup+cos'],
  ['cosine', 'cosine'],
  ['linear', 'linear'],
  ['wsd', 'WSD'],
  ['step', 'step'],
  ['inv-sqrt', '1/√t'],
  ['cyclic', 'cyclic'],
];
side.append(
  section(
    'LR schedule',
    h('div', { class: 'sched-wrap' }, schedEditor.el, peakSlider),
    h(
      'div',
      { class: 'chips', style: 'margin-top:6px' },
      ...presets.map(([p, label]) => h('button', { onclick: () => schedEditor.set(presetSchedule(p, 0.05)) }, label)),
      h('button', { onclick: () => schedEditor.smooth(), title: 'Smooth the drawn curve' }, '〰 smooth'),
    ),
  ),
);

// --- Systems
side.append(
  section(
    'Systems',
    row(
      'Precision',
      seg<Precision>(
        [
          ['fp32', 'fp32'],
          ['mp', 'mixed'],
          ['bf16', 'bf16'],
        ],
        () => cfg.precision ?? 'fp32',
        (v) => (cfg.precision = v),
        {
          fp32: 'Everything in float32',
          mp: 'Mixed precision (like torch.autocast): matmul inputs rounded to bfloat16, fp32 accumulation and fp32 weights',
          bf16: 'Pure bfloat16: matmul inputs, outputs and the weights themselves are bf16, so small updates can be rounded away',
        },
      ),
    ),
    row(
      'Micro-batches',
      seg<number>(
        [
          [0, 'auto'],
          [2, '2'],
          [4, '4'],
          [8, '8'],
          [16, '16'],
        ],
        () => cfg.microBatches ?? 0,
        (v) => (cfg.microBatches = v),
        { 0: 'Split only when a batch does not fit in GPU memory' },
      ),
    ),
    info(() => {
      const w: string[] = [];
      if ((cfg.precision ?? 'fp32') !== 'fp32') w.push('bf16 is emulated by rounding fp32 values, so it changes the numerics but not the speed.');
      if (cfg.microBatches) w.push('Gradients are accumulated over the micro-batches; only BatchNorm (per-micro-batch statistics) and speed change.');
      return w.join('\n');
    }),
  ),
);

/** What the next Train or Run sweep would start: the remote GPU sets these up in advance. */
function upcomingConfigs(): TrainConfig[] {
  return [cfg, ...sweepConfigs().slice(0, pool.status?.runsPerGpu ?? 8).map((s) => s.c)];
}

function onConfigChanged() {
  // Deferred: this also runs during startup, before `pool` is initialized.
  queueMicrotask(() => backend === 'remote' && pool.prepare(upcomingConfigs()));
  const L = buildLayout(cfg.model, 3 * cfg.data.resolution ** 2);
  renderSchematic(schematicEl, cfg.model, L, setModel);
  sweepInfoUpdate();
}

// ============================================================================
// Tabs
// ============================================================================

const tabsEl = document.getElementById('tabs')!;
const tabNames: [string, string][] = [
  ['curves', 'Main'],
  ['sweep', 'Sweeps'],
  ['scaling', 'Scaling laws'],
];
let activeTab = 'curves';
const tabBtns = tabNames.map(([id, label]) => {
  const b = h('button', { class: 'tabbtn', onclick: () => setTab(id) }, label);
  tabsEl.append(b);
  return [id, b] as const;
});
tabsEl.append(h('div', { class: 'spacer' }));
const queueInfo = h('span', { class: 'muted' });
tabsEl.append(queueInfo);

function setTab(id: string) {
  activeTab = id;
  tabBtns.forEach(([t, b]) => b.classList.toggle('on', t === id));
  for (const [t] of tabNames) document.getElementById(`tab-${t}`)!.classList.toggle('on', t === id);
  requestRender();
}

// ============================================================================
// Curves tab
// ============================================================================

type XAxis = 'step' | 'epoch' | 'examples' | 'flops';
const view = { xAxis: 'examples' as XAxis, smoothing: 0, aggregate: false, logY: false };
const curvesTab = document.getElementById('tab-curves')!;
const smoothSlider = h('input', { type: 'range', min: 0, max: 0.995, step: 0.005, value: String(view.smoothing) });
smoothSlider.addEventListener('input', () => {
  view.smoothing = +smoothSlider.value;
  requestRender();
});
const toolbar = h(
  'div',
  { class: 'toolbar' },
  h(
    'label',
    {},
    'x-axis',
    sel<XAxis>(
      [
        ['step', 'steps'],
        ['epoch', 'epochs'],
        ['examples', 'data'],
        ['flops', 'compute (FLOPs)'],
      ],
      () => view.xAxis,
      (v) => ((view.xAxis = v), requestRender()),
    ),
  ),
  h('label', {}, 'smoothing', smoothSlider),
  check('log-y loss', () => view.logY, (v) => ((view.logY = v), requestRender())),
  check('aggregate seeds (mean ± std)', () => view.aggregate, (v) => ((view.aggregate = v), requestRender()), 'Runs that differ only in seed are merged into one band'),
);
curvesTab.append(toolbar);
const grid = h('div', { class: 'grid' });
curvesTab.append(grid);
const plotTrainLoss = new Plot(grid, { title: 'Train loss', logY: true, height: 230 });
const plotTestLoss = new Plot(grid, { title: 'Test loss', logY: true, height: 230 });
const plotTestAcc = new Plot(grid, { title: 'Test accuracy', height: 230 });
const plotLr = new Plot(grid, { title: 'Learning rate', height: 230 });
const plotGnorm = new Plot(grid, { title: 'Gradient norm ‖g‖', logY: true, height: 230 });
const plotPnorm = new Plot(grid, { title: 'Parameter norm ‖θ‖', height: 230 });
const plotLogitMax = new Plot(grid, { title: 'Max logit |z|', logY: true, height: 230 });
const plotActMax = new Plot(grid, { title: 'Max activation |h| (residual stream)', logY: true, height: 230 });
const filterCard = h('div', { class: 'card' });
const filterTitle = h('h4', {}, 'First-layer weights');
const filterCanvas = h('canvas', { id: 'filters' });
filterCard.append(filterTitle, filterCanvas);
grid.append(filterCard);
const dataCard = h('div', { class: 'card data-card' });
const dataTitle = h('h4', {}, 'Data');
const dataCanvas = h('canvas', { id: 'batch' });
const dataInfo = h('div', { class: 'info' });
const zoomCanvas = h('canvas');
const zoomCaption = h('div', { class: 'batch-zoom-caption' });
const dataZoom = h('div', { class: 'batch-zoom' }, zoomCanvas, zoomCaption);
dataCard.append(dataTitle, dataCanvas, dataInfo, dataZoom);
grid.append(dataCard);
new ResizeObserver(() => renderBatch(true)).observe(dataCard);
const runsTableWrap = h('div');
curvesTab.append(runsTableWrap);

function xOf(r: Run, step: number): number {
  switch (view.xAxis) {
    case 'step':
      return step;
    case 'epoch':
      return (step * r.B) / r.cfg.data.trainSize;
    case 'examples':
      return step * r.B;
    case 'flops':
      return step * r.B * r.flopsPerEx;
  }
}

function aggKey(r: Run): string {
  return JSON.stringify({ ...r.cfg, initSeed: 0, dataSeed: 0, schedule: r.cfg.schedule === schedule ? 'live' : r.cfg.schedule, g: r.group });
}

function decimate(x: number[], y: number[], lo?: number[], hi?: number[], max = 1500) {
  if (x.length <= max) return { x, y, lo, hi };
  const s = Math.ceil(x.length / max);
  const pick = <T>(a: T[]) => a.filter((_, i) => i % s === 0 || i === a.length - 1);
  return { x: pick(x), y: pick(y), lo: lo && pick(lo), hi: hi && pick(hi) };
}

function curveSeries(field: 'loss' | 'acc' | 'gnorm' | 'pnorm' | 'logitMax' | 'actMax' | 'lr' | 'evLoss' | 'evAcc', smooth: boolean): Series[] {
  const visible = runs.filter((r) => r.visible && r.step.length + r.evStep.length > 0);
  const groups = new Map<string, Run[]>();
  for (const r of visible) {
    const k = view.aggregate ? aggKey(r) : String(r.id);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(r);
  }
  const out: Series[] = [];
  for (const rs of groups.values()) {
    const r0 = rs[0];
    const isEval = field === 'evLoss' || field === 'evAcc';
    const ys = rs.map((r) => {
      const raw = r[field];
      return smooth ? ema(raw, view.smoothing) : raw.slice();
    });
    const steps = isEval ? r0.evStep : r0.step;
    const running = rs.some((r) => r.status === 'running');
    const width = running ? 2.2 : 1.5;
    if (rs.length === 1) {
      const d = decimate(
        steps.map((s) => xOf(r0, s)),
        ys[0],
      );
      out.push({ label: r0.label, color: r0.color, x: d.x, y: d.y, width, points: isEval && steps.length < 60, pointRadius: 2 });
    } else {
      const n = Math.min(...ys.map((y) => y.length));
      const x: number[] = [];
      const m: number[] = [];
      const lo: number[] = [];
      const hi: number[] = [];
      for (let i = 0; i < n; i++) {
        const st = meanStd(ys.map((y) => y[i]));
        x.push(xOf(r0, steps[i]));
        m.push(st.mean);
        lo.push(st.mean - st.std);
        hi.push(st.mean + st.std);
      }
      const d = decimate(x, m, lo, hi);
      out.push({ label: `${r0.baseLabel} (n=${rs.length})`, color: r0.color, x: d.x, y: d.y, lo: d.lo, hi: d.hi, width });
    }
  }
  return out;
}

function renderCurves() {
  const xl = { step: 'step', epoch: 'epoch', examples: 'instances', flops: 'FLOPs' }[view.xAxis];
  const logX = view.xAxis === 'flops';
  const xTickFormat = view.xAxis === 'examples' ? formatCount : undefined;
  for (const p of [plotTrainLoss, plotTestLoss, plotTestAcc, plotGnorm, plotPnorm, plotLogitMax, plotActMax, plotLr]) p.setOpts({ xLabel: xl, logX, xTickFormat });
  plotTrainLoss.setOpts({ logY: view.logY });
  plotTestLoss.setOpts({ logY: view.logY });
  plotTrainLoss.setSeries(curveSeries('loss', true));
  plotTestLoss.setSeries(curveSeries('evLoss', false));
  plotTestAcc.setSeries(curveSeries('evAcc', false));
  plotGnorm.setSeries(curveSeries('gnorm', true));
  plotPnorm.setSeries(curveSeries('pnorm', false));
  plotLogitMax.setSeries(curveSeries('logitMax', true));
  plotActMax.setSeries(curveSeries('actMax', true));
  plotLr.setSeries(curveSeries('lr', false));
  renderFilters();
  renderBatch();
}

function renderFilters() {
  if (!lastStem) return;
  const { w, run } = lastStem;
  const res = run.cfg.data.resolution;
  const d = run.cfg.model.width;
  const hw = res * res;
  const nShow = Math.min(d, 64);
  const cols = Math.min(nShow, 16);
  const rowsN = Math.ceil(nShow / cols);
  const tile = res + 1;
  const W = cols * tile + 1;
  const H = rowsN * tile + 1;
  filterCanvas.width = W;
  filterCanvas.height = H;
  const ctx = filterCanvas.getContext('2d')!;
  const img = ctx.createImageData(W, H);
  img.data.fill(0);
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;
  for (let u = 0; u < nShow; u++) {
    let mx = 1e-12;
    for (let k = 0; k < 3 * hw; k++) mx = Math.max(mx, Math.abs(w[k * d + u]));
    const ox = (u % cols) * tile + 1;
    const oy = Math.floor(u / cols) * tile + 1;
    for (let y = 0; y < res; y++)
      for (let x = 0; x < res; x++) {
        const p = ((oy + y) * W + ox + x) * 4;
        for (let c = 0; c < 3; c++) img.data[p + c] = Math.round(128 + 127 * (w[(c * hw + y * res + x) * d + u] / mx));
      }
  }
  ctx.putImageData(img, 0, 0);
  filterTitle.textContent = `First-layer weights — ${run.label}`;
}

const CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

let batchDrawn: BatchSample | null = null;
function renderBatch(force = false) {
  if (!lastBatch || (!force && batchDrawn === lastBatch.b)) return;
  batchDrawn = lastBatch.b;
  const { b, run, mean, std } = lastBatch;
  const res = run.cfg.data.resolution;
  const hw = res * res;
  const D = 3 * hw;
  const dpr = window.devicePixelRatio || 1;
  const avail = dataCanvas.parentElement!.clientWidth - 20;
  if (avail <= 0) return;
  const tile = b.n <= 4 ? 64 : b.n <= 16 ? 40 : b.n <= 64 ? 28 : 22;
  const gap = 3;
  const cols = Math.max(1, Math.min(b.n, Math.floor((avail + gap) / (tile + gap))));
  const rowsN = Math.ceil(b.n / cols);
  const W = cols * (tile + gap) - gap;
  const H = rowsN * (tile + gap) - gap;
  dataCanvas.width = Math.round(W * dpr);
  dataCanvas.height = Math.round(H * dpr);
  dataCanvas.style.width = W + 'px';
  dataCanvas.style.height = H + 'px';
  const ctx = dataCanvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  const off = document.createElement('canvas');
  off.width = res;
  off.height = res;
  const octx = off.getContext('2d')!;
  const img = octx.createImageData(res, res);
  const fill = (i: number) => {
    for (let p = 0; p < hw; p++) {
      for (let c = 0; c < 3; c++) img.data[p * 4 + c] = Math.max(0, Math.min(255, Math.round((b.inputs[i * D + c * hw + p] * std[c] + mean[c]) * 255)));
      img.data[p * 4 + 3] = 255;
    }
    return img;
  };
  const predOf = (i: number) => {
    let arg = 0;
    for (let k = 1; k < 10; k++) if (b.logits[i * 10 + k] > b.logits[i * 10 + arg]) arg = k;
    return arg;
  };
  let correct = 0;
  for (let i = 0; i < b.n; i++) {
    octx.putImageData(fill(i), 0, 0);
    const ok = predOf(i) === b.labels[i];
    if (ok) correct++;
    const x = (i % cols) * (tile + gap);
    const y = Math.floor(i / cols) * (tile + gap);
    ctx.drawImage(off, x + 2, y + 2, tile - 4, tile - 4);
    ctx.strokeStyle = ok ? '#4fd18b' : '#ff6b8b';
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, tile - 2, tile - 2);
  }
  const B = run.B;
  const shown = b.n < B ? `showing ${b.n} of ${B.toLocaleString()}` : `all ${B}`;
  dataTitle.textContent = `Data — step ${b.step.toLocaleString()} · batch ${B.toLocaleString()} (${shown})`;
  dataInfo.textContent = `${correct}/${b.n} correct (${((100 * correct) / b.n).toFixed(0)}%)`;
  // Hover: show the example enlarged, centered on its tile and kept inside the card.
  zoomCanvas.width = res;
  zoomCanvas.height = res;
  const zctx = zoomCanvas.getContext('2d')!;
  let hovered = -1;
  const hideZoom = () => {
    hovered = -1;
    dataZoom.style.display = 'none';
  };
  dataCanvas.onmouseleave = hideZoom;
  dataCanvas.onmousemove = (e) => {
    const r = dataCanvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - r.left) / (tile + gap));
    const row = Math.floor((e.clientY - r.top) / (tile + gap));
    const i = row * cols + col;
    if (col < 0 || col >= cols || i < 0 || i >= b.n) return hideZoom();
    if (i === hovered) return;
    hovered = i;
    zctx.putImageData(fill(i), 0, 0);
    const pred = predOf(i);
    const ok = pred === b.labels[i];
    dataZoom.classList.toggle('wrong', !ok);
    zoomCaption.textContent = ok ? CIFAR_CLASSES[pred] : `${CIFAR_CLASSES[pred]} (is ${CIFAR_CLASSES[b.labels[i]]})`;
    dataZoom.style.display = 'block';
    const card = dataCard.getBoundingClientRect();
    const zw = dataZoom.offsetWidth;
    const zh = dataZoom.offsetHeight;
    const cx = r.left - card.left + col * (tile + gap) + tile / 2;
    const cy = r.top - card.top + row * (tile + gap) + tile / 2;
    dataZoom.style.left = `${Math.max(4, Math.min(card.width - zw - 4, cx - zw / 2))}px`;
    dataZoom.style.top = `${Math.max(4, cy - zh / 2)}px`;
  };
}

function lastFinite(a: number[]): number {
  for (let i = a.length - 1; i >= 0; i--) if (isFinite(a[i])) return a[i];
  return NaN;
}

function finalTrainLoss(r: Run): number {
  const n = r.loss.length;
  if (!n) return NaN;
  const k = Math.max(1, Math.floor(n * 0.05));
  return meanStd(r.loss.slice(n - k)).mean;
}

let tableTimer = 0;
function renderRunsTable() {
  const now = performance.now();
  if (now - tableTimer < 250) return;
  tableTimer = now;
  const t = h(
    'table',
    { class: 'runs' },
    h(
      'tr',
      {},
      ...['', '', 'run', 'status', 'params', 'data', 'steps', 'B', 'train loss', 'test loss', 'test acc', 'steps/s', ''].map((c) => h('th', {}, c)),
    ),
  );
  for (const r of [...runs].reverse()) {
    const cb = h('input', { type: 'checkbox', checked: r.visible });
    cb.addEventListener('change', () => {
      r.visible = cb.checked;
      requestRender();
    });
    const del = h('button', { style: 'padding:0 6px', title: 'remove', onclick: () => removeRun(r) }, '×');
    t.append(
      h(
        'tr',
        {},
        h('td', {}, cb),
        h('td', {}, h('span', { class: 'swatch', style: `background:${r.color}` })),
        h('td', { title: r.error ?? '' }, r.label),
        h('td', { class: `st-${r.status}` }, r.status === 'running' ? `${Math.round((100 * (r.step.at(-1) ?? 0)) / r.T)}%` : r.status),
        h('td', { class: 'num' }, formatCount(r.numParams)),
        h('td', { class: 'num', title: 'training-set size' }, formatCount(r.cfg.data.trainSize)),
        h('td', { class: 'num' }, r.T.toLocaleString()),
        h('td', { class: 'num' }, String(r.B)),
        h('td', { class: 'num' }, fmtNum(finalTrainLoss(r))),
        h('td', { class: 'num' }, fmtNum(lastFinite(r.evLoss))),
        h('td', { class: 'num' }, r.evAcc.length ? (100 * lastFinite(r.evAcc)).toFixed(1) + '%' : ''),
        h('td', { class: 'num' }, r.sps ? Math.round(r.sps).toLocaleString() : ''),
        h('td', {}, del),
      ),
    );
  }
  runsTableWrap.replaceChildren(
    h(
      'div',
      { class: 'toolbar' },
      h('b', { style: 'color:var(--text)' }, `Runs (${runs.length})`),
      h('button', { onclick: () => (runs.forEach((r) => (r.visible = true)), requestRender(true)) }, 'show all'),
      h('button', { onclick: () => (runs.forEach((r) => (r.visible = false)), requestRender(true)) }, 'hide all'),
    ),
    t,
  );
}

// ============================================================================
// Sweep tab
// ============================================================================

const sweepTab = document.getElementById('tab-sweep')!;
const PRESET_LRS = '3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1';
const PRESET_BATCHES = '1024, 256, 64, 16, 4, 2, 1';
const PRESET_WDS = '0, 0.01, 0.1, 1, 10';
const SEEDS_8 = '0, 1, 2, 3, 4, 5, 6, 7';
const sweepState = {
  p1: 'lr',
  v1: PRESET_LRS,
  p2: 'batchSize',
  v2: PRESET_BATCHES,
  repeats: 1,
  vary: 'both' as 'init' | 'data' | 'both',
  metric: 'testLoss' as 'testLoss' | 'testAcc' | 'trainLoss' | 'bestTestLoss',
};
let lastSweep: { group: string; p1: string; p2: string; v1: (number | string)[]; v2: (number | string)[] } | null = null;
let sweepCounter = 0;

const paramOpts: [string, string][] = Object.entries(SWEEP_PARAMS).map(([k, p]) => [k, p.label]);
const v1Input = h('input', { class: 'grow', type: 'text' });
v1Input.addEventListener('change', () => ((sweepState.v1 = v1Input.value), sync()));
updaters.push(() => (v1Input.value = sweepState.v1));
const v2Input = h('input', { class: 'grow', type: 'text' });
v2Input.addEventListener('change', () => ((sweepState.v2 = v2Input.value), sync()));
updaters.push(() => {
  v2Input.value = sweepState.v2;
  v2Input.disabled = !sweepState.p2;
});
const sweepInfo = h('div', { class: 'info' });

function sweepConfigs(): { c: TrainConfig; v1?: number | string; v2?: number | string; rep: number }[] {
  const p1 = SWEEP_PARAMS[sweepState.p1];
  const p2 = sweepState.p2 ? SWEEP_PARAMS[sweepState.p2] : null;
  const vs1 = parseValues(p1, sweepState.v1);
  const vs2 = p2 ? parseValues(p2, sweepState.v2) : [undefined];
  const out: { c: TrainConfig; v1?: number | string; v2?: number | string; rep: number }[] = [];
  for (const v2 of vs2)
    for (const v1 of vs1)
      for (let rep = 0; rep < sweepState.repeats; rep++) {
        const c = structuredClone({ ...cfg, schedule: cfg.schedule.slice() });
        if (sweepState.repeats > 1) {
          if (sweepState.vary !== 'data') c.initSeed = cfg.initSeed + rep;
          if (sweepState.vary !== 'init') c.dataSeed = cfg.dataSeed + rep;
        }
        p1.apply(c, v1);
        if (p2 && v2 !== undefined) p2.apply(c, v2);
        out.push({ c, v1, v2, rep });
      }
  return out;
}

function sweepInfoUpdate() {
  const cs = sweepConfigs();
  let C = 0;
  for (const { c } of cs) C += totalSteps(c) * effectiveBatch(c) * flopsPerExample(buildLayout(c.model, 3 * c.data.resolution ** 2));
  sweepInfo.textContent = `${cs.length} runs · total ≈ ${formatSci(C)} FLOPs. Every other setting comes from the sidebar.`;
}

function runSweep() {
  const cs = sweepConfigs();
  if (!cs.length) return;
  const group = `sweep${++sweepCounter}`;
  const p1 = SWEEP_PARAMS[sweepState.p1];
  const p2 = sweepState.p2 ? SWEEP_PARAMS[sweepState.p2] : null;
  const vs1 = parseValues(p1, sweepState.v1);
  const vs2 = p2 ? parseValues(p2, sweepState.v2) : [];
  lastSweep = { group, p1: sweepState.p1, p2: sweepState.p2, v1: vs1, v2: vs2 };
  for (const r of runs) r.visible = false;
  const rs = cs.map(({ c, v1, v2, rep }) => {
    const i1 = vs1.indexOf(v1!);
    const i2 = p2 ? vs2.indexOf(v2!) : 0;
    const color = p2 && vs1.length <= 1 ? PALETTE[i2 % PALETTE.length] : cmap(PLASMA, vs1.length > 1 ? i1 / (vs1.length - 1) : 0.5);
    const base = `${shortName(sweepState.p1)}=${fmtVal(v1!)}` + (p2 ? ` ${shortName(sweepState.p2)}=${fmtVal(v2!)}` : '');
    return makeRun(c, {
      group,
      color,
      label: base + (sweepState.repeats > 1 ? ` #${rep}` : ''),
      baseLabel: base,
      sweep: { v1, v2, rep },
    });
  });
  enqueue(rs);
}

const shortName = (k: string) => ({ lr: 'lr', batchSize: 'B', width: 'd', depth: 'L', mlpRatio: 'r', initSeed: 'iseed', dataSeed: 'dseed', trainSize: 'N' })[k] ?? k;
const fmtVal = (v: number | string) => (typeof v === 'number' ? fmtNum(v) : v);

const sweepSummary = h('div', { class: 'two' });
const plotSweep = new Plot(sweepSummary, { title: 'Sweep summary', height: 300, legend: true, padX: true });
const sweepHeatmap = new Heatmap(sweepSummary, 300);
sweepHeatmap.set(null);
const plotSweepCurves = new Plot(sweepSummary, { title: 'Test loss during training', logY: true, height: 300 });
const plotSweepExamples = new Plot(sweepSummary, { title: 'Test loss vs data', xLabel: 'instances', xTickFormat: formatCount, logY: true, height: 300 });

sweepTab.append(
  h(
    'div',
    { class: 'section', style: 'margin-top:12px' },
    h('h3', {}, 'Grid sweep'),
    row(
      'Parameter 1',
      sel(paramOpts, () => sweepState.p1, (v) => ((sweepState.p1 = v), (sweepState.v1 = SWEEP_PARAMS[v].defaults))),
      v1Input,
    ),
    row(
      'Parameter 2',
      sel<string>([['', '— none —'], ...paramOpts], () => sweepState.p2, (v) => ((sweepState.p2 = v), (sweepState.v2 = v ? SWEEP_PARAMS[v].defaults : ''))),
      v2Input,
    ),
    row(
      'Repeats',
      num(() => sweepState.repeats, (v) => (sweepState.repeats = v), { int: true, min: 1, max: 20 }),
      h('span', { class: 'muted' }, 'varying'),
      seg(
        [
          ['init', 'init seed'],
          ['data', 'data order'],
          ['both', 'both'],
        ],
        () => sweepState.vary,
        (v) => (sweepState.vary = v),
      ),
    ),
    row(
      'Metric',
      seg(
        [
          ['testLoss', 'final test loss'],
          ['bestTestLoss', 'best test loss'],
          ['testAcc', 'final test acc'],
          ['trainLoss', 'final train loss'],
        ],
        () => sweepState.metric,
        (v) => ((sweepState.metric = v), requestRender()),
      ),
    ),
    h('div', { class: 'row' }, h('button', { class: 'primary', onclick: () => runSweep() }, '▶ Run sweep'), h('button', { onclick: () => stopAll() }, '■ Stop')),
    sweepInfo,
    h(
      'div',
      { class: 'chips' },
      h('span', { class: 'muted' }, 'preset:'),
      h('button', { onclick: () => quickSweep('lr', '', 1) }, 'LR'),
      h('button', { onclick: () => quickSweep('lr', 'batchSize', 1, PRESET_LRS, PRESET_BATCHES) }, 'LR × batch size'),
      h('button', { onclick: () => quickSweep('init', '', 3) }, 'init schemes × 3 seeds'),
      h('button', { onclick: () => quickSweep('lr', 'wd', 1, PRESET_LRS, PRESET_WDS) }, 'LR × weight decay'),
      h('button', { onclick: () => quickSweep('initSeed', 'dataSeed', 1, SEEDS_8, SEEDS_8) }, '8 init seeds × 8 data orders'),
      h('button', { onclick: () => quickSweep('act', 'norm', 1) }, 'act × norm'),
      h('button', { onclick: () => quickSweep('batchMode', '', 1) }, 'full vs mini vs SGD'),
    ),
  ),
  sweepSummary,
);

function quickSweep(p1: string, p2: string, repeats: number, v1?: string, v2?: string) {
  sweepState.p1 = p1;
  sweepState.v1 = v1 ?? SWEEP_PARAMS[p1].defaults;
  sweepState.p2 = p2;
  sweepState.v2 = p2 ? (v2 ?? SWEEP_PARAMS[p2].defaults) : '';
  sweepState.repeats = repeats;
  sync();
}

function runMetric(r: Run): number {
  if (r.status === 'diverged') return sweepState.metric === 'testAcc' ? 0.1 : NaN;
  switch (sweepState.metric) {
    case 'testLoss':
      return lastFinite(r.evLoss);
    case 'bestTestLoss':
      return r.evLoss.length ? Math.min(...r.evLoss.filter(isFinite)) : NaN;
    case 'testAcc':
      return lastFinite(r.evAcc);
    case 'trainLoss':
      return finalTrainLoss(r);
  }
}

function renderSweep() {
  if (!lastSweep) {
    plotSweep.setSeries([]);
    plotSweepCurves.setSeries([]);
    plotSweepExamples.setSeries([]);
    sweepHeatmap.set(null);
    return;
  }
  const S = lastSweep;
  const p1 = SWEEP_PARAMS[S.p1];
  const p2 = S.p2 ? SWEEP_PARAMS[S.p2] : null;
  const rs = runs.filter((r) => r.group === S.group);
  const numericX = p1.numeric;
  const xsRaw = S.v1.map((v, i) => (numericX ? (v as number) : i));
  const pos = xsRaw.filter((x) => x > 0);
  const logX = numericX && pos.length === xsRaw.length && Math.max(...pos) / Math.min(...pos) >= 10;
  const series: Series[] = [];
  const v2s = p2 ? S.v2 : [undefined];
  v2s.forEach((v2, j) => {
    const x: number[] = [];
    const y: number[] = [];
    const lo: number[] = [];
    const hi: number[] = [];
    S.v1.forEach((v1, i) => {
      const done = rs.filter((r) => r.sweep?.v1 === v1 && r.sweep?.v2 === v2 && r.status !== 'queued' && r.status !== 'running');
      if (!done.length) return;
      const st = meanStd(done.map(runMetric));
      x.push(xsRaw[i]);
      y.push(st.mean);
      lo.push(st.mean - st.std);
      hi.push(st.mean + st.std);
    });
    series.push({
      label: p2 ? `${p2.label} = ${fmtVal(v2!)}` : '',
      color: p2 ? PALETTE[j % PALETTE.length] : '#ffaa3c',
      x,
      y,
      lo: rs.some((r) => (r.sweep?.rep ?? 0) > 0) ? lo : undefined,
      hi: rs.some((r) => (r.sweep?.rep ?? 0) > 0) ? hi : undefined,
      points: true,
      pointRadius: 3.5,
      width: 2,
    });
  });
  const metricLabel = { testLoss: 'final test loss', bestTestLoss: 'best test loss', testAcc: 'final test accuracy', trainLoss: 'final train loss' }[sweepState.metric];
  const diverged = rs.filter((r) => r.status === 'diverged').length;
  plotSweep.setOpts({
    title: `${metricLabel} vs ${p1.label}${diverged ? `   (${diverged} diverged)` : ''}`,
    xLabel: p1.label,
    logX,
    logY: sweepState.metric !== 'testAcc',
    xCategories: numericX ? undefined : (S.v1 as string[]),
  });
  plotSweep.setSeries(series);
  if (p2) {
    const z: number[][] = [];
    const missing: ('pending' | 'diverged' | null)[][] = [];
    for (const v2 of S.v2) {
      const zr: number[] = [];
      const mr: ('pending' | 'diverged' | null)[] = [];
      for (const v1 of S.v1) {
        const cellRuns = rs.filter((r) => r.sweep?.v1 === v1 && r.sweep?.v2 === v2);
        const done = cellRuns.filter((r) => r.status !== 'queued' && r.status !== 'running');
        const vals = done.filter((r) => r.status !== 'diverged').map(runMetric).filter(isFinite);
        zr.push(vals.length ? meanStd(vals).mean : NaN);
        mr.push(vals.length ? null : done.length && done.every((r) => r.status === 'diverged') ? 'diverged' : 'pending');
      }
      z.push(zr);
      missing.push(mr);
    }
    sweepHeatmap.set({
      title: `${metricLabel} · ${p1.label} × ${p2.label}`,
      xLabel: p1.label,
      yLabel: p2.label,
      xTicks: S.v1.map(fmtVal),
      yTicks: S.v2.map(fmtVal),
      z,
      missing,
      lowerIsBetter: sweepState.metric !== 'testAcc',
      logScale: sweepState.metric !== 'testAcc',
    });
  } else {
    sweepHeatmap.set(null);
  }
  const sub = runs.filter((r) => r.group === S.group);
  const saved = runs.map((r) => r.visible);
  runs.forEach((r) => (r.visible = sub.includes(r)));
  plotSweepCurves.setOpts({ xLabel: view.xAxis, logX: view.xAxis === 'flops', xTickFormat: view.xAxis === 'examples' ? formatCount : undefined });
  plotSweepCurves.setSeries(curveSeries('evLoss', false));
  const xAxis = view.xAxis;
  view.xAxis = 'examples';
  plotSweepExamples.setSeries(curveSeries('evLoss', false));
  view.xAxis = xAxis;
  runs.forEach((r, i) => (r.visible = saved[i]));
}

// ============================================================================
// Scaling-law tab
// ============================================================================

const scalingTab = document.getElementById('tab-scaling')!;
const scaleState = { widths: '16, 32, 64, 128, 256, 512', source: 'test' as 'test' | 'train', skip: 25, evals: 20 };
let lastScaling: string | null = null;
let scaleCounter = 0;
const widthsInput = h('input', { class: 'grow', type: 'text' });
widthsInput.addEventListener('change', () => ((scaleState.widths = widthsInput.value), sync()));
updaters.push(() => (widthsInput.value = scaleState.widths));

function scalingConfigs(): TrainConfig[] {
  return scaleState.widths
    .split(/[,\s]+/)
    .map(Number)
    .filter((w) => w > 0)
    .map((w) => {
      const c = structuredClone({ ...cfg, schedule: cfg.schedule.slice() });
      c.model.width = Math.round(w);
      c.numEvals = scaleState.evals;
      c.logEvals = true;
      return c;
    });
}

function runScaling() {
  const cs = scalingConfigs();
  if (!cs.length) return;
  const group = `scale${++scaleCounter}`;
  lastScaling = group;
  for (const r of runs) r.visible = false;
  enqueue(
    cs.map((c, i) => {
      const label = `d=${c.model.width} (${formatCount(buildLayout(c.model, 3 * c.data.resolution ** 2).numParams)})`;
      return makeRun(c, { group, color: cmap(VIRIDIS, cs.length > 1 ? 0.1 + (0.9 * i) / (cs.length - 1) : 0.5), label, baseLabel: label });
    }),
  );
}

const fitText = h('div', { class: 'fit' });
const scaleWrap = h('div', { class: 'two' });
const plotScale = new Plot(scaleWrap, { title: 'Loss vs training compute', logX: true, logY: true, height: 360, xLabel: 'C = 6·N·examples (FLOPs)' });
const plotScaleN = new Plot(scaleWrap, { title: 'Final loss vs parameters', logX: true, logY: true, height: 360, xLabel: 'N (params)', padX: true });

scalingTab.append(
  h(
    'div',
    { class: 'section', style: 'margin-top:12px' },
    h('h3', {}, 'Tiny compute scaling law · L(C) = α · C^β'),
    row('Widths', widthsInput),
    row(
      'Fit on',
      seg(
        [
          ['test', 'test loss'],
          ['train', 'train loss (smoothed)'],
        ],
        () => scaleState.source,
        (v) => ((scaleState.source = v), requestRender()),
      ),
    ),
    row(
      'Skip first',
      num(() => scaleState.skip, (v) => ((scaleState.skip = v), requestRender()), { min: 0, max: 95 }),
      h('span', { class: 'muted' }, '% of the log-compute range (early transient)'),
    ),
    row('Eval points', num(() => scaleState.evals, (v) => (scaleState.evals = v), { min: 4, max: 200, int: true }), h('span', { class: 'muted' }, 'log-spaced per run')),
    h('div', { class: 'row' }, h('button', { class: 'primary', onclick: () => runScaling() }, '▶ Run scaling sweep'), h('button', { onclick: () => stopAll() }, '■ Stop')),
  ),
  fitText,
  scaleWrap,
);

function renderScaling() {
  if (!lastScaling) {
    plotScale.setSeries([]);
    plotScaleN.setSeries([]);
    fitText.textContent = '';
    return;
  }
  const rs = runs.filter((r) => r.group === lastScaling);
  const series: Series[] = [];
  const pts: { c: number; l: number }[] = [];
  for (const r of rs) {
    let C: number[];
    let Lv: number[];
    if (scaleState.source === 'test') {
      C = r.evStep.map((s) => s * r.B * r.flopsPerEx);
      Lv = r.evLoss;
    } else {
      const sm = ema(r.loss, 0.95);
      const d = decimate(
        r.step.map((s) => s * r.B * r.flopsPerEx),
        sm,
        undefined,
        undefined,
        400,
      );
      C = d.x;
      Lv = d.y;
    }
    C.forEach((c, i) => pts.push({ c, l: Lv[i] }));
    series.push({ label: r.label, color: r.color, x: C, y: Lv, width: 1.5, alpha: 0.85, points: scaleState.source === 'test', pointRadius: 1.8 });
  }
  const front = frontier(pts);
  let fit: PowerFit | null = null;
  let used: { c: number; l: number }[] = [];
  if (front.length >= 2) {
    const lo = Math.log(front[0].c);
    const hi = Math.log(front[front.length - 1].c);
    const cut = Math.exp(lo + (scaleState.skip / 100) * (hi - lo));
    used = front.filter((p) => p.c >= cut);
    fit = fitPowerLaw(
      used.map((p) => p.c),
      used.map((p) => p.l),
    );
  }
  series.push({ label: 'frontier (fit points)', color: '#ffffff', x: used.map((p) => p.c), y: used.map((p) => p.l), line: false, points: true, pointRadius: 3.5 });
  plotScale.overlay = (ctx, tf) => {
    if (!fit) return;
    const c0 = tf.dom.xmin;
    const c1 = tf.dom.xmax;
    ctx.strokeStyle = '#ffaa3c';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    const N = 40;
    for (let i = 0; i <= N; i++) {
      const c = Math.exp(Math.log(c0) + (i / N) * (Math.log(c1) - Math.log(c0)));
      const l = fit.alpha * Math.pow(c, fit.beta);
      const x = tf.x(c);
      const y = tf.y(l);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  plotScale.setSeries(series);
  fitText.innerHTML = fit
    ? `L(C) = <b>${fit.alpha.toPrecision(3)}</b> · C<sup><b>${fit.beta.toFixed(4)}</b></sup> &nbsp;&nbsp; <span class="muted">R² = ${fit.r2.toFixed(3)}, ${fit.n} frontier points · 10× compute → loss × ${Math.pow(10, fit.beta).toFixed(3)}</span>`
    : '<span class="muted">Waiting for runs…</span>';

  // Final loss vs N
  const done = rs.filter((r) => r.evLoss.length && r.status !== 'running' && r.status !== 'queued');
  const N = done.map((r) => r.numParams);
  const Lf = done.map((r) => (scaleState.source === 'test' ? lastFinite(r.evLoss) : finalTrainLoss(r)));
  const fitN = fitPowerLaw(N, Lf);
  plotScaleN.overlay = (ctx, tf) => {
    if (!fitN) return;
    ctx.strokeStyle = '#7aa2ff';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tf.x(tf.dom.xmin), tf.y(fitN.alpha * Math.pow(tf.dom.xmin, fitN.beta)));
    ctx.lineTo(tf.x(tf.dom.xmax), tf.y(fitN.alpha * Math.pow(tf.dom.xmax, fitN.beta)));
    ctx.stroke();
    ctx.setLineDash([]);
  };
  plotScaleN.setOpts({ title: fitN ? `Final loss vs parameters — L ∝ N^${fitN.beta.toFixed(3)}` : 'Final loss vs parameters' });
  plotScaleN.setSeries(done.map((r, i) => ({ label: r.label, color: r.color, x: [N[i]], y: [Lf[i]], points: true, line: false, pointRadius: 5 })));
}

// ============================================================================
// Run management
// ============================================================================

function describe(c: TrainConfig): string {
  const m = c.model;
  const short = { layernorm: 'LN', rmsnorm: 'RMS', batchnorm: 'BN' };
  const nt = short[m.norm];
  const np = short[postNormKind(m)];
  const norm =
    m.preNorm && m.postNorm && nt !== np
      ? `${nt}-pre+${np}-post`
      : m.preNorm || m.postNorm
        ? `${m.preNorm ? nt : np}-${m.preNorm && m.postNorm ? 'pre+post' : m.preNorm ? 'pre' : 'post'}`
        : 'noNorm';
  const opt = { sgd: 'sgd', momentum: 'mom', adam: 'adam' }[c.opt.kind];
  const B = c.batchMode === 'full' ? 'full' : c.batchMode === 'sgd' ? 'B1' : `B${c.batchSize}`;
  const branch = m.fc1 === false || m.fc2 === false ? ` [${m.fc1 === false ? '' : 'fc1'}${m.fc2 === false ? '' : '+fc2'}]` : '';
  const extra =
    (m.dropout ? ` drop${m.dropout}` + (DROP_SITES.every((s) => dropAt(m, s)) ? '' : `@${DROP_SITES.filter((s) => dropAt(m, s)).join(',')}`) : '') +
    (c.opt.wd ? ` wd${fmtNum(c.opt.wd)}` : '') +
    (c.precision && c.precision !== 'fp32' ? ` ${c.precision}` : '') +
    (c.microBatches ? ` μ${c.microBatches}` : '') +
    (c.data.noise ? ` noise${fmtNum(c.data.noise)}` : '') +
    (c.data.labelNoise ? ` lblErr${fmtNum(c.data.labelNoise)}` : '');
  return `d${m.width}×L${m.depth} ${m.actOn === false ? 'noAct' : m.act}${branch} ${norm}${m.residual ? '' : ' noRes'} ${m.init === 'zero' ? '0init' : m.init} ${opt} lr${fmtNum(c.opt.lr)} ${B}${extra}`;
}

function makeRun(c: TrainConfig, o: { group?: string; color?: string; label?: string; baseLabel?: string; sweep?: Run['sweep'] } = {}): Run {
  const L = buildLayout(c.model, 3 * c.data.resolution ** 2);
  const base = o.baseLabel ?? describe(c);
  return {
    id: nextId++,
    label: o.label ?? `${base} s${c.initSeed}/${c.dataSeed}`,
    baseLabel: base,
    color: o.color ?? PALETTE[(nextId - 2) % PALETTE.length],
    group: o.group ?? null,
    cfg: c,
    numParams: L.numParams,
    flopsPerEx: flopsPerExample(L),
    B: effectiveBatch(c),
    T: totalSteps(c),
    step: [],
    loss: [],
    acc: [],
    gnorm: [],
    pnorm: [],
    logitMax: [],
    actMax: [],
    lr: [],
    evStep: [],
    evLoss: [],
    evAcc: [],
    status: 'queued',
    visible: true,
    sps: 0,
    sweep: o.sweep,
  };
}

function trainSingle() {
  if (rerollSeeds) {
    cfg.initSeed = Math.floor(Math.random() * 10000);
    cfg.dataSeed = Math.floor(Math.random() * 10000);
    sync();
  }
  const c = structuredClone({ ...cfg, schedule: [] as number[] });
  c.schedule = schedule; // shared with the editor, so drawing during the run takes effect live
  enqueue([makeRun(c)]);
}

function enqueue(rs: Run[]) {
  runs.push(...rs);
  if (backend === 'remote') {
    for (const r of rs) void executeRemote(r);
    requestRender(true);
    return;
  }
  queue.push(...rs);
  requestRender(true);
  void pump();
}

async function executeRemote(r: Run) {
  const rr = new RemoteRun(r.cfg);
  remote.set(r, rr);
  pool.start(); // brings an auto-off'd GPU back (dot goes yellow → green); no-op if already on
  pool.touch();
  try {
    await rr.run(
      () => {
        if (r.status === 'queued') r.status = 'running';
        requestRender(true);
      },
      (u) => {
        pool.touch();
        onUpdate(r, u);
      },
      (pct) => pool.setUtil(pct),
    );
    if (r.status === 'running' || r.status === 'queued') r.status = 'done';
  } catch (e) {
    r.status = 'error';
    r.error = String(e);
    setStatus(`Error: ${e}`);
  } finally {
    remote.delete(r);
    if (!remote.size) pool.setUtil(0);
    pool.touch();
    requestRender(true);
  }
}

let pumping = false;
async function pump() {
  if (pumping) return;
  pumping = true;
  try {
    while (queue.length) {
      const r = queue.shift()!;
      if (r.status !== 'queued') continue;
      await execute(r);
      requestRender(true);
    }
  } finally {
    pumping = false;
    updateLive();
    if (!/error/i.test(statusInfo.textContent ?? '')) setStatus('');
  }
}

let readyPromise: Promise<void> | null = null;
async function execute(r: Run) {
  await readyPromise;
  if (!engine || !cifar) {
    r.status = 'error';
    r.error = 'WebGPU or data not available';
    return;
  }
  let tr: Trainer;
  let ds: GpuDataset;
  try {
    ds = getGpuDataset(engine.device, cifar, r.cfg.data.resolution, r.cfg.data.trainSize, r.cfg.data.testSize, {
      noise: r.cfg.data.noise ?? 0,
      labelNoise: r.cfg.data.labelNoise ?? 0,
      seed: r.cfg.dataSeed,
    });
    tr = new Trainer(engine, ds, r.cfg);
  } catch (e) {
    r.status = 'error';
    r.error = String(e);
    setStatus(`Error: ${e}`);
    return;
  }
  r.status = 'running';
  current = { run: r, trainer: tr };
  try {
    await tr.run(
      (u) => {
        if (u.batch) lastBatch = { b: u.batch, run: r, mean: ds.mean, std: ds.std };
        onUpdate(r, u);
      },
      { weightsEveryMs: 300, batchEveryMs: 250 },
    );
    if (r.status === 'running') r.status = 'done';
  } catch (e) {
    r.status = 'error';
    r.error = String(e);
    setStatus(`Error: ${e}`);
  } finally {
    tr.destroy();
    current = null;
    if (r.cfg.schedule === schedule) {
      schedEditor.progress = -1;
      schedEditor.draw();
    }
  }
}

function onUpdate(r: Run, u: ChunkUpdate) {
  for (const t of u.train) {
    r.step.push(t.step);
    r.loss.push(t.loss);
    r.acc.push(t.acc);
    r.gnorm.push(t.gnorm);
    r.pnorm.push(t.pnorm ?? NaN);
    r.logitMax.push(t.logitMax ?? NaN);
    r.actMax.push(t.actMax ?? NaN);
    r.lr.push(t.lr);
  }
  for (const e of u.evals) {
    r.evStep.push(e.step);
    r.evLoss.push(e.loss);
    r.evAcc.push(e.acc);
  }
  r.sps = r.sps ? 0.7 * r.sps + 0.3 * u.stepsPerSec : u.stepsPerSec;
  if (u.diverged) r.status = 'diverged';
  if (u.stemWeights) lastStem = { w: u.stemWeights, run: r };
  if (r.cfg.schedule === schedule) {
    schedEditor.progress = u.step / u.totalSteps;
    schedEditor.draw();
  }
  requestRender();
}

function stopAll() {
  for (const r of queue) if (r.status === 'queued') r.status = 'stopped';
  queue.length = 0;
  if (current) {
    current.run.status = 'stopped';
    current.trainer.stop();
  }
  stopRemote();
  requestRender(true);
}

function stopRemote() {
  for (const [r, rr] of remote) {
    r.status = 'stopped';
    rr.stop();
  }
}

function removeRun(r: Run) {
  if (r.status === 'running' && current?.run === r) current.trainer.stop();
  remote.get(r)?.stop();
  r.status = r.status === 'queued' ? 'stopped' : r.status;
  const i = runs.indexOf(r);
  if (i >= 0) runs.splice(i, 1);
  requestRender(true);
}

function clearRuns() {
  for (const r of queue) if (r.status === 'queued') r.status = 'stopped';
  queue.length = 0;
  stopRemote();
  for (let i = runs.length - 1; i >= 0; i--) if (current?.run !== runs[i]) runs.splice(i, 1);
  requestRender(true);
}

/** m:ss, or h:mm:ss past an hour. */
function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return hh ? `${hh}:${String(mm).padStart(2, '0')}:${ss}` : `${mm}:${ss}`;
}

function setStatus(s: string) {
  statusInfo.textContent = s;
  statusInfo.style.display = s ? '' : 'none';
}

function updateLive() {
  const q = queue.filter((r) => r.status === 'queued').length;
  queueInfo.textContent = current ? `running 1 · queued ${q}` : q ? `queued ${q}` : '';
  stopBtn.disabled = !current && !q && !remote.size;
  if (!current && remote.size) {
    const rs = [...remote.keys()];
    const started = rs.filter((r) => r.status === 'running');
    const done = rs.reduce((a, r) => a + (r.step.at(-1) ?? 0), 0);
    const total = rs.reduce((a, r) => a + r.T, 0);
    const sps = started.reduce((a, r) => a + r.sps, 0);
    queueInfo.textContent = '';
    progressBar.style.width = `${(100 * done) / Math.max(1, total)}%`;
    const eta = sps > 0 ? `≈ ${fmtDuration((total - done) / sps)} until completion` : 'estimating time until completion…';
    liveInfo.textContent = `${rs.length} remote run(s) · ${started.length}/${rs.length} running\nstep ${done.toLocaleString()}/${total.toLocaleString()} total · ${Math.round(sps).toLocaleString()} steps/s\n${eta}`;
    return;
  }
  if (!current) {
    progressBar.style.width = '0';
    liveInfo.textContent = 'idle';
    return;
  }
  const r = current.run;
  const s = r.step.at(-1) ?? 0;
  progressBar.style.width = `${(100 * s) / r.T}%`;
  const ev = r.evAcc.length ? ` · test acc ${(100 * lastFinite(r.evAcc)).toFixed(1)}%` : '';
  liveInfo.textContent = `${r.label}\nstep ${s.toLocaleString()}/${r.T.toLocaleString()} · ${Math.round(r.sps).toLocaleString()} steps/s · ${formatCount(r.sps * r.B)} ex/s\ntrain loss ${fmtNum(ema(r.loss, 0.9).at(-1) ?? NaN)}${ev}`;
}

let renderQueued = false;
function requestRender(forceTable = false) {
  if (forceTable) tableTimer = 0;
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    updateLive();
    if (activeTab === 'curves') {
      renderCurves();
      renderRunsTable();
    }
    if (activeTab === 'sweep') renderSweep();
    if (activeTab === 'scaling') renderScaling();
  });
}

// ============================================================================
// Boot
// ============================================================================

async function boot() {
  setTab('curves');
  sync();
  updateLive();
  readyPromise = (async () => {
    const [eng, data] = await Promise.allSettled([Engine.create(), loadCifar((m) => setStatus(m))]);
    if (eng.status === 'rejected') {
      setStatus(`WebGPU unavailable: ${eng.reason}`);
      return;
    }
    engine = eng.value;
    engine.device.addEventListener('uncapturederror', (e) => {
      const msg = (e as GPUUncapturedErrorEvent).error.message;
      console.error(msg);
      setStatus(`GPU error: ${msg}`);
    });
    document.getElementById('gpu')!.textContent = engine.adapterInfo;
    if (data.status === 'rejected') {
      setStatus(String(data.reason));
      return;
    }
    cifar = data.value;
    setStatus('');
  })();
  await readyPromise;
}

void boot();
