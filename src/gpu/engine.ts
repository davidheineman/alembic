import * as K from './kernels';
import {
  buildLayout,
  decayMask,
  effectiveBatch,
  initParams,
  lrMultiplier,
  totalSteps,
  type ModelLayout,
  type ParamSpec,
  type TrainConfig,
  NUM_CLASSES,
  branchShape,
  roundBf16,
  dropAt,
} from '../model';
import type { GpuDataset } from '../data';
import { Rng } from '../rng';

type BindingType = 'uniform' | 'read' | 'rw';

const KERNELS: Record<string, [string, BindingType[]]> = {
  gather: [K.gatherWGSL, ['uniform', 'read', 'read', 'read', 'rw', 'rw', 'read']],
  mmNN: [K.matmulWGSL('NN'), ['uniform', 'read', 'read', 'read', 'read', 'rw', 'rw']],
  mmNT: [K.matmulWGSL('NT'), ['uniform', 'read', 'read', 'read', 'read', 'rw', 'rw']],
  mmTN: [K.matmulWGSL('TN'), ['uniform', 'read', 'read', 'read', 'read', 'rw', 'rw']],
  mmReduce: [K.mmReduceWGSL, ['uniform', 'read', 'read', 'read', 'rw']],
  colsum: [K.colsumWGSL, ['uniform', 'read', 'rw']],
  actFwd: [K.actFwdWGSL, ['uniform', 'read', 'rw']],
  actBwd: [K.actBwdWGSL, ['uniform', 'read', 'read', 'rw']],
  add: [K.addWGSL, ['uniform', 'read', 'read', 'rw']],
  rowAbsMax: [K.rowAbsMaxWGSL, ['uniform', 'read', 'rw']],
  lnFwd: [K.lnFwdWGSL, ['uniform', 'read', 'read', 'rw', 'rw']],
  lnBwd: [K.lnBwdWGSL, ['uniform', 'read', 'read', 'read', 'read', 'rw']],
  lnParamGrad: [K.lnParamGradWGSL, ['uniform', 'read', 'read', 'read', 'rw']],
  bnFwd: [K.bnFwdWGSL, ['uniform', 'read', 'read', 'rw', 'rw', 'rw']],
  bnBwd: [K.bnBwdWGSL, ['uniform', 'read', 'read', 'read', 'read', 'rw', 'rw']],
  ce: [K.ceWGSL, ['uniform', 'read', 'read', 'rw', 'rw']],
  lossReduce: [K.lossReduceWGSL, ['read', 'rw']],
  gnPartial: [K.gradNormPartialWGSL, ['uniform', 'read', 'rw']],
  gnFinal: [K.gradNormFinalWGSL, ['read', 'rw', 'uniform']],
  optimizer: [K.optimizerWGSL, ['uniform', 'rw', 'read', 'rw', 'rw', 'read', 'read']],
};

interface Kernel {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

export class Engine {
  readonly passLayout: GPUBindGroupLayout;
  private kernels = new Map<string, Kernel>();

  private constructor(
    readonly device: GPUDevice,
    readonly adapterInfo: string,
  ) {
    this.passLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    });
  }

  static async create(): Promise<Engine> {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });
    device.lost.then((info) => console.error('WebGPU device lost:', info.message));
    const info = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : '';
    const engine = new Engine(device, info || 'WebGPU');
    await engine.compile();
    return engine;
  }

  private async compile() {
    const entries = Object.entries(KERNELS);
    await Promise.all(
      entries.map(async ([name, [code, types]]) => {
        const layout = this.device.createBindGroupLayout({
          entries: types.map((t, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: t === 'uniform' ? 'uniform' : t === 'read' ? 'read-only-storage' : 'storage' },
          })),
        });
        const module = this.device.createShaderModule({ code, label: name });
        const pipeline = await this.device.createComputePipelineAsync({
          label: name,
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, this.passLayout] }),
          compute: { module, entryPoint: 'main' },
        });
        this.kernels.set(name, { pipeline, layout });
      }),
    );
  }

  kernel(name: string): Kernel {
    const k = this.kernels.get(name);
    if (!k) throw new Error(`unknown kernel ${name}`);
    return k;
  }
}

// ---------------------------------------------------------------------------

interface Seg {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

interface Op {
  pipe: GPUComputePipeline;
  bg: GPUBindGroup;
  wg: (M: number) => [number, number, number];
}

export interface StepRecord {
  step: number;
  loss: number;
  acc: number;
  gnorm: number;
  pnorm?: number; // ‖θ‖ after the update
  logitMax?: number; // max |logit| over the batch
  actMax?: number; // max |h| over the batch and every residual-stream position (stem and block outputs)
  lr: number;
}

export interface EvalRecord {
  step: number;
  loss: number;
  acc: number;
}

export interface ChunkUpdate {
  step: number;
  totalSteps: number;
  train: StepRecord[];
  evals: EvalRecord[];
  stepsPerSec: number;
  diverged: boolean;
  stemWeights?: Float32Array;
  batch?: BatchSample;
}

/** First rows of the last micro-batch of a training step, as the model saw them (augmented, normalized). */
export interface BatchSample {
  step: number;
  n: number;
  inputs: Float32Array; // n × D, CHW per row
  logits: Float32Array; // n × NUM_CLASSES, from the forward pass before that step's update
  labels: Uint32Array;
}

export const BATCH_SAMPLE_CAP = 256;

/** Test hook: force split-K on every matmul whose inner dimension allows it. */
export const debugOptions = { forceSplits: 1 };

const MAX_ENTRIES = 8192;
const MAX_IDX = 1 << 22;
const ACT_BUDGET_FLOATS = (448 * 1024 * 1024) / 4;
const MICRO_CAP = 1024;

const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

class Sampler {
  private perm: Uint32Array;
  private pos: number;
  private rng: Rng;

  constructor(
    private n: number,
    seed: number,
    private flip: boolean,
    private crop: boolean,
    private cutout: boolean,
    private res: number,
  ) {
    this.rng = new Rng(seed, 2);
    this.perm = new Uint32Array(n);
    for (let i = 0; i < n; i++) this.perm[i] = i;
    this.pos = n;
  }

  fill(out: Uint32Array, augOut: Uint32Array, count: number) {
    const pad = Math.max(1, this.res / 8);
    for (let i = 0; i < count; i++) {
      if (this.pos >= this.n) {
        for (let j = this.n - 1; j > 0; j--) {
          const k = this.rng.int(j + 1);
          const t = this.perm[j];
          this.perm[j] = this.perm[k];
          this.perm[k] = t;
        }
        this.pos = 0;
      }
      let v = this.perm[this.pos++];
      if (this.flip && (this.rng.u32() & 1)) v = (v | 0x80000000) >>> 0;
      out[i] = v;
      let a = 0;
      if (this.crop || this.cutout) {
        a = (1 << 31) | 8 | (8 << 4);
        if (this.crop) a = (1 << 31) | (8 - pad + this.rng.int(2 * pad + 1)) | ((8 - pad + this.rng.int(2 * pad + 1)) << 4);
        if (this.cutout) a |= (1 << 20) | (this.rng.int(this.res) << 8) | (this.rng.int(this.res) << 14);
      }
      augOut[i] = a >>> 0;
    }
  }
}

export function evalSchedule(T: number, n: number, log: boolean): number[] {
  const s = new Set<number>([0, T]);
  if (log) {
    for (let i = 0; i < n; i++) s.add(Math.round(Math.pow(T, i / Math.max(1, n - 1))));
  } else {
    for (let i = 0; i < n; i++) s.add(Math.round((T * i) / Math.max(1, n - 1)));
  }
  return [...s].filter((x) => x >= 0 && x <= T).sort((a, b) => a - b);
}

export class Trainer {
  readonly layout: ModelLayout;
  readonly T: number;
  readonly B: number;
  readonly Mmax: number;
  readonly micro: number[];
  readonly evalM: number;
  private device: GPUDevice;
  private owned: GPUBuffer[] = [];
  private params!: GPUBuffer;
  private grads!: GPUBuffer;
  private stats!: GPUBuffer;
  private statsRead!: GPUBuffer;
  private passBuf!: GPUBuffer;
  private passBG!: GPUBindGroup;
  private idxTrain!: GPUBuffer;
  private augTrain!: GPUBuffer;
  private X0!: GPUBuffer;
  private Y!: GPUBuffer;
  private Z!: GPUBuffer;
  private batchRead: GPUBuffer | null = null;
  private trainOps: Op[] = [];
  private evalOps: Op[] = [];
  private bwdOps: Op[] = [];
  private optOps: Op[] = [];
  private sampler: Sampler;
  private evalSteps: Set<number>;
  private stopped = false;

  constructor(
    private engine: Engine,
    private data: GpuDataset,
    readonly cfg: TrainConfig,
  ) {
    this.device = engine.device;
    this.layout = buildLayout(cfg.model, data.D);
    this.T = totalSteps(cfg);
    this.B = effectiveBatch(cfg);
    const L = this.layout;
    const m = cfg.model;
    const perRow =
      L.inDim + 2 + (m.depth + 1) * L.width + m.depth * (2 * L.width + L.hidden1 + L.hidden) + 4 * L.width + L.hidden + L.hidden1 + 4 * NUM_CLASSES + 8;
    this.Mmax = Math.max(1, Math.min(MICRO_CAP, Math.floor(ACT_BUDGET_FLOATS / perRow)));
    const nMicro = Math.max(ceilDiv(this.B, this.Mmax), Math.min(this.B, cfg.microBatches ?? 0));
    this.micro = [];
    for (let i = 0; i < nMicro; i++) this.micro.push(Math.floor(this.B / nMicro) + (i < this.B % nMicro ? 1 : 0));
    this.evalM = Math.min(data.test.N, this.Mmax);
    this.sampler = new Sampler(data.train.N, cfg.dataSeed, cfg.data.flip, !!cfg.data.crop, !!cfg.data.cutout, data.res);
    this.evalSteps = new Set(evalSchedule(this.T, cfg.numEvals, !!cfg.logEvals));
    this.build();
  }

  // --- buffer helpers -------------------------------------------------------
  private buf(floats: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil((floats * 4) / 16) * 16), usage });
    this.owned.push(b);
    return b;
  }

  private ubuf(values: (number | [number, 'f'])[]): GPUBuffer {
    const n = Math.ceil(values.length / 4) * 4;
    const b = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    const ab = b.getMappedRange();
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    values.forEach((v, i) => {
      if (Array.isArray(v)) f[i] = v[0];
      else u[i] = v;
    });
    b.unmap();
    this.owned.push(b);
    return b;
  }

  private op(kernel: string, bindings: (GPUBuffer | Seg)[], wg: Op['wg']): Op {
    const k = this.engine.kernel(kernel);
    const bg = this.device.createBindGroup({
      layout: k.layout,
      entries: bindings.map((b, i) => ({ binding: i, resource: b instanceof GPUBuffer ? { buffer: b } : b })),
    });
    return { pipe: k.pipeline, bg, wg };
  }

  private pseg(s: ParamSpec): Seg {
    return { buffer: this.params, offset: s.offset * 4, size: s.size * 4 };
  }

  private gseg(s: ParamSpec): Seg {
    return { buffer: this.grads, offset: s.offset * 4, size: s.size * 4 };
  }

  // --- graph construction ---------------------------------------------------
  private build() {
    const L = this.layout;
    const m = this.cfg.model;
    const d = L.width;
    const M = this.Mmax;
    const dev = this.device;
    const dummy = this.buf(4);

    this.params = this.buf(L.totalFloats);
    this.grads = this.buf(L.totalFloats);
    const m1 = this.buf(this.cfg.opt.kind === 'sgd' ? 4 : L.totalFloats);
    const m2 = this.buf(this.cfg.opt.kind === 'adam' ? L.totalFloats : 4);
    const mask = this.buf(L.totalFloats);
    const p0 = initParams(L, m, this.cfg.initSeed);
    dev.queue.writeBuffer(this.params, 0, this.cfg.precision === 'bf16' ? roundBf16(p0) : p0);
    dev.queue.writeBuffer(mask, 0, decayMask(L));

    this.stats = this.buf(MAX_ENTRIES * K.STAT_STRIDE);
    this.statsRead = this.buf(MAX_ENTRIES * K.STAT_STRIDE, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.passBuf = dev.createBuffer({ size: MAX_ENTRIES * K.PASS_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.owned.push(this.passBuf);
    this.passBG = dev.createBindGroup({
      layout: this.engine.passLayout,
      entries: [{ binding: 0, resource: { buffer: this.passBuf, size: K.PASS_FLOATS * 4 } }],
    });
    const idxCap = Math.min(MAX_IDX, Math.max(this.B, 1) * 64);
    this.idxTrain = this.buf(Math.max(idxCap, this.B));
    this.augTrain = this.buf(Math.max(idxCap, this.B));
    const idxEval = this.buf(this.data.test.N);
    const augEval = this.buf(this.data.test.N);
    dev.queue.writeBuffer(idxEval, 0, Uint32Array.from({ length: this.data.test.N }, (_, i) => i));

    // Activations
    const X0 = (this.X0 = this.buf(M * L.inDim));
    const Y = (this.Y = this.buf(M));
    const H = Array.from({ length: m.depth + 1 }, () => this.buf(M * d));
    const Z = (this.Z = this.buf(M * NUM_CLASSES));
    const dZ = this.buf(M * NUM_CLASSES);
    const rowLoss = this.buf(4 * M);
    const GA = this.buf(M * d);
    const GB = this.buf(M * d);
    const dG = this.buf(M * L.hidden);
    const dA = this.buf(M * L.hidden1);
    const partial = this.buf(256);

    // Pre- and post-norm may be different kinds; each norm segment carries its own.
    const isBNs = (s: ParamSpec) => s.norm === 'batchnorm';
    const rmsOf = (s: ParamSpec) => (s.norm === 'rmsnorm' ? 1 : 0);
    const normStatsFloats = 2 * Math.max(d, M);
    const newRunning = () => {
      const r = this.buf(2 * d);
      const init = new Float32Array(2 * d);
      init.fill(1, d);
      dev.queue.writeBuffer(r, 0, init);
      return r;
    };

    const rowsWG = (cols: number) => (Mr: number): [number, number, number] => [ceilDiv(cols, 256), Mr, 1];
    const colWG = (cols: number) => (): [number, number, number] => [ceilDiv(cols, 256), 1, 1];
    const col16WG = (cols: number) => (): [number, number, number] => [ceilDiv(cols, 16), 1, 1];

    // Split-K: when a matmul has few output tiles but a long inner loop, the GPU is mostly idle.
    // Split the inner dimension across workgroups and sum the partials in a second pass.
    const microMax = Math.max(...this.micro);
    type MM = { rows: number; cols: number; inner: number; dynRows?: boolean; dynInner?: boolean; bias?: Seg; res?: Seg; acc: 0 | 1 | 2 };
    interface PendingMM {
      o: MM;
      variant: K.MatmulVariant;
      A: Seg;
      Bm: Seg;
      C: Seg;
      splits: number;
      ops: Op[];
    }
    const pending: PendingMM[] = [];
    const prec = this.cfg.precision ?? 'fp32';
    const precFlags = (prec !== 'fp32' ? 64 : 0) | (prec === 'bf16' ? 128 : 0);
    const mm = (variant: K.MatmulVariant, o: MM, A: Seg, Bm: Seg, C: Seg): Op[] => {
      const rowsEst = o.dynRows ? microMax : o.rows;
      const innerEst = o.dynInner ? microMax : o.inner;
      const tiles = ceilDiv(o.cols, 64) * ceilDiv(rowsEst, 64);
      const auto = Math.max(1, Math.min(Math.floor(128 / tiles), Math.floor(innerEst / 128), 32));
      const splits = Math.max(auto, Math.min(debugOptions.forceSplits, Math.ceil(innerEst / 16)));
      const ops: Op[] = [];
      pending.push({ o, variant, A, Bm, C, splits, ops });
      return ops;
    };
    const finalizeMatmuls = (scratch: GPUBuffer, dummyRW: GPUBuffer) => {
      for (const p of pending) {
        const { o, splits } = p;
        const flags = (o.dynRows ? 1 : 0) | (o.dynInner ? 2 : 0) | (o.bias ? 4 : 0) | (o.res ? 8 : 0) | (o.acc << 4) | precFlags;
        const u = this.ubuf([o.rows, o.cols, o.inner, flags]);
        p.ops.push(
          this.op(`mm${p.variant}`, [u, p.A, p.Bm, o.bias ?? dummy, o.res ?? dummy, splits > 1 ? dummyRW : p.C, splits > 1 ? scratch : dummyRW], (Mr) => [
            ceilDiv(o.cols, 64),
            ceilDiv(o.dynRows ? Mr : o.rows, 64),
            splits,
          ]),
        );
        if (splits > 1) {
          const ru = this.ubuf([o.rows, o.cols, splits, flags]);
          p.ops.push(
            this.op('mmReduce', [ru, scratch, o.bias ?? dummy, o.res ?? dummy, p.C], (Mr) => {
              const n = (o.dynRows ? Mr : o.rows) * o.cols;
              const wgs = ceilDiv(n, 256);
              const nx = Math.min(wgs, 32768);
              return [nx, ceilDiv(wgs, nx), 1];
            }),
          );
        }
      }
    };
    const scratchFloats = () => Math.max(4, ...pending.filter((p) => p.splits > 1).map((p) => p.splits * (p.o.dynRows ? M : p.o.rows) * p.o.cols));
    const colsum = (cols: number, dY: Seg, out: Seg) => this.op('colsum', [this.ubuf([cols, 2]), dY, out], col16WG(cols));

    const normFwd = (spec: ParamSpec, X: GPUBuffer, Yb: GPUBuffer, st: GPUBuffer): Op => {
      if (isBNs(spec)) return this.op('bnFwd', [this.ubuf([d, [0.1, 'f']]), X, this.pseg(spec), Yb, st, newRunning()], colWG(d));
      return this.op('lnFwd', [this.ubuf([d, rmsOf(spec)]), X, this.pseg(spec), Yb, st], (Mr) => [Mr, 1, 1]);
    };
    const normBwd = (spec: ParamSpec, dY: GPUBuffer, X: GPUBuffer, st: GPUBuffer, out: GPUBuffer, accX: 0 | 1): Op[] => {
      if (isBNs(spec)) return [this.op('bnBwd', [this.ubuf([d, accX, 2]), dY, X, this.pseg(spec), st, out, this.gseg(spec)], colWG(d))];
      return [
        this.op('lnBwd', [this.ubuf([d, rmsOf(spec), accX]), dY, X, this.pseg(spec), st, out], (Mr) => [Mr, 1, 1]),
        this.op('lnParamGrad', [this.ubuf([d, rmsOf(spec), 2]), dY, X, st, this.gseg(spec)], col16WG(d)),
      ];
    };
    const actKind = branchShape(m).actKind;
    const actUs: GPUBuffer[] = [];
    const pStem = dropAt(m, 'stem');
    const pAct = dropAt(m, 'act');
    const pBranch = dropAt(m, 'branch');
    // Dropout mask seeds per site, so different sites never share a mask.
    const SEED_BRANCH = 1000;
    const SEED_STEM = 2000;
    const actU = (l: number) => (actUs[l] ??= this.ubuf([L.hidden, actKind, l, [pAct, 'f']]));
    const addOp = (x: GPUBuffer, r: GPUBuffer | null, out: GPUBuffer, o: { acc?: 0 | 1; seed?: number; p?: number } = {}) =>
      this.op('add', [this.ubuf([d, r ? 1 : 0, o.acc ?? 0, o.seed ?? 0, [o.p ?? 0, 'f']]), x, r ?? x, out], rowsWG(d));
    const actMaxU = [this.ubuf([d, 0]), this.ubuf([d, 1])];
    const actMax = (X: GPUBuffer, acc: 0 | 1) => this.op('rowAbsMax', [actMaxU[acc], X, rowLoss], (Mr) => [Mr, 1, 1]);

    // Forward
    const fwd: (Op | Op[])[] = [];
    if (pStem > 0) {
      const raw = this.buf(M * d);
      fwd.push(mm('NN', { rows: 0, cols: d, inner: L.inDim, dynRows: true, bias: this.pseg(L.stem.b), acc: 0 }, { buffer: X0 }, this.pseg(L.stem.W), { buffer: raw }));
      fwd.push(addOp(raw, null, H[0], { seed: SEED_STEM, p: pStem }));
    } else {
      fwd.push(mm('NN', { rows: 0, cols: d, inner: L.inDim, dynRows: true, bias: this.pseg(L.stem.b), acc: 0 }, { buffer: X0 }, this.pseg(L.stem.W), { buffer: H[0] }));
    }
    fwd.push(actMax(H[0], 0));
    // Block: u = preNorm ? N(h) : h;  s = fc2(act(fc1(u))) + h;  h' = postNorm ? N(s) : s
    interface Saved { u: GPUBuffer; A: GPUBuffer; G: GPUBuffer; S?: GPUBuffer; stPre?: GPUBuffer; stPost?: GPUBuffer }
    const saved: Saved[] = [];
    for (let l = 0; l < m.depth; l++) {
      const blk = L.blocks[l];
      const hIn = H[l];
      const hOut = H[l + 1];
      const G = this.buf(M * L.hidden);
      const res = m.residual ? { buffer: hIn } : undefined;
      const sv: Saved = { u: hIn, A: hIn, G };
      if (blk.normPre) {
        sv.u = sv.A = this.buf(M * d);
        sv.stPre = this.buf(normStatsFloats);
        fwd.push(normFwd(blk.normPre, hIn, sv.u, sv.stPre));
      }
      if (blk.fc1) {
        sv.A = this.buf(M * L.hidden1);
        fwd.push(mm('NN', { rows: 0, cols: L.hidden1, inner: d, dynRows: true, bias: this.pseg(blk.fc1.b), acc: 0 }, { buffer: sv.u }, this.pseg(blk.fc1.W), { buffer: sv.A }));
      }
      fwd.push(this.op('actFwd', [actU(l), sv.A, G], rowsWG(L.hidden)));
      const sumOut = blk.normPost ? (sv.S = this.buf(M * d)) : hOut;
      const rIn = m.residual ? hIn : null;
      const brDrop = { seed: SEED_BRANCH + l, p: pBranch };
      if (blk.fc2 && pBranch === 0) {
        fwd.push(mm('NN', { rows: 0, cols: d, inner: L.hidden, dynRows: true, bias: this.pseg(blk.fc2.b), res, acc: 0 }, { buffer: G }, this.pseg(blk.fc2.W), { buffer: sumOut }));
      } else if (blk.fc2) {
        // Branch dropout must hit fc2's output before the residual add, so the add can't be fused.
        const T = this.buf(M * d);
        fwd.push(mm('NN', { rows: 0, cols: d, inner: L.hidden, dynRows: true, bias: this.pseg(blk.fc2.b), acc: 0 }, { buffer: G }, this.pseg(blk.fc2.W), { buffer: T }));
        fwd.push(addOp(T, rIn, sumOut, brDrop));
      } else {
        fwd.push(addOp(G, rIn, sumOut, brDrop));
      }
      if (blk.normPost) {
        sv.stPost = this.buf(normStatsFloats);
        fwd.push(normFwd(blk.normPost, sv.S!, hOut, sv.stPost));
      }
      fwd.push(actMax(hOut, 1));
      saved.push(sv);
    }
    let F = H[m.depth];
    let FS: GPUBuffer | undefined;
    if (L.finalNorm) {
      F = this.buf(M * d);
      FS = this.buf(normStatsFloats);
      fwd.push(normFwd(L.finalNorm, H[m.depth], F, FS));
    }
    fwd.push(mm('NN', { rows: 0, cols: NUM_CLASSES, inner: d, dynRows: true, bias: this.pseg(L.head.b), acc: 0 }, { buffer: F }, this.pseg(L.head.W), { buffer: Z }));
    fwd.push(this.op('ce', [this.ubuf([NUM_CLASSES]), Z, Y, dZ, rowLoss], (Mr) => [ceilDiv(Mr, 256), 1, 1]));
    fwd.push(this.op('lossReduce', [rowLoss, this.stats], () => [1, 1, 1]));

    const gather = (split: 'train' | 'test', idx: GPUBuffer, aug: GPUBuffer) => {
      const s = this.data[split];
      const res = this.data.res;
      const u = this.ubuf([
        this.data.D,
        res * res,
        res,
        res / 2,
        [this.data.mean[0], 'f'],
        [this.data.mean[1], 'f'],
        [this.data.mean[2], 'f'],
        [0, 'f'],
        [this.data.std[0], 'f'],
        [this.data.std[1], 'f'],
        [this.data.std[2], 'f'],
        [1, 'f'],
      ]);
      return this.op('gather', [u, s.data, s.labels, idx, X0, Y, aug], (Mr) => [ceilDiv(this.data.D, 256), Mr, 1]);
    };
    const gatherTrain = gather('train', this.idxTrain, this.augTrain);
    const gatherEval = gather('test', idxEval, augEval);

    // Backward
    const bwd: (Op | Op[])[] = [];
    bwd.push(mm('TN', { rows: d, cols: NUM_CLASSES, inner: 0, dynInner: true, acc: 2 }, { buffer: F }, { buffer: dZ }, this.gseg(L.head.W)));
    bwd.push(colsum(NUM_CLASSES, { buffer: dZ }, this.gseg(L.head.b)));
    bwd.push(mm('NT', { rows: 0, cols: d, inner: NUM_CLASSES, dynRows: true, acc: 0 }, { buffer: dZ }, this.pseg(L.head.W), { buffer: GA }));
    let cur = GA;
    let oth = GB;
    const swap = () => {
      const t = cur;
      cur = oth;
      oth = t;
    };
    if (L.finalNorm) {
      bwd.push(...normBwd(L.finalNorm, cur, H[m.depth], FS!, oth, 0));
      swap();
    }
    const resAcc: 0 | 1 = m.residual ? 1 : 0;
    let dBranch: GPUBuffer | undefined;
    for (let l = m.depth - 1; l >= 0; l--) {
      const blk = L.blocks[l];
      const sv = saved[l];
      // cur holds dL/dh'. After the post-norm backward it holds dL/ds, which is both the
      // branch-output gradient and (if residual) the skip-path gradient for h.
      if (blk.normPost) {
        bwd.push(...normBwd(blk.normPost, cur, sv.S!, sv.stPost!, oth, 0));
        swap();
      }
      // dBr: gradient of the branch output. Branch dropout masks it; the skip path keeps cur as is.
      let dBr = cur;
      if (pBranch > 0) {
        dBranch ??= this.buf(M * d);
        bwd.push(addOp(cur, null, dBranch, { seed: SEED_BRANCH + l, p: pBranch }));
        dBr = dBranch;
      }
      let dGb = dBr;
      if (blk.fc2) {
        bwd.push(mm('TN', { rows: L.hidden, cols: d, inner: 0, dynInner: true, acc: 2 }, { buffer: sv.G }, { buffer: dBr }, this.gseg(blk.fc2.W)));
        bwd.push(colsum(d, { buffer: dBr }, this.gseg(blk.fc2.b)));
        bwd.push(mm('NT', { rows: 0, cols: L.hidden, inner: d, dynRows: true, acc: 0 }, { buffer: dBr }, this.pseg(blk.fc2.W), { buffer: dG }));
        dGb = dG;
      }
      bwd.push(this.op('actBwd', [actU(l), sv.A, dGb, dA], rowsWG(L.hidden)));
      const fc1 = blk.fc1;
      if (!fc1) {
        // dL/du = dA directly.
        if (blk.normPre) bwd.push(...normBwd(blk.normPre, dA, H[l], sv.stPre!, cur, resAcc));
        else bwd.push(addOp(dA, null, cur, { acc: resAcc }));
        continue;
      }
      bwd.push(mm('TN', { rows: d, cols: L.hidden1, inner: 0, dynInner: true, acc: 2 }, { buffer: sv.u }, { buffer: dA }, this.gseg(fc1.W)));
      bwd.push(colsum(L.hidden1, { buffer: dA }, this.gseg(fc1.b)));
      const dInput = (out: GPUBuffer, acc: 0 | 1) =>
        mm('NT', { rows: 0, cols: d, inner: L.hidden1, dynRows: true, acc }, { buffer: dA }, this.pseg(fc1.W), { buffer: out });
      if (blk.normPre) {
        bwd.push(dInput(oth, 0));
        bwd.push(...normBwd(blk.normPre, oth, H[l], sv.stPre!, cur, resAcc));
      } else {
        bwd.push(dInput(cur, resAcc));
      }
    }
    if (pStem > 0) {
      bwd.push(addOp(cur, null, oth, { seed: SEED_STEM, p: pStem }));
      swap();
    }
    bwd.push(mm('TN', { rows: L.inDim, cols: d, inner: 0, dynInner: true, acc: 2 }, { buffer: X0 }, { buffer: cur }, this.gseg(L.stem.W)));
    bwd.push(colsum(d, { buffer: cur }, this.gseg(L.stem.b)));

    finalizeMatmuls(this.buf(scratchFloats()), this.buf(4));
    const flat = (xs: (Op | Op[])[]) => xs.flatMap((x) => (Array.isArray(x) ? x : [x]));
    this.trainOps = [gatherTrain, ...flat(fwd)];
    this.evalOps = [gatherEval, ...flat(fwd)];
    this.bwdOps = flat(bwd);

    // Optimizer
    const o = this.cfg.opt;
    const n = L.totalFloats;
    const wgTotal = ceilDiv(n, 256);
    const nx = Math.min(wgTotal, 32768);
    const ny = ceilDiv(wgTotal, nx);
    const kind = o.kind === 'sgd' ? 0 : o.kind === 'momentum' ? 1 : 2;
    this.optOps = [
      this.op('gnPartial', [this.ubuf([n]), this.grads, partial], () => [256, 1, 1]),
      this.op('gnFinal', [partial, this.stats, this.ubuf([2])], () => [1, 1, 1]),
      this.op(
        'optimizer',
        [
          this.ubuf([n, kind, nx * 256, this.cfg.precision === 'bf16' ? 1 : 0, [o.momentum, 'f'], [o.beta1, 'f'], [o.beta2, 'f'], [1e-8, 'f'], [o.wd, 'f'], [o.clip, 'f'], [0, 'f'], [0, 'f']]),
          this.params,
          this.grads,
          m1,
          m2,
          mask,
          this.stats,
        ],
        () => [nx, ny, 1],
      ),
      this.op('gnPartial', [this.ubuf([n]), this.params, partial], () => [256, 1, 1]),
      this.op('gnFinal', [partial, this.stats, this.ubuf([3])], () => [1, 1, 1]),
    ];
  }

  // --- execution --------------------------------------------------------------
  private encodeOps(pass: GPUComputePassEncoder, ops: Op[], entry: number, M: number) {
    pass.setBindGroup(1, this.passBG, [entry * K.PASS_STRIDE]);
    for (const op of ops) {
      const [x, y, z] = op.wg(M);
      if (x === 0 || y === 0) continue;
      pass.setPipeline(op.pipe);
      pass.setBindGroup(0, op.bg);
      pass.dispatchWorkgroups(x, y, z);
    }
  }

  lrAt(step: number): number {
    return this.cfg.opt.lr * lrMultiplier(this.cfg.schedule, step / Math.max(1, this.T - 1));
  }

  stop() {
    this.stopped = true;
  }

  async run(onUpdate: (u: ChunkUpdate) => void, opts: { weightsEveryMs?: number; batchEveryMs?: number } = {}): Promise<void> {
    const dev = this.device;
    const B = this.B;
    const testN = this.data.test.N;
    const evalPasses = ceilDiv(testN, this.evalM);
    const maxStepsByEntries = Math.max(1, Math.floor((MAX_ENTRIES - 2 * evalPasses) / (this.micro.length + evalPasses)));
    const maxStepsByIdx = Math.max(1, Math.floor(this.idxTrain.size / 4 / B));
    const entries = new ArrayBuffer(MAX_ENTRIES * K.PASS_STRIDE);
    const eu = new Uint32Array(entries);
    const ef = new Float32Array(entries);
    const idx = new Uint32Array(this.idxTrain.size / 4);
    const augW = new Uint32Array(this.augTrain.size / 4);
    let step = 0;
    let K_ = 2;
    let lastWeights = 0;
    let lastBatch = -Infinity;
    const stemW = this.layout.stem.W;
    const D = this.data.D;
    const lastMicro = this.micro[this.micro.length - 1];
    const nSample = Math.min(lastMicro, BATCH_SAMPLE_CAP);

    while (step < this.T && !this.stopped) {
      const kSteps = Math.max(1, Math.min(K_, this.T - step, maxStepsByEntries, maxStepsByIdx));
      this.sampler.fill(idx, augW, kSteps * B);
      let entry = 0;
      let slot = 0;
      const slotInfo: ({ kind: 'train'; step: number; lr: number } | { kind: 'eval'; step: number })[] = [];
      const enc = dev.createCommandEncoder();
      enc.clearBuffer(this.stats, 0, Math.min(this.stats.size, (2 * kSteps + 2) * K.STAT_STRIDE * 4));
      let pass = enc.beginComputePass();
      const captureBatch = opts.batchEveryMs !== undefined && performance.now() - lastBatch > opts.batchEveryMs;
      let batchStep = 0;
      const setEntry = (M: number, off: number, s: number, accum: number, lr: number, t: number, train: number) => {
        const b = entry * (K.PASS_STRIDE / 4);
        eu[b] = M;
        eu[b + 1] = off;
        eu[b + 2] = s;
        eu[b + 3] = accum;
        ef[b + 4] = lr;
        ef[b + 5] = 1 / B;
        ef[b + 6] = t;
        eu[b + 7] = train;
        return entry++;
      };
      const encodeEval = (atStep: number) => {
        const s = slot++;
        slotInfo.push({ kind: 'eval', step: atStep });
        for (let off = 0; off < testN; off += this.evalM) {
          const Me = Math.min(this.evalM, testN - off);
          this.encodeOps(pass, this.evalOps, setEntry(Me, off, s, 0, 0, 1, 0), Me);
        }
      };
      if (step === 0 && this.evalSteps.has(0)) encodeEval(0);
      for (let k = 0; k < kSteps; k++) {
        const g = step + k;
        const lr = this.lrAt(g);
        const s = slot++;
        slotInfo.push({ kind: 'train', step: g + 1, lr });
        let off = k * B;
        let last = 0;
        this.micro.forEach((Mi, i) => {
          last = setEntry(Mi, off, s, i > 0 ? 1 : 0, lr, g + 1, 1);
          this.encodeOps(pass, this.trainOps, last, Mi);
          this.encodeOps(pass, this.bwdOps, last, Mi);
          off += Mi;
        });
        this.encodeOps(pass, this.optOps, last, 1);
        if (captureBatch && k === kSteps - 1) {
          // Copy before any eval pass reuses X0/Y/Z for test examples; copies can't happen inside a compute pass.
          pass.end();
          this.batchRead ??= this.buf(BATCH_SAMPLE_CAP * (D + NUM_CLASSES + 1), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          enc.copyBufferToBuffer(this.X0, 0, this.batchRead, 0, nSample * D * 4);
          enc.copyBufferToBuffer(this.Z, 0, this.batchRead, nSample * D * 4, nSample * NUM_CLASSES * 4);
          enc.copyBufferToBuffer(this.Y, 0, this.batchRead, nSample * (D + NUM_CLASSES) * 4, nSample * 4);
          pass = enc.beginComputePass();
          batchStep = g + 1;
        }
        if (this.evalSteps.has(g + 1)) encodeEval(g + 1);
      }
      pass.end();
      const SS = K.STAT_STRIDE;
      enc.copyBufferToBuffer(this.stats, 0, this.statsRead, 0, slot * SS * 4);
      dev.queue.writeBuffer(this.passBuf, 0, entries, 0, entry * K.PASS_STRIDE);
      dev.queue.writeBuffer(this.idxTrain, 0, idx, 0, kSteps * B);
      dev.queue.writeBuffer(this.augTrain, 0, augW, 0, kSteps * B);
      const t0 = performance.now();
      dev.queue.submit([enc.finish()]);
      await this.statsRead.mapAsync(GPUMapMode.READ, 0, slot * SS * 4);
      const st = new Float32Array(this.statsRead.getMappedRange(0, slot * SS * 4).slice(0));
      this.statsRead.unmap();
      const dt = performance.now() - t0;
      let batch: BatchSample | undefined;
      if (captureBatch && this.batchRead) {
        const bytes = nSample * (D + NUM_CLASSES + 1) * 4;
        await this.batchRead.mapAsync(GPUMapMode.READ, 0, bytes);
        const raw = this.batchRead.getMappedRange(0, bytes).slice(0);
        this.batchRead.unmap();
        batch = {
          step: batchStep,
          n: nSample,
          inputs: new Float32Array(raw, 0, nSample * D),
          logits: new Float32Array(raw, nSample * D * 4, nSample * NUM_CLASSES),
          labels: new Uint32Array(raw, nSample * (D + NUM_CLASSES) * 4, nSample),
        };
        lastBatch = performance.now();
      }

      const train: StepRecord[] = [];
      const evals: EvalRecord[] = [];
      let diverged = false;
      slotInfo.forEach((info, i) => {
        if (info.kind === 'train') {
          const o = SS * i;
          const loss = st[o] / B;
          if (!isFinite(loss)) diverged = true;
          train.push({
            step: info.step,
            loss,
            acc: st[o + 1] / B,
            gnorm: Math.sqrt(st[o + 2]),
            pnorm: Math.sqrt(st[o + 3]),
            logitMax: st[o + 4],
            actMax: st[o + 5],
            lr: info.lr,
          });
        } else {
          const o = SS * i;
          const loss = st[o] / testN;
          if (!isFinite(loss)) diverged = true;
          evals.push({ step: info.step, loss, acc: st[o + 1] / testN });
        }
      });
      step += kSteps;
      K_ = Math.max(1, Math.min(4096, Math.round((kSteps * 60) / Math.max(1, dt))));
      let stemWeights: Float32Array | undefined;
      const now = performance.now();
      if (opts.weightsEveryMs !== undefined && (now - lastWeights > opts.weightsEveryMs || step >= this.T || diverged)) {
        lastWeights = now;
        stemWeights = await this.readParams(stemW);
      }
      onUpdate({ step, totalSteps: this.T, train, evals, stepsPerSec: (kSteps * 1000) / Math.max(dt, 1e-3), diverged, stemWeights, batch });
      if (diverged) break;
    }
  }

  async readParams(spec?: ParamSpec, which: 'params' | 'grads' = 'params'): Promise<Float32Array> {
    const src = which === 'params' ? this.params : this.grads;
    const offset = spec ? spec.offset * 4 : 0;
    const size = spec ? spec.size * 4 : src.size;
    const rb = this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, rb, 0, rb.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, size));
    rb.unmap();
    rb.destroy();
    return out;
  }

  writeParams(p: Float32Array) {
    this.device.queue.writeBuffer(this.params, 0, p);
  }

  /** Forward (+ optional backward) on the first `M` training examples. Returns mean loss. For gradient checks. */
  async debugStep(M: number, backward: boolean): Promise<number> {
    const dev = this.device;
    const idx = Uint32Array.from({ length: M }, (_, i) => i);
    const entries = new ArrayBuffer(K.PASS_STRIDE);
    const eu = new Uint32Array(entries);
    const ef = new Float32Array(entries);
    eu[0] = M;
    eu[1] = 0;
    eu[2] = 0;
    eu[3] = 0;
    ef[4] = 0;
    ef[5] = 1 / M;
    ef[6] = 1;
    eu[7] = 1;
    dev.queue.writeBuffer(this.passBuf, 0, entries);
    dev.queue.writeBuffer(this.idxTrain, 0, idx);
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(this.stats, 0, K.STAT_STRIDE * 4);
    const pass = enc.beginComputePass();
    this.encodeOps(pass, this.trainOps, 0, M);
    if (backward) this.encodeOps(pass, this.bwdOps, 0, M);
    pass.end();
    enc.copyBufferToBuffer(this.stats, 0, this.statsRead, 0, 16);
    dev.queue.submit([enc.finish()]);
    await this.statsRead.mapAsync(GPUMapMode.READ, 0, 16);
    const loss = new Float32Array(this.statsRead.getMappedRange(0, 16))[0] / M;
    this.statsRead.unmap();
    return loss;
  }

  destroy() {
    this.stopped = true;
    for (const b of this.owned) b.destroy();
    this.owned = [];
  }
}
