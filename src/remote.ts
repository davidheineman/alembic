// Client for the Modal GPU backend (remote/modal_app.py). Each run is one streaming POST;
// the server replies with newline-delimited JSON in the same shape as the engine's ChunkUpdate.
import type { ChunkUpdate } from './gpu/engine';
import type { TrainConfig } from './model';

const URL_ = import.meta.env.VITE_REMOTE_URL as string | undefined;
const TOKEN = import.meta.env.VITE_REMOTE_TOKEN as string | undefined;
// The A10G worker's own endpoint (Worker.web); skips the hop through the `api` function.
const TRAIN_URL = (import.meta.env.VITE_REMOTE_TRAIN_URL as string | undefined) ?? URL_;

export const remoteAvailable = !!(URL_ && TOKEN);

const headers = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' };

// GPU types offered by remote/modal_app.py (GPU_TYPES) with rough Modal on-demand prices, $/GPU-hour.
export const GPU_PRICES: Record<string, number> = { T4: 0.59, L4: 0.8, A10G: 1.1, L40S: 1.95, A100: 2.1, H100: 3.95 };
export const GPU_TYPES = Object.keys(GPU_PRICES);
const DEFAULT_GPU = 'A10G';

export let remoteGpu = GPU_TYPES.includes(localStorage.getItem('remoteGpu') ?? '') ? localStorage.getItem('remoteGpu')! : DEFAULT_GPU;

/** Each GPU type has its own worker class, served at ...-worker<type>-web... (A10G is plain `worker`). */
function trainUrl(): string | undefined {
  if (!TRAIN_URL?.includes('-worker-web')) return TRAIN_URL;
  return remoteGpu === DEFAULT_GPU ? TRAIN_URL : TRAIN_URL.replace('-worker-web', `-worker${remoteGpu.toLowerCase()}-web`);
}

/** Whether a CUDA device name ("NVIDIA A100-SXM4-40GB", "Tesla T4") is the given type. */
function isType(name: string, type: string): boolean {
  const n = name.replace(/^NVIDIA |^Tesla /, '');
  return n === type || n.startsWith(type + ' ') || n.startsWith(type + '-');
}

export interface PoolStatus {
  want: number;
  ready: number;
  active: number;
  gpus: string[];
  runsPerGpu: number;
  released?: boolean;
}

export type PoolState = 'off' | 'warming' | 'ready' | 'stopping' | 'error';

/**
 * Keeps one GPU container warm on Modal while the remote backend is selected, so runs start
 * instantly. Sends a heartbeat every 30 s. stop() shuts the container down (state 'stopping' until
 * it is gone). The GPU is also shut down after IDLE_MS without remote training, and server-side
 * (remote/modal_app.py `reaper`) if heartbeats stop or nothing trains for 2 minutes.
 */
export class RemotePool {
  status: PoolStatus | null = null;
  error = '';
  /** Set when the GPU was shut down for inactivity; cleared by start(). */
  released = false;
  private on = false;
  private stopping = false;
  private explicitStart = false;
  private timer = 0;
  private gen = 0;
  private lastActive = Date.now();
  private prepBody = '';
  private prepSent = '';
  private prepTimer = 0;
  static readonly IDLE_MS = 2 * 60 * 1000;

  constructor(private onChange: () => void) {}

  get state(): PoolState {
    if (this.stopping) return 'stopping';
    if (!this.on) return 'off';
    if (this.error) return 'error';
    // Right after switching types the old container may still report in; it doesn't count.
    return this.status && this.status.ready >= 1 && this.status.gpus.some((g) => isType(g, remoteGpu)) ? 'ready' : 'warming';
  }

  get gpuLabel(): string {
    const g = this.status?.gpus.find((n) => isType(n, remoteGpu));
    return g ? g.replace(/^NVIDIA |^Tesla /, '') : remoteGpu;
  }

  get costPerHour(): number {
    return GPU_PRICES[remoteGpu];
  }

  /** Switches the warm container to another GPU type; runs on the old one are lost. */
  setGpu(type: string) {
    if (type === remoteGpu || !GPU_TYPES.includes(type)) return;
    remoteGpu = type;
    localStorage.setItem('remoteGpu', type);
    this.status = null;
    this.prepSent = '';
    if (this.on && !this.stopping) {
      this.explicitStart = true;
      this.onChange();
      void this.tick();
    }
  }

  /** explicit=false (e.g. restoring after a page reload) does not override an inactivity shutdown. */
  start(explicit = true) {
    if (this.on && !this.error) return;
    this.gen++;
    this.on = true;
    this.stopping = false;
    this.released = false;
    this.explicitStart = explicit;
    this.error = '';
    this.status = null;
    this.lastActive = Date.now();
    this.onChange();
    void this.tick();
  }

  /** Shuts the GPU down; with released=true the UI explains it was an inactivity timeout. */
  stop(released = false) {
    clearTimeout(this.timer);
    const gen = ++this.gen;
    this.released = released;
    if (!this.on && !this.stopping) return;
    this.on = false;
    this.stopping = true;
    this.onChange();
    const deadline = Date.now() + 60000;
    const poll = async () => {
      if (gen !== this.gen) return;
      try {
        const st = await this.request('/status');
        if (gen !== this.gen) return;
        this.status = st;
        if (st.ready > 0 && Date.now() < deadline) {
          this.timer = window.setTimeout(poll, 1500);
          return;
        }
      } catch {
        // fall through: the server reaper still guarantees shutdown
      }
      this.stopping = false;
      this.status = null;
      this.onChange();
    };
    this.post(0, false)
      .catch(() => {})
      .then(poll);
  }

  /** The page is going away: unpins the GPU without stopping it. It exits after ~20 s without runs
   * unless a reloaded page pins it again first, which then gets the warm container back. */
  leave() {
    if (!this.on) return;
    void fetch(`${URL_}/warm`, { method: 'POST', headers, body: JSON.stringify({ gpus: 0, leaving: true }), keepalive: true }).catch(() => {});
  }

  touch() {
    this.lastActive = Date.now();
  }

  /** GPU utilization (%) last reported by a run; 0 when nothing is training. */
  util = 0;

  setUtil(pct: number) {
    if (Math.round(pct) === this.util) return;
    this.util = Math.round(pct);
    this.onChange();
  }

  /** Asks the GPU to set up the runs about to start (one step graph per concurrent run), so they
   * start without building datasets or capturing graphs. */
  prepare(cfgs: TrainConfig[]) {
    // Seeds, LR, schedule and run length don't change the step graph, so runs differing only in
    // those (an LR or seed sweep) need one graph each of the same shape.
    const groups = new Map<string, { cfg: TrainConfig; count: number }>();
    for (const c of cfgs) {
      const { initSeed: _i, dataSeed: _d, schedule: _s, budget: _b, numEvals: _n, ...shape } = c;
      const key = JSON.stringify({ ...shape, opt: { ...c.opt, lr: 0 } });
      const g = groups.get(key);
      if (g) g.count++;
      else groups.set(key, { cfg: { ...c, schedule: [], initSeed: 0, dataSeed: 0 }, count: 1 });
    }
    const body = JSON.stringify({ configs: [...groups.values()] });
    if (body === this.prepBody) return;
    this.prepBody = body;
    clearTimeout(this.prepTimer);
    this.prepTimer = window.setTimeout(() => this.sendPrepare(), 250);
  }

  private sendPrepare() {
    if (this.state !== 'ready' || !this.prepBody || this.prepBody === this.prepSent) return;
    const body = this.prepBody;
    this.prepSent = body;
    void fetch(`${trainUrl()}/prepare`, { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: JSON.stringify({ ...JSON.parse(body), token: TOKEN }) }).catch(() => {
      if (this.prepSent === body) this.prepSent = '';
    });
  }

  private async request(path: string, body?: object): Promise<PoolStatus> {
    const res = await fetch(`${URL_}${path}`, body ? { method: 'POST', headers, body: JSON.stringify(body), keepalive: true } : { headers });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  private post(gpus: number, start: boolean): Promise<PoolStatus> {
    return this.request('/warm', { gpus, start, gpu: remoteGpu });
  }

  private async tick() {
    clearTimeout(this.timer);
    const gen = this.gen;
    if (!this.on) return;
    // A cold start (slow for some GPU types) doesn't count as idle; the server gives up on it eventually.
    if (this.state === 'warming') this.touch();
    if (Date.now() - this.lastActive > RemotePool.IDLE_MS && !this.status?.active) {
      this.stop(true);
      return;
    }
    try {
      const st = await this.post(1, this.explicitStart);
      if (gen !== this.gen) return;
      this.explicitStart = false;
      if (st.released) {
        this.stop(true);
        return;
      }
      if (st.active) this.touch();
      if (!this.status?.ready && st.ready) this.prepSent = ''; // a fresh container has nothing prepared
      this.status = st;
      this.error = '';
      // Keeps the browser's connection to the GPU container open, so pressing Train doesn't pay
      // for a new TCP/TLS handshake.
      if (st.ready) void fetch(`${trainUrl()}/ping`).catch(() => {});
      this.sendPrepare();
    } catch (e) {
      if (gen !== this.gen) return;
      this.error = String(e);
    }
    this.onChange();
    this.timer = window.setTimeout(() => this.tick(), this.state === 'ready' ? 30000 : 2000);
  }
}

type Msg =
  | { type: 'queued' | 'ping' | 'retry' }
  | { type: 'start'; gpu: string }
  | { type: 'update'; step: number; totalSteps: number; train: (number | null)[][]; evals: (number | null)[][]; stepsPerSec: number; diverged: boolean; util?: number }
  | { type: 'final'; stemW16: string }
  | { type: 'error'; error: string };

const num = (x: number | null) => (x === null ? NaN : x);

function halfToFloat(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const s = h[i] & 0x8000 ? -1 : 1;
    const e = (h[i] >> 10) & 0x1f;
    const f = h[i] & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class RemoteRun {
  private ctrl = new AbortController();
  private stopped = false;
  private readonly id = Date.now().toString(36) + Math.random().toString(36).slice(2);

  constructor(readonly cfg: TrainConfig) {}

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.ctrl.abort();
    void fetch(`${URL_}/cancel`, { method: 'POST', headers, body: JSON.stringify({ runIds: [this.id] }), keepalive: true }).catch(() => {});
  }

  /** Resolves when training finishes or is stopped; rejects on server errors. onUtil gets the GPU's
   * utilization (%) as reported alongside each update. */
  async run(onStart: (gpu: string) => void, onUpdate: (u: ChunkUpdate) => void, onUtil?: (pct: number) => void): Promise<void> {
    // A request can land on a GPU container that is shutting down; it says so and the next try
    // reaches (or starts) a fresh one.
    for (let i = 0; i < 5; i++) {
      if (!(await this.attempt(onStart, onUpdate, onUtil)) || this.stopped) return;
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error('remote backend: GPU kept shutting down');
  }

  /** True if the run should be resent. */
  private async attempt(onStart: (gpu: string) => void, onUpdate: (u: ChunkUpdate) => void, onUtil?: (pct: number) => void): Promise<boolean> {
    let res: Response;
    try {
      res = await fetch(`${trainUrl()}/train`, {
        method: 'POST',
        // text/plain with the token in the body is a CORS "simple request", so there is no
        // preflight round trip before each run.
        headers: { 'Content-Type': 'text/plain' },
        body: JSON.stringify({ ...this.cfg, runId: this.id, token: TOKEN, gpu: remoteGpu }),
        signal: this.ctrl.signal,
      });
    } catch (e) {
      if (this.stopped) return false;
      throw e;
    }
    if (!res.ok || !res.body) throw new Error(`remote backend: HTTP ${res.status} ${await res.text()}`);
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = '';
    let step = 0;
    let T = 0;
    let sps = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += value;
        let nl: number;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          if (!line.trim()) continue;
          const m = JSON.parse(line) as Msg;
          if (m.type === 'retry') return true;
          if (m.type === 'start') onStart(m.gpu);
          else if (m.type === 'error') throw new Error(`remote backend: ${m.error}`);
          else if (m.type === 'update') {
            step = m.step;
            T = m.totalSteps;
            sps = m.stepsPerSec;
            if (m.util !== undefined) onUtil?.(m.util);
            onUpdate({
              step,
              totalSteps: T,
              train: m.train.map(([s, loss, acc, gnorm, lr, pnorm, actMax]) => ({
                step: num(s),
                loss: num(loss),
                acc: num(acc),
                gnorm: num(gnorm),
                lr: num(lr),
                pnorm: pnorm === undefined ? undefined : num(pnorm),
                actMax: actMax === undefined ? undefined : num(actMax),
              })),
              evals: m.evals.map(([s, loss, acc]) => ({ step: num(s), loss: num(loss), acc: num(acc) })),
              stepsPerSec: m.stepsPerSec,
              diverged: m.diverged,
            });
          } else if (m.type === 'final') {
            const bytes = Uint8Array.from(atob(m.stemW16), (c) => c.charCodeAt(0));
            onUpdate({ step, totalSteps: T, train: [], evals: [], stepsPerSec: sps, diverged: false, stemWeights: halfToFloat(new Uint16Array(bytes.buffer)) });
          }
        }
      }
    } catch (e) {
      if (this.stopped) return false;
      throw e;
    }
    return false;
  }
}
