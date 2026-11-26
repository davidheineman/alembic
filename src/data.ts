// CIFAR-10 loading. Binary format: records of [label(1), R(1024), G(1024), B(1024)].
import { Rng } from './rng';

export const CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
const REC = 3073;

export interface CpuSplit {
  N: number;
  labels: Uint8Array;
  pixels: Uint8Array; // N x 3 x 32 x 32
}

export interface Cifar {
  train: CpuSplit;
  test: CpuSplit;
}

export interface GpuSplit {
  N: number;
  data: GPUBuffer; // packed uint8 CHW at the chosen resolution
  labels: GPUBuffer; // u32
}

export interface GpuDataset {
  res: number;
  D: number;
  mean: number[];
  std: number[];
  train: GpuSplit;
  test: GpuSplit;
}

// Sized from the known file size rather than Content-Length: servers that gzip (GitHub Pages does)
// report the compressed length, while the stream yields decompressed bytes.
async function fetchWithProgress(url: string, size: number, onProgress: (frac: number) => void): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url} (${resp.status}). Run \`npm run get-data\` first.`);
  if (!resp.body) return new Uint8Array(await resp.arrayBuffer());
  let out = new Uint8Array(size);
  const reader = resp.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) {
      const bigger = new Uint8Array(Math.max(out.length * 2, got + value.length));
      bigger.set(out.subarray(0, got));
      out = bigger;
    }
    out.set(value, got);
    got += value.length;
    onProgress(Math.min(1, got / size));
  }
  return out.subarray(0, got);
}

// Kept in Cache Storage so later visits skip the ~180 MB download; bump the name if the files change.
const DATA_CACHE = 'cifar10-v1';

async function cachedFetch(url: string, size: number, onProgress: (frac: number) => void): Promise<Uint8Array> {
  const cache = 'caches' in globalThis ? await caches.open(DATA_CACHE).catch(() => null) : null;
  const hit = await cache?.match(url).catch(() => undefined);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    if (bytes.length === size) {
      onProgress(1);
      return bytes;
    }
  }
  const bytes = await fetchWithProgress(url, size, onProgress);
  if (cache && bytes.length === size) cache.put(url, new Response(bytes)).catch(() => {}); // e.g. over storage quota
  return bytes;
}

function parse(bytes: Uint8Array): CpuSplit {
  const N = Math.floor(bytes.length / REC);
  const labels = new Uint8Array(N);
  const pixels = new Uint8Array(N * 3072);
  for (let i = 0; i < N; i++) {
    labels[i] = bytes[i * REC];
    pixels.set(bytes.subarray(i * REC + 1, (i + 1) * REC), i * 3072);
  }
  return { N, labels, pixels };
}

export async function loadCifar(onProgress: (msg: string) => void): Promise<Cifar> {
  const base = import.meta.env.BASE_URL + 'data/';
  let a = 0;
  let b = 0;
  const report = () => onProgress(`Downloading CIFAR-10… ${Math.round(((a * 5 + b) / 6) * 100)}%`);
  const [trainBytes, testBytes] = await Promise.all([
    cachedFetch(base + 'cifar10_train.bin', 50000 * REC, (f) => ((a = f), report())),
    cachedFetch(base + 'cifar10_test.bin', 10000 * REC, (f) => ((b = f), report())),
  ]);
  onProgress('Parsing…');
  return { train: parse(trainBytes), test: parse(testBytes) };
}

/** Box-filter downsample of the first n images to res x res. */
function downsample(split: CpuSplit, n: number, res: number): Uint8Array {
  const D = 3 * res * res;
  const out = new Uint8Array(Math.ceil((n * D) / 4) * 4);
  if (res === 32) {
    out.set(split.pixels.subarray(0, n * 3072));
    return out;
  }
  const f = 32 / res;
  const inv = 1 / (f * f);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const src = i * 3072 + c * 1024;
      const dst = i * D + c * res * res;
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          let s = 0;
          for (let dy = 0; dy < f; dy++) {
            const row = src + (y * f + dy) * 32 + x * f;
            for (let dx = 0; dx < f; dx++) s += split.pixels[row + dx];
          }
          out[dst + y * res + x] = Math.round(s * inv);
        }
      }
    }
  }
  return out;
}

function channelStats(px: Uint8Array, n: number, res: number) {
  const hw = res * res;
  const mean = [0, 0, 0];
  const std = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < n; i++) {
      const base = i * 3 * hw + c * hw;
      for (let k = 0; k < hw; k++) {
        const v = px[base + k] / 255;
        s += v;
        s2 += v * v;
      }
    }
    const cnt = n * hw;
    mean[c] = s / cnt;
    std[c] = Math.sqrt(Math.max(1e-8, s2 / cnt - mean[c] * mean[c]));
  }
  return { mean, std };
}

export interface Corruption {
  noise: number;
  labelNoise: number;
  seed: number;
}

/** Fixed (per-dataset, not per-step) Gaussian pixel noise and label flips, applied in place. */
function corrupt(px: Uint8Array, labels: Uint32Array, n: number, D: number, c: Corruption) {
  const rng = new Rng(c.seed, 3);
  if (c.noise > 0) {
    const s = c.noise * 255;
    for (let i = 0; i < n * D; i++) px[i] = Math.max(0, Math.min(255, Math.round(px[i] + s * rng.normal())));
  }
  if (c.labelNoise > 0) {
    const K = CLASSES.length;
    for (let i = 0; i < n; i++) if (rng.uniform() < c.labelNoise) labels[i] = (labels[i] + 1 + rng.int(K - 1)) % K;
  }
}

const gpuCache = new Map<string, GpuDataset>();

export function getGpuDataset(
  device: GPUDevice,
  cifar: Cifar,
  res: number,
  trainSize: number,
  testSize: number,
  corruption: Corruption = { noise: 0, labelNoise: 0, seed: 0 },
): GpuDataset {
  trainSize = Math.min(trainSize, cifar.train.N);
  testSize = Math.min(testSize, cifar.test.N);
  const corrupted = corruption.noise > 0 || corruption.labelNoise > 0;
  const key = `${res}/${trainSize}/${testSize}` + (corrupted ? `/${corruption.noise}/${corruption.labelNoise}/${corruption.seed}` : '');
  const hit = gpuCache.get(key);
  if (hit) return hit;
  // Keep at most a couple of datasets resident.
  if (gpuCache.size >= 2) {
    const [k, v] = gpuCache.entries().next().value as [string, GpuDataset];
    for (const s of [v.train, v.test]) {
      s.data.destroy();
      s.labels.destroy();
    }
    gpuCache.delete(k);
  }
  const upload = (split: CpuSplit, n: number, c?: Corruption): { split: GpuSplit; px: Uint8Array } => {
    const px = downsample(split, n, res);
    const lab = Uint32Array.from(split.labels.subarray(0, n));
    if (c) corrupt(px, lab, n, 3 * res * res, c);
    const data = device.createBuffer({ size: px.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(data, 0, px);
    const labels = device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(labels, 0, lab);
    return { split: { N: n, data, labels }, px };
  };
  const tr = upload(cifar.train, trainSize, corrupted ? corruption : undefined);
  const te = upload(cifar.test, testSize);
  const { mean, std } = channelStats(tr.px, trainSize, res);
  const ds: GpuDataset = { res, D: 3 * res * res, mean, std, train: tr.split, test: te.split };
  gpuCache.set(key, ds);
  return ds;
}
