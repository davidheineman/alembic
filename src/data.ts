// CIFAR-10 loading. Binary format: records of [label(1), R(1024), G(1024), B(1024)].

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

async function fetchWithProgress(url: string, onProgress: (frac: number) => void): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url} (${resp.status}). Run \`npm run get-data\` first.`);
  const total = Number(resp.headers.get('content-length')) || 0;
  if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = resp.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out;
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
    fetchWithProgress(base + 'cifar10_train.bin', (f) => ((a = f), report())),
    fetchWithProgress(base + 'cifar10_test.bin', (f) => ((b = f), report())),
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

const gpuCache = new Map<string, GpuDataset>();

export function getGpuDataset(device: GPUDevice, cifar: Cifar, res: number, trainSize: number, testSize: number): GpuDataset {
  trainSize = Math.min(trainSize, cifar.train.N);
  testSize = Math.min(testSize, cifar.test.N);
  const key = `${res}/${trainSize}/${testSize}`;
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
  const upload = (split: CpuSplit, n: number): { split: GpuSplit; px: Uint8Array } => {
    const px = downsample(split, n, res);
    const data = device.createBuffer({ size: px.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(data, 0, px);
    const labels = device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(labels, 0, Uint32Array.from(split.labels.subarray(0, n)));
    return { split: { N: n, data, labels }, px };
  };
  const tr = upload(cifar.train, trainSize);
  const te = upload(cifar.test, testSize);
  const { mean, std } = channelStats(tr.px, trainSize, res);
  const ds: GpuDataset = { res, D: 3 * res * res, mean, std, train: tr.split, test: te.split };
  gpuCache.set(key, ds);
  return ds;
}
