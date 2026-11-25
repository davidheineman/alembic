// Runs a few configs on the WebGPU engine and prints per-step records as JSON, for
// comparing against the PyTorch port (remote/compare.py). Open /remote/webgpu_ref.html.
import { Engine, Trainer, type StepRecord, type EvalRecord } from '../src/gpu/engine';
import { loadCifar, getGpuDataset } from '../src/data';
import type { TrainConfig } from '../src/model';
import { CASES } from './parity_cases';

const out = document.getElementById('out')!;

async function main() {
  const engine = await Engine.create();
  const cifar = await loadCifar((m) => (out.textContent = m));
  const results: Record<string, { train: StepRecord[]; evals: EvalRecord[] }> = {};
  for (const [name, cfg] of Object.entries(CASES) as [string, TrainConfig][]) {
    out.textContent = `running ${name}`;
    const ds = getGpuDataset(engine.device, cifar, cfg.data.resolution, cfg.data.trainSize, cfg.data.testSize);
    const tr = new Trainer(engine, ds, cfg);
    const r = { train: [] as StepRecord[], evals: [] as EvalRecord[] };
    await tr.run((u) => {
      r.train.push(...u.train);
      r.evals.push(...u.evals);
    });
    tr.destroy();
    results[name] = r;
  }
  out.textContent = JSON.stringify(results);
  out.dataset.done = '1';
}

main().catch((e) => (out.textContent = 'ERROR ' + e));
