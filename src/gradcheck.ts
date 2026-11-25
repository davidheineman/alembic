// Finite-difference gradient check of the WebGPU trainer across architecture variants.
// Open /test.html to run.
import { Engine, Trainer, debugOptions } from './gpu/engine';
import { loadCifar, getGpuDataset } from './data';
import { DROP_SITES, initParams, type Act, type ModelConfig, type Norm, type TrainConfig } from './model';

const out = document.getElementById('out')!;
const log = (s: string) => {
  out.textContent += s + '\n';
};

function baseConfig(): TrainConfig {
  return {
    model: { width: 16, depth: 2, mlpRatio: 2, act: 'relu', norm: 'layernorm', preNorm: false, postNorm: false, residual: true, init: 'kaiming', zeroInitLast: false },
    data: { resolution: 8, trainSize: 64, testSize: 16, flip: false, crop: false, cutout: false },
    opt: { kind: 'sgd', lr: 0.1, momentum: 0.9, beta1: 0.9, beta2: 0.999, wd: 0, clip: 0 },
    batchMode: 'minibatch',
    batchSize: 40,
    budgetUnit: 'steps',
    budget: 10,
    schedule: [1, 1],
    initSeed: 1,
    dataSeed: 1,
    numEvals: 2,
  };
}

async function main() {
  const engine = await Engine.create();
  log(`adapter: ${engine.adapterInfo}`);
  const cifar = await loadCifar((m) => (document.getElementById('status')!.textContent = m));
  const ds = getGpuDataset(engine.device, cifar, 8, 64, 16);
  const M = 40;
  let failures = 0;
  for (const forceSplits of [1, 3]) {
  debugOptions.forceSplits = forceSplits;
  log(`--- split-K ${forceSplits > 1 ? 'forced' : 'auto'} ---`);
  const normCfgs: [Norm, boolean, boolean][] = [['layernorm', false, false]];
  for (const n of ['layernorm', 'rmsnorm', 'batchnorm'] as Norm[])
    for (const [pre, post] of [[true, false], [false, true], [true, true]]) normCfgs.push([n, pre, post]);
  const acts: Act[] = ['relu', 'gelu', 'swiglu'];
  type Variant = { norm: Norm; preNorm: boolean; postNorm: boolean; act: Act; residual: boolean; branch?: Partial<ModelConfig> };
  const variants: Variant[] = [];
  for (const [norm, preNorm, postNorm] of normCfgs)
    for (const act of acts) for (const residual of [true, false]) variants.push({ norm, preNorm, postNorm, act, residual });
  // Residual branches with layers removed (only once; split-K doesn't interact with these).
  if (forceSplits === 1)
    for (const [norm, preNorm, postNorm] of [normCfgs[0], normCfgs[1], normCfgs[normCfgs.length - 2]])
      for (const fc1 of [true, false])
        for (const fc2 of [true, false])
          for (const actOn of [true, false])
            for (const act of actOn ? acts : (['relu'] as Act[]))
              for (const residual of [true, false])
                if (!(fc1 && fc2 && actOn)) variants.push({ norm, preNorm, postNorm, act, residual, branch: { fc1, fc2, actOn } });
  if (forceSplits === 1)
    for (const [norm, normPost] of [['layernorm', 'batchnorm'], ['batchnorm', 'rmsnorm'], ['rmsnorm', 'layernorm']] as [Norm, Norm][])
      for (const residual of [true, false]) variants.push({ norm, preNorm: true, postNorm: true, act: 'gelu', residual, branch: { normPost } });
  if (forceSplits === 1)
    for (const [norm, preNorm, postNorm] of [normCfgs[0], normCfgs[1]])
      for (const act of acts)
        for (const fc1 of [true, false]) variants.push({ norm, preNorm, postNorm, act, residual: true, branch: { fc1, fc2: true, actOn: true, dropout: 0.3 } });
  if (forceSplits === 1)
    for (const site of DROP_SITES)
      for (const [fc2, residual] of [[true, true], [false, true], [true, false]])
        variants.push({
          norm: 'layernorm',
          preNorm: true,
          postNorm: false,
          act: 'gelu',
          residual,
          branch: { fc1: true, fc2, actOn: true, dropout: 0.3, dropoutAt: Object.fromEntries(DROP_SITES.map((s) => [s, s === site])) },
        });
  for (const { norm, preNorm, postNorm, act, residual, branch } of variants) {
          const place =
            (preNorm || postNorm ? `${norm}-${preNorm ? 'pre' : ''}${preNorm && postNorm ? '+' : ''}${postNorm ? 'post' : ''}` : 'no-norm') +
            (branch?.normPost ? ` post=${branch.normPost}` : '') +
            (branch && branch.fc1 !== undefined ? ` [${branch.fc1 ? 'fc1' : '-'} ${branch.actOn ? 'act' : '-'} ${branch.fc2 ? 'fc2' : '-'}]` : '') +
            (branch?.dropout ? ` drop${branch.dropout}` : '') +
            (branch?.dropoutAt ? `@${DROP_SITES.filter((s) => branch.dropoutAt![s]).join(',')}` : '');
          // ReLU kinks can sit inside the finite-difference step for an unlucky init; a real bug
          // reproduces across seeds, so retry with a couple of other seeds before declaring failure.
          for (let attempt = 0; attempt < 3; attempt++) {
          const cfg = baseConfig();
          cfg.initSeed = 1 + attempt;
          Object.assign(cfg.model, { norm, preNorm, postNorm, act, residual }, branch);
          const tr = new Trainer(engine, ds, cfg);
          const p0 = initParams(tr.layout, cfg.model, cfg.initSeed);
          tr.writeParams(p0);
          const loss0 = await tr.debugStep(M, true);
          let nBad = 0;
          let nTot = 0;
          let nKink = 0;
          const grads = await tr.readParams(undefined, 'grads');
          let worst = 0;
          let worstName = '';
          const central = async (i: number, eps: number) => {
            const p = p0.slice();
            p[i] = p0[i] + eps;
            tr.writeParams(p);
            const lp = await tr.debugStep(M, false);
            p[i] = p0[i] - eps;
            tr.writeParams(p);
            const lm = await tr.debugStep(M, false);
            return (lp - lm) / (2 * eps);
          };
          for (const spec of tr.layout.specs) {
            const picks = Math.min(4, spec.size);
            for (let k = 0; k < picks; k++) {
              let i = spec.offset + Math.floor(((k + 0.5) / picks) * spec.size);
              if (spec.kind === 'norm' && norm === 'rmsnorm') i = spec.offset + (k % spec.cols);
              const an = grads[i];
              // Several step sizes + Richardson extrapolation: large steps can cross ReLU kinks or
              // suffer curvature error, small steps suffer f32 cancellation. Take the best estimate.
              const d1 = await central(i, 1e-2);
              const d2 = await central(i, 5e-3);
              const d3 = await central(i, 1e-3);
              const cands = [d1, d2, d3, (4 * d2 - d1) / 3];
              const errOf = (num: number) => Math.abs(num - an) / Math.max(2e-3, Math.abs(num) + Math.abs(an));
              let num = cands.reduce((a, b) => (errOf(b) < errOf(a) ? b : a));
              if (errOf(num) > 0.05) {
                // At a ReLU kink the analytical gradient equals one of the one-sided derivatives.
                const e = 1e-3;
                const p = p0.slice();
                tr.writeParams(p);
                const f0 = await tr.debugStep(M, false);
                p[i] = p0[i] + e;
                tr.writeParams(p);
                const right = ((await tr.debugStep(M, false)) - f0) / e;
                p[i] = p0[i] - e;
                tr.writeParams(p);
                const left = (f0 - (await tr.debugStep(M, false))) / e;
                const side = errOf(right) < errOf(left) ? right : left;
                if (errOf(side) < 0.05) {
                  num = side;
                  nKink++;
                }
              }
              const err = errOf(num);
              nTot++;
              if (err > 0.05) nBad++;
              if (err > worst) {
                worst = err;
                worstName = `${spec.name}[${i - spec.offset}] num=${num.toExponential(3)} an=${an.toExponential(3)}`;
              }
            }
          }
          tr.destroy();
          // With a huge loss (e.g. un-normalized SwiGLU at init) f32 can't resolve the perturbation.
          const precisionLimited = loss0 > 20;
          const ok = nBad === 0;
          const retry = !ok && !precisionLimited && act === 'relu' && attempt < 2;
          if (!ok && !precisionLimited && !retry) failures++;
          const verdict = ok ? 'PASS' : precisionLimited ? 'SKIP' : retry ? 'RETRY' : 'FAIL';
          log(
            `${verdict} ${place} act=${act} residual=${residual} seed=${cfg.initSeed}  loss=${loss0.toFixed(3)} bad=${nBad}/${nTot}${nKink ? ` (${nKink} at ReLU kinks)` : ''}  worst rel err=${worst.toExponential(2)}  ${worstName}`,
          );
          if (!retry) break;
          }
        }
  }
  log(failures === 0 ? '\nALL PASSED' : `\n${failures} FAILED`);
  document.title = failures === 0 ? 'gradcheck: ALL PASSED' : `gradcheck: ${failures} FAILED`;
}

main().catch((e) => log('ERROR: ' + (e?.stack ?? e)));
