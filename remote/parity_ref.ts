import { Rng } from '../src/rng';
import { buildLayout, initParams } from '../src/model';
const out: any = {};
for (const [seed, stream] of [[0, 1], [3, 1], [7, 2], [123, 1]]) {
  const r = new Rng(seed, stream);
  out[`n_${seed}_${stream}`] = Array.from({ length: 7 }, () => Math.fround(r.normal()));
}
// sampler replica (same code as engine.ts Sampler)
function sample(n: number, seed: number, flip: boolean, count: number) {
  const rng = new Rng(seed, 2);
  const perm = new Uint32Array(n); for (let i = 0; i < n; i++) perm[i] = i;
  let pos = n; const o: number[] = [];
  for (let i = 0; i < count; i++) {
    if (pos >= n) { for (let j = n - 1; j > 0; j--) { const k = rng.int(j + 1); const t = perm[j]; perm[j] = perm[k]; perm[k] = t; } pos = 0; }
    let v = perm[pos++]; if (flip && (rng.u32() & 1)) v += n; o.push(v);
  }
  return o;
}
out.s1 = sample(10, 0, false, 25);
out.s2 = sample(1000, 5, true, 2500).slice(-20);
const models = [
  { width: 32, depth: 2, mlpRatio: 2, act: 'swiglu', norm: 'layernorm', preNorm: true, postNorm: false, residual: true, init: 'kaiming', zeroInitLast: true },
  { width: 24, depth: 3, mlpRatio: 3, act: 'gelu', norm: 'rmsnorm', preNorm: true, postNorm: true, residual: false, init: 'uniform', zeroInitLast: false },
  { width: 16, depth: 1, mlpRatio: 4, act: 'relu', norm: 'batchnorm', preNorm: false, postNorm: true, residual: true, init: 'xavier', zeroInitLast: false },
];
out.models = models.map((m: any) => {
  const L = buildLayout(m, 192);
  const p = initParams(L, m, 4);
  return { m, specs: L.specs.map((s) => [s.name, s.size, Array.from(p.subarray(s.offset, s.offset + Math.min(5, s.size))), p.subarray(s.offset, s.offset + s.size).reduce((a, b) => a + b, 0)]) };
});
console.log(JSON.stringify(out));
