// WGSL kernels for a residual-MLP trainer. Every kernel shares the per-pass
// uniform `P` (group 1, dynamic offset) so a whole chunk of training steps can be
// encoded into a single command buffer with no CPU round-trips.

export const PASS_STRIDE = 256;
export const PASS_FLOATS = 8;

const PASS_HEADER = /* wgsl */ `
struct Pass {
  M: u32,          // rows in this (micro)batch
  sampleOff: u32,  // offset into the index buffer
  slot: u32,       // stats slot for loss / acc / grad-norm
  accum: u32,      // 1 if gradients should accumulate into existing values
  lr: f32,
  scale: f32,      // 1 / logical batch size
  t: f32,          // optimizer step (1-based)
  train: u32,
};
@group(1) @binding(0) var<uniform> P: Pass;

fn doAcc(mode: u32) -> bool {
  return mode == 1u || (mode == 2u && P.accum != 0u);
}

// Round to the nearest bfloat16 (8-bit mantissa), ties to even; result stays an f32.
fn bf16(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

fn hash(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

// Inverted-dropout multiplier for element (m, j) while training. The mask is a hash of
// (step, sample offset, row, column, site seed), so the backward pass regenerates it exactly.
fn dropScale(seed: u32, m: u32, j: u32, p: f32) -> f32 {
  if (P.train == 0u || p <= 0.0) { return 1.0; }
  let r = hash(hash(hash(u32(P.t) ^ (seed * 0x9e3779b9u)) + P.sampleOff + m) + j);
  return select(0.0, 1.0 / (1.0 - p), f32(r >> 8u) * (1.0 / 16777216.0) >= p);
}
`;

const withHeader = (src: string) => PASS_HEADER + src;

// ---------------------------------------------------------------------------
// Data: gather a batch of uint8 images (CHW) by index and normalize.
// Bit 31 of an index means "horizontal flip".
// aug word (0 = none): bit 31 active; bits 0-3 / 4-7 crop shift dx+8 / dy+8;
// bit 20 cutout on, centered at (bits 8-13, bits 14-19). Padding and cutout pixels are 0 after
// normalization, i.e. the dataset-mean color.
export const gatherWGSL = withHeader(/* wgsl */ `
struct U { D: u32, HW: u32, W: u32, cut: u32, mean: vec4f, stdv: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read> idx: array<u32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<u32>;
@group(0) @binding(6) var<storage, read> aug: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let d = g.x;
  let m = g.y;
  if (m >= P.M || d >= u.D) { return; }
  let raw = idx[P.sampleOff + m];
  let n = raw & 0x7fffffffu;
  if (d == 0u) { Y[m] = labels[n]; }
  let flip = (raw >> 31u) == 1u;
  let c = d / u.HW;
  let r = d % u.HW;
  var y = i32(r / u.W);
  var x = i32(r % u.W);
  let W = i32(u.W);
  let a = aug[P.sampleOff + m];
  if ((a >> 31u) == 1u) {
    if (((a >> 20u) & 1u) == 1u) {
      let lo = vec2i(i32((a >> 8u) & 63u), i32((a >> 14u) & 63u)) - vec2i(i32(u.cut) / 2);
      if (x >= lo.x && x < lo.x + i32(u.cut) && y >= lo.y && y < lo.y + i32(u.cut)) {
        X[m * u.D + d] = 0.0;
        return;
      }
    }
    x += i32(a & 15u) - 8;
    y += i32((a >> 4u) & 15u) - 8;
    if (x < 0 || x >= W || y < 0 || y >= W) {
      X[m * u.D + d] = 0.0;
      return;
    }
  }
  if (flip) { x = W - 1 - x; }
  let src = n * u.D + c * u.HW + u32(y) * u.W + u32(x);
  let b = (data[src >> 2u] >> ((src & 3u) * 8u)) & 255u;
  X[m * u.D + d] = (f32(b) / 255.0 - u.mean[c]) / u.stdv[c];
}
`);

// ---------------------------------------------------------------------------
// Tiled matmul. C = op(A) * op(B) [+ bias] [+ R] [+ C]
// Variants (compile-time): NN (forward), NT (dX = dY W^T), TN (dW = X^T dY).
export type MatmulVariant = 'NN' | 'NT' | 'TN';

export function matmulWGSL(variant: MatmulVariant): string {
  const transA = variant === 'TN';
  const transB = variant === 'NT';
  const loadA = transA
    ? `let i = e % 64u; let kk = e / 64u; let gi = row0 + i; let gk = k0 + kk;
       var v = 0.0; if (gi < rows && gk < kEnd) { v = A[gk * rows + gi]; }
       As[kk * 64u + i] = select(v, bf16(v), rndIn);`
    : `let i = e / 16u; let kk = e % 16u; let gi = row0 + i; let gk = k0 + kk;
       var v = 0.0; if (gi < rows && gk < kEnd) { v = A[gi * inner + gk]; }
       As[kk * 64u + i] = select(v, bf16(v), rndIn);`;
  const loadB = transB
    ? `let j = e / 16u; let kk = e % 16u; let gj = col0 + j; let gk = k0 + kk;
       var v = 0.0; if (gj < cols && gk < kEnd) { v = B[gj * inner + gk]; }
       Bs[kk * 64u + j] = select(v, bf16(v), rndIn);`
    : `let j = e % 64u; let kk = e / 64u; let gj = col0 + j; let gk = k0 + kk;
       var v = 0.0; if (gj < cols && gk < kEnd) { v = B[gk * cols + gj]; }
       Bs[kk * 64u + j] = select(v, bf16(v), rndIn);`;
  return withHeader(/* wgsl */ `
// flags: bit0 rows=P.M, bit1 inner=P.M, bit2 bias, bit3 residual, bits4-5 accMode,
// bit6 round operands to bf16, bit7 round the output to bf16
// Split-K: when dispatched with z > 1, workgroup z handles a slice of the inner
// dimension and writes raw partial sums to S; mmReduce finishes the epilogue.
struct U { rows: u32, cols: u32, inner: u32, flags: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> R: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> S: array<f32>;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  var rows = u.rows;
  var inner = u.inner;
  let cols = u.cols;
  if ((u.flags & 1u) != 0u) { rows = P.M; }
  if ((u.flags & 2u) != 0u) { inner = P.M; }
  let row0 = wid.y * 64u;
  let col0 = wid.x * 64u;
  if (row0 >= rows) { return; }
  let chunk = ((inner + nwg.z - 1u) / nwg.z + 15u) / 16u * 16u;
  let kBeg = wid.z * chunk;
  let kEnd = min(inner, kBeg + chunk);
  let tid = lid.y * 16u + lid.x;
  let rndIn = (u.flags & 64u) != 0u;
  var acc: array<f32, 16>;
  for (var k0 = kBeg; k0 < kEnd; k0 += 16u) {
    for (var l = 0u; l < 4u; l++) {
      let e = tid + l * 256u;
      { ${loadA} }
      { ${loadB} }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var r = 0u; r < 4u; r++) { a[r] = As[kk * 64u + lid.y + 16u * r]; }
      for (var c = 0u; c < 4u; c++) { b[c] = Bs[kk * 64u + lid.x + 16u * c]; }
      for (var r = 0u; r < 4u; r++) {
        for (var c = 0u; c < 4u; c++) {
          acc[r * 4u + c] = fma(a[r], b[c], acc[r * 4u + c]);
        }
      }
    }
    workgroupBarrier();
  }
  if (nwg.z > 1u) {
    let base = wid.z * rows * cols;
    for (var r = 0u; r < 4u; r++) {
      let i = row0 + lid.y + 16u * r;
      if (i >= rows) { continue; }
      for (var c = 0u; c < 4u; c++) {
        let j = col0 + lid.x + 16u * c;
        if (j < cols) { S[base + i * cols + j] = acc[r * 4u + c]; }
      }
    }
    return;
  }
  let hasBias = (u.flags & 4u) != 0u;
  let hasRes = (u.flags & 8u) != 0u;
  let addOld = doAcc((u.flags >> 4u) & 3u);
  for (var r = 0u; r < 4u; r++) {
    let i = row0 + lid.y + 16u * r;
    if (i >= rows) { continue; }
    for (var c = 0u; c < 4u; c++) {
      let j = col0 + lid.x + 16u * c;
      if (j >= cols) { continue; }
      let o = i * cols + j;
      var v = acc[r * 4u + c];
      if (hasBias) { v += bias[j]; }
      if (hasRes) { v += R[o]; }
      if (addOld) { v += C[o]; }
      if ((u.flags & 128u) != 0u) { v = bf16(v); }
      C[o] = v;
    }
  }
}
`);
}

// Sum split-K partials and apply the matmul epilogue.
export const mmReduceWGSL = withHeader(/* wgsl */ `
struct U { rows: u32, cols: u32, splits: u32, flags: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  var rows = u.rows;
  if ((u.flags & 1u) != 0u) { rows = P.M; }
  let n = rows * u.cols;
  let o = g.y * nwg.x * 256u + g.x;
  if (o >= n) { return; }
  var v = 0.0;
  for (var z = 0u; z < u.splits; z++) { v += S[z * n + o]; }
  if ((u.flags & 4u) != 0u) { v += bias[o % u.cols]; }
  if ((u.flags & 8u) != 0u) { v += R[o]; }
  if (doAcc((u.flags >> 4u) & 3u)) { v += C[o]; }
  if ((u.flags & 128u) != 0u) { v = bf16(v); }
  C[o] = v;
}
`);

// Column sums over the batch (bias gradients). 16 columns x 16 row-slices per workgroup.
const COL_REDUCE = /* wgsl */ `
var<workgroup> cr: array<f32, 256>;
var<workgroup> cr2: array<f32, 256>;
fn colReduce2(a: f32, b: f32, lx: u32, ly: u32) -> vec2f {
  cr[ly * 16u + lx] = a;
  cr2[ly * 16u + lx] = b;
  workgroupBarrier();
  for (var s = 8u; s > 0u; s >>= 1u) {
    if (ly < s) {
      cr[ly * 16u + lx] += cr[(ly + s) * 16u + lx];
      cr2[ly * 16u + lx] += cr2[(ly + s) * 16u + lx];
    }
    workgroupBarrier();
  }
  return vec2f(cr[lx], cr2[lx]);
}
`;

export const colsumWGSL = withHeader(COL_REDUCE + /* wgsl */ `
struct U { cols: u32, accMode: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let j = wid.x * 16u + lid.x;
  var s = 0.0;
  if (j < u.cols) {
    for (var m = lid.y; m < P.M; m += 16u) { s += dY[m * u.cols + j]; }
  }
  let r = colReduce2(s, 0.0, lid.x, lid.y);
  if (lid.y == 0u && j < u.cols) {
    var v = r.x;
    if (doAcc(u.accMode)) { v += out[j]; }
    out[j] = v;
  }
}
`);

// ---------------------------------------------------------------------------
// Activations. kind: 0 relu, 1 gelu (tanh approx), 2 swiglu (A has 2*n columns),
// 3 self-gated silu(x)*x (SwiGLU with no fc1 to produce separate gate/value), 4 identity.
// Optional dropout on the output (seed identifies the site).
const ACT_COMMON = /* wgsl */ `
struct U { n: u32, kind: u32, seed: u32, drop: f32 };
@group(0) @binding(0) var<uniform> u: U;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn tanhSafe(z: f32) -> f32 {
  let c = clamp(z, -15.0, 15.0);
  let e = exp(2.0 * c);
  return (e - 1.0) / (e + 1.0);
}
const GK: f32 = 0.7978845608;
`;

export const actFwdWGSL = withHeader(ACT_COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  let m = g.y;
  if (m >= P.M || j >= u.n) { return; }
  if (u.kind == 2u) {
    let x = A[m * 2u * u.n + j];
    let y = A[m * 2u * u.n + u.n + j];
    G[m * u.n + j] = x * sigmoid(x) * y * dropScale(u.seed, m, j, u.drop);
    return;
  }
  let x = A[m * u.n + j];
  var v = x;
  if (u.kind == 0u) { v = max(x, 0.0); }
  else if (u.kind == 1u) { v = 0.5 * x * (1.0 + tanhSafe(GK * (x + 0.044715 * x * x * x))); }
  else if (u.kind == 3u) { v = x * sigmoid(x) * x; }
  G[m * u.n + j] = v * dropScale(u.seed, m, j, u.drop);
}
`);

export const actBwdWGSL = withHeader(ACT_COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> dG: array<f32>;
@group(0) @binding(3) var<storage, read_write> dA: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  let m = g.y;
  if (m >= P.M || j >= u.n) { return; }
  let dg = dG[m * u.n + j] * dropScale(u.seed, m, j, u.drop);
  if (u.kind == 2u) {
    let ia = m * 2u * u.n + j;
    let x = A[ia];
    let y = A[ia + u.n];
    let s = sigmoid(x);
    let silu = x * s;
    dA[ia] = dg * y * s * (1.0 + x * (1.0 - s));
    dA[ia + u.n] = dg * silu;
    return;
  }
  let x = A[m * u.n + j];
  var d = 1.0;
  if (u.kind == 0u) { d = select(0.0, 1.0, x > 0.0); }
  else if (u.kind == 3u) {
    let s = sigmoid(x);
    d = x * s * (2.0 + x * (1.0 - s));
  }
  else if (u.kind == 1u) {
    let x2 = x * x;
    let t = tanhSafe(GK * (x + 0.044715 * x2 * x));
    d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GK * (1.0 + 3.0 * 0.044715 * x2);
  }
  dA[m * u.n + j] = dg * d;
}
`);

// out = dropout(x) + (useR ? r : 0) + (acc ? out : 0), over M rows of n columns.
export const addWGSL = withHeader(/* wgsl */ `
struct U { n: u32, useR: u32, acc: u32, seed: u32, drop: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> r: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  let m = g.y;
  if (m >= P.M || j >= u.n) { return; }
  let i = m * u.n + j;
  var v = x[i] * dropScale(u.seed, m, j, u.drop);
  if (u.useR == 1u) { v += r[i]; }
  if (u.acc == 1u) { v += out[i]; }
  out[i] = v;
}
`);

// ---------------------------------------------------------------------------
// LayerNorm / RMSNorm (row-wise). Norm params are [gamma(D), beta(D)].
const ROW_REDUCE = /* wgsl */ `
var<workgroup> red: array<f32, 64>;
fn wgSum(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

export const lnFwdWGSL = withHeader(ROW_REDUCE + /* wgsl */ `
struct U { D: u32, rms: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> gb: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l3: vec3u, @builtin(workgroup_id) w3: vec3u) {
  let lid = l3.x;
  let m = w3.x;
  if (m >= P.M) { return; }
  let D = u.D;
  let base = m * D;
  var mean = 0.0;
  if (u.rms == 0u) {
    var s = 0.0;
    for (var d = lid; d < D; d += 64u) { s += X[base + d]; }
    mean = wgSum(s, lid) / f32(D);
  }
  var s2 = 0.0;
  for (var d = lid; d < D; d += 64u) { let c = X[base + d] - mean; s2 += c * c; }
  let rstd = inverseSqrt(wgSum(s2, lid) / f32(D) + 1e-5);
  for (var d = lid; d < D; d += 64u) {
    var v = (X[base + d] - mean) * rstd * gb[d];
    if (u.rms == 0u) { v += gb[D + d]; }
    Y[base + d] = v;
  }
  if (lid == 0u) { stats[2u * m] = mean; stats[2u * m + 1u] = rstd; }
}
`);

export const lnBwdWGSL = withHeader(ROW_REDUCE + /* wgsl */ `
struct U { D: u32, rms: u32, accMode: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> gb: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> dX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l3: vec3u, @builtin(workgroup_id) w3: vec3u) {
  let lid = l3.x;
  let m = w3.x;
  if (m >= P.M) { return; }
  let D = u.D;
  let base = m * D;
  let mean = stats[2u * m];
  let rstd = stats[2u * m + 1u];
  var s1 = 0.0;
  var s2 = 0.0;
  for (var d = lid; d < D; d += 64u) {
    let dxh = dY[base + d] * gb[d];
    let xh = (X[base + d] - mean) * rstd;
    s1 += dxh;
    s2 += dxh * xh;
  }
  var m1 = wgSum(s1, lid) / f32(D);
  let m2 = wgSum(s2, lid) / f32(D);
  if (u.rms != 0u) { m1 = 0.0; }
  let addOld = doAcc(u.accMode);
  for (var d = lid; d < D; d += 64u) {
    let dxh = dY[base + d] * gb[d];
    let xh = (X[base + d] - mean) * rstd;
    var v = rstd * (dxh - m1 - xh * m2);
    if (addOld) { v += dX[base + d]; }
    dX[base + d] = v;
  }
}
`);

// Column-wise gamma/beta gradients for LN/RMS.
export const lnParamGradWGSL = withHeader(COL_REDUCE + /* wgsl */ `
struct U { D: u32, rms: u32, accMode: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgb: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let j = wid.x * 16u + lid.x;
  let D = u.D;
  var sg = 0.0;
  var sb = 0.0;
  if (j < D) {
    for (var m = lid.y; m < P.M; m += 16u) {
      let dy = dY[m * D + j];
      sg += dy * (X[m * D + j] - stats[2u * m]) * stats[2u * m + 1u];
      sb += dy;
    }
  }
  let r = colReduce2(sg, sb, lid.x, lid.y);
  if (lid.y == 0u && j < D) {
    var g = r.x;
    var b = r.y;
    if (u.rms != 0u) { b = 0.0; }
    if (doAcc(u.accMode)) { g += dgb[j]; b += dgb[D + j]; }
    dgb[j] = g;
    dgb[D + j] = b;
  }
}
`);

// BatchNorm (column-wise). stats: [mean(D), rstd(D)], running: [mean(D), var(D)].
export const bnFwdWGSL = withHeader(/* wgsl */ `
struct U { D: u32, momentum: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> gb: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> running: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  let D = u.D;
  if (j >= D) { return; }
  let M = P.M;
  var mean = 0.0;
  var rstd = 1.0;
  if (P.train != 0u) {
    var s = 0.0;
    for (var m = 0u; m < M; m++) { s += X[m * D + j]; }
    mean = s / f32(M);
    var s2 = 0.0;
    for (var m = 0u; m < M; m++) { let c = X[m * D + j] - mean; s2 += c * c; }
    let v = s2 / f32(M);
    rstd = inverseSqrt(v + 1e-5);
    stats[j] = mean;
    stats[D + j] = rstd;
    let unbiased = select(v, s2 / f32(M - 1u), M > 1u);
    running[j] = (1.0 - u.momentum) * running[j] + u.momentum * mean;
    running[D + j] = (1.0 - u.momentum) * running[D + j] + u.momentum * unbiased;
  } else {
    mean = running[j];
    rstd = inverseSqrt(running[D + j] + 1e-5);
  }
  let ga = gb[j];
  let be = gb[D + j];
  for (var m = 0u; m < M; m++) {
    Y[m * D + j] = (X[m * D + j] - mean) * rstd * ga + be;
  }
}
`);

export const bnBwdWGSL = withHeader(/* wgsl */ `
struct U { D: u32, accX: u32, accP: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> gb: array<f32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> dX: array<f32>;
@group(0) @binding(6) var<storage, read_write> dgb: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  let D = u.D;
  if (j >= D) { return; }
  let M = P.M;
  let mean = stats[j];
  let rstd = stats[D + j];
  var s1 = 0.0;
  var s2 = 0.0;
  for (var m = 0u; m < M; m++) {
    let dy = dY[m * D + j];
    s1 += dy;
    s2 += dy * (X[m * D + j] - mean) * rstd;
  }
  let ga = gb[j];
  let addX = doAcc(u.accX);
  let fm = f32(M);
  for (var m = 0u; m < M; m++) {
    let o = m * D + j;
    let xh = (X[o] - mean) * rstd;
    var v = ga * rstd * (dY[o] - s1 / fm - xh * s2 / fm);
    if (addX) { v += dX[o]; }
    dX[o] = v;
  }
  var dg = s2;
  var db = s1;
  if (doAcc(u.accP)) { dg += dgb[j]; db += dgb[D + j]; }
  dgb[j] = dg;
  dgb[D + j] = db;
}
`);

// ---------------------------------------------------------------------------
// Softmax cross-entropy. Writes per-row [loss, correct] and dlogits (train only).
export const ceWGSL = withHeader(/* wgsl */ `
struct U { C: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<u32>;
@group(0) @binding(3) var<storage, read_write> dZ: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowLoss: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let m = g.x;
  if (m >= P.M) { return; }
  let C = u.C;
  let base = m * C;
  var mx = Z[base];
  var am = 0u;
  var amax = abs(mx);
  for (var c = 1u; c < C; c++) {
    let z = Z[base + c];
    if (z > mx) { mx = z; am = c; }
    amax = max(amax, abs(z));
  }
  var se = 0.0;
  for (var c = 0u; c < C; c++) { se += exp(Z[base + c] - mx); }
  let lse = mx + log(se);
  let y = Y[m];
  rowLoss[4u * m] = lse - Z[base + y];
  rowLoss[4u * m + 1u] = select(0.0, 1.0, am == y);
  rowLoss[4u * m + 2u] = amax;
  if (P.train != 0u) {
    for (var c = 0u; c < C; c++) {
      let p = exp(Z[base + c] - lse);
      dZ[base + c] = (p - select(0.0, 1.0, c == y)) * P.scale;
    }
  }
}
`);

// rowLoss[4m+3] = max(acc ? old : 0, max_j |X[m, j]|); one workgroup per row.
export const rowAbsMaxWGSL = withHeader(/* wgsl */ `
struct U { n: u32, acc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> rowLoss: array<f32>;
var<workgroup> r: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l3: vec3u, @builtin(workgroup_id) w3: vec3u) {
  let m = w3.x;
  if (m >= P.M) { return; }
  let lid = l3.x;
  var v = 0.0;
  for (var j = lid; j < u.n; j += 64u) { v = max(v, abs(X[m * u.n + j])); }
  r[lid] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid < s) { r[lid] = max(r[lid], r[lid + s]); }
    workgroupBarrier();
  }
  if (lid == 0u) {
    var o = r[0];
    if (u.acc == 1u) { o = max(o, rowLoss[4u * m + 3u]); }
    rowLoss[4u * m + 3u] = o;
  }
}
`);

// Single-workgroup reduction of per-row loss into stats[slot].
// Stats slot (8 floats): loss sum, correct, |grad|^2, |params|^2, max |logit|, max |residual stream|.
export const STAT_STRIDE = 8;
export const lossReduceWGSL = withHeader(/* wgsl */ `
@group(0) @binding(0) var<storage, read> rowLoss: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
var<workgroup> rl: array<f32, 256>;
var<workgroup> rc: array<f32, 256>;
var<workgroup> rm: array<f32, 256>;
var<workgroup> rh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3u) {
  let lid = l3.x;
  var sl = 0.0;
  var sc = 0.0;
  var sm = 0.0;
  var sh = 0.0;
  for (var m = lid; m < P.M; m += 256u) {
    sl += rowLoss[4u * m];
    sc += rowLoss[4u * m + 1u];
    sm = max(sm, rowLoss[4u * m + 2u]);
    sh = max(sh, rowLoss[4u * m + 3u]);
  }
  rl[lid] = sl;
  rc[lid] = sc;
  rm[lid] = sm;
  rh[lid] = sh;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      rl[lid] += rl[lid + s];
      rc[lid] += rc[lid + s];
      rm[lid] = max(rm[lid], rm[lid + s]);
      rh[lid] = max(rh[lid], rh[lid + s]);
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    stats[8u * P.slot] += rl[0];
    stats[8u * P.slot + 1u] += rc[0];
    stats[8u * P.slot + 4u] = max(stats[8u * P.slot + 4u], rm[0]);
    stats[8u * P.slot + 5u] = max(stats[8u * P.slot + 5u], rh[0]);
  }
}
`);

// ---------------------------------------------------------------------------
// Global gradient norm (two stages) and the optimizer update.
export const gradNormPartialWGSL = withHeader(/* wgsl */ `
struct U { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
var<workgroup> r: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3u, @builtin(workgroup_id) w3: vec3u) {
  let lid = l3.x;
  var s = 0.0;
  for (var i = w3.x * 256u + lid; i < u.n; i += 65536u) { let g = G[i]; s += g * g; }
  r[lid] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid < k) { r[lid] += r[lid + k]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partial[w3.x] = r[0]; }
}
`);

// u.at: which stats field receives the squared norm (2 = gradients, 3 = parameters).
export const gradNormFinalWGSL = withHeader(/* wgsl */ `
struct U { at: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
var<workgroup> r: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l3: vec3u) {
  let lid = l3.x;
  r[lid] = partial[lid];
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid < k) { r[lid] += r[lid + k]; }
    workgroupBarrier();
  }
  if (lid == 0u) { stats[8u * P.slot + u.at] = r[0]; }
}
`);

export const optimizerWGSL = withHeader(/* wgsl */ `
// kind: 0 sgd, 1 sgd+momentum, 2 adamw. rnd: store weights in bf16 (pure-bf16 training).
struct U { n: u32, kind: u32, strideX: u32, rnd: u32, mom: f32, b1: f32, b2: f32, eps: f32, wd: f32, clip: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> W: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> m1: array<f32>;
@group(0) @binding(4) var<storage, read_write> m2: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read> stats: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y * u.strideX + g.x;
  if (i >= u.n) { return; }
  var gr = G[i];
  if (u.clip > 0.0) {
    let gn = sqrt(stats[8u * P.slot + 2u]);
    if (gn > u.clip) { gr *= u.clip / gn; }
  }
  let w = W[i];
  let decay = u.wd * mask[i] * w;
  let lr = P.lr;
  var nw: f32;
  if (u.kind == 0u) {
    nw = w - lr * (gr + decay);
  } else if (u.kind == 1u) {
    let v = u.mom * m1[i] + gr + decay;
    m1[i] = v;
    nw = w - lr * v;
  } else {
    let a = u.b1 * m1[i] + (1.0 - u.b1) * gr;
    let b = u.b2 * m2[i] + (1.0 - u.b2) * gr * gr;
    m1[i] = a;
    m2[i] = b;
    let ah = a / (1.0 - pow(u.b1, P.t));
    let bh = b / (1.0 - pow(u.b2, P.t));
    nw = w - lr * (ah / (sqrt(bh) + u.eps) + decay);
  }
  W[i] = select(nw, bf16(nw), u.rnd != 0u);
}
`);
