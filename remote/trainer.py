"""PyTorch port of src/gpu/engine.ts + src/model.ts.

Same parameterization, seeds, data order, kernels (tanh-GELU, eps=1e-5 norms, running-stat
BatchNorm at eval), and optimizer update rules as the WebGPU trainer, so a config produces
the same curves in either backend up to float rounding.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from rng import Sampler, normals, uniforms

NUM_CLASSES = 10
REC = 3073


_gpu_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data


def load_cifar_bins(train_path: str, test_path: str):
    def parse(path):
        raw = np.fromfile(path, dtype=np.uint8)
        recs = raw[: (len(raw) // REC) * REC].reshape(-1, REC)
        return recs[:, 0].copy(), recs[:, 1:].copy()

    return parse(train_path), parse(test_path)


def downsample(pixels: np.ndarray, n: int, res: int) -> np.ndarray:
    px = pixels[:n]
    if res == 32:
        return px
    f = 32 // res
    s = px.reshape(n, 3, res, f, res, f).astype(np.float64).sum(axis=(3, 5))
    return np.floor(s / (f * f) + 0.5).astype(np.uint8).reshape(n, -1)


@dataclass
class Dataset:
    res: int
    D: int
    n_train: int
    n_test: int
    X_train: torch.Tensor  # [2 * n_train, D]: rows n.. are horizontally flipped copies
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def make_dataset_gpu(raw_gpu, res: int, train_size: int, test_size: int) -> Dataset:
    """make_dataset on device-resident uint8 pixels (see upload_raw); same math, a few ms instead of ~1 s."""
    (ytr, ptr), (yte, pte) = raw_gpu
    ntr = min(train_size, len(ytr))
    nte = min(test_size, len(yte))
    hw = res * res

    def down(px, n):
        px = px[:n]
        if res == 32:
            return px.float()
        f = 32 // res
        s = px.view(n, 3, res, f, res, f).float().sum(dim=(3, 5))
        return torch.floor(s / (f * f) + 0.5)

    tr = down(ptr, ntr)
    te = down(pte, nte)
    v = tr.reshape(ntr, 3, hw).double() / 255
    mean = v.mean(dim=(0, 2))
    std = torch.sqrt(torch.clamp((v * v).mean(dim=(0, 2)) - mean * mean, min=1e-8))
    mean_t = mean.float().view(1, 3, 1, 1)
    std_t = std.float().view(1, 3, 1, 1)

    def norm(px, n):
        return (px.view(n, 3, res, res) / 255.0 - mean_t) / std_t

    xtr = norm(tr, ntr)
    X_train = torch.cat([xtr, xtr.flip(-1)]).reshape(2 * ntr, -1).contiguous()
    return Dataset(
        res=res,
        D=3 * hw,
        n_train=ntr,
        n_test=nte,
        X_train=X_train,
        y_train=torch.cat([ytr[:ntr], ytr[:ntr]]),
        X_test=norm(te, nte).reshape(nte, -1).contiguous(),
        y_test=yte[:nte],
    )


def upload_raw(raw, device):
    (ytr, ptr), (yte, pte) = raw
    up = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
    return ((up(ytr, torch.int64), up(ptr, torch.uint8)), (up(yte, torch.int64), up(pte, torch.uint8)))


def make_dataset(raw, res: int, train_size: int, test_size: int, device) -> Dataset:
    (ytr, ptr), (yte, pte) = raw
    ntr = min(train_size, len(ytr))
    nte = min(test_size, len(yte))
    tr = downsample(ptr, ntr, res)
    te = downsample(pte, nte, res)
    hw = res * res
    v = tr.reshape(ntr, 3, hw).astype(np.float64) / 255
    mean = v.mean(axis=(0, 2))
    std = np.sqrt(np.maximum(1e-8, (v * v).mean(axis=(0, 2)) - mean * mean))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def norm(px, n):
        x = torch.from_numpy(np.ascontiguousarray(px)).to(device).view(n, 3, res, res).float() / 255.0
        return (x - mean_t) / std_t

    xtr = norm(tr, ntr)
    X_train = torch.cat([xtr, xtr.flip(-1)]).reshape(2 * ntr, -1).contiguous()
    X_test = norm(te, nte).reshape(nte, -1).contiguous()
    ytr_t = torch.from_numpy(ytr[:ntr].astype(np.int64)).to(device)
    return Dataset(
        res=res,
        D=3 * hw,
        n_train=ntr,
        n_test=nte,
        X_train=X_train,
        y_train=torch.cat([ytr_t, ytr_t]),
        X_test=X_test,
        y_test=torch.from_numpy(yte[:nte].astype(np.int64)).to(device),
    )


# ---------------------------------------------------------------------------
# Config helpers (mirror src/model.ts)


def hidden_width(m) -> int:
    h = m["width"] * m["mlpRatio"]
    if m["act"] == "swiglu":
        return max(8, round_half_up(h * 2 / 3 / 8) * 8)
    return max(1, round_half_up(h))


def branch_shape(m):
    """(fc1, act, fc2, hidden, hidden1, act_kind), mirroring branchShape in src/model.ts."""
    fc1 = m.get("fc1", True) is not False
    act = m.get("actOn", True) is not False
    fc2 = m.get("fc2", True) is not False
    glu = act and m["act"] == "swiglu"
    if not fc1 or not fc2:
        hidden = m["width"]
    elif glu:
        hidden = hidden_width(m)
    else:
        hidden = max(1, round_half_up(m["width"] * m["mlpRatio"]))
    hidden1 = 2 * hidden if glu and fc1 else hidden
    if not act:
        kind = "none"
    elif m["act"] == "swiglu" and not fc1:
        kind = "selfglu"
    else:
        kind = m["act"]
    return fc1, act, fc2, hidden, hidden1, kind


def norm_kind(m, name: str) -> str:
    """Post-norms may use a different type than pre/final norms (normPost in src/model.ts)."""
    return (m.get("normPost") or m["norm"]) if ".postNorm" in name else m["norm"]


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def effective_batch(cfg) -> int:
    if cfg["batchMode"] == "full":
        return cfg["data"]["trainSize"]
    if cfg["batchMode"] == "sgd":
        return 1
    return max(1, min(cfg["batchSize"], cfg["data"]["trainSize"]))


def total_steps(cfg) -> int:
    B = effective_batch(cfg)
    if cfg["budgetUnit"] == "steps":
        return max(1, round_half_up(cfg["budget"]))
    return max(1, round_half_up(cfg["budget"] * cfg["data"]["trainSize"] / B))


def lr_multiplier(schedule, frac: float) -> float:
    n = len(schedule)
    x = min(1.0, max(0.0, frac)) * (n - 1)
    i = min(n - 2, math.floor(x))
    t = x - i
    return schedule[i] * (1 - t) + schedule[i + 1] * t


def eval_schedule(T: int, n: int, log: bool) -> list[int]:
    s = {0, T}
    for i in range(n):
        if log:
            s.add(round_half_up(T ** (i / max(1, n - 1))))
        else:
            s.add(round_half_up(T * i / max(1, n - 1)))
    return sorted(x for x in s if 0 <= x <= T)


def browser_micro_batches(cfg, in_dim: int, B: int) -> list[int]:
    """The engine's micro-batch split. Only matters for BatchNorm, whose statistics are per micro-batch."""
    m = cfg["model"]
    d = m["width"]
    _, _, _, hidden, hidden1, _ = branch_shape(m)
    per_row = in_dim + 2 + (m["depth"] + 1) * d + m["depth"] * (2 * d + hidden1 + hidden) + 4 * d + hidden + hidden1 + 4 * NUM_CLASSES + 8
    mmax = max(1, min(1024, (448 * 1024 * 1024 // 4) // per_row))
    n = max(-(-B // mmax), min(B, cfg.get("microBatches") or 0))
    return [B // n + (1 if i < B % n else 0) for i in range(n)]


def split_even(B: int, cap: int) -> list[int]:
    n = -(-B // cap)
    return [B // n + (1 if i < B % n else 0) for i in range(n)]


# ---------------------------------------------------------------------------
# Model


@dataclass
class Spec:
    name: str
    kind: str  # matrix | bias | norm
    rows: int
    cols: int
    is_last: bool = False


@dataclass
class Model:
    cfg: dict
    in_dim: int
    params: dict[str, torch.Tensor] = field(default_factory=dict)
    running: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    precision: str = "fp32"

    def to(self, device) -> "Model":
        self.params = {k: v.detach().to(device).requires_grad_(True) for k, v in self.params.items()}
        self.running = {k: (m.to(device), v.to(device)) for k, (m, v) in self.running.items()}
        return self

    def specs(self) -> list[Spec]:
        m = self.cfg
        d = m["width"]
        has_fc1, _, has_fc2, hidden, hidden1, _ = branch_shape(m)
        out: list[Spec] = []

        def linear(name, fin, fout, last=False):
            out.append(Spec(f"{name}.W", "matrix", fin, fout, last))
            out.append(Spec(f"{name}.b", "bias", 1, fout, last))

        linear("stem", self.in_dim, d)
        for l in range(m["depth"]):
            if m["preNorm"]:
                out.append(Spec(f"block{l}.preNorm", "norm", 1, d))
            if has_fc1:
                linear(f"block{l}.fc1", d, hidden1)
            if has_fc2:
                linear(f"block{l}.fc2", hidden, d, True)
            if m["postNorm"]:
                out.append(Spec(f"block{l}.postNorm", "norm", 1, d))
        if m["preNorm"] and not m["postNorm"]:
            out.append(Spec("final.norm", "norm", 1, d))
        linear("head", d, NUM_CLASSES, True)
        return out


def init_model(m: dict, in_dim: int, seed: int, device) -> Model:
    model = Model(m, in_dim)
    specs = model.specs()
    draws = []
    for s in specs:
        random = s.kind == "matrix" and m["init"] != "zero" and not (m["zeroInitLast"] and s.is_last)
        draws.append(s.rows * s.cols if random else 0)
    # Every random matrix uses the same distribution, so one sequential stream reproduces the JS loop.
    z = uniforms(seed, 1, sum(draws)) if m["init"] == "uniform" else normals(seed, 1, sum(draws))
    off = 0
    for s, n in zip(specs, draws):
        if s.kind == "norm":
            kind = norm_kind(m, s.name)
            model.params[s.name + ".g"] = torch.ones(s.cols)
            if kind != "rmsnorm":
                model.params[s.name + ".b"] = torch.zeros(s.cols)
            if kind == "batchnorm":
                model.running[s.name] = (torch.zeros(s.cols, device=device), torch.ones(s.cols, device=device))
            continue
        if n == 0:
            model.params[s.name] = torch.zeros(s.rows, s.cols) if s.kind == "matrix" else torch.zeros(s.cols)
            continue
        if m["init"] == "uniform":
            vals = (2 * z[off : off + n] - 1) * (1 / math.sqrt(s.rows))
        else:
            std = math.sqrt(2 / s.rows) if m["init"] == "kaiming" else math.sqrt(2 / (s.rows + s.cols))
            vals = z[off : off + n] * std
        w = torch.from_numpy(vals.astype(np.float32)).view(s.rows, s.cols)
        off += n
        model.params[s.name] = w
    model.params = {k: v.to(device).requires_grad_(True) for k, v in model.params.items()}
    return model


def act_fn(a: torch.Tensor, kind: str, hidden: int) -> torch.Tensor:
    if kind == "relu":
        return F.relu(a)
    if kind == "gelu":
        return F.gelu(a, approximate="tanh")
    if kind == "none":
        return a
    if kind == "selfglu":
        return F.silu(a) * a
    return F.silu(a[:, :hidden]) * a[:, hidden:]


def norm_fn(model: Model, name: str, x: torch.Tensor, train: bool) -> torch.Tensor:
    kind = norm_kind(model.cfg, name)
    p = model.params
    g = p[name + ".g"]
    if kind == "layernorm":
        return F.layer_norm(x, (x.shape[1],), g, p[name + ".b"], eps=1e-5)
    if kind == "rmsnorm":
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5) * g
    rm, rv = model.running[name]
    if train:
        mean = x.mean(0)
        var = (x - mean).pow(2).mean(0)
        with torch.no_grad():
            M = x.shape[0]
            unbiased = var * (M / (M - 1)) if M > 1 else var
            rm.mul_(0.9).add_(0.1 * mean)
            rv.mul_(0.9).add_(0.1 * unbiased)
        return (x - mean) * torch.rsqrt(var + 1e-5) * g + p[name + ".b"]
    return (x - rm) * torch.rsqrt(rv + 1e-5) * g + p[name + ".b"]


def forward(model: Model, x: torch.Tensor, train: bool) -> torch.Tensor:
    # mp and bf16 both run matmuls under bf16 autocast; bf16 additionally keeps the weights in bf16 (see bf16_round_).
    if model.precision != "fp32":
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            return _forward(model, x, train).float()
    return _forward(model, x, train)


@torch.no_grad()
def bf16_round_(ts: list[torch.Tensor]):
    for t in ts:
        t.copy_(t.to(torch.bfloat16).float())


def _forward(model: Model, x: torch.Tensor, train: bool) -> torch.Tensor:
    m = model.cfg
    p = model.params
    has_fc1, _, has_fc2, hidden, _, kind = branch_shape(m)
    sites = m.get("dropoutAt") or {}

    def drop(t: torch.Tensor, site: str) -> torch.Tensor:
        p_drop = (m.get("dropout") or 0.0) if sites.get(site, True) is not False else 0.0
        return F.dropout(t, p_drop, training=True) if train and p_drop > 0 else t

    h = drop(torch.addmm(p["stem.b"], x, p["stem.W"]), "stem")
    for l in range(m["depth"]):
        blk = f"block{l}"
        u = norm_fn(model, blk + ".preNorm", h, train) if m["preNorm"] else h
        a = torch.addmm(p[blk + ".fc1.b"], u, p[blk + ".fc1.W"]) if has_fc1 else u
        s = drop(act_fn(a, kind, hidden), "act")
        if has_fc2:
            s = torch.addmm(p[blk + ".fc2.b"], s, p[blk + ".fc2.W"])
        s = drop(s, "branch")
        if m["residual"]:
            s = s + h
        h = norm_fn(model, blk + ".postNorm", s, train) if m["postNorm"] else s
    if m["preNorm"] and not m["postNorm"]:
        h = norm_fn(model, "final.norm", h, train)
    return torch.addmm(p["head.b"], h, p["head.W"])


# ---------------------------------------------------------------------------
# Training


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit(do_not_specialize=["n", "n_decay"])
    def _update_kernel(P, G, M1, M2, LR, BC1, BC2, SCALE, n, n_decay, w1, b2, w2, mom, wd, KIND: tl.constexpr, CLIP: tl.constexpr, BLOCK: tl.constexpr):
        """Optimizer.step's update rules (same operation order) in one pass. KIND: 0 adam, 1 momentum, 2 sgd."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        p = tl.load(P + offs, mask=mask)
        g = tl.load(G + offs, mask=mask)
        if CLIP:
            g = g * tl.load(SCALE)
        decay = offs < n_decay
        if KIND == 0:
            m = tl.load(M1 + offs, mask=mask)
            v = tl.load(M2 + offs, mask=mask)
            m = m + w1 * (g - m)  # torch.lerp for weights < 0.5
            v = v * b2 + w2 * g * g
            tl.store(M1 + offs, m, mask=mask)
            tl.store(M2 + offs, v, mask=mask)
            upd = tl.div_rn(tl.div_rn(m, tl.load(BC1)), tl.sqrt_rn(tl.div_rn(v, tl.load(BC2))) + 1e-8)
            upd = tl.where(decay, upd + wd * p, upd)
        elif KIND == 1:
            m = tl.load(M1 + offs, mask=mask)
            m = m * mom + g
            m = tl.where(decay, m + wd * p, m)
            tl.store(M1 + offs, m, mask=mask)
            upd = m
        else:
            upd = tl.where(decay, g + wd * p, g)
        tl.store(P + offs, p + upd * -tl.load(LR), mask=mask)


def warm_kernels(device):
    """Compiles every _update_kernel variant (~2 s in all) so runs don't pay for it. Runs that start
    meanwhile wait on the lock."""
    if triton is None or device != "cuda":
        return
    with _gpu_lock:
        torch.cuda.set_stream(torch.cuda.Stream())
        for kind in ("adam", "momentum", "sgd"):
            for clip in (0, 1):
                m = Model({}, 0, {"w.W": torch.zeros(4, 4, device=device, requires_grad=True)})
                opt = Optimizer(m, {"kind": kind, "beta1": 0.9, "beta2": 0.99, "momentum": 0.9, "wd": 0, "clip": clip})
                one = torch.ones((), device=device)
                opt.collect_grads()
                opt.step(one, one, one)
        torch.cuda.current_stream().synchronize()


class Optimizer:
    """On CUDA all parameters, gradients and moments live in flat buffers (P, G, M1, M2; the per-tensor
    lists are views), so the update is one full-width kernel. PyTorch's foreach and fused optimizers
    split work into 64K-element chunks: a 2M-parameter model gets ~30 thread blocks, and their update
    took ~45% of its step."""

    def __init__(self, model: Model, o: dict):
        self.o = o
        names = [k for k in model.params if k.endswith(".W")] + [k for k in model.params if not k.endswith(".W")]
        self.flat = triton is not None and next(iter(model.params.values())).is_cuda
        if self.flat:
            sizes = [model.params[k].numel() for k in names]
            self.P = torch.cat([model.params[k].detach().reshape(-1) for k in names])
            self.G = torch.zeros_like(self.P)
            self.M1 = torch.zeros_like(self.P) if o["kind"] != "sgd" else None
            self.M2 = torch.zeros_like(self.P) if o["kind"] == "adam" else None
            self.n_decay = sum(s for k, s in zip(names, sizes) if k.endswith(".W"))
            views = lambda flat: [v.view(model.params[k].shape) for k, v in zip(names, flat.split(sizes))]
            for k, p in zip(names, views(self.P)):
                model.params[k] = p.requires_grad_(True)
            self.m1 = views(self.M1) if self.M1 is not None else None
            self.m2 = views(self.M2) if self.M2 is not None else None
        else:
            zeros = lambda: [torch.zeros_like(model.params[k]) for k in names]
            self.m1 = zeros() if o["kind"] != "sgd" else None
            self.m2 = zeros() if o["kind"] == "adam" else None
        self.decay = [model.params[k] for k in names if k.endswith(".W")]
        self.other = [model.params[k] for k in names if not k.endswith(".W")]
        self.params = self.decay + self.other

    def state(self) -> list[torch.Tensor]:
        if self.flat:
            return [t for t in (self.P, self.M1, self.M2) if t is not None]
        return self.params + (self.m1 or []) + (self.m2 or [])

    def moments(self) -> list[torch.Tensor]:
        if self.flat:
            return [t for t in (self.M1, self.M2) if t is not None]
        return (self.m1 or []) + (self.m2 or [])

    def weights(self) -> list[torch.Tensor]:
        return [self.P] if self.flat else self.params

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def collect_grads(self) -> torch.Tensor:
        """Gathers the gradients for step(); returns their squared norm."""
        if self.flat:
            # One copy into G is cheaper than presetting .grad to views of G, which makes backward
            # accumulate into them with a separate add per tensor.
            torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device) for p in self.params], out=self.G)
            return self.G.pow(2).sum()
        grads = [p.grad for p in self.params if p.grad is not None]
        return torch.stack(torch._foreach_norm(grads)).pow(2).sum()

    @torch.no_grad()
    def step(self, lr: torch.Tensor, t: torch.Tensor, gn2: torch.Tensor):
        """lr and t are 0-dim device tensors so the update can be captured in a CUDA graph."""
        o = self.o
        scale = torch.clamp(o["clip"] / gn2.sqrt(), max=1.0) if o["clip"] > 0 else None
        if self.flat:
            kind = {"adam": 0, "momentum": 1, "sgd": 2}[o["kind"]]
            b1, b2 = o.get("beta1", 0.9), o.get("beta2", 0.999)
            bc1 = 1 - torch.pow(b1, t) if kind == 0 else lr
            bc2 = 1 - torch.pow(b2, t) if kind == 0 else lr
            n, BLOCK = self.P.numel(), 1024
            m1 = self.M1 if self.M1 is not None else self.P
            m2 = self.M2 if self.M2 is not None else self.P
            _update_kernel[(triton.cdiv(n, BLOCK),)](self.P, self.G, m1, m2, lr, bc1, bc2, scale if scale is not None else lr, n, self.n_decay, 1 - b1, b2, 1 - b2, o.get("momentum", 0.0), o["wd"], KIND=kind, CLIP=scale is not None, BLOCK=BLOCK)
            return
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        if scale is not None:
            grads = torch._foreach_mul(grads, scale)
        nd = len(self.decay)
        if o["kind"] == "adam":
            b1, b2 = o["beta1"], o["beta2"]
            torch._foreach_lerp_(self.m1, grads, 1 - b1)
            torch._foreach_mul_(self.m2, b2)
            torch._foreach_addcmul_(self.m2, grads, grads, value=1 - b2)
            bc1 = 1 - torch.pow(b1, t)
            bc2 = 1 - torch.pow(b2, t)
            denom = torch._foreach_sqrt(torch._foreach_div(self.m2, bc2))
            torch._foreach_add_(denom, 1e-8)
            upd = torch._foreach_div(torch._foreach_div(self.m1, bc1), denom)
        else:
            upd = grads if o["kind"] == "sgd" else None
            if o["kind"] == "momentum":
                torch._foreach_mul_(self.m1, o["momentum"])
                torch._foreach_add_(self.m1, grads)
        if o["wd"] > 0 and nd:
            if o["kind"] == "momentum":
                torch._foreach_add_(self.m1[:nd], self.decay, alpha=o["wd"])
            else:
                upd = list(upd)
                upd[:nd] = torch._foreach_add(upd[:nd], self.decay, alpha=o["wd"])
        if o["kind"] == "momentum":
            upd = self.m1
        torch._foreach_add_(self.params, torch._foreach_mul(upd, -lr))


def capture(step_fn, state: list[torch.Tensor], opt: "Optimizer", ptr: torch.Tensor):
    """Records one training step as a CUDA graph. Warm-up iterations are rolled back afterwards."""
    saved = [t.detach().clone() for t in state]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step_fn()
            ptr.zero_()
    torch.cuda.current_stream().wait_stream(side)
    with torch.no_grad():
        for t, s in zip(state, saved):
            t.copy_(s)
    opt.zero_grad()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        step_fn()
    ptr.zero_()
    return graph.replay


def capture_eval(eval_fn):
    """Eager evals cost ~2 ms of host time each (under _gpu_lock), which held back every other run."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eval_fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        eval_fn()
    return graph.replay


@dataclass
class Session:
    """A model, optimizer and step graph that later runs with the same shapes and baked-in settings
    reuse by resetting values in place, so they skip graph capture (~0.3-0.7 s) entirely."""

    model: Model
    opt: "Optimizer"
    k_max: int
    idx_buf: torch.Tensor
    sched_buf: torch.Tensor
    stats: torch.Tensor
    ptr: torch.Tensor
    idx_host: torch.Tensor
    sched_host: torch.Tensor
    stats_host: torch.Tensor
    train_step: object
    eval_step: object  # writes [test loss, test acc] to eval_out
    eval_out: torch.Tensor
    replay: object = None
    eval_replay: object = None
    stream: object = None

    def capture(self):
        self.replay = capture(self.train_step, self.state(), self.opt, self.ptr)
        self.eval_replay = capture_eval(self.eval_step)

    def state(self) -> list[torch.Tensor]:
        return self.opt.state() + [t for pair in self.model.running.values() for t in pair]

    @torch.no_grad()
    def reset(self, fresh: Model):
        for name, t in self.model.params.items():
            t.copy_(fresh.params[name])
        for mean, var in self.model.running.values():
            mean.zero_()
            var.fill_(1)
        for m in self.opt.moments():
            m.zero_()


_sessions: dict[str, list[Session]] = {}  # idle sessions by key
_live: dict[str, int] = {}  # sessions per key, idle or in use
_MAX_SESSIONS = 24


def _take_session(key: str) -> Session | None:
    idle = _sessions.get(key)
    return idle.pop() if idle else None


def _put_session(key: str, sess: Session):
    with _gpu_lock:
        _sessions.setdefault(key, []).append(sess)
        while sum(len(v) for v in _sessions.values()) > _MAX_SESSIONS:
            oldest = next(k for k, v in _sessions.items() if v)
            _sessions[oldest].pop(0)
            _live[oldest] -= 1
            if not _sessions[oldest]:
                del _sessions[oldest]


def _new_session(model: Model, cfg: dict, ds: Dataset, B: int, micro: list[int], precision: str, device) -> Session:
    """Call under _gpu_lock."""
    cuda = device == "cuda"
    model.precision = precision
    pure_bf16 = precision == "bf16"
    opt = Optimizer(model, cfg["opt"])
    # Per-chunk inputs live on the device and the step reads row `ptr` of each, so a chunk of
    # steps runs as back-to-back graph replays with no host work in between.
    k_max = max(2, min(1024, (1 << 22) // B))
    idx_buf = torch.zeros(k_max, B, dtype=torch.int64, device=device)
    sched_buf = torch.ones(2, k_max, device=device)  # rows: lr, optimizer step t
    stats = torch.zeros(k_max, 3, device=device)
    ptr = torch.zeros(1, dtype=torch.int64, device=device)

    def train_step():
        idx = idx_buf.index_select(0, ptr).view(-1)
        lr = sched_buf[0].index_select(0, ptr).view(())
        t = sched_buf[1].index_select(0, ptr).view(())
        opt.zero_grad()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        off = 0
        for Mi in micro:
            bi = idx[off : off + Mi]
            off += Mi
            z = forward(model, ds.X_train[bi], True)
            y = ds.y_train[bi]
            l = F.cross_entropy(z, y, reduction="sum")
            (l / B).backward()
            loss_sum += l.detach()
            correct += (z.detach().argmax(1) == y).sum()
        gn2 = opt.collect_grads()
        opt.step(lr, t, gn2)
        if pure_bf16:
            bf16_round_(opt.weights())
        stats.index_copy_(0, ptr, torch.stack([loss_sum / B, correct / B, gn2]).view(1, 3))
        ptr.add_(1)

    eval_out = torch.zeros(2, device=device)
    # A captured eval keeps its activations allocated; this caps each at 32 MB (8192 rows up to 1024 wide).
    widest = max(model.cfg["width"], branch_shape(model.cfg)[4])
    rows = max(512, min(8192, (1 << 23) // widest))

    @torch.no_grad()
    def eval_step():
        loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(0, ds.n_test, rows):
            z = forward(model, ds.X_test[i : i + rows], False)
            y = ds.y_test[i : i + rows]
            loss += F.cross_entropy(z, y, reduction="sum")
            correct += (z.argmax(1) == y).sum()
        eval_out.copy_(torch.stack([loss / ds.n_test, correct / ds.n_test]))

    return Session(
        model=model,
        opt=opt,
        k_max=k_max,
        idx_buf=idx_buf,
        sched_buf=sched_buf,
        stats=stats,
        ptr=ptr,
        idx_host=torch.zeros(k_max, B, dtype=torch.int64, pin_memory=cuda),
        sched_host=torch.zeros(2, k_max, pin_memory=cuda),
        stats_host=torch.zeros(k_max, 3, pin_memory=cuda),
        train_step=train_step,
        eval_step=eval_step,
        eval_out=eval_out,
    )


def _dataset(cfg: dict, raw, device, ds_cache: dict | None, raw_gpu) -> Dataset:
    """Cached per (resolution, trainSize, testSize). Call under _gpu_lock."""
    dc = cfg["data"]
    key = (dc["resolution"], dc["trainSize"], dc["testSize"])
    ds = ds_cache.get(key) if ds_cache is not None else None
    if ds is None:
        ds = make_dataset_gpu(raw_gpu, *key) if raw_gpu is not None else make_dataset(raw, *key, device)
        if ds_cache is not None:
            while len(ds_cache) >= 8:
                ds_cache.pop(next(iter(ds_cache)))
            ds_cache[key] = ds
    if device == "cuda":
        torch.cuda.current_stream().synchronize()
    return ds


def _session_key(cfg: dict, ds: Dataset) -> tuple[str, int, list[int], str]:
    """(key, batch, micro-batches, precision). The key is everything a captured graph bakes in; the
    LR schedule, step counter, batch indices and seeds are inputs, so LR sweeps and reseeded reruns
    share one session."""
    precision = cfg.get("precision") or "fp32"
    B = effective_batch(cfg)
    mc = cfg["model"]
    uses_bn = (mc["preNorm"] and mc["norm"] == "batchnorm") or (mc["postNorm"] and norm_kind(mc, ".postNorm") == "batchnorm")
    micro = browser_micro_batches(cfg, ds.D, B) if uses_bn or cfg.get("microBatches") else split_even(B, 16384)
    dc = cfg["data"]
    key = json.dumps([mc, {k: v for k, v in cfg["opt"].items() if k != "lr"}, precision, B, list(micro), [dc["resolution"], dc["trainSize"], dc["testSize"]]], sort_keys=True)
    return key, B, micro, precision


class _Waiting:
    """Runs waiting to set up. Background preparation yields the GPU lock to them."""

    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self._n += 1

    def __exit__(self, *exc):
        with self._lock:
            self._n -= 1

    def yield_to_runs(self):
        while self._n:
            time.sleep(0.005)


_waiting = _Waiting()
_PREP_MIN_FREE = 4 << 30  # stop preparing when less GPU memory than this is free


class _Preparer:
    """Builds sessions ahead of time in one background thread, so the runs the browser is about to
    start (the sidebar config, the first runs of the sweep) skip setup and graph capture. Each
    session holds the lock for at most one capture (~0.1 s) at a time."""

    def __init__(self):
        self.cv = threading.Condition()
        self.want = None
        self.thread = None

    def request(self, want: list[tuple[dict, int]], ctx: tuple):
        with self.cv:
            self.want = (want, ctx)
            self.cv.notify()
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def _loop(self):
        torch.cuda.set_stream(torch.cuda.Stream())
        while True:
            with self.cv:
                while self.want is None:
                    self.cv.wait()
                want, ctx = self.want
                self.want = None
            for cfg, count in want:
                try:
                    while self.want is None and self._build(cfg, count, *ctx):
                        pass
                except Exception as e:  # noqa: BLE001
                    print(f"prepare failed: {type(e).__name__}: {e}")

    def _build(self, cfg, count, raw, device, ds_cache, raw_gpu) -> bool:
        """Adds one idle session for cfg unless it already has `count`. False when done."""
        _waiting.yield_to_runs()
        with _gpu_lock:
            ds = _dataset(cfg, raw, device, ds_cache, raw_gpu)
            cfg = {**cfg, "data": {**cfg["data"], "trainSize": ds.n_train, "testSize": ds.n_test}}
            skey, B, micro, precision = _session_key(cfg, ds)
            if _live.get(skey, 0) >= count or torch.cuda.mem_get_info()[0] < _PREP_MIN_FREE:
                return False
            _live[skey] = _live.get(skey, 0) + 1
        try:
            with _gpu_lock:
                sess = _new_session(init_model(cfg["model"], ds.D, cfg["initSeed"], device), cfg, ds, B, micro, precision, device)
            _waiting.yield_to_runs()
            with _gpu_lock:
                sess.replay = capture(sess.train_step, sess.state(), sess.opt, sess.ptr)
            _waiting.yield_to_runs()
            with _gpu_lock:
                sess.eval_replay = capture_eval(sess.eval_step)
                sess.stream = torch.cuda.current_stream()
                sess.stream.synchronize()
        except BaseException:
            with _gpu_lock:
                _live[skey] -= 1
            raise
        _put_session(skey, sess)
        return True


_preparer = _Preparer()


def prepare(want: list[tuple[dict, int]], raw, device, ds_cache: dict | None = None, raw_gpu=None):
    """Asks for `count` ready sessions per config, replacing any earlier request. Returns at once."""
    if device == "cuda":
        _preparer.request(want, (raw, device, ds_cache, raw_gpu))


def train(cfg: dict, raw, device, ds_cache: dict | None = None, target_update_s: float = 0.05, use_graphs: bool = True, raw_gpu=None) -> Iterator[dict]:
    """Yields engine-style ChunkUpdates: {step, totalSteps, train, evals, stepsPerSec, diverged}.

    Several runs may share a GPU from different threads, each on its own stream. Setup (allocation,
    init, graph capture) is not safe while another thread is capturing, so it happens under _gpu_lock;
    the training loop only uses graph replays, async copies to pinned memory and event waits, which
    are. Must be iterated from a single thread (the stream is thread-local). Concurrent runs are only
    bit-identical to solo runs with CUBLAS_WORKSPACE_CONFIG=:0:0 (see modal_app.py).
    """
    cuda = device == "cuda"
    # Generating the initial weights is ~15 ms of CPU; outside the lock, concurrent runs do it in parallel.
    fresh = init_model(cfg["model"], 3 * cfg["data"]["resolution"] ** 2, cfg["initSeed"], "cpu")
    with _waiting:
        _gpu_lock.acquire()
    try:
        if cuda:
            torch.cuda.set_stream(torch.cuda.Stream())
        ds = _dataset(cfg, raw, device, ds_cache, raw_gpu)
        cfg = {**cfg, "data": {**cfg["data"], "trainSize": ds.n_train, "testSize": ds.n_test}}
        skey, B, micro, precision = _session_key(cfg, ds)
        pure_bf16 = precision == "bf16"
        T = total_steps(cfg)
        sampler = Sampler(ds.n_train, cfg["dataSeed"], cfg["data"]["flip"])
        eval_steps = set(eval_schedule(T, cfg["numEvals"], bool(cfg.get("logEvals"))))
        evals_host = torch.zeros(2, len(eval_steps), 2, pin_memory=cuda)
        evals_dev = torch.zeros(2, len(eval_steps), 2, device=device)

        sess = _take_session(skey) if cuda and use_graphs else None
        if sess is None:
            sess = _new_session(fresh.to(device), cfg, ds, B, micro, precision, device)
            _live[skey] = _live.get(skey, 0) + 1
        else:
            sess.reset(fresh)
        sess.stream = torch.cuda.current_stream() if cuda else None
        if pure_bf16:
            bf16_round_(sess.opt.weights())
        model, k_max = sess.model, sess.k_max
        idx_buf, sched_buf, stats, ptr = sess.idx_buf, sess.sched_buf, sess.stats, sess.ptr
        idx_host, sched_host, stats_host = sess.idx_host, sess.sched_host, sess.stats_host
        train_step = sess.train_step
        # Captured before the first step: eager steps can differ from graph replays in the last bit,
        # and a run must not depend on whether its session was new or reused.
        if cuda and use_graphs and sess.replay is None:
            sess.capture()
        run_step = sess.replay or train_step
        run_eval = sess.eval_replay or sess.eval_step
        if cuda:
            torch.cuda.current_stream().synchronize()
    finally:
        _gpu_lock.release()

    sched = cfg["schedule"]
    base_lr = cfg["opt"]["lr"]

    def lr_at(step):
        return base_lr * lr_multiplier(sched, step / max(1, T - 1))

    def wait():
        if cuda:
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()

    # Two chunks are in flight: chunk n is queued on the GPU before chunk n-1's results are read back
    # and yielded, so frequent updates don't leave the GPU idle. Chunks alternate between two halves
    # (slots) of the input/stat buffers; the step reads row `ptr`, which starts at the slot offset.
    half = k_max // 2

    def launch(step, k, slot):
        off = slot * half
        idx_host[off : off + k].numpy()[:] = sampler.fill(k * B).reshape(k, B)
        lrs = [lr_at(step + j) for j in range(k)]
        sched_host[0, off : off + k] = torch.tensor(lrs, dtype=torch.float32)
        sched_host[1, off : off + k] = torch.arange(step + 1, step + k + 1, dtype=torch.float32)
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            ev0.record()
        idx_buf[off : off + k].copy_(idx_host[off : off + k], non_blocking=True)
        sched_buf[:, off : off + k].copy_(sched_host[:, off : off + k], non_blocking=True)
        ptr.fill_(off)
        ev_steps = []

        def evaluate(s):
            run_eval()
            evals_dev[slot, len(ev_steps)].copy_(sess.eval_out)
            ev_steps.append(s)

        if step == 0 and 0 in eval_steps:
            evaluate(0)
        for j in range(k):
            run_step()
            if step + j + 1 in eval_steps:
                evaluate(step + j + 1)
        stats_host[off : off + k].copy_(stats[off : off + k], non_blocking=True)
        if ev_steps:
            evals_host[slot, : len(ev_steps)].copy_(evals_dev[slot, : len(ev_steps)], non_blocking=True)
        ev1 = None
        if cuda:
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
        return {"step": step, "k": k, "slot": slot, "lrs": lrs, "ev_steps": ev_steps, "t0": t0, "ev0": ev0, "ev1": ev1}

    def collect(c):
        if cuda:
            c["ev1"].synchronize()
            dt = c["ev0"].elapsed_time(c["ev1"]) / 1000
        else:
            dt = time.perf_counter() - c["t0"]
        step, k, off = c["step"], c["k"], c["slot"] * half
        st = stats_host[off : off + k].numpy()
        evs = evals_host[c["slot"], : len(c["ev_steps"])].numpy()
        train_recs = [[step + j + 1, float(st[j, 0]), float(st[j, 1]), math.sqrt(max(0.0, float(st[j, 2]))) if np.isfinite(st[j, 2]) else float("nan"), c["lrs"][j]] for j in range(k)]
        eval_recs = [[s, float(e[0]), float(e[1])] for s, e in zip(c["ev_steps"], evs)]
        diverged = any(not math.isfinite(r[1]) for r in train_recs) or any(not math.isfinite(r[1]) for r in eval_recs)
        return {"step": step + k, "totalSteps": T, "train": train_recs, "evals": eval_recs, "stepsPerSec": k / max(dt, 1e-6), "diverged": diverged}, dt

    step = 0
    k_steps = 1  # the first point goes out as soon as possible
    pending = None
    slot = 0
    reusable = False
    try:
        while step < T or pending:
            nxt = None
            if step < T:
                k = min(k_steps, T - step, half)
                # Launching allocates (evals, eager steps), which must not overlap another thread's
                # capture. It only queues work, so the lock is held briefly.
                with _gpu_lock:
                    nxt = launch(step, k, slot)
                step += k
                slot ^= 1
            if pending:
                u, dt = collect(pending)
                yield u
                if u["diverged"]:
                    return
                k_steps = max(1, min(4096, round(pending["k"] * target_update_s / max(dt, 1e-6))))
            elif nxt and step < T:
                k_steps = 4  # second chunk is launched before the first is measured
            pending = nxt
        with _gpu_lock:
            stem = torch.empty_like(model.params["stem.W"], device="cpu", pin_memory=cuda)
            stem.copy_(model.params["stem.W"].detach(), non_blocking=True)
        wait()
        yield {"final": True, "stemW": stem.numpy()}
        reusable = True
    except GeneratorExit:
        reusable = True  # stopped early; the session is reset before its next use anyway
        raise
    finally:
        if reusable and cuda and use_graphs:
            sess.stream.synchronize()
            _put_session(skey, sess)
        else:
            with _gpu_lock:
                _live[skey] -= 1
