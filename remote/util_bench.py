"""How well concurrent runs fill one GPU: sweep throughput at several concurrency levels.

  cd remote && modal run util_bench.py
"""

import modal

from modal_app import GPU, gpu_image

app = modal.App("cifar-alchemy-utilbench")
image = gpu_image.add_local_python_source("modal_app")


def base_cfg(**over):
    import math

    sched = [min(1.0, i / 2) * 0.5 * (1 + math.cos(math.pi * i / 40)) for i in range(41)]
    cfg = {
        "model": dict(width=256, depth=3, mlpRatio=4, act="gelu", norm="layernorm", preNorm=True, postNorm=False, residual=True, init="kaiming", zeroInitLast=False),
        "data": dict(resolution=16, trainSize=10000, testSize=10000, flip=False),
        "opt": dict(kind="adam", lr=1e-3, momentum=0.9, beta1=0.9, beta2=0.99, wd=0, clip=0),
        "batchMode": "minibatch",
        "batchSize": 128,
        "budgetUnit": "epochs",
        "budget": 5,
        "schedule": sched,
        "initSeed": 0,
        "dataSeed": 42,
        "numEvals": 101,
    }
    for k, v in over.items():
        if k in ("width", "depth"):
            cfg["model"][k] = v
        elif k in ("resolution", "trainSize"):
            cfg["data"][k] = v
        else:
            cfg[k] = v
    return cfg


@app.function(image=image, gpu=GPU.split(","), cpu=4, max_containers=1, timeout=1800)
def measure(cases: list[tuple[str, dict]], ns: list[int]) -> list[str]:
    import threading
    import time

    import pynvml
    import torch

    import trainer
    from trainer import load_cifar_bins, train, upload_raw

    torch.backends.cuda.matmul.allow_tf32 = False
    raw = load_cifar_bins("/data/cifar10_train.bin", "/data/cifar10_test.bin")
    raw_gpu = upload_raw(raw, "cuda")
    cache = {}
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    out = [f"GPU: {torch.cuda.get_device_name()}"]

    held = [0.0]
    real = trainer._gpu_lock

    class Timed:
        def __enter__(self):
            real.acquire()
            self.t = time.perf_counter()

        def __exit__(self, *a):
            held[0] += time.perf_counter() - self.t
            real.release()

    trainer._gpu_lock = Timed()
    lr_of = lambda i, n: 10 ** (-4 + 2 * i / max(1, n - 1))

    def one(cfg, lr):
        c = {**cfg, "opt": {**cfg["opt"], "lr": lr}}
        tr, ev = [], []
        for u in train(c, raw, "cuda", cache, raw_gpu=raw_gpu):
            if not u.get("final"):
                tr += u["train"]
                ev += u["evals"]
        return tr, ev

    def sweep(cfg, n):
        res = [None] * n

        def run(i):
            res[i] = one(cfg, lr_of(i, n))

        samples = []
        done = threading.Event()

        def sample():
            while not done.is_set():
                samples.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                time.sleep(0.05)

        held[0] = 0.0
        sm = threading.Thread(target=sample)
        sm.start()
        t0 = time.perf_counter()
        ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        done.set()
        sm.join()
        steps = sum(len(r[0]) for r in res)
        util = sum(samples) / max(1, len(samples))
        return res, wall, steps, util, held[0]

    for name, cfg in cases:
        sweep(cfg, 2)  # warm caches, sessions, numba
        for n in ns:
            res, wall, steps, util, lk = sweep(cfg, n)
            out.append(f"{name:28s} n={n:2d}  wall={wall:6.2f}s  {steps / wall:6.0f} steps/s  {n / wall:5.2f} runs/s  nvml util={util:3.0f}%  lock held {100 * lk / wall:3.0f}%")
            if n == max(ns):
                same = all(one(cfg, lr_of(i, n)) == res[i] for i in range(n))
                out.append(f"  concurrent runs identical to solo runs: {same}")
    return out


@app.function(image=image, gpu=GPU.split(","), cpu=4, max_containers=1, timeout=1800)
def micro(cfg: dict) -> list[str]:
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    import trainer as tr

    torch.backends.cuda.matmul.allow_tf32 = False
    raw = tr.load_cifar_bins("/data/cifar10_train.bin", "/data/cifar10_test.bin")
    ds = tr._dataset(cfg, raw, "cuda", {}, tr.upload_raw(raw, "cuda"))
    out = [f"GPU: {torch.cuda.get_device_name()}"]
    torch.cuda.set_stream(torch.cuda.Stream())
    skey, B, micro_, prec = tr._session_key(cfg, ds)
    sess = tr._new_session(tr.init_model(cfg["model"], ds.D, 0, "cuda"), cfg, ds, B, micro_, prec, "cuda")
    sess.capture()
    torch.cuda.synchronize()

    def prof(fn, label):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        ev = [e for e in p.events() if e.device_type.name == "CUDA"]
        out.append(f"{label}: {len(ev) / 10:.0f} kernels, {sum(e.device_time for e in ev) / 10:.0f} us summed kernel time")
        agg = {}
        for e in ev:
            a = agg.setdefault(e.name[:80].replace("\n", " "), [0, 0.0])
            a[0] += 1
            a[1] += e.device_time / 10
        for name, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:8]:
            out.append(f"  {t:6.1f} us  x{c // 10:3d}  {name}")

    def timed(fn, n, label):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        out.append(f"{label}: {1e6 * (time.perf_counter() - t0) / n:.0f} us")

    sess.ptr.zero_()
    prof(sess.train_step, "eager train step")
    sess.ptr.zero_()

    def rep():
        if sess.ptr.item() >= sess.k_max - 1:
            sess.ptr.zero_()
        sess.replay()

    timed(lambda: [sess.replay() for _ in range(sess.k_max // 2)] and sess.ptr.zero_(), 4, f"{sess.k_max // 2} train replays")
    prof(sess.eval_step, "eager eval")
    timed(sess.eval_replay, 30, "graph eval")
    return out


@app.function(image=image, gpu=GPU.split(","), cpu=4, max_containers=1, timeout=1800)
def setup_cost(cfgs: list[dict]) -> list[str]:
    import time

    import torch

    import trainer as tr

    torch.backends.cuda.matmul.allow_tf32 = False
    raw = tr.load_cifar_bins("/data/cifar10_train.bin", "/data/cifar10_test.bin")
    raw_gpu = tr.upload_raw(raw, "cuda")
    tr.warm_kernels("cuda")
    cache = {}
    out = []
    for cfg in cfgs:
        torch.cuda.set_stream(torch.cuda.Stream())
        ds = tr._dataset(cfg, raw, "cuda", cache, raw_gpu)
        skey, B, micro_, prec = tr._session_key(cfg, ds)
        for rep in range(2):
            t0 = time.perf_counter()
            fresh = tr.init_model(cfg["model"], ds.D, 0, "cuda")
            t1 = time.perf_counter()
            sess = tr._new_session(fresh, cfg, ds, B, micro_, prec, "cuda")
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            sess.replay = tr.capture(sess.train_step, sess.state(), sess.opt, sess.ptr)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            sess.eval_replay = tr.capture_eval(sess.eval_step)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            out.append(f"B={B:5d} micro={len(micro_)} rep{rep}: init {1e3 * (t1 - t0):5.0f} ms  session {1e3 * (t2 - t1):5.0f} ms  train capture {1e3 * (t3 - t2):5.0f} ms  eval capture {1e3 * (t4 - t3):5.0f} ms")
        # Time to first update for a fresh run (no idle session).
        tr._sessions.clear()
        t0 = time.perf_counter()
        it = tr.train({**cfg, "initSeed": 1}, raw, "cuda", cache, raw_gpu=raw_gpu)
        next(it)
        out.append(f"          first update of an unprepared run: {1e3 * (time.perf_counter() - t0):.0f} ms")
        it.close()
    return out


@app.local_entrypoint()
def setup():
    for line in setup_cost.remote([base_cfg(batchSize=b) for b in (1024, 128, 16, 1)]):
        print(line)


@app.local_entrypoint()
def prof():
    for line in micro.remote(base_cfg()):
        print(line)


@app.local_entrypoint()
def main():
    cases = [
        ("default (d256 L3 r16 N10k)", base_cfg()),
        ("default, numEvals=11", base_cfg(numEvals=11)),
        ("bigger (d512 r32 N50k)", base_cfg(width=512, resolution=32, trainSize=50000, budget=2)),
    ]
    for line in measure.remote(cases, [1, 2, 4, 8, 16]):
        print(line)
