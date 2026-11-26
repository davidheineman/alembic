"""Remote GPU backend on Modal.

  modal deploy remote/modal_app.py          # deploy the web endpoint + GPU workers
  modal run remote/modal_app.py::bench      # time a parallel LR sweep from the CLI

The browser POSTs one TrainConfig per run to /train and reads newline-delimited JSON
updates in the same shape as the WebGPU engine's ChunkUpdate. Everything runs on a single
GPU container (MAX_GPUS = 1), RUNS_PER_GPU runs at a time.

Warm GPU: while "Remote GPUs" is selected the browser POSTs /warm every 30 s, which pins
min_containers=1 so runs start instantly. The pin is released when the browser switches back
or closes, and independently by the `reaper` cron (every minute) when heartbeats stop for
HEARTBEAT_TIMEOUT_S or no run has started for NO_RUNS_TIMEOUT_S. Without the pin, the GPU
container exits IDLE_GPU_SHUTDOWN_S after its last run.
"""

import os
import time
from pathlib import Path

import modal

GPU = os.environ.get("CIFAR_GPU", "L4,A10G,T4,L40S")  # run_cases only; "none" = CPU containers
# GPU types the browser can pick from; one worker class each (see the bottom of the Worker section).
GPU_TYPES = ["T4", "L4", "A10G", "L40S", "A100", "H100"]
DEFAULT_GPU = "A10G"
# Hard cap: never more than one GPU container per type, and only the selected type is kept warm;
# switching types shuts the others down. Runs beyond RUNS_PER_GPU wait in Modal's queue.
MAX_GPUS = 1
RUNS_PER_GPU = 8
IDLE_GPU_SHUTDOWN_S = 20  # an unpinned GPU container exits this long after its last run
HEARTBEAT_TIMEOUT_S = 90  # warm GPU is shut down if the browser stops checking in...
NO_RUNS_TIMEOUT_S = 120  # ...or if nothing has trained for this long, even with the tab open
MAX_WARMUP_S = 600  # a GPU type that never comes up (e.g. none available) is given up on after this long
HERE = Path(__file__).parent
DATA = HERE.parent / "public" / "data"

app = modal.App("cifar-alchemy")
# "pool": {"want": 0|1, "hb": last browser heartbeat, ...}; "w:<task id>": worker liveness and active runs;
# "shutdown": time of the last shutdown request; "cancel:<run id>": runs the browser abandoned.
state = modal.Dict.from_name("cifar-alchemy-state", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "numpy", "numba", "fastapi[standard]", "nvidia-ml-py")
    # Capturing a CUDA graph frees every cuBLAS workspace, including ones baked into graphs other
    # runs are still replaying; with no workspace, concurrent runs stay bit-identical to solo runs.
    .env({"NUMBA_CACHE_DIR": "/tmp/numba", "CUBLAS_WORKSPACE_CONFIG": ":0:0"})
    .add_local_file(DATA / "cifar10_train.bin", "/data/cifar10_train.bin", copy=True)
    .add_local_file(DATA / "cifar10_test.bin", "/data/cifar10_test.bin", copy=True)
    .add_local_python_source("rng", "trainer")
)

web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("fastapi[standard]")


def clean(x):
    # JSON has no NaN/Infinity; the browser treats null as non-finite.
    import math

    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


auth_secret = modal.Secret.from_name("cifar-remote-auth")
# Modal's proxy zstd-compresses responses for browsers, which holds back the whole stream until the
# run ends. Declaring the body already encoded makes it forward each NDJSON line as it is written.
NO_BUFFERING = {"Content-Encoding": "identity", "Cache-Control": "no-cache, no-transform"}


class WorkerBase:
    gpu_type = DEFAULT_GPU

    @modal.enter()
    def load(self):
        import threading

        import modal.experimental
        import torch

        from rng import Sampler, normals
        from trainer import load_cifar_bins, upload_raw, warm_kernels

        self.raw = load_cifar_bins("/data/cifar10_train.bin", "/data/cifar10_test.bin")
        self.cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.raw_gpu = upload_raw(self.raw, self.device) if self.device == "cuda" else None
        threading.Thread(target=warm_kernels, args=(self.device,), daemon=True).start()
        self.gpu_name = torch.cuda.get_device_name() if self.device == "cuda" else "CPU"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        Sampler(10, 0, True).fill(20)
        normals(0, 1, 4)

        self.key = f"w:{os.environ.get('MODAL_TASK_ID', id(self))}"
        self.alive = True
        self.active = 0
        self.started = time.time()
        self.util = 0

        def watch_util():
            # nvidia-smi's "GPU util": the share of time any kernel was running, not how much of the
            # GPU those kernels used.
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while self.alive:
                self.util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                time.sleep(0.5)

        if self.device == "cuda":
            threading.Thread(target=watch_util, daemon=True).start()

        def beat():
            n = 0
            while self.alive:
                if n % 10 == 0:
                    state[self.key] = {"t": time.time(), "gpu": self.gpu_name, "type": self.gpu_type, "started": self.started, "active": self.active}
                if (state.get("shutdown") or 0) > self.started:
                    modal.experimental.stop_fetching_inputs()
                    self.alive = False
                    state.pop(self.key, None)
                    break
                n += 1
                time.sleep(1)

        threading.Thread(target=beat, daemon=True).start()

    @modal.exit()
    def leave(self):
        self.alive = False
        try:
            state.pop(self.key)
        except KeyError:
            pass

    def run(self, cfg: dict, stop=None):
        """Training updates for one run; ends early when stopped, cancelled via /cancel, or shutting down."""
        import base64
        import threading

        from trainer import train

        def note_activity():
            threading.Thread(target=state.put, args=("lastRun", time.time()), daemon=True).start()

        cancel_key = f"cancel:{cfg.get('runId')}"
        cancelled = threading.Event()

        def check_cancelled():
            # A modal.Dict lookup is a network round trip, so it stays off the path to the first update.
            if state.get(cancel_key):
                cancelled.set()

        if not self.alive:
            # This container is shutting down but still got the request; the browser resends it.
            yield {"type": "retry"}
            return
        threading.Thread(target=check_cancelled, daemon=True).start()
        self.active += 1
        note_activity()
        try:
            yield {"type": "start", "gpu": self.gpu_name}
            checked = time.time()
            for u in train(cfg, self.raw, self.device, self.cache, raw_gpu=self.raw_gpu):
                if u.get("final"):
                    w = u["stemW"]
                    yield {"type": "final", "stemW": base64.b64encode(w.tobytes()).decode()}
                else:
                    yield {"type": "update", **u, "util": self.util}
                if cancelled.is_set() or (stop is not None and stop.is_set()):
                    return
                if time.time() - checked > 0.5:
                    checked = time.time()
                    threading.Thread(target=check_cancelled, daemon=True).start()
                if not self.alive:
                    return
        finally:
            self.active -= 1
            note_activity()

    @modal.method()
    def train(self, cfg: dict):
        yield from self.run(cfg)

    @modal.asgi_app()
    def web(self):
        """POST /train streams NDJSON updates straight from this GPU container (no hop through `api`)."""
        import asyncio
        import json
        import threading

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse

        web = FastAPI()
        web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        token = os.environ["AUTH_TOKEN"]

        @web.get("/ping")
        async def ping():
            return {"ok": True}

        @web.post("/prepare")
        async def prepare(req: Request):
            """{"configs": [{"cfg": TrainConfig, "count": n}, ...]}: the runs the browser is about to
            start (sidebar config, first runs of the sweep). Sets up `count` sessions per config in the
            background so those runs skip setup. Same text/plain + token-in-body auth as /train."""
            from trainer import prepare as prep

            body = json.loads(await req.body())
            if body.get("token") != token:
                raise HTTPException(401, "bad token")
            prep([(c["cfg"], int(c.get("count", 1))) for c in body["configs"]], self.raw, self.device, self.cache, raw_gpu=self.raw_gpu)
            return {"ok": True}

        @web.post("/train")
        async def train(req: Request):
            cfg = json.loads(await req.body())
            # The browser sends the token in the body as text/plain, which makes this a CORS "simple
            # request": no preflight round trip before each run.
            if req.headers.get("authorization") != f"Bearer {token}" and cfg.pop("token", None) != token:
                raise HTTPException(401, "bad token")
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue()
            stop = threading.Event()

            def work():
                # One thread for the whole run: the trainer's CUDA stream is thread-local.
                try:
                    for u in self.run(cfg, stop):
                        loop.call_soon_threadsafe(q.put_nowait, u)
                except Exception as e:  # noqa: BLE001
                    loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "error": f"{type(e).__name__}: {e}"})
                loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=work, daemon=True).start()

            async def stream():
                try:
                    while True:
                        try:
                            u = await asyncio.wait_for(q.get(), timeout=2)
                        except asyncio.TimeoutError:
                            yield json.dumps({"type": "ping"}) + "\n"
                            continue
                        if u is None:
                            break
                        yield json.dumps(clean(u)) + "\n"
                finally:
                    stop.set()

            return StreamingResponse(stream(), media_type="application/x-ndjson", headers=NO_BUFFERING)

        return web


def worker_cls(gpu: str):
    return app.cls(image=gpu_image, gpu=gpu, cpu=4, max_containers=MAX_GPUS, scaledown_window=IDLE_GPU_SHUTDOWN_S, timeout=3600, secrets=[auth_secret])


# Modal needs each class at module scope under its own name. The A10G one keeps the name `Worker`
# so its web URL (VITE_REMOTE_TRAIN_URL) is unchanged; the others are at ...-worker<type>-web...
@worker_cls("A10G")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class Worker(WorkerBase):
    gpu_type = "A10G"


@worker_cls("T4")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class WorkerT4(WorkerBase):
    gpu_type = "T4"


@worker_cls("L4")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class WorkerL4(WorkerBase):
    gpu_type = "L4"


@worker_cls("L40S")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class WorkerL40S(WorkerBase):
    gpu_type = "L40S"


@worker_cls("A100")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class WorkerA100(WorkerBase):
    gpu_type = "A100"


@worker_cls("H100")
@modal.concurrent(max_inputs=RUNS_PER_GPU)
class WorkerH100(WorkerBase):
    gpu_type = "H100"


WORKERS = {"T4": WorkerT4, "L4": WorkerL4, "A10G": Worker, "L40S": WorkerL40S, "A100": WorkerA100, "H100": WorkerH100}
assert list(WORKERS) == GPU_TYPES


def pick_gpu(name) -> str:
    return name if name in WORKERS else DEFAULT_GPU


@app.function(image=web_image, secrets=[auth_secret], timeout=3600, scaledown_window=300)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def api():
    import asyncio
    import json
    import math

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    web = FastAPI()
    web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    token = os.environ["AUTH_TOKEN"]

    def auth(req: Request):
        if req.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "bad token")

    @web.get("/health")
    async def health():
        return {"ok": True, "gpu": GPU, "maxGpus": MAX_GPUS}

    @web.post("/warm")
    async def warm(req: Request):
        """Heartbeat + desired pool size. Call every ~30 s while the remote backend is selected."""
        auth(req)
        body = await req.json()
        want = 1 if int(body.get("gpus", 0)) > 0 else 0
        pool = await asyncio.to_thread(state.get, "pool", {}) or {}
        now = time.time()
        current = pick_gpu(pool.get("gpu"))
        gpu = pick_gpu(body.get("gpu")) if want else current
        if want and pool.get("released") and not body.get("start"):
            # The reaper released the GPU for inactivity; only an explicit user action re-warms it.
            return {**(await asyncio.to_thread(pool_status, 0)), "released": True, "gpu": gpu}
        if want and pool.get("want") and gpu != current:
            # Switching GPU type: stop the old container (and its runs) before pinning the new one.
            await asyncio.to_thread(shut_down)
            pool["since"] = now
        if pool.get("want") != want:
            pool["since"] = now
        if want or pool.get("want"):
            # Re-applied on every heartbeat: a redeploy resets min_containers to the decorator's value.
            await asyncio.to_thread(set_pool, want, gpu)
        if not want and not body.get("leaving"):
            # Leaving (page closed or reloading) only unpins: the container exits IDLE_GPU_SHUTDOWN_S
            # after its last run, and a reloaded page gets it back warm.
            await asyncio.to_thread(shut_down)
        await asyncio.to_thread(state.put, "pool", {"want": want, "hb": now, "since": pool.get("since", now), "gpu": gpu})
        return {**(await asyncio.to_thread(pool_status, want)), "gpu": gpu}

    @web.get("/status")
    async def status(req: Request):
        auth(req)
        pool = await asyncio.to_thread(state.get, "pool", {}) or {}
        return await asyncio.to_thread(pool_status, pool.get("want", 0))

    @web.post("/cancel")
    async def cancel(req: Request):
        """Drops the given runs, whether they are training or still queued for the GPU."""
        auth(req)
        now = time.time()
        await asyncio.to_thread(state.update, {f"cancel:{i}": now for i in (await req.json()).get("runIds", [])})
        return {"ok": True}

    @web.post("/train")
    async def train(req: Request):
        cfg = json.loads(await req.body())
        if req.headers.get("authorization") != f"Bearer {token}" and cfg.pop("token", None) != token:
            raise HTTPException(401, "bad token")
        run_id = cfg.get("runId")
        worker = WORKERS[pick_gpu(cfg.pop("gpu", None))]
        await asyncio.to_thread(state.put, "lastRun", time.time())

        async def stream():
            q: asyncio.Queue = asyncio.Queue()

            async def pump():
                try:
                    async for u in worker().train.remote_gen.aio(cfg):
                        await q.put(u)
                except Exception as e:  # noqa: BLE001
                    await q.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                await q.put(None)

            task = asyncio.create_task(pump())
            finished = False
            yield json.dumps({"type": "queued"}) + "\n"
            try:
                while True:
                    try:
                        u = await asyncio.wait_for(q.get(), timeout=2)
                    except asyncio.TimeoutError:
                        yield json.dumps({"type": "ping"}) + "\n"
                        continue
                    if u is None:
                        finished = True
                        break
                    yield json.dumps(clean(u)) + "\n"
            finally:
                task.cancel()
                await asyncio.to_thread(state.put, "lastRun", time.time())
                if not finished and run_id:
                    # Browser went away or stopped the run: make the worker drop it, even if still queued.
                    await asyncio.to_thread(state.put, f"cancel:{run_id}", time.time())

        return StreamingResponse(stream(), media_type="application/x-ndjson", headers=NO_BUFFERING)

    return web


def set_pool(n: int, gpu: str):
    WORKERS[gpu]().update_autoscaler(min_containers=n)


def shut_down():
    """Unpins every GPU type and tells running workers to exit now instead of after the idle window."""
    for g in GPU_TYPES:
        set_pool(0, g)
    state["shutdown"] = time.time()


def pool_status(want: int) -> dict:
    now = time.time()
    workers = [v for k, v in state.items() if k.startswith("w:") and now - v["t"] < 30]
    return {"want": want, "ready": len(workers), "active": sum(w.get("active", 0) for w in workers), "gpus": sorted({w["gpu"] for w in workers}), "runsPerGpu": RUNS_PER_GPU}


@app.function(image=web_image, schedule=modal.Period(minutes=1))
def reaper():
    """Releases the warm GPU when the browser stops checking in or no runs are started, and prunes dead worker entries."""
    now = time.time()
    pool = state.get("pool", {}) or {}
    if pool.get("want"):
        silent = now - pool.get("hb", 0)
        live = [v for k, v in state.items() if k.startswith("w:") and now - v["t"] < 30]
        active = sum(v.get("active", 0) for v in live)
        up = [v.get("started", 0) for v in live if v.get("type", DEFAULT_GPU) == pick_gpu(pool.get("gpu"))]
        if not up:
            # Still cold-starting: that isn't idleness, but give up if the GPU never arrives.
            idle = 0 if now - pool.get("since", 0) < MAX_WARMUP_S else float("inf")
        else:
            idle = 0 if active else now - max(state.get("lastRun", 0) or 0, pool.get("since", 0), min(up))
        if silent > HEARTBEAT_TIMEOUT_S or idle > NO_RUNS_TIMEOUT_S:
            state["pool"] = {"want": 0, "hb": pool.get("hb", 0), "since": now, "released": True}
            shut_down()
            print(f"shut down warm GPU (no heartbeat for {silent:.0f}s, idle for {idle:.0f}s)")
    # Re-asserted every minute in case an earlier release call failed: only the selected type may be pinned.
    pool = state.get("pool", {}) or {}
    for g in GPU_TYPES:
        if not pool.get("want") or g != pick_gpu(pool.get("gpu")):
            set_pool(0, g)
    for k, v in list(state.items()):
        if (k.startswith("w:") and now - v["t"] > 120) or (k.startswith("cancel:") and now - v > 3600):
            state.pop(k)


@app.function(image=gpu_image, gpu=None if GPU == "none" else GPU.split(","), max_containers=1, scaledown_window=2, timeout=600)
def run_cases(cases: dict, use_graphs: bool, gpu_ds: bool = True) -> dict:
    from trainer import load_cifar_bins, train, upload_raw

    raw = load_cifar_bins("/data/cifar10_train.bin", "/data/cifar10_test.bin")
    raw_gpu = upload_raw(raw, "cuda") if gpu_ds else None
    out = {}
    for name, cfg in cases.items():
        tr, ev = [], []
        for u in train(cfg, raw, "cuda", {}, use_graphs=use_graphs, raw_gpu=raw_gpu):
            if not u.get("final"):
                tr += u["train"]
                ev += u["evals"]
        out[name] = {"train": tr, "evals": ev}
    return out


@app.local_entrypoint()
def parity(ref: str = "/tmp/webgpu.json", gpu_ds: bool = True):
    """Runs remote/parity_cases.json on a GPU (eager and CUDA-graph) and compares with WebGPU output from webgpu_ref.html."""
    import json

    cases = json.load(open(HERE / "parity_cases.json"))
    want = json.load(open(ref))
    for graphs in (False, True):
        got = run_cases.remote(cases, graphs, gpu_ds)
        for name in cases:
            a, b = got[name]["train"], want[name]["train"]
            rel = max(abs(x[1] - y["loss"]) / max(1e-6, abs(y["loss"])) for x, y in zip(a, b))
            print(f"g={int(graphs)} err={rel:.1e} test={got[name]['evals'][-1][1]:.4f}/{want[name]['evals'][-1]['loss']:.4f} {name[:20]}")


@app.local_entrypoint()
def bench(n: int = 16, epochs: float = 5, train_size: int = 50000, res: int = 32, width: int = 256, depth: int = 3, batch: int = 128):
    """Runs an n-point LR sweep in parallel and reports wall-clock time."""
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor

    sched = [min(1.0, i / 2) * 0.5 * (1 + math.cos(math.pi * i / 40)) for i in range(41)]
    base = {
        "model": dict(width=width, depth=depth, mlpRatio=4, act="gelu", norm="layernorm", preNorm=True, postNorm=False, residual=True, init="kaiming", zeroInitLast=False),
        "data": dict(resolution=res, trainSize=train_size, testSize=10000, flip=True),
        "opt": dict(kind="adam", lr=1e-3, momentum=0.9, beta1=0.9, beta2=0.99, wd=0, clip=0),
        "batchMode": "minibatch",
        "batchSize": batch,
        "budgetUnit": "epochs",
        "budget": epochs,
        "schedule": sched,
        "initSeed": 0,
        "dataSeed": 0,
        "numEvals": 25,
    }
    lrs = [10 ** (-4.5 + 2.5 * i / max(1, n - 1)) for i in range(n)]
    t0 = time.time()

    def one(lr):
        cfg = {**base, "opt": {**base["opt"], "lr": lr}}
        first = None
        last = None
        sps = []
        gpu = None
        for u in Worker().train.remote_gen(cfg):
            if u["type"] == "start":
                first = time.time() - t0
                gpu = u["gpu"]
            elif u["type"] == "update":
                last = u
                sps.append(u["stepsPerSec"])
        ev = last["evals"][-1] if last["evals"] else None
        return lr, gpu, first, time.time() - t0, last["totalSteps"], sorted(sps)[len(sps) // 2], ev

    with ThreadPoolExecutor(n) as ex:
        results = list(ex.map(one, lrs))
    for lr, gpu, first, done, T, sps, ev in results:
        print(f"lr={lr:.2e} {gpu} start={first:.1f}s done={done:.1f}s steps={T} median {sps:.0f} steps/s  test loss={ev[1]:.3f} acc={ev[2]:.3f}")
    print(f"{n} runs wall clock: {time.time() - t0:.1f}s")
