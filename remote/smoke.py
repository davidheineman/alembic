"""Local smoke test: python smoke.py [device]"""

import sys
import time

from trainer import load_cifar_bins, train

DATA = "../public/data/"


def default_cfg():
    sched = [min(1.0, i / 2) * 0.5 * (1 + __import__("math").cos(__import__("math").pi * i / 40)) for i in range(41)]
    return {
        "model": dict(width=256, depth=3, mlpRatio=4, act="gelu", norm="layernorm", preNorm=True, postNorm=False, residual=True, init="kaiming", zeroInitLast=False),
        "data": dict(resolution=16, trainSize=10000, testSize=2000, flip=False),
        "opt": dict(kind="adam", lr=1e-3, momentum=0.9, beta1=0.9, beta2=0.99, wd=0, clip=0),
        "batchMode": "minibatch",
        "batchSize": 128,
        "budgetUnit": "epochs",
        "budget": 2,
        "schedule": sched,
        "initSeed": 0,
        "dataSeed": 0,
        "numEvals": 5,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    raw = load_cifar_bins(DATA + "cifar10_train.bin", DATA + "cifar10_test.bin")
    for variant in [{}, {"norm": "batchnorm", "act": "swiglu"}, {"norm": "rmsnorm", "preNorm": False, "postNorm": True, "init": "uniform"}]:
        cfg = default_cfg()
        cfg["model"].update(variant)
        for kind in ["adam", "momentum"]:
            cfg["opt"]["kind"] = kind
            cfg["opt"]["wd"] = 1e-3 if kind == "momentum" else 0
            cfg["opt"]["clip"] = 1.0 if kind == "momentum" else 0
            cfg["opt"]["lr"] = 1e-3 if kind == "adam" else 3e-2
            t0 = time.time()
            first = last = None
            evals = []
            for u in train(cfg, raw, device):
                if u.get("final"):
                    continue
                if first is None:
                    first = u["train"][0]
                last = u["train"][-1]
                evals += u["evals"]
            print(variant, kind, f"{time.time() - t0:.1f}s", "first", [round(x, 4) for x in first[:4]], "last", [round(x, 4) for x in last[:4]], "evals", [(e[0], round(e[1], 3), round(e[2], 3)) for e in evals])
