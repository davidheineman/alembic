"""Compares PyTorch training curves to WebGPU reference output from webgpu_ref.html.

  python compare.py webgpu.json [device]
"""

import json
import sys

import numpy as np

from trainer import load_cifar_bins, train

ref = json.load(open(sys.argv[1]))
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
cases = json.load(open("parity_cases.json"))
raw = load_cifar_bins("../public/data/cifar10_train.bin", "../public/data/cifar10_test.bin")

for name, cfg in cases.items():
    tr, ev = [], []
    for u in train(cfg, raw, device):
        if not u.get("final"):
            tr += u["train"]
            ev += u["evals"]
    r = ref[name]
    a = np.array([[t[1], t[2], t[3]] for t in tr])
    b = np.array([[t["loss"], t["acc"], t["gnorm"]] for t in r["train"]])
    n = min(len(a), len(b))
    rel = np.abs(a[:n] - b[:n]) / np.maximum(1e-6, np.abs(b[:n]))
    first_loss = (a[0, 0], b[0, 0])
    ea = np.array([e[1] for e in ev])
    eb = np.array([e["loss"] for e in r["evals"]])
    print(
        f"{name:38s} steps {len(a)}/{len(b)}  step1 loss {first_loss[0]:.6f} vs {first_loss[1]:.6f}  "
        f"max rel err: loss {rel[:, 0].max():.1e} gnorm {rel[:, 2].max():.1e}  final loss {a[-1, 0]:.4f} vs {b[-1, 0]:.4f}  "
        f"final test loss {ea[-1]:.4f} vs {eb[-1]:.4f}"
    )
