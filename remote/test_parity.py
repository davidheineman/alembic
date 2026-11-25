"""Checks the Python RNG / init / sampler against reference values produced by the TS sources.

Usage (from the repo root):
  npx esbuild remote/parity_ref.ts --bundle --platform=node --outfile=/tmp/parity_ref.js
  node /tmp/parity_ref.js > /tmp/ts.json && (cd remote && uv run --with torch --with numpy --with numba python test_parity.py /tmp/ts.json)
"""

import json
import sys

import numpy as np
import torch

from rng import Sampler, normals
from trainer import init_model

ref = json.load(open(sys.argv[1]))
for key in [k for k in ref if k.startswith("n_")]:
    _, seed, stream = key.split("_")
    got = normals(int(seed), int(stream), 7).astype(np.float32)
    assert np.array_equal(got, np.array(ref[key], dtype=np.float32)), (key, got, ref[key])

assert Sampler(10, 0, False).fill(25).tolist() == ref["s1"]
assert Sampler(1000, 5, True).fill(2500)[-20:].tolist() == ref["s2"]

for entry in ref["models"]:
    m = entry["m"]
    model = init_model(m, 192, 4, "cpu")
    names = [s.name for s in model.specs()]
    assert names == [s[0] for s in entry["specs"]], (names, entry["specs"])
    for name, size, head, total in entry["specs"]:
        if name.endswith("Norm") or name.endswith(".norm"):
            g = model.params[name + ".g"]
            t = torch.cat([g, model.params.get(name + ".b", torch.zeros_like(g))])
        else:
            t = model.params[name]
        flat = t.detach().flatten().numpy()
        assert flat.size == size, (name, flat.size, size)
        assert np.array_equal(flat[:5], np.array(head, dtype=np.float32)), (name, flat[:5], head)
        assert abs(float(flat.astype(np.float64).sum()) - total) < 1e-4, (name, flat.sum(), total)
print("parity OK")
