"""Bit-exact port of src/rng.ts (sfc32 seeded via splitmix32) and the engine's Sampler.

Matching the browser's RNG means a given (initSeed, dataSeed) produces the same initial
weights and the same example order locally and remotely.
"""

import math

import numpy as np
from numba import njit, prange

M32 = 0xFFFFFFFF


@njit(cache=True)
def _new_state(seed, stream):
    a = ((seed * 0x2545F491) & M32) ^ ((stream * 0x6C8E9CF5) & M32)
    st = np.zeros(4, dtype=np.int64)
    for i in range(4):
        a = (a + 0x9E3779B9) & M32
        t = a ^ (a >> 16)
        t = (t * 0x21F0AAAD) & M32
        t = t ^ (t >> 15)
        t = (t * 0x735A2D97) & M32
        st[i] = (t ^ (t >> 15)) & M32
    for _ in range(12):
        _u32(st)
    return st


@njit(cache=True)
def _u32(st):
    a, b, c, d = st[0], st[1], st[2], st[3]
    t = (a + b + d) & M32
    d = (d + 1) & M32
    a = b ^ (b >> 9)
    b = (c + (c << 3)) & M32
    c = ((c << 21) | (c >> 11)) & M32
    c = (c + t) & M32
    st[0], st[1], st[2], st[3] = a, b, c, d
    return t


@njit(cache=True)
def _uniform(st):
    return (_u32(st) + 0.5) / 4294967296.0


@njit(cache=True)
def _normals(st, n):
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        u = _uniform(st)
        v = _uniform(st)
        r = math.sqrt(-2.0 * math.log(u))
        out[i] = r * math.cos(2.0 * math.pi * v)
        if i + 1 < n:
            out[i + 1] = r * math.sin(2.0 * math.pi * v)
        i += 2
    return out


@njit(cache=True)
def _uniform_stream(st, n):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _uniform(st)
    return out


@njit(parallel=True, cache=True)
def _box_muller(uv, n):
    # Same per-pair arithmetic as _normals; only the (independent) pairs are spread across cores.
    out = np.empty(n, dtype=np.float64)
    for p in prange((n + 1) // 2):
        i = 2 * p
        r = math.sqrt(-2.0 * math.log(uv[i]))
        out[i] = r * math.cos(2.0 * math.pi * uv[i + 1])
        if i + 1 < n:
            out[i + 1] = r * math.sin(2.0 * math.pi * uv[i + 1])
    return out


def normals(seed: int, stream: int, n: int) -> np.ndarray:
    """The first n values of Rng(seed, stream).normal() (float64, like JS)."""
    st = _new_state(seed, stream)
    if n < 1 << 16:
        return _normals(st, n)
    return _box_muller(_uniform_stream(st, 2 * ((n + 1) // 2)), n)


@njit(cache=True)
def _uniforms(st, n):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _uniform(st)
    return out


def uniforms(seed: int, stream: int, n: int) -> np.ndarray:
    """The first n values of Rng(seed, stream).uniform()."""
    return _uniforms(_new_state(seed, stream), n)


@njit(cache=True)
def _fill(st, perm, pos, flip, count, out):
    n = perm.shape[0]
    for i in range(count):
        if pos >= n:
            for j in range(n - 1, 0, -1):
                k = int(math.floor(_uniform(st) * (j + 1)))
                tmp = perm[j]
                perm[j] = perm[k]
                perm[k] = tmp
            pos = 0
        v = perm[pos]
        pos += 1
        if flip and (_u32(st) & 1):
            v += n
        out[i] = v
    return pos


class Sampler:
    """Same stream as the engine's Sampler. Flipped examples are encoded as index + n."""

    def __init__(self, n: int, seed: int, flip: bool):
        self.st = _new_state(seed, 2)
        self.perm = np.arange(n, dtype=np.int64)
        self.pos = n
        self.flip = flip

    def fill(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.int64)
        self.pos = _fill(self.st, self.perm, self.pos, self.flip, count, out)
        return out
