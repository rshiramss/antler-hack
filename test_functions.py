"""
Diverse test functions for Swarm A analysis and ranking.
Covers pure numeric hot loops, matrix ops, numpy, side effects,
recursion, randomness, strings, external deps, and simple getters.
"""
import math
import random
import time
import hashlib
import json

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# --- globals used by impure functions ---
_call_count = 0
_registry = {}


# ── 1. Mandelbrot grid — pure numeric, 3 nested loops ────────────────────────
def mandelbrot_grid(width, height, max_iter=64):
    result = []
    for py in range(height):
        row = []
        for px in range(width):
            c_re = (px - width / 2.0) * 3.5 / width
            c_im = (py - height / 2.0) * 2.0 / height
            z_re, z_im = 0.0, 0.0
            n = 0
            while n < max_iter:
                z_re2 = z_re * z_re - z_im * z_im + c_re
                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re2
                if z_re * z_re + z_im * z_im > 4.0:
                    break
                n += 1
            row.append(n)
        result.append(row)
    return result


# ── 2. Matrix multiply — pure Python, 3 nested loops ─────────────────────────
def matmul_pure(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


# ── 3. Dot product with L2 normalisation — numeric, single loop ───────────────
def dot_product_cosine(a, b):
    total = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    denom = (norm_a * norm_b) ** 0.5
    if denom > 0.0:
        return total / denom
    return 0.0


# ── 4. ReLU forward pass — 2 nested loops, pure ───────────────────────────────
def relu_forward(matrix):
    result = []
    for row in matrix:
        out_row = []
        for val in row:
            out_row.append(val if val > 0.0 else 0.0)
        result.append(out_row)
    return result


# ── 5. micrograd-style scalar accumulation loop ───────────────────────────────
def micrograd_forward(weights, inputs, bias=0.0):
    activation = bias
    for w, x in zip(weights, inputs):
        activation += w * x
    out = math.tanh(activation)
    return out


# ── 6. Softmax with numpy ─────────────────────────────────────────────────────
def softmax_numpy(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# ── 7. Fibonacci — recursive, deterministic ───────────────────────────────────
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ── 8. String normalisation and tokenisation ──────────────────────────────────
def tokenize_and_normalize(text):
    text = text.lower().strip()
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        tok = tok.strip(".,!?;:\"'()[]")
        if len(tok) > 1:
            cleaned.append(tok)
    return " ".join(cleaned)


# ── 9. Cryptographic hash — many external calls, no loops ────────────────────
def compute_hash_signature(data, secret):
    body = json.dumps(data, sort_keys=True)
    sig = hashlib.sha256(f"{body}{secret}".encode()).hexdigest()
    truncated = sig[:32]
    return {"signature": truncated, "algorithm": "sha256", "length": len(sig)}


# ── 10. Simple registry lookup — getter style ─────────────────────────────────
def lookup_registry(key, default=None):
    return _registry.get(key, default)


# ── 11. Side effects: print + global mutation ─────────────────────────────────
def noisy_multiply(a, b):
    global _call_count
    _call_count += 1
    result = a * b
    print(f"[call {_call_count}] {a} * {b} = {result}")
    return result


# ── 12. Stochastic perturbation — random + time calls ─────────────────────────
def stochastic_perturbation(values, noise_scale=0.01):
    t0 = time.time()
    result = [v + random.gauss(0.0, noise_scale) for v in values]
    elapsed = time.time() - t0
    return result, elapsed
