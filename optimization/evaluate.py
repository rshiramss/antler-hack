# evaluate.py  —  FROZEN. The metric. The agent cannot touch this, so it can't game it.
#
# The candidate under evolution emits CODE ONLY. It never influences the inputs it is
# judged on and never authors expected outputs:
#   * correctness oracle  = the original Python `reference`, run at eval time;
#   * verification inputs  = generated HERE, with a fixed seed unseen by the mutation
#     model, at randomized shapes — a reviewer can confirm hardcoding is impossible.
#
# The champion is "the fastest port that regresses on NOTHING": a candidate must clear
# lexicographic gates (correct → no size regression → memory → stable) before its
# geometric-mean speedup is allowed to compete. Artifact-agnostic: `candidate_fn` may be
# today's Python solve(x) or tomorrow's imported Rust extension — the metric is unchanged.
import inspect
import os
import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from target import reference as _DEFAULT_REFERENCE, make_inputs


# ── Config (named constants; all env-overridable) ────────────────────────────────
def _envfloat(key, default):
    return float(os.environ.get(key, default))


def _envint(key, default):
    return int(os.environ.get(key, default))


RTOL, ATOL       = _envfloat("RF_RTOL", 1e-9), _envfloat("RF_ATOL", 1e-9)
N_FUZZ           = _envint("RF_N_FUZZ", 20)            # correctness fuzz inputs
FUZZ_MAX_LEN     = _envint("RF_FUZZ_MAX_LEN", 5000)
SIZES            = tuple(int(float(s)) for s in
                         os.environ.get("RF_SIZES", "100,10000,1000000").split(","))
BEST_OF          = _envint("RF_BEST_OF", 5)           # timing samples per measurement; take the MIN
REGRESSION_FLOOR = _envfloat("RF_REGRESSION_FLOOR", 0.85)
MEM_CEILING      = _envfloat("RF_MEM_CEILING", 1.5)
STABILITY_SAMPLES = _envint("RF_STABILITY_SAMPLES", 5)
STABILITY_COV    = _envfloat("RF_STABILITY_COV", 0.15)

# Harness input-generation seed. Fixed (deterministic) and unseen by the mutation model —
# this file is frozen and is never shown to the LLM, so a constant seed is both
# reproducible and unfakeable.
_GEN_SEED = 0xC0FFEE


# ── 1a. Input-schema inference (deterministic — no LLM) ──────────────────────────
@dataclass
class ArgSchema:
    name: str
    kind: str                 # "ndarray" | "scalar" | "sequence"
    dtype: str | None = None  # e.g. "float64" for ndarrays
    ndim: int | None = None
    scales: bool = False      # does this arg's length scale the workload?


@dataclass
class Schema:
    args: list = field(default_factory=list)

    @property
    def scaling_arg(self) -> "ArgSchema | None":
        return next((a for a in self.args if a.scales), None)


def infer_schema(reference, sample_inputs) -> Schema:
    """Read the signature + a representative sample to classify each argument.

    Picks the size-bearing argument deterministically (the largest array/sequence),
    so the sweep knows which input to grow. No LLM — signature + samples only.
    """
    params = list(inspect.signature(reference).parameters.values())
    args = []
    sizes = []
    for i, val in enumerate(sample_inputs):
        name = params[i].name if i < len(params) else f"arg{i}"
        if isinstance(val, np.ndarray):
            args.append(ArgSchema(name, "ndarray", str(val.dtype), val.ndim))
            sizes.append(val.size)
        elif isinstance(val, (list, tuple)):
            args.append(ArgSchema(name, "sequence"))
            sizes.append(len(val))
        else:
            args.append(ArgSchema(name, "scalar"))
            sizes.append(-1)
    if sizes:
        # The argument with the largest sample size carries the scaling axis.
        scale_idx = int(np.argmax(sizes))
        if sizes[scale_idx] > 0:
            args[scale_idx].scales = True
    return Schema(args)


def _make_args(schema: Schema, sample_inputs, size: int, rng) -> tuple:
    """Generate one fresh argument tuple at the given workload size (unseen seed)."""
    out = []
    for arg, sample in zip(schema.args, sample_inputs):
        if arg.kind == "ndarray":
            n = size if arg.scales else int(np.asarray(sample).size)
            dt = arg.dtype or "float64"
            out.append(rng.standard_normal(n).astype(dt))
        elif arg.kind == "sequence":
            n = size if arg.scales else len(sample)
            out.append(rng.standard_normal(n).tolist())
        else:  # scalar — reuse the representative value (don't scale)
            out.append(sample)
    return tuple(out)


# ── Timing + memory primitives ───────────────────────────────────────────────────
def _best_of(fn, args) -> float:
    """Minimum wall-time over BEST_OF runs. Min (not median): noise only ADDS time."""
    fn(*args)  # warm up
    best = float("inf")
    for _ in range(BEST_OF):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _peak_mem(fn, args) -> int:
    """Peak Python-tracked allocation (bytes) for one call, via tracemalloc.

    Note: tracemalloc tracks Python-level allocations; native (e.g. numpy C buffer)
    memory is only partially visible. Good enough to catch gratuitous Python-side
    blow-ups; swap for peak RSS if native accuracy is later required.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _geomean(values) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


# ── The metric ────────────────────────────────────────────────────────────────────
def evaluate(candidate_fn, reference=_DEFAULT_REFERENCE, sample_input=None) -> dict:
    """Score one candidate against the frozen oracle across a size sweep, gated.

    Returns a dict with the ratchet's compared number `speedup` (the gated geomean,
    0.0 if ineligible) plus a full research record: per-size speedups, geomean,
    mem_ratio, max_abs_error, stable, and a human-readable verdict.
    """
    rng = np.random.default_rng(_GEN_SEED)  # unseen by the mutation model

    # Representative sample → schema (drives all subsequent generation).
    if sample_input is None:
        sample_input = make_inputs(SIZES[0])
    sample_inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    schema = infer_schema(reference, sample_inputs)

    base = {
        "passed": False, "speedup": 0.0, "geomean_speedup": 0.0,
        "speedup_by_size": {}, "mem_ratio": 0.0, "max_abs_error": None,
        "stable": False, "verdict": "", "error": None,
    }

    # ── Gate 1: correctness (differential fuzz on unseen inputs) ──────────────────
    max_abs_error = 0.0
    for _ in range(N_FUZZ):
        n = int(rng.integers(1, FUZZ_MAX_LEN))
        args = _make_args(schema, sample_inputs, n, rng)
        try:
            got = np.asarray(candidate_fn(*args), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            return {**base, "error": repr(e), "verdict": f"crash on fuzz input -> {e!r}"}
        ref_out = np.asarray(reference(*args), dtype=np.float64)
        if got.shape != ref_out.shape or not np.allclose(got, ref_out, rtol=RTOL, atol=ATOL):
            diff = float(np.max(np.abs(got - ref_out))) if got.shape == ref_out.shape else float("inf")
            return {**base, "error": "differential mismatch",
                    "max_abs_error": diff,
                    "verdict": f"correctness FAIL (max abs error {diff:.2e}) -> discarded"}
        max_abs_error = max(max_abs_error, float(np.max(np.abs(got - ref_out))))

    # ── 1b. Multi-size regression sweep (best-of-N min, same input both sides) ────
    speedup_by_size = {}
    for size in SIZES:
        args = _make_args(schema, sample_inputs, size, rng)
        try:
            ref_t = _best_of(reference, args)
            cand_t = _best_of(candidate_fn, args)
        except Exception as e:  # noqa: BLE001
            return {**base, "max_abs_error": max_abs_error, "error": repr(e),
                    "verdict": f"crash during benchmark @ size {size:.0e} -> {e!r}"}
        speedup_by_size[size] = (ref_t / cand_t) if cand_t > 0 else 0.0

    geomean = _geomean(speedup_by_size.values())

    # ── 1c. Peak memory at the largest size ───────────────────────────────────────
    largest = max(SIZES)
    mem_args = _make_args(schema, sample_inputs, largest, rng)
    ref_peak = _peak_mem(reference, mem_args) or 1
    cand_peak = _peak_mem(candidate_fn, mem_args)
    mem_ratio = cand_peak / ref_peak

    # ── 1d. Stability: spread of independent best-of-N samples at the mid size ────
    def _measure_stability() -> bool:
        mid = SIZES[len(SIZES) // 2]
        samples = []
        for _ in range(STABILITY_SAMPLES):
            args = _make_args(schema, sample_inputs, mid, rng)
            samples.append(_best_of(candidate_fn, args))
        mean = float(np.mean(samples))
        cov = (float(np.std(samples)) / mean) if mean > 0 else 0.0
        return cov <= STABILITY_COV

    stable = _measure_stability()
    if not stable:
        stable = _measure_stability()  # re-run once, then distrust

    record = {
        "passed": True,
        "geomean_speedup": geomean,
        "speedup_by_size": speedup_by_size,
        "mem_ratio": mem_ratio,
        "max_abs_error": max_abs_error,
        "stable": stable,
        "error": None,
    }

    # ── 1e. Gated (lexicographic) eligibility, then 1f. headline = gated geomean ──
    worst_size = min(speedup_by_size, key=speedup_by_size.get)
    worst = speedup_by_size[worst_size]
    if worst < REGRESSION_FLOOR:
        verdict = (f"geomean {geomean:.2f}x but {worst:.2f}x@{worst_size:.0e} "
                   f"< floor {REGRESSION_FLOOR} -> SIZE REGRESSION -> discarded")
        return {**record, "speedup": 0.0, "verdict": verdict}
    if mem_ratio > MEM_CEILING:
        verdict = (f"geomean {geomean:.2f}x but mem {mem_ratio:.2f}x "
                   f"> ceiling {MEM_CEILING} -> MEMORY -> discarded")
        return {**record, "speedup": 0.0, "verdict": verdict}
    if not stable:
        verdict = f"geomean {geomean:.2f}x but timing unstable -> distrusted -> discarded"
        return {**record, "speedup": 0.0, "verdict": verdict}

    sizes_str = ", ".join(f"{v:.2f}x@{s:.0e}" for s, v in speedup_by_size.items())
    verdict = f"geomean {geomean:.2f}x ({sizes_str}), mem {mem_ratio:.2f}x, stable -> kept"
    return {**record, "speedup": geomean, "verdict": verdict}
