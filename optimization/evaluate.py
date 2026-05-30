# evaluate.py  —  FROZEN. The metric. The agent cannot touch this, so it can't game it.
import time
import numpy as np
from target import reference, make_inputs

RTOL, ATOL = 1e-9, 1e-9      # differential-correctness tolerance
N_FUZZ      = 20             # random input batches the candidate never saw at write time
N_BENCH     = 5             # benchmark repeats; take the median


def evaluate(candidate_fn) -> dict:
    """Score one candidate. speedup is gated to 0.0 on any correctness failure."""
    rng = np.random.default_rng(12345)   # NOT the target seed — unseen inputs

    # 1) Correctness gate: differential test on fresh random inputs.
    for _ in range(N_FUZZ):
        x = rng.standard_normal(rng.integers(1, 5000))
        try:
            got = candidate_fn(x)
        except Exception as e:
            return {"passed": False, "speedup": 0.0, "error": repr(e)}
        if not np.allclose(got, reference(x), rtol=RTOL, atol=ATOL):
            return {"passed": False, "speedup": 0.0, "error": "differential mismatch"}

    # 2) Benchmark: median wall-time of candidate vs reference on the same input.
    x = make_inputs()
    ref_t = _median_time(reference, x)
    cand_t = _median_time(candidate_fn, x)
    return {"passed": True, "speedup": ref_t / cand_t, "error": ""}


def _median_time(fn, x) -> float:
    fn(x)                                 # warm up
    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter(); fn(x); times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]
