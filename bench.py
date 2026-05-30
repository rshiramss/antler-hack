import timeit
import numpy as np


def benchmark(rust_fn, oracle_fn, iterations=10, rounds=50):
    """Time rust_fn vs oracle_fn on a fixed input.

    Returns {"speedup": float, "python_ms": float, "rust_ms": float}.
    Speedup = median_python / median_rust.
    """
    fixed_input = np.array([0.5, -0.3], dtype=np.float64)

    rust_times = timeit.repeat(
        lambda: rust_fn(fixed_input),
        number=iterations,
        repeat=rounds,
    )
    python_times = timeit.repeat(
        lambda: oracle_fn(fixed_input),
        number=iterations,
        repeat=rounds,
    )

    rust_median = sorted(rust_times)[rounds // 2] / iterations
    python_median = sorted(python_times)[rounds // 2] / iterations

    return {
        "speedup": python_median / rust_median,
        "python_ms": python_median * 1000,
        "rust_ms": rust_median * 1000,
    }
