import timeit
import numpy as np
import rustforge_port

rng = np.random.default_rng(42)
a = rng.random((80, 201))
b = rng.random((201, 200))

rust_result = rustforge_port.matmul(a, b)
ref_result = np.dot(a, b)

passed = np.allclose(rust_result, ref_result, rtol=1e-5, atol=1e-8)

n_runs = 50
rust_times = timeit.repeat(lambda: rustforge_port.matmul(a, b), number=1, repeat=n_runs)
np_times = timeit.repeat(lambda: np.dot(a, b), number=1, repeat=n_runs)

rust_median = sorted(rust_times)[n_runs // 2]
np_median = sorted(np_times)[n_runs // 2]
speedup = np_median / rust_median

print(f"Rust median:  {rust_median * 1000:.3f} ms")
print(f"NumPy median: {np_median * 1000:.3f} ms")
print(f"Speedup:      {speedup:.2f}x")

if passed:
    print("STAGE 1 GATE: PASS")
else:
    max_diff = np.max(np.abs(rust_result - ref_result))
    print(f"STAGE 1 GATE: FAIL — max abs diff: {max_diff}")
