import numpy as np


def verify(rust_fn, oracle_fn, spec=None, n_examples=200):
    """Fuzz-test rust_fn against oracle_fn over the spec's input profile.

    The input shape, element range, and tolerances come from the TargetSpec instead
    of being hardcoded. Falls back to the legacy 2-element [-1, 1] profile when no
    spec is given so the function stays usable standalone.

    Returns {"passed": bool, "error": str | None}.
    """
    if spec is not None:
        shape = tuple(spec.input_shape)
        low, high = spec.input_low, spec.input_high
        rtol, atol = spec.rtol, spec.atol
    else:
        shape, low, high, rtol, atol = (2,), -1.0, 1.0, 1e-5, 1e-6

    try:
        from hypothesis.extra.numpy import arrays
        from hypothesis.strategies import floats
        from hypothesis import given, settings, HealthCheck

        results = {"passed": True, "error": None}

        @given(
            arrays(
                np.float64,
                shape=shape,
                elements=floats(low, high, allow_nan=False, allow_infinity=False),
            )
        )
        @settings(max_examples=n_examples, suppress_health_check=[HealthCheck.too_slow])
        def _test(inputs):
            oracle_result = np.asarray(oracle_fn(inputs))
            rust_result = np.asarray(rust_fn(inputs))
            if not np.allclose(oracle_result, rust_result, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(oracle_result - rust_result))
                raise AssertionError(
                    f"allclose failed: max diff {max_diff:.3e}, "
                    f"oracle={oracle_result}, rust={rust_result}"
                )

        try:
            _test()
        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)

        return results

    except ImportError:
        # Fallback: manual random generation
        rng = np.random.default_rng(0)
        for i in range(n_examples):
            inputs = rng.uniform(low, high, size=shape).astype(np.float64)
            try:
                oracle_result = np.asarray(oracle_fn(inputs))
                rust_result = np.asarray(rust_fn(inputs))
                if not np.allclose(oracle_result, rust_result, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(oracle_result - rust_result))
                    return {
                        "passed": False,
                        "error": f"allclose failed at example {i}: max diff {max_diff:.3e}",
                    }
            except Exception as e:
                return {"passed": False, "error": f"exception at example {i}: {e}"}
        return {"passed": True, "error": None}
