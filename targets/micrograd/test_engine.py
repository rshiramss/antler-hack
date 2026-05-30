import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from targets.micrograd.engine import Value
from targets.micrograd.oracle import run_reference


def test_add():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul():
    a, b = Value(3.0), Value(-4.0)
    c = a * b
    c.backward()
    assert c.data == -12.0
    assert a.grad == -4.0
    assert b.grad == 3.0


def test_relu_positive():
    a = Value(2.0)
    b = a.relu()
    b.backward()
    assert b.data == 2.0
    assert a.grad == 1.0


def test_relu_negative():
    a = Value(-3.0)
    b = a.relu()
    b.backward()
    assert b.data == 0.0
    assert a.grad == 0.0


def test_chain():
    x = Value(1.5)
    y = (x * x + x * Value(2.0)).relu()
    y.backward()
    assert abs(y.data - (1.5 * 1.5 + 1.5 * 2.0)) < 1e-10
    assert abs(x.grad - (2 * 1.5 + 2.0)) < 1e-10  # d/dx(x^2+2x) = 2x+2


def test_oracle_shape():
    result = run_reference(np.array([0.5, -0.3]))
    assert result.shape == (3,), f"Expected (3,), got {result.shape}"
    assert result.dtype == np.float64


def test_oracle_deterministic():
    inp = np.array([0.5, -0.3])
    assert np.allclose(run_reference(inp), run_reference(inp))


def test_oracle_varies_with_input():
    r1 = run_reference(np.array([0.5, -0.3]))
    r2 = run_reference(np.array([-0.8, 0.1]))
    assert not np.allclose(r1, r2)


if __name__ == "__main__":
    for fn in [test_add, test_mul, test_relu_positive, test_relu_negative,
               test_chain, test_oracle_shape, test_oracle_deterministic, test_oracle_varies_with_input]:
        fn()
        print(f"  {fn.__name__}: PASS")
    print("All engine tests passed.")
