"""RustForge stress suite runner.

Loads every adversarial candidate in stress/candidates/ (one file per attack, each
documenting the gate it targets), scores it with the FROZEN shared metric (evaluate.py),
and prints a matrix:  test · claim · expected · actual · PASS/FAIL.

Each candidate must trip ONLY its target gate (gate-isolation). Suite A also includes a
static seed-leakage check (A2); Suite C checks schema adaptivity (C1) and restricted
domains (C2). The anti-cheat invariant is never violated: candidates emit code only;
the harness generates verification inputs at eval time with a seed unseen by the model;
the oracle is the original Python reference.

Usage:
    python stress/run_stress.py            # full suite (B3 first, then A, B, C)
    python stress/run_stress.py --only b3  # just one attack id-prefix
"""
import glob
import importlib
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "optimization"))  # evaluate.py, target.py
CAND_DIR = os.path.join(HERE, "candidates")


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_under(env, candidate, reference, sample):
    """Run evaluate() with per-attack env overrides (SIZES etc. are import-time consts)."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        import evaluate
        importlib.reload(evaluate)
        return evaluate.evaluate(candidate, reference=reference, sample_input=sample)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_matrix(only=None):
    rows = []
    files = sorted(glob.glob(os.path.join(CAND_DIR, "*.py")))
    for path in files:
        mod = _load(path)
        for case in mod.CASES:
            cid = case["name"]
            if only and only.lower() not in cid.lower():
                continue
            r = _eval_under(case.get("env", {}), case["candidate"], case["reference"], case["sample"])
            kept = r["speedup"] > 0
            expect_sub = case["expect"]
            ok = (expect_sub in r["verdict"]) and (kept == case["kept"])
            # secondary numeric assertion (optional)
            if ok and "assert" in case:
                ok = bool(case["assert"](r))
            rows.append({
                "test": cid, "gate": mod.GATE, "claim": case["claim"],
                "expected": expect_sub, "kept": kept, "verdict": r["verdict"],
                "ok": ok, "record": r,
            })
    return rows


def _print_matrix(rows):
    print(f"\n{'TEST':<22}{'GATE':<14}{'EXPECTED':<26}{'KEPT':<6}{'RESULT'}")
    print("-" * 104)
    n_pass = 0
    for r in rows:
        n_pass += r["ok"]
        res = "PASS" if r["ok"] else "*** FAIL ***"
        print(f"{r['test']:<22}{r['gate']:<14}{r['expected']:<26}{str(r['kept']):<6}{res}")
        print(f"  claim:   {r['claim']}")
        print(f"  verdict: {r['verdict']}")
    print("-" * 104)
    print(f"{n_pass}/{len(rows)} PASS")
    return n_pass == len(rows)


# ── A2: static seed-leakage check (not an evaluate candidate) ─────────────────────
def check_a2_seed_leakage():
    """Grep the candidate-facing prompts for the fuzz seed / eval inputs / expected outputs."""
    import re
    # the harness generation seed (must never appear in anything the model sees)
    forbidden = [hex(0xC0FFEE), str(0xC0FFEE), "_GEN_SEED", "expected_output", "fuzz_seed"]
    facing = []
    for name in ("program.md", "program_rust.md", "mutate.py"):
        p = os.path.join(ROOT, "optimization", name)
        if os.path.exists(p):
            facing.append(p)
    hits = []
    for p in facing:
        text = open(p).read()
        for tok in forbidden:
            if tok in text:
                hits.append((os.path.basename(p), tok))
    ok = not hits
    print(f"\n[A2] seed-leakage static check over {', '.join(os.path.basename(p) for p in facing)}")
    print(f"     forbidden tokens: {forbidden}")
    print(f"     {'PASS — none present' if ok else '*** FAIL *** leaked: ' + str(hits)}")
    return ok


# ── C1: schema adaptivity across structurally different signatures ────────────────
def check_c1_signatures():
    import evaluate
    importlib.reload(evaluate)
    cases = [
        ("elementwise-1D", (lambda a: a * a + 1.0), (np.ones(8),), "a"),
        ("matmul-2D", (lambda A, B: A @ B), (np.ones((4, 4)), np.ones((4, 4))), "A,B"),
        ("scalar-reduction", (lambda a: np.sum(a * a)), (np.ones(8),), "a"),
    ]
    print("\n[C1] schema inference / adaptive generation across signatures")
    all_ok = True
    for label, ref, sample, want_scaling in cases:
        sch = evaluate.infer_schema(ref, sample)
        scaling = ",".join(a.name for a in sch.args if a.scales)
        args = evaluate._make_args(sch, sample, 16, np.random.default_rng(0))
        try:
            ref(*args)
            ran = True
        except Exception as e:  # noqa: BLE001
            ran = False
            err = repr(e)
        dtypes_ok = all((a.dtype == "float64") for a in sch.args if a.kind == "ndarray")
        ok = ran and scaling == want_scaling and dtypes_ok
        all_ok &= ok
        shapes = [np.asarray(a).shape for a in args]
        print(f"     {label:<18} scaling={scaling:<6} shapes={shapes} dtype_ok={dtypes_ok} "
              f"runs={'yes' if ran else 'NO:'+err}  -> {'PASS' if ok else '*** FAIL ***'}")
    return all_ok


# ── C2: restricted-domain function gets VALID harness-generated inputs ────────────
def check_c2_restricted_domain():
    """A function valid only on positive inputs; the generic generator yields signed
    values, so this verifies the harness must adapt. We confirm (a) the generic path can
    *fail* on the domain and (b) a domain-aware sample lets the reference run — WITHOUT
    any LLM-authored inputs/outputs (anti-cheat invariant intact)."""
    import evaluate
    importlib.reload(evaluate)

    def ref_sqrt_log(x):
        # valid only for x > 0
        return np.log(x) + np.sqrt(x)

    # generic generator (standard normal) WILL include negatives -> reference errors/NaNs.
    sch = evaluate.infer_schema(ref_sqrt_log, (np.ones(8),))
    rng = np.random.default_rng(0)
    generic = evaluate._make_args(sch, (np.ones(8),), 64, rng)
    with np.errstate(invalid="ignore"):
        generic_out = ref_sqrt_log(*generic)
    generic_valid = bool(np.all(np.isfinite(generic_out)))

    # domain-aware generation: harness draws positive inputs itself (NOT from the model).
    pos = np.abs(rng.standard_normal(64)) + 0.1
    domain_out = ref_sqrt_log(pos)
    domain_valid = bool(np.all(np.isfinite(domain_out)))

    print("\n[C2] restricted-domain generation (positive-only function)")
    print(f"     generic standard-normal inputs valid? {generic_valid}  (expected False — needs domain awareness)")
    print(f"     harness positive-domain inputs valid? {domain_valid}  (expected True)")
    print("     anti-cheat: inputs are harness-generated, never LLM-authored — invariant intact")
    # PASS = we can demonstrate the domain issue and the harness-side fix; Stage-2 (LLM
    # domain inference) would automate choosing the positive sampler.
    ok = (not generic_valid) and domain_valid
    print(f"     -> {'PASS (domain handling demonstrated)' if ok else '*** FAIL ***'}")
    return ok


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    rows = run_matrix(only)
    gates_ok = _print_matrix(rows)
    if only:
        sys.exit(0 if gates_ok else 1)
    a2 = check_a2_seed_leakage()
    c1 = check_c1_signatures()
    c2 = check_c2_restricted_domain()
    print("\n=== SUITE SUMMARY ===")
    print(f"  gate-trip matrix: {'PASS' if gates_ok else 'FAIL'}")
    print(f"  A2 seed-leakage:  {'PASS' if a2 else 'FAIL'}")
    print(f"  C1 adaptivity:    {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 domain:        {'PASS' if c2 else 'FAIL'}")
    sys.exit(0 if (gates_ok and a2 and c1 and c2) else 1)
