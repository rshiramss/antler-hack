"""
Swarm A debug runner — maximum verbosity, parallelism proof.

Usage:
    modal run debug_swarm_a.py --filepath test_functions.py
    python debug_swarm_a.py [filepath]
"""
import ast
import os
import sys
import time

import modal

app = modal.App("rustforge-swarm-a-debug")
image = modal.Image.debian_slim(python_version="3.11")


# ── Debug worker ──────────────────────────────────────────────────────────────

@app.function(image=image)
def score_function_debug(fn_info: dict) -> dict:
    """Score one function with full debug instrumentation."""
    import ast
    import os
    import socket
    import textwrap
    import time

    name = fn_info["name"]
    source = fn_info["source"]
    pid = os.getpid()
    hostname = socket.gethostname()
    t_start = time.time()

    print(
        f"[WORKER START] fn={name} hostname={hostname} pid={pid} "
        f"time={t_start:.6f}"
    )

    try:
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        t_end = time.time()
        elapsed_ms = (t_end - t_start) * 1000
        result = {"name": name, "score": 0.0, "features": {"error": str(exc)}, "source": source,
                  "_pid": pid, "_hostname": hostname}
        print(
            f"[WORKER DONE] fn={name} pid={pid} score=0.0 elapsed={elapsed_ms:.1f}ms"
        )
        return result

    fn_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fn_node = node
            break

    if fn_node is None:
        t_end = time.time()
        elapsed_ms = (t_end - t_start) * 1000
        result = {"name": name, "score": 0.0, "features": {}, "source": source,
                  "_pid": pid, "_hostname": hostname}
        print(
            f"[WORKER DONE] fn={name} pid={pid} score=0.0 elapsed={elapsed_ms:.1f}ms"
        )
        return result

    # ── Feature 1: Purity ────────────────────────────────────────────────────
    SIDE_EFFECT_CALLS = {"print", "open", "input", "exit", "quit"}
    purity = 1
    for node in ast.walk(fn_node):
        if purity == 0:
            break
        if isinstance(node, ast.Global):
            purity = 0
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SIDE_EFFECT_CALLS:
                purity = 0
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"write", "flush"}:
                    purity = 0
    if purity == 1:
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        purity = 0
                        break
            if isinstance(node, ast.AugAssign):
                if (
                    isinstance(node.target, ast.Attribute)
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                ):
                    purity = 0

    # ── Feature 2: Numeric density ───────────────────────────────────────────
    NUMERIC_OP_TYPES = (
        ast.Mult, ast.Div, ast.Pow, ast.Add, ast.Sub,
        ast.Mod, ast.FloorDiv, ast.MatMult,
    )
    NUMERIC_BUILTINS = {"sum", "abs", "min", "max", "round", "pow"}
    NUMERIC_MODULES = {"math", "np", "numpy", "cmath"}

    numeric_count = 0
    for node in ast.walk(fn_node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, NUMERIC_OP_TYPES):
            numeric_count += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in NUMERIC_BUILTINS:
                numeric_count += 1
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in NUMERIC_MODULES
            ):
                numeric_count += 1

    n_lines = max(len([l for l in dedented.splitlines() if l.strip()]), 1)
    numeric_density = min(numeric_count / n_lines, 3.0)

    # ── Feature 3: Loop score ────────────────────────────────────────────────
    loop_score = 0.0
    for node in ast.walk(fn_node):
        if isinstance(node, (ast.For, ast.While)):
            nested = sum(
                1
                for child in ast.walk(node)
                if child is not node and isinstance(child, (ast.For, ast.While))
            )
            loop_score += 2.0 if nested > 0 else 1.0
    loop_score = min(loop_score, 4.0)

    # ── Feature 4: Dependency penalty ────────────────────────────────────────
    ALL_BUILTINS = {
        "abs", "all", "any", "bin", "bool", "callable", "chr", "classmethod",
        "dict", "dir", "divmod", "enumerate", "exit", "filter", "float",
        "format", "frozenset", "getattr", "globals", "hasattr", "hash", "hex",
        "id", "input", "int", "isinstance", "issubclass", "iter", "len",
        "list", "locals", "map", "max", "min", "next", "object", "oct", "open",
        "ord", "pow", "print", "property", "quit", "range", "repr", "reversed",
        "round", "set", "setattr", "slice", "sorted", "staticmethod", "str",
        "sum", "super", "tuple", "type", "vars", "zip",
    }
    ext_calls = 0
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id not in {"self", "cls"}
            ):
                ext_calls += 1
            elif isinstance(node.func, ast.Name) and node.func.id not in ALL_BUILTINS:
                ext_calls += 1
    dependency_penalty = min(ext_calls * 0.3, 3.0)

    # ── Feature 5: Determinism ───────────────────────────────────────────────
    NON_DET_MODULES = {"random", "time", "datetime", "uuid", "secrets"}
    NON_DET_ATTRS = {
        "time", "now", "utcnow", "today", "random", "randint", "randrange",
        "choice", "choices", "shuffle", "gauss", "uniform", "getpid", "urandom",
        "seed",
    }
    determinism = 1
    for node in ast.walk(fn_node):
        if determinism == 0:
            break
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in NON_DET_MODULES
                ):
                    determinism = 0
                elif node.func.attr in NON_DET_ATTRS:
                    determinism = 0
            elif isinstance(node.func, ast.Name) and node.func.id in {"time", "random"}:
                determinism = 0

    # ── Final score ──────────────────────────────────────────────────────────
    score = (
        purity * 2.0
        + numeric_density
        + loop_score
        + determinism * 2.0
        - dependency_penalty
    )
    score = round(score, 2)

    t_end = time.time()
    elapsed_ms = (t_end - t_start) * 1000
    print(
        f"[WORKER DONE] fn={name} pid={pid} score={score} elapsed={elapsed_ms:.1f}ms"
    )

    return {
        "name": name,
        "score": score,
        "features": {
            "purity": purity,
            "numeric_density": round(numeric_density, 2),
            "loop_score": loop_score,
            "dependency_penalty": round(dependency_penalty, 2),
            "determinism": determinism,
        },
        "source": source,
        "_pid": pid,
        "_hostname": hostname,
    }


# ── Local helpers (copied from analyze.py, no modification to analyze.py) ────

def _extract_functions(filepath: str) -> list[dict]:
    with open(filepath) as f:
        source_code = f.read()
    tree = ast.parse(source_code)
    lines = source_code.splitlines()
    fn_infos = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            fn_infos.append({"name": node.name, "source": fn_source})
    return fn_infos


def _print_table(results: list[dict]) -> None:
    col_name = max(len(r["name"]) for r in results)
    col_name = max(col_name, 8)
    header = (
        f"{'Rank':<5} {'Function':<{col_name}} {'Score':>7}  "
        f"{'Pure':>4}  {'NumD':>4}  {'Loops':>5}  {'Det':>3}  {'Dep-':>4}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        f = r["features"]
        print(
            f"{i:<5} {r['name']:<{col_name}} {r['score']:>7.2f}  "
            f"{f.get('purity', 0):>4}  {f.get('numeric_density', 0):>4.2f}  "
            f"{f.get('loop_score', 0):>5.1f}  {f.get('determinism', 0):>3}  "
            f"{f.get('dependency_penalty', 0):>4.2f}"
        )


# ── Modal local entrypoint ────────────────────────────────────────────────────

@app.local_entrypoint()
def main(filepath: str = "test_functions.py") -> None:
    filepath = os.path.abspath(filepath)
    fn_infos = _extract_functions(filepath)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SWARM A DEBUG — pre-flight")
    print(f"{'='*60}")
    print(f"File       : {filepath}")
    print(f"Functions  : {len(fn_infos)} total")
    for i, fn in enumerate(fn_infos, 1):
        print(f"  {i:>2}. {fn['name']}")
    print(f"{'='*60}\n")

    if not fn_infos:
        print("No top-level functions found — aborting.")
        return

    print(f"Fanning out {len(fn_infos)} Modal workers (order_outputs=False)...\n")

    # ── Fan-out timing ────────────────────────────────────────────────────────
    t_fanout_start = time.time()
    print(f"[FAN-OUT START] timestamp={t_fanout_start:.6f}")

    results_raw = []
    for result in score_function_debug.map(fn_infos, order_outputs=False):
        arrived_at = time.time()
        print(
            f"[RESULT] fn={result['name']} score={result['score']} "
            f"arrived_at={arrived_at:.6f}"
        )
        results_raw.append(result)

    t_fanout_end = time.time()
    wall_clock_s = t_fanout_end - t_fanout_start
    print(f"\n[FAN-OUT END] timestamp={t_fanout_end:.6f}")
    print(f"[FAN-OUT TIMING] total wall-clock time = {wall_clock_s:.3f}s for {len(fn_infos)} workers\n")

    # ── Parallelism proof ─────────────────────────────────────────────────────
    pids = {r["_pid"] for r in results_raw}
    hostnames = {r["_hostname"] for r in results_raw}

    print(f"{'='*60}")
    print("PARALLELISM ANALYSIS")
    print(f"{'='*60}")
    print(f"Distinct PIDs      : {sorted(pids)}")
    print(f"Distinct hostnames : {sorted(hostnames)}")

    n_distinct = len(hostnames) if len(hostnames) > 1 else len(pids)
    if len(hostnames) > 1 or len(pids) > 1:
        print(f"PARALLELISM CONFIRMED: {n_distinct} distinct containers observed")
    else:
        print("WARNING: all workers ran on same pid — may be sequential")
    print(f"{'='*60}\n")

    # ── Ranked table ──────────────────────────────────────────────────────────
    results_sorted = sorted(results_raw, key=lambda r: r["score"], reverse=True)
    print("FINAL RANKED TABLE")
    _print_table(results_sorted)

    top = results_sorted[0]
    print(f"\nTop candidate: {top['name']}  (score {top['score']:.2f})")
    print("Handoff contract — name, score, and source ready for Swarm B.\n")
    print("SWARM A DEBUG: PASS")


# ── Direct invocation: python debug_swarm_a.py [filepath] ────────────────────

if __name__ == "__main__":
    import subprocess

    filepath = sys.argv[1] if len(sys.argv) > 1 else "test_functions.py"
    subprocess.run(
        [
            "modal",
            "run",
            os.path.abspath(__file__),
            "--filepath",
            os.path.abspath(filepath),
        ],
        check=True,
    )
