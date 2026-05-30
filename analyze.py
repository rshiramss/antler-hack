"""
Swarm A — repo analysis and function ranking.

Usage:
    python analyze.py test_functions.py
    python analyze.py https://github.com/karpathy/micrograd
    modal run analyze.py --filepath test_functions.py
    modal run analyze.py --filepath https://github.com/karpathy/micrograd

Each top-level function gets its own Modal worker.
Workers score functions using objective AST features only — no LLM.
Results stream back in completion order and are ranked by score.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile

import modal

app = modal.App("rustforge-swarm-a")
image = modal.Image.debian_slim(python_version="3.11")


# ── Worker — runs remotely, one container per function ───────────────────────

@app.function(image=image)
def score_function(fn_info: dict) -> dict:
    """Score one function on optimization potential using pure AST analysis."""
    import ast
    import textwrap

    name = fn_info["name"]
    source = fn_info["source"]

    try:
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        return {"name": name, "score": 0.0, "features": {"error": str(exc)}, "source": source}

    fn_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fn_node = node
            break

    if fn_node is None:
        return {"name": name, "score": 0.0, "features": {}, "source": source}

    # ── Feature 1: Purity ────────────────────────────────────────────────────
    # 0 if: declares a global, calls print/file I/O, or imports inside body.
    # self.attr mutations are normal method behavior — not counted as impure.
    SIDE_EFFECT_CALLS = {"print", "open", "input", "exit", "quit"}

    purity = 1
    for node in ast.walk(fn_node):
        if purity == 0:
            break
        if isinstance(node, ast.Global):
            purity = 0
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            purity = 0
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SIDE_EFFECT_CALLS:
                purity = 0
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"write", "flush"}:
                    purity = 0

    # ── Feature 2: Numeric density ───────────────────────────────────────────
    # Counts arithmetic BinOps and calls to numeric builtins/modules, normalised
    # by non-blank line count. Capped at 3.0.
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
    # Each loop: +1. Loops that contain inner loops: +2 instead (2x weight).
    # Capped at 4.0.
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
    # Counts external function/method calls inside the function body.
    # Fewer dependencies = more portable = better candidate for Rust porting.
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

    local_names = set(fn_info.get("local_names") or [])

    ext_calls = 0
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id not in {"self", "cls"}
            ):
                ext_calls += 1
            elif isinstance(node.func, ast.Name) and node.func.id not in ALL_BUILTINS | local_names:
                ext_calls += 1

    dependency_penalty = min(ext_calls * 0.3, 3.0)

    # ── Feature 5: Determinism ───────────────────────────────────────────────
    # 0 if the function calls anything non-deterministic (random, time, etc.)
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
    # score = purity*2 + numeric_density + loop_score + determinism*2 - dep_pen
    score = (
        purity * 2.0
        + numeric_density
        + loop_score
        + determinism * 2.0
        - dependency_penalty
    )

    return {
        "name": name,
        "score": round(score, 2),
        "features": {
            "purity": purity,
            "numeric_density": round(numeric_density, 2),
            "loop_score": loop_score,
            "dependency_penalty": round(dependency_penalty, 2),
            "determinism": determinism,
        },
        "source": source,
    }


# ── Local helpers ─────────────────────────────────────────────────────────────

_SKIP_DUNDERS = {"__init__", "__repr__", "__str__", "__del__", "__new__"}


def _extract_functions(filepath: str, file_label: str = "") -> list[dict]:
    """Parse function definitions from a Python source file.

    Collects module-level functions and class methods (skipping boilerplate
    dunders). Methods are named ClassName.method_name so they remain unique
    across files.
    """
    with open(filepath) as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    lines = source_code.splitlines()
    label = file_label or os.path.basename(filepath)

    local_names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]

    fn_infos = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            fn_infos.append({"name": node.name, "source": fn_source, "file": label, "local_names": local_names})
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name not in _SKIP_DUNDERS:
                    fn_source = "\n".join(lines[item.lineno - 1 : item.end_lineno])
                    fn_infos.append({
                        "name": f"{class_name}.{item.name}",
                        "source": fn_source,
                        "file": label,
                        "local_names": local_names,
                    })
    return fn_infos


_SKIP_DIRS = {"__pycache__", ".git"}


def _clone_and_extract_repo(url: str) -> tuple[list[dict], str]:
    """Clone a GitHub repo depth-1 and extract top-level functions from all .py files.

    Returns (fn_infos, tmpdir). Caller is responsible for cleaning up tmpdir.
    """
    tmpdir = tempfile.mkdtemp(prefix="rustforge_swarm_a_")
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, tmpdir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")

    fn_infos = []
    for dirpath, dirnames, filenames in os.walk(tmpdir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname == "setup.py":
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, tmpdir)
            try:
                extracted = _extract_functions(full_path, file_label=rel_path)
            except SyntaxError as exc:
                print(f"  [warn] skipping {rel_path} (syntax error): {exc}", file=sys.stderr)
                continue
            except Exception as exc:
                print(f"  [warn] skipping {rel_path}: {exc}", file=sys.stderr)
                continue
            for fn in extracted:
                if fn["source"]:
                    fn_infos.append(fn)
                else:
                    print(
                        f"  [warn] skipping {fn['name']} in {rel_path}: empty source",
                        file=sys.stderr,
                    )
    return fn_infos, tmpdir


def _print_table(results: list[dict]) -> None:
    show_file = any(r.get("file") for r in results)
    col_name = max(len(r["name"]) for r in results)
    col_name = max(col_name, 8)

    if show_file:
        col_file = max((len(r.get("file", "")) for r in results), default=4)
        col_file = max(col_file, 4)
        header = (
            f"{'Rank':<5} {'Function':<{col_name}} {'File':<{col_file}} {'Score':>7}  "
            f"{'Pure':>4}  {'NumD':>4}  {'Loops':>5}  {'Det':>3}  {'Dep-':>4}"
        )
    else:
        col_file = 0
        header = (
            f"{'Rank':<5} {'Function':<{col_name}} {'Score':>7}  "
            f"{'Pure':>4}  {'NumD':>4}  {'Loops':>5}  {'Det':>3}  {'Dep-':>4}"
        )

    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        f = r["features"]
        if show_file:
            print(
                f"{i:<5} {r['name']:<{col_name}} {r.get('file', ''):<{col_file}} {r['score']:>7.2f}  "
                f"{f.get('purity', 0):>4}  {f.get('numeric_density', 0):>4.2f}  "
                f"{f.get('loop_score', 0):>5.1f}  {f.get('determinism', 0):>3}  "
                f"{f.get('dependency_penalty', 0):>4.2f}"
            )
        else:
            print(
                f"{i:<5} {r['name']:<{col_name}} {r['score']:>7.2f}  "
                f"{f.get('purity', 0):>4}  {f.get('numeric_density', 0):>4.2f}  "
                f"{f.get('loop_score', 0):>5.1f}  {f.get('determinism', 0):>3}  "
                f"{f.get('dependency_penalty', 0):>4.2f}"
            )


# ── Modal local entrypoint ────────────────────────────────────────────────────

@app.local_entrypoint()
def main(filepath: str, emit_json: str = "") -> None:
    is_url = filepath.startswith("http")
    tmpdir = None

    try:
        if is_url:
            print(f"Swarm A: cloning {filepath} ...")
            fn_infos, tmpdir = _clone_and_extract_repo(filepath)
            source_label = filepath
        else:
            filepath = os.path.abspath(filepath)
            fn_infos = _extract_functions(filepath)
            source_label = os.path.basename(filepath)

        if not fn_infos:
            print(f"No top-level functions found in {source_label}")
            return

        # Build lookup to restore the file field after scoring (worker doesn't return it)
        file_map = {(info["name"], info["source"]): info.get("file", "") for info in fn_infos}

        print(f"Swarm A: {len(fn_infos)} functions found in {source_label}")
        print(f"Fanning out {len(fn_infos)} Modal workers (order_outputs=False)...\n")

        results = list(score_function.map(fn_infos, order_outputs=False))

        for r in results:
            r["file"] = file_map.get((r["name"], r["source"]), "")

        results.sort(key=lambda r: r["score"], reverse=True)
        _print_table(results)

        top = results[0]
        print(f"\nTop candidate: {top['name']}  (score {top['score']:.2f})")
        if top.get("file"):
            print(f"  File: {top['file']}")
        print("Handoff contract — name, score, and source ready for Swarm B.\n")

        # Persist the ranked results so Swarm B (run_stage2.py --swarm-a) can consume them.
        if emit_json:
            import json as _json
            with open(emit_json, "w") as f:
                _json.dump(results, f, indent=2)
            print(f"  [handoff] wrote ranked candidates → {emit_json}")

        print("SWARM A GATE: PASS")

    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"  [cleanup] removed {tmpdir}")


# ── Direct invocation: python analyze.py <filepath|url> ──────────────────────

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_functions.py"
    emit_json = sys.argv[2] if len(sys.argv) > 2 else ""
    cmd = [
        "modal",
        "run",
        os.path.abspath(__file__),
        "--filepath",
        target,
    ]
    if emit_json:
        cmd += ["--emit-json", os.path.abspath(emit_json)]
    subprocess.run(cmd, check=True)
