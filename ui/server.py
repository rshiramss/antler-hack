"""
RustForge — live dashboard server (Stage 5).

FastAPI + WebSocket. Serves ui/index.html and streams swarm events to it.
Closes build.md open-question O-2 (no websocket example in the vendored docs):
the working `@app.websocket("/ws")` + broadcast-hub + thread-safe push pattern
is implemented here from scratch.

RUN
---
  pip install fastapi uvicorn
  python ui/server.py            # serve + drive the built-in simulator (great for demo)
  python ui/server.py --serve    # serve only; events come from evolve.py via push_event()
  open  http://localhost:8000

WIRE THE REAL SWARM
-------------------
`evolve.py` imports `push_event` and calls it as each Modal container reports.
Because Modal `.spawn_map()` is fire-and-forget (build.md O-6), stream results
out-of-band: each `try_one_port` worker returns a dict; the driver turns it into
one event and calls push_event(...).  The CHAMPION event must carry the new Rust
`code` (and ideally `why`) so the dashboard can render the diff + lineage — that
visible code improvement is the entire point of the demo.

EVENT SCHEMA  (identical to the JS simulator in index.html — keep them in sync)
------------------------------------------------------------------------------
  {"type":"round_start", "round":int, "target":str, "n_cells":int}
  {"type":"cell_update", "round":int, "cell":int, "status":str, "model":str}
       status ∈ queued|spawning|compiling|testing|benchmarking|passed|failed|crashed|champion
  {"type":"cell_result", "round":int, "cell":int, "passed":bool, "speedup":float,
                         "status":str, "description":str, "reward_hack":bool}
  {"type":"champion",    "round":int, "speedup":float, "code":str, "why":str,
                         "description":str, "cell":int}      # code/why drive the diff + lineage
  {"type":"journal",     "id":int, "speedup":float, "status":"keep|discard|crash",
                         "description":str, "reward_hack":bool}
  {"type":"round_end",   "round":int, "best_speedup":float, "kept":bool}

The dashboard DERIVES the narration feed and the process funnel from these
events client-side, so no extra event types are needed.
"""
from __future__ import annotations
import asyncio, json, os, random, sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

HERE = Path(__file__).parent
app = FastAPI(title="RustForge Swarm Console")


class Hub:
    """Fan-out broadcast hub + a thread-safe push entry point for evolve.py."""
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.history: list[dict] = []      # replay to late-joining browsers

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        for evt in self.history[-600:]:
            await ws.send_text(json.dumps(evt))

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, evt: dict) -> None:
        if evt.get("type") == "round_start" and evt.get("round") == 1:
            self.history.clear()           # a fresh run resets the replay buffer
        self.history.append(evt)
        for ws in list(self.clients):
            try:
                await ws.send_text(json.dumps(evt))
            except Exception:
                self.disconnect(ws)


hub = Hub()


def push_event(evt: dict) -> None:
    """Call from ANY thread (e.g. the Modal driver in evolve.py)."""
    if hub.loop is None:
        return
    asyncio.run_coroutine_threadsafe(hub.broadcast(evt), hub.loop)


# ---------------------------------------------------------------------------
# Replay mode (adapter B): stream a captured ui/events.jsonl of REAL swarm events
# over the same websocket, with demo pacing. The browser uses the SAME applyEvent()
# path, so replay looks identical to live. Every value comes from the captured file;
# only the inter-event PACING is added here.
# ---------------------------------------------------------------------------
async def _replayer(path: str) -> None:
    await asyncio.sleep(0.5)
    speed = float(os.environ.get("RF_REPLAY_SPEED", "1.0"))   # >1 = faster
    with open(path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    pace = {"round_start": 0.45, "cell_update": 0.10, "cell_result": 0.05,
            "journal": 0.04, "champion": 0.85, "round_end": 1.10}
    for evt in events:
        await hub.broadcast(evt)
        d = pace.get(evt.get("type"), 0.05)
        if evt.get("type") == "cell_update" and evt.get("status") == "spawning":
            d = 0.04
        await asyncio.sleep(d / speed)


def _replay_path() -> str:
    i = sys.argv.index("--replay")
    nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    if nxt and not nxt.startswith("-") and os.path.exists(nxt):
        return nxt
    return str(HERE / "events.jsonl")   # default


@app.on_event("startup")
async def _startup() -> None:
    hub.loop = asyncio.get_running_loop()
    if "--replay" in sys.argv:              # adapter B: replay captured REAL events
        asyncio.create_task(_replayer(_replay_path()))
    elif "--serve" not in sys.argv:         # default: also run the demo simulator
        asyncio.create_task(_simulator())


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()        # keep the socket open; ignore inbound
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)


# ---------------------------------------------------------------------------
# Built-in simulator — proves the live websocket pipe end-to-end and gives a
# churning demo before the Modal swarm is wired.  Emits the SAME events (incl.
# real, progressively-optimized champion code) as the JS simulator, so the diff
# + lineage panes light up identically over a real websocket.
# ---------------------------------------------------------------------------
N = 50
MUT = ["loop reorder i-k-j", "smaller tiles", "scalar fallback", "extra bounds check",
       "redundant copy", "non-contiguous access"]            # labels for *slower* variants
CRASH = ["borrow checker E0502", "f32/f64 mismatch at .dot()", "cannot move out of borrowed",
         "unresolved import ndarray::Zip", "lifetime 'py escapes function"]

# Each step's `code` differs from the previous by a few lines -> a meaningful diff.
SEQ = {
 "whisper_mel": [
   {"sp": 1.0, "desc": "seed naive port", "why": "Literal port — correct, no faster than NumPy.",
    "code": "let mel = f.as_array().dot(&m.as_array());\nlet out = mel.mapv(|v| v.max(1e-10).log10());\nout.into_pyarray(py)"},
   {"sp": 1.9, "desc": "rayon par_iter + fuse log", "why": "Rows run in parallel; matmul and log fused into one pass.",
    "code": "out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)| {\n    for j in 0..b.ncols() {\n        let mut acc = 0.0f32;\n        for k in 0..a.ncols() { acc += a[[i,k]] * b[[k,j]]; }\n        row[j] = acc.max(1e-10).log10();   // fused\n    }\n});"},
   {"sp": 3.1, "desc": "uget() bounds elision", "why": "Drop per-element bounds checks in the hot loop with unsafe uget().",
    "code": "for k in 0..a.ncols() {\n    unsafe { acc += a.uget((i,k)) * b.uget((k,j)); }   // no bounds check\n}\nrow[j] = acc.max(1e-10).log10();"},
   {"sp": 4.4, "desc": "transpose b for locality", "why": "Transpose b once so the inner product walks contiguous memory.",
    "code": "let bt = b.t();\nlet ai = a.row(i);\nfor j in 0..bt.nrows() {\n    let bj = bt.row(j);\n    let mut acc = 0.0f32;\n    for k in 0..ai.len() { unsafe { acc += ai.uget(k) * bj.uget(k); } }\n    row[j] = acc.max(1e-10).log10();\n}"},
   {"sp": 5.6, "desc": "4 accumulators (ILP)", "why": "Four independent accumulators expose instruction-level parallelism.",
    "code": "let (n, mut acc) = (ai.len(), [0.0f32; 4]);\nlet mut k = 0;\nwhile k + 4 <= n { unsafe {\n    acc[0]+=ai.uget(k)*bj.uget(k);     acc[1]+=ai.uget(k+1)*bj.uget(k+1);\n    acc[2]+=ai.uget(k+2)*bj.uget(k+2); acc[3]+=ai.uget(k+3)*bj.uget(k+3);\n} k += 4; }\nlet mut s = acc[0]+acc[1]+acc[2]+acc[3];\nwhile k < n { unsafe { s += ai.uget(k)*bj.uget(k); } k += 1; }\nrow[j] = s.max(1e-10).log10();"},
   {"sp": 6.8, "desc": "std::simd f32x8", "why": "Explicit f32x8 SIMD — eight multiply-adds per instruction.",
    "code": "let mut v = f32x8::splat(0.0);\nlet mut k = 0;\nwhile k + 8 <= n {\n    v += f32x8::from_slice(&ai_s[k..]) * f32x8::from_slice(&bj_s[k..]);\n    k += 8;\n}\nlet mut s = v.reduce_sum();\nwhile k < n { s += ai_s[k]*bj_s[k]; k += 1; }\nrow[j] = s.max(1e-10).log10();"},
 ],
 "mandelbrot": [
   {"sp": 1.0, "desc": "seed escape-time loop", "why": "Per-pixel escape-time, single thread.",
    "code": "for y in 0..h { for x in 0..w {\n    let (cx,cy) = px_to_c(x,y,w,h);\n    let (mut zx,mut zy,mut n) = (0.0,0.0,0);\n    while zx*zx+zy*zy<=4.0 && n<max_it { let t=zx*zx-zy*zy+cx; zy=2.0*zx*zy+cy; zx=t; n+=1; }\n    out[[y,x]] = n;\n}}"},
   {"sp": 6.0, "desc": "rayon par_chunks rows", "why": "Each scanline renders on its own core via par_chunks_mut.",
    "code": "buf.par_chunks_mut(w).enumerate().for_each(|(y,row)| {\n    for x in 0..w {\n        let (cx,cy) = px_to_c(x,y,w,h);\n        let (mut zx,mut zy,mut n) = (0.0,0.0,0);\n        while zx*zx+zy*zy<=4.0 && n<max_it { let t=zx*zx-zy*zy+cx; zy=2.0*zx*zy+cy; zx=t; n+=1; }\n        row[x] = n;\n    }\n});"},
   {"sp": 18.0, "desc": "reuse z² terms", "why": "Track x² and y² — 3 multiplies per iteration instead of 5.",
    "code": "let (mut zx,mut zy,mut x2,mut y2,mut n) = (0.0,0.0,0.0,0.0,0);\nwhile x2+y2<=4.0 && n<max_it {\n    zy = 2.0*zx*zy + cy;\n    zx = x2 - y2 + cx;\n    x2 = zx*zx; y2 = zy*zy;   // reuse squares\n    n += 1;\n}\nrow[x] = n;"},
   {"sp": 34.0, "desc": "cardioid + bulb early-out", "why": "Analytic test skips the two biggest interior regions instantly.",
    "code": "let q = (cx-0.25).powi(2) + cy*cy;\nif q*(q+(cx-0.25)) <= 0.25*cy*cy || (cx+1.0).powi(2)+cy*cy <= 0.0625 {\n    row[x] = max_it; continue;   // interior — no iteration\n}"},
   {"sp": 52.0, "desc": "f64x8 SIMD over 8 pixels", "why": "Eight horizontal pixels share one lane-masked escape-time loop.",
    "code": "let cx = f64x8::from_array(std::array::from_fn(|l| px_x(x+l,w)));\nlet cy = f64x8::splat(px_y(y,h));\nlet (mut zx,mut zy,mut n) = (f64x8::splat(0.0),f64x8::splat(0.0),u32x8::splat(0));\nfor _ in 0..max_it {\n    let m = (zx*zx+zy*zy).simd_le(f64x8::splat(4.0));\n    if !m.any() { break; }\n    let t = zx*zx-zy*zy+cx; zy = f64x8::splat(2.0)*zx*zy+cy; zx = t;\n    n += m.select(u32x8::splat(1), u32x8::splat(0));\n}"},
   {"sp": 88.0, "desc": "SIMD + early-out combined", "why": "Interior early-out plus f64x8 lanes together — the full kernel.",
    "code": "let interior = (q*(q+cx-0.25)).simd_le(0.25*cy*cy);\nlet mut n = interior.select(u32x8::splat(max_it), u32x8::splat(0));\nlet mut live = !interior;\nfor _ in 0..max_it {\n    live &= (zx*zx+zy*zy).simd_le(f64x8::splat(4.0));\n    if !live.any() { break; }\n    let t = zx*zx-zy*zy+cx; zy = f64x8::splat(2.0)*zx*zy+cy; zx = t;\n    n += live.cast().select(u32x8::splat(1), u32x8::splat(0));\n}"},
 ],
}


async def _simulator(target: str = "whisper_mel") -> None:
    await asyncio.sleep(0.5)
    seq = SEQ[target]
    best = 1.0
    jid = 0
    await hub.broadcast({"type": "journal", "id": (jid := jid + 1), "speedup": 1.0,
                         "status": "keep", "description": "seed naive port (baseline)"})
    await hub.broadcast({"type": "champion", "round": 0, "speedup": seq[0]["sp"], "cell": None,
                         "description": seq[0]["desc"], "why": seq[0]["why"], "code": seq[0]["code"]})
    for rnd in range(1, len(seq)):
        await hub.broadcast({"type": "round_start", "round": rnd, "target": target, "n_cells": N})
        peak = seq[rnd]["sp"]; winner = random.randrange(N); hack = random.randrange(N)
        best_round = 0.0; best_cell = -1
        for i in range(N):
            await hub.broadcast({"type": "cell_update", "round": rnd, "cell": i, "status": "spawning"})
        await asyncio.sleep(0.4)
        for i in range(N):
            await hub.broadcast({"type": "cell_update", "round": rnd, "cell": i, "status": "compiling"})
            if random.random() < (0.32 if rnd <= 2 else 0.16):
                await hub.broadcast({"type": "cell_result", "round": rnd, "cell": i, "passed": False,
                                     "status": "crashed", "speedup": 0.0, "description": CRASH[i % len(CRASH)]})
                await hub.broadcast({"type": "journal", "id": (jid := jid + 1), "speedup": 0.0,
                                     "status": "crash", "description": CRASH[i % len(CRASH)]})
                continue
            await hub.broadcast({"type": "cell_update", "round": rnd, "cell": i, "status": "testing"})
            if i == hack:
                msg = "hardcoded outputs — passed unit tests, FAILED fuzz"
                await hub.broadcast({"type": "cell_result", "round": rnd, "cell": i, "passed": False,
                                     "status": "failed", "speedup": 0.0, "reward_hack": True, "description": msg})
                await hub.broadcast({"type": "journal", "id": (jid := jid + 1), "speedup": 0.0,
                                     "status": "discard", "reward_hack": True, "description": msg})
                continue
            await hub.broadcast({"type": "cell_update", "round": rnd, "cell": i, "status": "benchmarking"})
            sp = peak * (0.96 + random.random() * 0.05) if i == winner else max(0.5, best * (0.5 + random.random() * 0.8))
            faster = sp > best
            await hub.broadcast({"type": "cell_result", "round": rnd, "cell": i, "passed": True,
                                 "speedup": sp, "status": "passed", "description": seq[rnd]["desc"]})
            if faster or random.random() < 0.10:
                await hub.broadcast({"type": "journal", "id": (jid := jid + 1), "speedup": sp,
                                     "status": "keep" if faster else "discard",
                                     "description": seq[rnd]["desc"] if faster else MUT[i % len(MUT)]})
            if sp > best_round:
                best_round, best_cell = sp, i
            await asyncio.sleep(0.02)
        kept = best_round > best
        if kept:
            best = seq[rnd]["sp"]
            await hub.broadcast({"type": "champion", "round": rnd, "speedup": best, "cell": best_cell,
                                 "description": seq[rnd]["desc"], "why": seq[rnd]["why"], "code": seq[rnd]["code"]})
        await hub.broadcast({"type": "round_end", "round": rnd, "best_speedup": best, "kept": kept})
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
