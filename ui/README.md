# RustForge — UI (Stage 5: live dashboard)

The demo-critical 4-pane "swarm console." The **container grid is the hero** — it shows the
50-wide Modal swarm churning (spawn → compile → fuzz-verify → benchmark → keep/discard), which is
the project's #1 goal: *a visibly parallel megastructure of agents doing real work.*

## Run it (two ways)

**A. Zero-setup (no server) — just open the file**
```
open ui/index.html
```
Opened via `file://`, the page auto-runs the built-in **simulator**: realistic churn, a climbing
speedup curve, an evolving champion `src/lib.rs`, the autoresearch journal filling with
keep/discard/crash rows — and the **reward-hack-rejected** beat (a candidate that passes the unit
tests but FAILS the differential fuzz → fitness gated to 0 → discarded). This is the fallback demo.

**B. Live websocket (proves the real pipe)**
```
pip install fastapi uvicorn
python ui/server.py          # serves + drives the simulator over a real websocket
# open http://localhost:8000
python ui/server.py --serve  # serve only; events come from evolve.py (no simulator)
```

## Wire the real swarm (≈10 lines in `evolve.py`)
`server.py` exposes a thread-safe `push_event(evt: dict)`. Because Modal `.spawn_map()` is
fire-and-forget (build.md O-6), stream results out-of-band — each container's report becomes one event:

```python
from ui.server import push_event
push_event({"type":"round_start","round":n,"target":"whisper_mel","n_cells":50})
push_event({"type":"cell_update","round":n,"cell":i,"status":"compiling"})
push_event({"type":"cell_result","round":n,"cell":i,"passed":ok,"speedup":sp,"status":"passed","description":desc})
push_event({"type":"champion","round":n,"speedup":sp,"code":rust_src,"description":desc,"cell":i})
push_event({"type":"journal","id":k,"speedup":sp,"status":"keep","description":desc})
push_event({"type":"round_end","round":n,"best_speedup":best,"kept":True})
```

The browser uses the **same `applyEvent()`** for live and simulated data, so the live demo looks
identical to the fallback. Full event schema is documented at the top of `server.py` and in a
comment block in `index.html` — **keep the two in sync.**

## Panes
1. **Swarm** (left, full height) — 50 Modal-container cells, color-coded by state. *The containerization.*
2. **Champion speedup** — best speedup per round vs. the Python baseline (climbs as rounds ratchet).
3. **Champion `src/lib.rs`** — the current best Rust, updated each time a new champion is crowned.
4. **Autoresearch journal** — `id · speedup · status · description`; keep-if-faster, fitness=0 on any correctness failure.
