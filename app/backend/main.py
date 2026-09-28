from __future__ import annotations

import os
import subprocess
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data_sources as ds
from . import fl_control
from . import sse

app = FastAPI(title="Predicția consumului energetic rezidențial prin învățare federată cu personalizare", version="0.1.0")


# hash -> momentul pornirii, ca interogările dese să nu lanseze generări duplicate
_comparison_inflight: dict[str, float] = {}


def _ensure_comparison() -> None:
    if ds.comparison_target().exists():
        return
    h   = ds.comparison_hash()
    now = time.time()
    t   = _comparison_inflight.get(h)
    if t and now - t < 120:
        return
    clients = ds.configured_clients()
    if not clients:
        return
    _comparison_inflight[h] = now
    ds.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    for old in ds.COMPARISON_DIR.glob("comparison_*.png"):
        if old.name != f"comparison_{h}.png":
            try: old.unlink()
            except OSError: pass
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.Popen(
            [sys.executable, "eda_compare_households.py",
             "--clients", *clients,
             "--out_dir", "outputs/flower/comparison",
             "--out_name", f"comparison_{h}"],
            cwd=str(ds.FL_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        _comparison_inflight.pop(h, None)


_persistence_inflight: dict[str, float] = {}


def _ensure_persistence() -> None:
    if ds.persistence_csv().exists():
        return
    h   = ds.comparison_hash()
    now = time.time()
    t   = _persistence_inflight.get(h)
    if t and now - t < 300:
        return
    clients = ds.configured_clients()
    if not clients:
        return
    _persistence_inflight[h] = now
    ds.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    for old in ds.COMPARISON_DIR.glob("persistence_*.csv"):
        if old.name != f"persistence_{h}.csv":
            try: old.unlink()
            except OSError: pass
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.Popen(
            [sys.executable, "compute_persistence.py",
             "--clients", *clients,
             "--out", f"outputs/flower/comparison/persistence_{h}.csv"],
            cwd=str(ds.FL_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        _persistence_inflight.pop(h, None)


_preslice_inflight: dict[str, float] = {}


def _ensure_presliced() -> None:
    h   = ds.comparison_hash()
    now = time.time()
    t   = _preslice_inflight.get(h)
    if t and now - t < 300:
        return
    clients = ds.configured_clients()
    if not clients:
        return
    _preslice_inflight[h] = now
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.Popen(
            [sys.executable, "preslice_data.py", "--clients", *clients],
            cwd=str(ds.FL_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        _preslice_inflight.pop(h, None)   # allow a retry; never block the API


_client_eda_inflight: dict[str, float] = {}


def _ensure_client_eda(lclid: str) -> None:
    """Generează în fundal figura EDA a unui client, dacă lipsește.

    run_eda nu antrenează nimic (doar încarcă + plotează), deci e ieftin și
    poate fi declanșat la prima deschidere a paginii de detalii.
    """
    if (ds.OUTPUTS / lclid / f"01_eda_{lclid}.png").exists():
        return
    if not ds.is_valid_client(lclid):
        return
    now = time.time()
    t   = _client_eda_inflight.get(lclid)
    if t and now - t < 300:
        return
    _client_eda_inflight[lclid] = now
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.Popen(
            [sys.executable, "compute_eda.py", "--clients", lclid],
            cwd=str(ds.FL_DIR), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        _client_eda_inflight.pop(lclid, None)   # permite o reîncercare


@app.on_event("startup")
def _on_startup() -> None:
    _ensure_comparison()
    _ensure_persistence()
    _ensure_presliced()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status":   "ok",
        "fl_dir":   str(ds.FL_DIR),
        "run_log":  str(ds.RUN_LOG),
        "has_log":  ds.RUN_LOG.exists(),
        "clients":  ds.configured_clients(),
    }


@app.get("/api/runs")
def get_runs() -> list[dict]:
    return ds.list_runs()


@app.get("/api/runs/{run_id}/rounds")
def get_run_rounds(run_id: str) -> list[dict]:
    rounds = ds.run_rounds(run_id)
    if not rounds:
        raise HTTPException(status_code=404, detail=f"No records for run_id={run_id}")
    return rounds


@app.get("/api/rounds/latest")
def get_latest_rounds() -> dict:
    rid = ds.latest_run_id()
    return {"run_id": rid, "records": ds.latest_rounds()}


@app.get("/api/runs/{run_id}/images")
def get_run_images(run_id: str) -> dict:
    return ds.run_images(run_id)


@app.get("/api/clients")
def get_clients() -> list[dict]:
    clients = ds.clients_status()
    proc_map = fl_control.client_process_map()
    for c in clients:
        c["process_up"] = proc_map.get(c["lclid"], c.get("process_up"))
    return clients



MAX_CLIENTS = 20


@app.get("/api/available-clients")
def get_available_clients() -> list[dict]:
    return ds.available_clients()


@app.get("/api/federation/clients")
def get_federation() -> dict:
    return {
        "clients":  ds.configured_clients(),
        "editable": not fl_control.status()["daemons_up"],
        "max":      MAX_CLIENTS,
    }


class FederationRequest(BaseModel):
    clients: list[str]


@app.post("/api/federation/clients")
def set_federation(req: FederationRequest) -> dict:
    if fl_control.status()["daemons_up"]:
        raise HTTPException(status_code=409,
                            detail="Stop the daemons before changing the client set.")
    seen: set[str] = set()
    clients = [c for c in (s.strip() for s in req.clients)
               if c and not (c in seen or seen.add(c))]
    if len(clients) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 clients for FL.")
    if len(clients) > MAX_CLIENTS:
        raise HTTPException(status_code=400,
                            detail=f"At most {MAX_CLIENTS} clients supported.")
    invalid = [c for c in clients if not ds.is_valid_client(c)]
    if invalid:
        raise HTTPException(status_code=400,
                            detail=f"Unknown LCLid(s): {', '.join(invalid)}")
    ds.set_configured_clients(clients)
    _ensure_comparison()
    _ensure_persistence()
    _ensure_presliced()
    return {"ok": True, "clients": clients}


@app.get("/api/clients/{lclid}/metrics")
def get_client_metrics(lclid: str) -> list[dict]:
    return ds.client_metrics(lclid)


@app.get("/api/clients/{lclid}/images")
def get_client_images(lclid: str) -> dict:
    _ensure_client_eda(lclid)
    return ds.client_images(lclid)


@app.get("/api/comparison")
def get_comparison() -> list[dict]:
    return ds.comparison()


@app.get("/api/comparison-live")
def get_comparison_live() -> dict:
    _ensure_persistence()
    return {"clients": ds.configured_clients(), "rows": ds.live_comparison()}


@app.get("/api/eda")
def get_eda() -> dict:
    _ensure_comparison()
    return {"images": ds.eda_images()}


@app.get("/api/stream")
async def stream(request: Request, run_id: str | None = None) -> StreamingResponse:
    return StreamingResponse(
        sse.event_stream(request, run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RunRequest(BaseModel):
    num_rounds: int = 20
    local_epochs: int = 5
    finetune_epochs: int = 10
    method: str = "perfedavg"     # fedavg | fedprox | perfedavg
    mu: float = 0.1               # coeficientul proximal FedProx
    device: str = "cpu"           # dispozitivul de antrenare al clientului


@app.get("/api/fl/status")
def fl_status() -> dict:
    return fl_control.status()


@app.post("/api/fl/up")
def fl_up() -> dict:
    return fl_control.up()


@app.post("/api/fl/down")
def fl_down() -> dict:
    return fl_control.down()


@app.post("/api/fl/run")
def fl_run(req: RunRequest) -> dict:
    return fl_control.run(req.num_rounds, req.local_epochs, req.finetune_epochs,
                          req.method, req.mu, req.device)


# servim arborele outputs/ (imagini EDA, antrenare, predicții, comparații)
if ds.OUTPUTS.exists():
    app.mount("/static", StaticFiles(directory=str(ds.OUTPUTS)), name="static")



_DIST = ds.FL_DIR / "app" / "frontend" / "dist"
if (_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        return FileResponse(str(_DIST / "index.html"))
