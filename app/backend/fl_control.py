from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

from . import data_sources as ds

FL_DIR  = ds.FL_DIR
PIDFILE = FL_DIR / "outputs" / "flower" / "daemons.pid"
PYTHON  = sys.executable
LAUNCHER = "start_flower.py"


def _child_env() -> dict:

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env

_run_proc: Optional[subprocess.Popen] = None
_run_id:   Optional[str] = None


def _running_pids() -> set[int]:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        pids = set()
        for line in out.splitlines():
            cols = line.split('","')
            if len(cols) >= 2:
                try:
                    pids.add(int(cols[1].strip('"')))
                except ValueError:
                    pass
        return pids
    import os
    pids = set()
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pids.add(int(entry.name))
        except ValueError:
            pass
    return pids


def _read_pidfile() -> list[int]:
    if not PIDFILE.exists():
        return []
    return [int(x) for x in PIDFILE.read_text(encoding="utf-8").split() if x.strip()]


def up() -> dict:
    proc = subprocess.run(
        [PYTHON, LAUNCHER, "up"],
        cwd=str(FL_DIR), capture_output=True, text=True, timeout=120,
        env=_child_env(),
    )
    return {"ok": proc.returncode == 0, "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-1000:], **status()}


def down() -> dict:
    global _run_proc, _run_id
    proc = subprocess.run(
        [PYTHON, LAUNCHER, "down"],
        cwd=str(FL_DIR), capture_output=True, text=True, timeout=60,
        env=_child_env(),
    )
    _run_proc, _run_id = None, None
    return {"ok": proc.returncode == 0, "stdout": proc.stdout[-2000:], **status()}


def run(num_rounds: int = 20, local_epochs: int = 5,
        finetune_epochs: int = 10, method: str = "perfedavg",
        mu: float = 0.1, device: str = "cpu") -> dict:

    global _run_proc, _run_id
    if _run_proc is not None and _run_proc.poll() is None:
        return {"ok": False, "error": "a run is already in progress",
                "run_id": _run_id, **status()}
    if not _read_pidfile():
        return {"ok": False, "error": "daemons not running — call /api/fl/up first",
                **status()}

    method = str(method).lower()
    if method not in ("fedavg", "fedprox", "perfedavg"):
        return {"ok": False, "error": f"unknown method: {method}", **status()}

    device = str(device).lower()
    if device not in ("cpu", "cuda", "auto"):
        return {"ok": False, "error": f"unknown device: {device}", **status()}


    n_clients = len(ds.configured_clients())

    run_id = dt.now().strftime("%Y%m%d_%H%M%S")
    _run_proc = subprocess.Popen(
        [PYTHON, LAUNCHER, "run",
         "--method", method,
         "--mu", str(mu),
         "--device", device,
         "--min_clients", str(n_clients),
         "--num_rounds", str(num_rounds),
         "--local_epochs", str(local_epochs),
         "--finetune_epochs", str(finetune_epochs),
         "--run_id", run_id],
        cwd=str(FL_DIR),
        env=_child_env(),
    )
    _run_id = run_id
    return {"ok": True, "run_id": run_id, "method": method, "device": device,
            "n_clients": n_clients, **status()}


def client_process_map() -> dict[str, Optional[bool]]:
    clients = ds.configured_clients()
    pids = _read_pidfile()
    if not pids:
        return {c: None for c in clients}
    running = _running_pids()
    node_pids = pids[1:]   # pids[0] este SuperLink
    out: dict[str, Optional[bool]] = {}
    for i, lclid in enumerate(clients):
        out[lclid] = (node_pids[i] in running) if i < len(node_pids) else None
    return out


def status() -> dict:
    pids    = _read_pidfile()
    running = _running_pids() if pids else set()
    node_pids = pids[1:] if pids else []
    daemons_up = bool(node_pids) and any(p in running for p in node_pids)
    run_active = _run_proc is not None and _run_proc.poll() is None
    return {
        "daemons_up":     daemons_up,
        "superlink_pid":  pids[0] if pids else None,
        "supernode_pids": node_pids,
        "run_active":     run_active,
        "run_id":         _run_id if run_active else None,
        "latest_run_id":  ds.latest_run_id(),
    }
