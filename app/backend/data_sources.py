

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

FL_DIR    = Path(__file__).resolve().parents[2]
OUTPUTS   = FL_DIR / "outputs"
FLOWER_DIR = OUTPUTS / "flower"
RUN_LOG   = FLOWER_DIR / "run_log.jsonl"
COMPARISON_DIR = FLOWER_DIR / "comparison"   # EDA comparativ pentru selecția curentă
CLIENTS_JSON = FLOWER_DIR / "clients.json"
INFO_CSV  = FL_DIR / "informations_households.csv"
COMPARISON_CSV = OUTPUTS / "comparison" / "all_metrics.csv"

sys.path.insert(0, str(FL_DIR))


DEFAULT_CLIENTS = ["MAC003235", "MAC001514", "MAC005020", "MAC004428", "MAC000100"]


def configured_clients() -> list[str]:
    try:
        data = json.loads(CLIENTS_JSON.read_text(encoding="utf-8"))
        cl = data.get("clients") if isinstance(data, dict) else data
        if isinstance(cl, list) and cl:
            return [str(x).strip() for x in cl if str(x).strip()]
    except Exception:
        pass
    return list(DEFAULT_CLIENTS)


def set_configured_clients(clients: list[str]) -> None:
    FLOWER_DIR.mkdir(parents=True, exist_ok=True)
    CLIENTS_JSON.write_text(
        json.dumps({"clients": list(clients)}, indent=1), encoding="utf-8")


_available_cache: Optional[list[dict]] = None


def available_clients() -> list[dict]:

    global _available_cache
    if _available_cache is None:
        df = _read_csv(INFO_CSV)
        if df.empty:
            _available_cache = []
        else:
            df.columns = [c.strip() for c in df.columns]
            rename = {"LCLid": "lclid", "Acorn": "acorn",
                      "Acorn_grouped": "acorn_grouped", "file": "file",
                      "stdorToU": "tariff"}
            present = {k: v for k, v in rename.items() if k in df.columns}
            sub = df[list(present)].rename(columns=present)
            recs = sub.astype(str).to_dict(orient="records")
            data_dir = FL_DIR / "halfhourly_dataset"
            blocks = {p.stem for p in data_dir.glob("*.csv")} if data_dir.exists() else set()
            if blocks:
                recs = [r for r in recs if str(r.get("file", "")).strip() in blocks]
            _available_cache = recs
    return _available_cache


def is_valid_client(lclid: str) -> bool:
    return any(c.get("lclid") == lclid for c in available_clients())


def read_run_log() -> list[dict]:
    if not RUN_LOG.exists():
        return []
    records: list[dict] = []
    with open(RUN_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def list_runs() -> list[dict]:
    records = read_run_log()
    runs: dict[str, dict] = {}
    for r in records:
        rid = r.get("run_id", "unknown")
        s = runs.setdefault(rid, {
            "run_id": rid, "rounds": 0, "clients": set(),
            "last_round": 0, "last_ts": None, "split": None, "method": None,
        })
        if r.get("method"):
            s["method"] = r["method"]
        rnd = r.get("round", 0) or 0
        s["last_round"] = max(s["last_round"], rnd)
        if r.get("scope") == "global":
            s["rounds"] = max(s["rounds"], rnd)
        lclid = r.get("lclid")
        if lclid and lclid != "__global__":
            s["clients"].add(lclid)
        if r.get("ts"):
            s["last_ts"] = r["ts"]
        if r.get("split"):
            s["split"] = r["split"]
    out = []
    for s in runs.values():
        s["clients"] = sorted(s["clients"])
        s["n_clients"] = len(s["clients"])
        out.append(s)
    out.sort(key=lambda s: s["last_ts"] or "", reverse=True)
    return out


def latest_run_id() -> Optional[str]:
    runs = list_runs()
    return runs[0]["run_id"] if runs else None


def run_rounds(run_id: str) -> list[dict]:
    return [r for r in read_run_log() if r.get("run_id") == run_id]


def latest_rounds() -> list[dict]:
    rid = latest_run_id()
    return run_rounds(rid) if rid else []


def clients_status(run_id: Optional[str] = None) -> list[dict]:
    """Starea fiecărui client, dedusă din jurnalul ultimei rulări."""
    rid = run_id or latest_run_id()
    recs = run_rounds(rid) if rid else []

    by_client: dict[str, dict] = {}
    for r in recs:
        lclid = r.get("lclid")
        if not lclid or lclid == "__global__" or r.get("scope") != "client":
            continue
        cur = by_client.get(lclid)
        if cur is None or (r.get("round", 0) or 0) >= cur.get("last_seen_round", -1):
            by_client[lclid] = {
                "lclid":           lclid,
                "last_seen_round": r.get("round"),
                "R2":              r.get("R2"),
                "MAE":             r.get("MAE"),
                "Spike_MAE":       r.get("Spike_MAE"),
                "drift":           r.get("drift"),
                "drift_rel":       r.get("drift_rel"),
            }

    # includem și clienții configurați care încă nu au apărut în această rulare
    configured = configured_clients()
    out = []
    for lclid in configured:
        entry = by_client.get(lclid, {"lclid": lclid, "last_seen_round": None})
        entry["configured"] = True
        entry["process_up"] = None   # completat de fl_control
        out.append(entry)
    # clienți văzuți în jurnal, dar absenți din lista configurată
    for lclid, entry in by_client.items():
        if lclid not in configured:
            entry["configured"] = False
            entry["process_up"] = None
            out.append(entry)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def comparison() -> list[dict]:
    df = _read_csv(COMPARISON_CSV)
    return df.to_dict(orient="records") if not df.empty else []


def client_metrics(lclid: str) -> list[dict]:
    df = _read_csv(COMPARISON_CSV)
    if df.empty or "household" not in df.columns:
        return []
    sub = df[df["household"].astype(str).str.strip() == lclid]
    return sub.to_dict(orient="records")


_METHOD_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "perfedavg": "PerFedAvg"}
_METRIC_KEYS   = ["MAE", "RMSE", "MAPE", "R2", "Spike_MAE"]


def persistence_csv() -> Path:
    return COMPARISON_DIR / f"persistence_{comparison_hash()}.csv"


def live_comparison() -> list[dict]:

    clients = set(configured_clients())
    rows: list[dict] = []

    pcsv = persistence_csv()
    if pcsv.exists():
        for r in _read_csv(pcsv).to_dict(orient="records"):
            if str(r.get("household")) in clients:
                rows.append({"household": r["household"], "method": "Persistence",
                             **{k: r.get(k) for k in _METRIC_KEYS}})

    test_rows = [r for r in read_run_log()
                 if r.get("scope") == "client" and r.get("split") == "test"]
    latest: dict[str, tuple[str, str]] = {}     # metodă -> (run_id, ts)
    for r in test_rows:
        m = str(r.get("method", "")).lower()
        if m not in _METHOD_LABELS:
            continue
        ts = r.get("ts", "")
        if m not in latest or ts > latest[m][1]:
            latest[m] = (r.get("run_id"), ts)

    for m, (rid, _ts) in latest.items():
        for r in test_rows:
            if (str(r.get("method", "")).lower() == m and r.get("run_id") == rid
                    and str(r.get("lclid")) in clients):
                rows.append({"household": r.get("lclid"), "method": _METHOD_LABELS[m],
                             "R2": r.get("R2"), "MAE": r.get("MAE"),
                             "RMSE": r.get("RMSE"), "Spike_MAE": r.get("Spike_MAE"),
                             "MAPE": None})
        # metricile personalizate PerFedAvg stau ca JSON în folderul rulării
        if m == "perfedavg" and rid:
            for jf in (FLOWER_DIR / "runs" / f"{rid}_perfedavg").glob("personalised_*.json"):
                try:
                    d = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(d.get("household")) in clients:
                    rows.append({"household": d["household"], "method": "PerFedAvg-Personal",
                                 "R2": d.get("R2"), "MAE": d.get("MAE"),
                                 "RMSE": d.get("RMSE"), "Spike_MAE": d.get("Spike_MAE"),
                                 "MAPE": d.get("MAPE")})
    return rows


def _rel(p: Path) -> Optional[str]:
    return str(p.relative_to(OUTPUTS)).replace("\\", "/") if p.exists() else None


def client_images(lclid: str) -> dict:
    cdir = OUTPUTS / lclid
    return {
        "eda":         _rel(cdir / f"01_eda_{lclid}.png"),
        "training":    _rel(cdir / f"02_training_{lclid}.png"),
        "predictions": _rel(cdir / f"03_predictions_{lclid}.png"),
    }


def comparison_hash(clients: Optional[list[str]] = None) -> str:
    """Identificator scurt și stabil al unui set de clienți."""
    cl = clients if clients is not None else configured_clients()
    return hashlib.md5(",".join(sorted(cl)).encode()).hexdigest()[:8]


def comparison_target() -> Path:
    """Calea figurii de comparație pentru selecția curentă."""
    return COMPARISON_DIR / f"comparison_{comparison_hash()}.png"


def eda_images() -> list[dict]:
    """Figura de comparație între gospodării, pentru selecția curentă.

    Numele fișierului este legat de hash-ul selecției, deci un rezultat gol
    înseamnă „încă negenerată pentru această selecție”: interfața afișează o
    stare de așteptare și reinterogează.
    """
    p = comparison_target()
    rel = _rel(p)
    if not rel:
        return []
    return [{"name": p.name, "title": ", ".join(configured_clients()), "url": rel}]


def run_images(run_id: str) -> dict:

    base = FLOWER_DIR / "runs"
    # folderul se numește "<run_id>_<metodă>": căutăm după prefixul run_id, ca
    # să funcționeze și pentru foldere vechi, numite doar cu run_id
    dirs   = sorted(base.glob(f"{run_id}*")) if base.exists() else []
    rdir   = dirs[0] if dirs else base / run_id
    method = rdir.name[len(run_id):].lstrip("_") or None
    items  = []
    for lclid in configured_clients():
        glob_png = _rel(rdir / f"03_predictions_{lclid}.png")
        pers_png = _rel(rdir / f"03_predictions_{lclid}_personalised.png")
        if glob_png or pers_png:
            items.append({"lclid": lclid,
                          "predictions": glob_png,
                          "personalised": pers_png})
    return {"run_id": run_id, "method": method, "clients": items}
