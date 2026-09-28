

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime as dt
from pathlib import Path


DEFAULT_CLIENTS = [
    "MAC003235",
    "MAC001514",
    "MAC005020",
    "MAC004428",
    "MAC000100",
]

FLEET_API_PORT        = 9092   # aici se conectează SuperNode-urile
SERVERAPP_API_PORT    = 9093   # aici se conectează flwr run
CLIENTAPPIO_BASE_PORT = 9094   # fiecare SuperNode primește portul lui: 9094, 9095, ...

HERE         = Path(__file__).parent
PIDFILE      = HERE / "outputs" / "flower" / "daemons.pid"
DAEMON_LOG   = HERE / "outputs" / "flower" / "daemons.log"
CLIENTS_JSON = HERE / "outputs" / "flower" / "clients.json"


def load_clients() -> list[str]:

    try:
        data = json.loads(CLIENTS_JSON.read_text(encoding="utf-8"))
        cl = data.get("clients") if isinstance(data, dict) else data
        if isinstance(cl, list) and cl:
            return [str(x).strip() for x in cl if str(x).strip()]
    except Exception:
        pass
    return list(DEFAULT_CLIENTS)


def _open_daemon_log():

    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    return open(DAEMON_LOG, "a", encoding="utf-8", errors="replace")


def _thread_cap(n_clients: int) -> int:

    physical = (os.cpu_count() or 4) // 2 or 1
    return max(1, physical // max(1, n_clients))


def _child_env(threads: int | None = None) -> dict:

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads)
    return env


def start_superlink() -> subprocess.Popen:
    print("[LAUNCHER] Starting SuperLink ...")
    proc = subprocess.Popen([
        "flower-superlink",
        "--insecure",
        f"--fleet-api-address=0.0.0.0:{FLEET_API_PORT}",
        f"--control-api-address=0.0.0.0:{SERVERAPP_API_PORT}",
    ], env=_child_env(), stdout=_open_daemon_log(), stderr=subprocess.STDOUT)
    time.sleep(3)   # așteptăm ca SuperLink să ocupe porturile
    return proc


def start_supernodes(clients: list[str] | None = None) -> list[subprocess.Popen]:
    if clients is None:
        clients = load_clients()
    # procesele clientapp moștenesc acest mediu, deci limita ajunge exact la
    # procesele care importă torch și antrenează
    threads = _thread_cap(len(clients))
    print(f"[LAUNCHER] {len(clients)} clients -> {threads} compute thread(s) each")
    env = _child_env(threads)
    procs = []
    for i, lclid in enumerate(clients):
        clientappio_port = CLIENTAPPIO_BASE_PORT + i
        print(f"[LAUNCHER] Starting SuperNode: {lclid} "
              f"(clientappio port {clientappio_port})")
        p = subprocess.Popen([
            "flower-supernode",
            f"--superlink=127.0.0.1:{FLEET_API_PORT}",
            f"--clientappio-api-address=0.0.0.0:{clientappio_port}",
            f'--node-config=lclid="{lclid}"',
            "--insecure",
        ], env=env, stdout=_open_daemon_log(), stderr=subprocess.STDOUT)
        procs.append(p)
    time.sleep(5)
    return procs


def write_pidfile(procs: list[subprocess.Popen]) -> None:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text("\n".join(str(p.pid) for p in procs), encoding="utf-8")
    print(f"[LAUNCHER] Daemon PIDs -> {PIDFILE}")


def read_pidfile() -> list[int]:
    if not PIDFILE.exists():
        return []
    return [int(x) for x in PIDFILE.read_text(encoding="utf-8").split() if x.strip()]


def stop_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.kill(pid, 9)
            print(f"[LAUNCHER] Stopped PID {pid}")
        except Exception as e:
            print(f"[LAUNCHER] Could not stop PID {pid}: {e}")


def _completion_in_log(log_path: Path, num_rounds: int, run_id: str) -> bool:

    if not log_path.exists():
        return False
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("run_id") == run_id
                        and rec.get("scope") == "global"
                        and rec.get("round") == num_rounds):
                    return True
    except OSError:
        pass
    return False


def trigger_run(args) -> int:

    log_path = Path(args.out_dir) / "run_log.jsonl"
    # apelantul poate impune run_id, ca să îl știe dinainte; fără două puncte,
    # altfel run-config îl respinge
    run_id   = getattr(args, "run_id", None) or dt.now().strftime("%Y%m%d_%H%M%S")

    min_clients = args.min_clients if args.min_clients else len(load_clients())

    print(f"[LAUNCHER] Triggering run id={run_id} method={args.method} "
          f"(rounds={args.num_rounds}, local_epochs={args.local_epochs}, "
          f"min_clients={min_clients}) ...")
    run_proc = subprocess.Popen([
        "flwr", "run", ".", "local-deployment",
        "--stream",
        "--run-config",
        (
            f'method="{args.method}" '
            f"mu={args.mu} "
            f'device="{args.device}" '
            f"min-clients={min_clients} "
            f"num-rounds={args.num_rounds} "
            f"local-epochs={args.local_epochs} "
            f"finetune-epochs={args.finetune_epochs} "
            f'out-dir="{args.out_dir}" '
            f'run-id="{run_id}"'
        ),
    ], env=_child_env())

    t0     = time.time()
    status = "timeout"
    while time.time() - t0 < args.timeout:
        if _completion_in_log(log_path, args.num_rounds, run_id):
            status = "done"
            break
        if run_proc.poll() is not None:
            time.sleep(2)
            status = "done" if _completion_in_log(log_path, args.num_rounds, run_id) else "exited"
            break
        time.sleep(2.0)

    elapsed = time.time() - t0
    if status == "done":
        print(f"[LAUNCHER] Run complete ({elapsed:.0f}s) — final round "
              f"{args.num_rounds} logged to {log_path}")
        time.sleep(2)
    elif status == "exited":
        print(f"[LAUNCHER] flwr run exited after {elapsed:.0f}s but no final "
              f"round in log — check output above.")
    else:
        print(f"[LAUNCHER] Timeout after {elapsed:.0f}s waiting for round "
              f"{args.num_rounds}.")

    # oprim driverul: pe Windows `flwr run --stream` se blochează la închidere
    if run_proc.poll() is None:
        run_proc.terminate()
        try:
            run_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            run_proc.kill()

    return 0 if status == "done" else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flower 2.x FL launcher — London Smart Meters")
    p.add_argument("mode", nargs="?", default="oneshot",
                   choices=["oneshot", "up", "run", "down"],
                   help="oneshot (default): up->run->down. "
                        "up/down: manage persistent daemons. "
                        "run: trigger a run on already-running daemons.")
    p.add_argument("--method",          default="perfedavg",
                   choices=["fedavg", "fedprox", "perfedavg"],
                   help="FL method: fedavg | fedprox | perfedavg (default).")
    p.add_argument("--mu",              type=float, default=0.1,
                   help="FedProx proximal coefficient (ignored by other methods).")
    p.add_argument("--device",          default="cpu",
                   choices=["cpu", "cuda", "auto"],
                   help="Client training device: cpu (default) | cuda | auto.")
    p.add_argument("--min_clients",     type=int, default=None,
                   help="Clients the server waits for (default: size of clients.json).")
    p.add_argument("--num_rounds",      type=int, default=20)
    p.add_argument("--local_epochs",    type=int, default=5)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--out_dir",         default="outputs/flower")
    p.add_argument("--run_id",          default=None,
                   help="Impose a run_id (else a timestamp is generated).")
    p.add_argument("--timeout",         type=int, default=1800,
                   help="Max seconds to wait for a run to finish (default 1800).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "down":
        pids = read_pidfile()
        if not pids:
            print("[LAUNCHER] No daemon PIDs recorded — nothing to stop.")
            return 0
        stop_pids(pids)
        PIDFILE.unlink(missing_ok=True)
        return 0

    if args.mode == "up":
        superlink = start_superlink()
        nodes     = start_supernodes()
        write_pidfile([superlink, *nodes])
        print("[LAUNCHER] Daemons running. Trigger a run with: "
              "python start_flower.py run")
        print("[LAUNCHER] Stop them with: python start_flower.py down")
        return 0

    if args.mode == "run":
        if not read_pidfile():
            print("[LAUNCHER] No running daemons found (no PID file). "
                  "Start them first with: python start_flower.py up")
            return 1
        return trigger_run(args)

    # oneshot (implicit): pornește tot, rulează o dată, oprește tot
    superlink = start_superlink()
    nodes     = start_supernodes()
    try:
        rc = trigger_run(args)
    finally:
        print("[LAUNCHER] Tearing down daemons ...")

        stop_pids([p.pid for p in [superlink, *nodes]])

    print(f"\n[LAUNCHER] Done (rc={rc}). Outputs in {args.out_dir}/")
    return rc


if __name__ == "__main__":
    sys.exit(main())
