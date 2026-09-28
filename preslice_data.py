import argparse
import json
import sys
from pathlib import Path

from smart_meter_with_weather import load_household_info, _preslice_block

HERE         = Path(__file__).parent
CLIENTS_JSON = HERE / "outputs" / "flower" / "clients.json"


def _load_clients() -> list[str]:
    try:
        data = json.loads(CLIENTS_JSON.read_text(encoding="utf-8"))
        cl = data.get("clients") if isinstance(data, dict) else data
        if isinstance(cl, list) and cl:
            return [str(x).strip() for x in cl if str(x).strip()]
    except Exception:
        pass
    return []


def preslice_clients(clients: list[str], data_dir: str, info_csv: str,
                     presliced_dir: str) -> int:

    info = load_household_info(info_csv)
    pdir = Path(presliced_dir)

    blocks_done: set[str] = set()
    for lclid in clients:
        if (pdir / f"{lclid}.csv").exists():
            continue                                   # deja în cache
        row = info[info["LCLid"] == lclid]
        if row.empty:
            print(f"[PRESLICE] Skip unknown LCLid: {lclid}")
            continue
        block_file = str(row["file"].values[0]).strip()
        if block_file in blocks_done:
            continue                                   # un client anterior a feliat-o
        block_path = Path(data_dir) / f"{block_file}.csv"
        if not block_path.exists():
            block_path = Path(data_dir) / block_file
        if not block_path.exists():
            print(f"[PRESLICE] Missing block for {lclid}: {block_path}")
            continue
        _preslice_block(block_path, pdir)
        blocks_done.add(block_file)

    print(f"[PRESLICE] Done - read {len(blocks_done)} block(s) for "
          f"{len(clients)} client(s); cache at {pdir}/")
    return len(blocks_done)


def main() -> int:
    p = argparse.ArgumentParser(description="Warm the presliced/ per-household cache.")
    p.add_argument("--clients", nargs="+", default=None,
                   help="LCLids to preslice (default: read clients.json).")
    p.add_argument("--data_dir",      default="halfhourly_dataset")
    p.add_argument("--info_csv",      default="informations_households.csv")
    p.add_argument("--presliced_dir", default="presliced")
    args = p.parse_args()

    clients = args.clients or _load_clients()
    if not clients:
        print("[PRESLICE] No clients to preslice.")
        return 0
    preslice_clients(clients, args.data_dir, args.info_csv, args.presliced_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
