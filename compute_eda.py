

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from smart_meter_with_weather import (
    get_cfg, set_seed, load_household_info,
    load_household_series, clean_series, run_eda,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generează figura EDA (01_eda_<lclid>.png) per gospodărie.")
    p.add_argument("--clients", nargs="+", required=True)
    p.add_argument("--data_dir",    default="halfhourly_dataset")
    p.add_argument("--info_csv",    default="informations_households.csv")
    p.add_argument("--out_dir",     default="outputs")
    args = p.parse_args()

    base = get_cfg({"data_dir": args.data_dir, "info_csv": args.info_csv})
    set_seed(base["seed"])
    info = load_household_info(args.info_csv)

    made = 0
    for lclid in args.clients:
        dest = Path(args.out_dir) / lclid / f"01_eda_{lclid}.png"
        if dest.exists():
            print(f"[eda] {lclid}: already exists -> {dest}")
            continue
        try:
            # out_dir per gospodărie, ca run_eda să scrie exact unde caută UI-ul
            cfg = get_cfg({"data_dir": args.data_dir, "info_csv": args.info_csv,
                           "out_dir": os.path.join(args.out_dir, lclid)})
            os.makedirs(cfg["out_dir"], exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                hh_raw = load_household_series(cfg["data_dir"], lclid, info, cfg)
                hh     = clean_series(hh_raw, cfg)
                run_eda(hh, lclid, cfg)
            made += 1
            print(f"[eda] {lclid}: wrote {dest}")
        except Exception as e:
            print(f"[eda] {lclid}: skipped ({e})")

    print(f"[eda] generated {made} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
