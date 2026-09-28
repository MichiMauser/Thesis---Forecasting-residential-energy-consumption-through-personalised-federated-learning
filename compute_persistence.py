

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from smart_meter_with_weather import (
    get_cfg, set_seed, load_household_info, load_weather, evaluate,
    PersistenceForecaster,
)
from fed_avg_baseline import prepare_client_data


def main() -> int:
    p = argparse.ArgumentParser(description="Linia de referință prin persistență.")
    p.add_argument("--clients", nargs="+", required=True)
    p.add_argument("--data_dir",    default="halfhourly_dataset")
    p.add_argument("--info_csv",    default="informations_households.csv")
    p.add_argument("--weather_csv", default="weather_hourly_darksky.csv")
    p.add_argument("--out",         required=True, help="Output CSV path.")
    args = p.parse_args()

    cfg = get_cfg({"data_dir": args.data_dir, "info_csv": args.info_csv,
                   "weather_csv": args.weather_csv})
    set_seed(cfg["seed"])
    info    = load_household_info(args.info_csv)
    weather = load_weather(args.weather_csv)
    model   = PersistenceForecaster().to(cfg["device"])

    rows = []
    for lclid in args.clients:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                d = prepare_client_data(lclid, cfg, info, weather)
                loader = DataLoader(d["test_ds"], batch_size=cfg["batch_size"],
                                    shuffle=False)
                _, _, m = evaluate(model, loader, d["tgt_scaler"], cfg)
            rows.append({"household": lclid, "method": "Persistence",
                         "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"],
                         "R2": m["R2"], "Spike_MAE": m["Spike_MAE"]})
            print(f"[persistence] {lclid}: R2={m['R2']:.4f} MAE={m['MAE']:.4f}")
        except Exception as e:
            print(f"[persistence] {lclid}: skipped ({e})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[persistence] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
