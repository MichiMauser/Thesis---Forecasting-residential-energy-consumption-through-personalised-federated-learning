

import contextlib
import copy
import io
import os
import random
import sys
import time
import argparse
import warnings
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from smart_meter_with_weather import (
    get_cfg, set_seed,
    load_household_info, load_household_series, clean_series,
    load_weather, add_time_features, make_splits,
    LSTMForecaster, build_model,
    evaluate, plot_predictions,
)
from fed_avg_baseline import (
    prepare_client_data,
    fed_avg_aggregate,
    eval_global_on_clients,
    plot_convergence,
)


CLIENTS = [
    "MAC003235",   # kurtosis = 0.4
    "MAC001514",   # kurtosis = 3.3
    "MAC005020",   # kurtosis = 5.4
    "MAC004428",   # kurtosis = 17.4
    "MAC000100",   # kurtosis = 14.0
    "MAC001000",
]

_FL_CFG = dict(
    num_rounds      = 20,
    local_epochs    = 5,
    client_fraction = 1.0,
    mu              = 0.1,   # coeficientul proximal; 0 = FedAvg simplu
    clients         = CLIENTS,
    data_dir        = "halfhourly_dataset",
    info_csv        = "informations_households.csv",
    weather_csv     = "weather_hourly_darksky.csv",
    out_dir         = "outputs/fedprox",
)


def local_train_prox(global_state: dict, client: dict,
                     cfg: dict) -> tuple[dict, int]:

    model = LSTMForecaster(
        input_size  = client["n_feat"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])
    model.load_state_dict({k: v.to(cfg["device"]) for k, v in global_state.items()})

    # copie înghețată a ponderilor globale, necesară termenului proximal
    global_weights = {k: v.to(cfg["device"]).detach()
                      for k, v in global_state.items()}

    model.train()
    loader    = DataLoader(client["train_ds"],
                           batch_size=cfg["batch_size"], shuffle=True)
    criterion = nn.HuberLoss(delta=cfg["huber_delta"])
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr           = cfg["lr"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        nesterov     = True,
    )

    mu = cfg["mu"]

    for _ in range(cfg["local_epochs"]):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(cfg["device"]), y_b.to(cfg["device"])
            optimiser.zero_grad()

            base_loss = criterion(model(X_b), y_b)

            # termenul proximal: penalizează abaterea de la modelul global
            prox = sum(
                torch.norm(p - global_weights[n]) ** 2
                for n, p in model.named_parameters()
                if n in global_weights
            )
            loss = base_loss + (mu / 2.0) * prox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimiser.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}, client["n_train_samples"]


def run_fedprox(fl_cfg: dict, base_cfg: dict) -> None:
    cfg = {**base_cfg, **fl_cfg}
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg["seed"])

    print("\n" + "=" * 72)
    print("  LONDON SMART METERS — FEDERATED LEARNING (FedProx)")
    print(f"  Clients      : {cfg['clients']}")
    print(f"  Rounds       : {cfg['num_rounds']}")
    print(f"  Local epochs : {cfg['local_epochs']}")
    print(f"  Mu (prox)    : {cfg['mu']}")
    print(f"  Device       : {cfg['device']}")
    print(f"  Output dir   : {cfg['out_dir']}/")
    print("=" * 72)

    print("\n[DATA] Preparing client datasets...")
    info_df = load_household_info(cfg["info_csv"])
    weather = load_weather(cfg["weather_csv"])

    clients_data = []
    for lclid in cfg["clients"]:
        print(f"\n  -- Client: {lclid}")
        c = prepare_client_data(lclid, cfg, info_df, weather)
        clients_data.append(c)

    n_feats = [c["n_feat"] for c in clients_data]
    assert len(set(n_feats)) == 1, \
        f"Feature count mismatch: {dict(zip(cfg['clients'], n_feats))}"
    n_feat = n_feats[0]
    print(f"\n[DATA] All {len(clients_data)} clients ready "
          f"| n_feat={n_feat} "
          f"| train sizes: {[c['n_train_samples'] for c in clients_data]}")

    global_model = build_model(n_feat, cfg)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    best_val_r2   = -999.0
    best_state    = copy.deepcopy(global_state)
    round_history = []
    t0            = time.time()

    print("\n" + "-" * 72)
    n_select = max(1, int(cfg["client_fraction"] * len(clients_data)))

    for rnd in range(1, cfg["num_rounds"] + 1):
        selected = random.sample(clients_data, n_select)

        local_states, sample_counts = [], []
        for c in selected:
            updated_state, n_samp = local_train_prox(global_state, c, cfg)
            local_states.append(updated_state)
            sample_counts.append(n_samp)

        # agregarea pe server e identică cu FedAvg
        global_state = fed_avg_aggregate(global_state, local_states, sample_counts)
        global_model.load_state_dict(
            {k: v.to(cfg["device"]) for k, v in global_state.items()})

        result  = eval_global_on_clients(global_model, clients_data, cfg,
                                          split="val", verbose=False)
        avg_m   = result["avg"]
        avg_r2  = avg_m["R2"]
        avg_mae = avg_m["MAE"]

        if avg_r2 > best_val_r2:
            best_val_r2 = avg_r2
            best_state  = copy.deepcopy(global_state)
            marker = "  ←"
        else:
            marker = ""

        print(f"  Round {rnd:>3d}/{cfg['num_rounds']} "
              f"| Avg Val R²={avg_r2:.4f}  MAE={avg_mae:.4f}  "
              f"SpikeMAE={avg_m['Spike_MAE']:.4f}"
              f"  [{time.time() - t0:.0f}s]{marker}")

        row = {
            "round":             rnd,
            "avg_val_r2":        avg_r2,
            "avg_val_mae":       avg_mae,
            "avg_val_rmse":      avg_m["RMSE"],
            "avg_val_mape":      avg_m["MAPE"],
            "avg_val_spike_mae": avg_m["Spike_MAE"],
        }
        for c in clients_data:
            cm = result["per_client"][c["lclid"]]
            row[f"{c['lclid']}_val_r2"]  = cm["R2"]
            row[f"{c['lclid']}_val_mae"] = cm["MAE"]
        round_history.append(row)

    print("\n" + "=" * 72)
    print("  FINAL TEST EVALUATION  (best global model by avg val R²)")
    print("=" * 72)

    global_model.load_state_dict(
        {k: v.to(cfg["device"]) for k, v in best_state.items()})

    ckpt_path = f"{cfg['out_dir']}/best_global_model.pt"
    torch.save(best_state, ckpt_path)
    print(f"[SAVE] Best global model -> {ckpt_path}  (best val R²={best_val_r2:.4f})\n")

    test_results = eval_global_on_clients(global_model, clients_data, cfg,
                                           split="test", verbose=True)
    for c in clients_data:
        loader = DataLoader(c["test_ds"], batch_size=cfg["batch_size"], shuffle=False)
        trues, preds, _ = evaluate(
            global_model, loader, c["tgt_scaler"], cfg,
            split_name=f"Test/{c['lclid']}")
        plot_predictions(trues, preds, c["lclid"], cfg)

    print(f"\n  {'Household':<14} {'R²':>7} {'MAE':>8} {'RMSE':>8} "
          f"{'MAPE':>8} {'SpikeMAE':>10}")
    print("  " + "-" * 62)
    for lclid, m in test_results["per_client"].items():
        print(f"  {lclid:<14} {m['R2']:>7.4f} {m['MAE']:>8.4f} "
              f"{m['RMSE']:>8.4f} {m['MAPE']:>7.2f}% {m['Spike_MAE']:>10.4f}")
    avg = test_results["avg"]
    print("  " + "-" * 62)
    print(f"  {'GLOBAL AVG':<14} {avg['R2']:>7.4f} {avg['MAE']:>8.4f} "
          f"{avg['RMSE']:>8.4f} {avg['MAPE']:>7.2f}% {avg['Spike_MAE']:>10.4f}")
    print("=" * 72)

    pd.DataFrame(round_history).to_csv(
        f"{cfg['out_dir']}/round_history.csv", index=False)
    print(f"[CSV] Round history    -> {cfg['out_dir']}/round_history.csv")

    ts         = dt.now().isoformat()
    final_rows = []
    for lclid, m in test_results["per_client"].items():
        final_rows.append({"household": lclid, "split": "test_fedprox",
                            "mu": cfg["mu"], **m, "timestamp": ts})
    final_rows.append({"household": "GLOBAL_AVG", "split": "test_fedprox",
                       "mu": cfg["mu"], **avg, "timestamp": ts})
    pd.DataFrame(final_rows).to_csv(
        f"{cfg['out_dir']}/final_metrics.csv", index=False)
    print(f"[CSV] Final metrics    -> {cfg['out_dir']}/final_metrics.csv")

    plot_convergence(round_history, cfg["out_dir"])
    print(f"\nDone. Total time: {(time.time() - t0) / 60:.1f} min")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedProx FL baseline — London Smart Meters")
    p.add_argument("--num_rounds",   type=int,   default=_FL_CFG["num_rounds"])
    p.add_argument("--local_epochs", type=int,   default=_FL_CFG["local_epochs"])
    p.add_argument("--mu",           type=float, default=_FL_CFG["mu"])
    p.add_argument("--clients",      nargs="+",  default=_FL_CFG["clients"],
                   help="Space-separated list of LCLids to use as clients")
    p.add_argument("--data_dir",     default=_FL_CFG["data_dir"])
    p.add_argument("--info_csv",     default=_FL_CFG["info_csv"])
    p.add_argument("--weather_csv",  default=_FL_CFG["weather_csv"])
    p.add_argument("--out_dir",      default=_FL_CFG["out_dir"])
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fl_cfg = {
        **_FL_CFG,
        "num_rounds":   args.num_rounds,
        "local_epochs": args.local_epochs,
        "mu":           args.mu,
        "clients":      args.clients,
        "data_dir":     args.data_dir,
        "info_csv":     args.info_csv,
        "weather_csv":  args.weather_csv,
        "out_dir":      args.out_dir,
    }
    base_cfg = get_cfg({"seed": args.seed})

    run_fedprox(fl_cfg, base_cfg)
