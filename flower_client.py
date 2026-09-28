import contextlib
import io
import json
import os
import sys
from pathlib import Path

#aceste variabile trebuie setate ÎNAINTE de importul bibliotecilor
# native, altfel fiecare proces rezervă arene pentru toate firele mașinii
# (~1.5 GB pe Windows). Rulează câte un proces per client, simultan, deci N
# astfel de procese epuizează memoria la mijlocul rundei.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

# OMP_NUM_THREADS acoperă doar pool-ul intra-op; cel inter-op se dimensionează
# separat și trebuie fixat înainte de prima regiune paralelă.
_threads = max(1, int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_threads(_threads)
with contextlib.suppress(RuntimeError):
    torch.set_num_interop_threads(_threads)

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

sys.path.insert(0, str(Path(__file__).parent))
from smart_meter_with_weather import (
    get_cfg, set_seed,
    load_household_info, load_weather,
    build_model, evaluate, plot_predictions,
)
from fed_avg_baseline import prepare_client_data, local_train
from fed_prox_baseline import local_train_prox


class SmartMeterClient(NumPyClient):
    def __init__(self, lclid: str, cfg: dict, out_dir: str):
        self.lclid   = lclid
        self.cfg     = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        info_df    = load_household_info(cfg["info_csv"])
        weather    = load_weather(cfg["weather_csv"])
        self.data  = prepare_client_data(lclid, cfg, info_df, weather)
        self.model = build_model(self.data["n_feat"], cfg)

        print(f"[{lclid}] Client ready | "
              f"n_feat={self.data['n_feat']} | "
              f"train={self.data['n_train_samples']} samples")

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def _set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = {
            k: torch.tensor(p)
            for k, p in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def _params_to_state_dict(self, parameters: list[np.ndarray]) -> dict:
        return {
            k: torch.tensor(p)
            for k, p in zip(self.model.state_dict().keys(), parameters)
        }

    def _run_dir(self) -> str:
        """Folderul cu imaginile produse de rularea curentă, afișate în interfață.

        Numele conține și run_id, și metoda, ca o rulare să nu suprascrie alta
        chiar dacă run_id se repetă.
        """
        run_id = str(self.cfg.get("run_id", "adhoc"))
        method = str(self.cfg.get("method", "perfedavg"))
        d = os.path.join(self.out_dir, "runs", f"{run_id}_{method}")
        os.makedirs(d, exist_ok=True)
        return d

    def fit(self, parameters: list[np.ndarray], config: dict):

        local_epochs = int(config.get("local_epochs", self.cfg["local_epochs"]))
        method       = str(config.get("method", self.cfg.get("method", "perfedavg"))).lower()
        mu           = float(config.get("mu", self.cfg.get("mu", 0.1)))
        fit_cfg      = {**self.cfg, "local_epochs": local_epochs, "mu": mu}

        global_state = self._params_to_state_dict(parameters)
        train_fn     = local_train_prox if method == "fedprox" else local_train
        updated_state, n_train = train_fn(global_state, self.data, fit_cfg)

        self.model.load_state_dict(
            {k: v.to(self.cfg["device"]) for k, v in updated_state.items()})

        # lclid travels with the update so the server can attribute weight drift.
        return self.get_parameters(config={}), n_train, {"lclid": self.lclid}

    def evaluate(self, parameters: list[np.ndarray], config: dict):
        """
        La ultima rundă (finetune=True) rulează și personalizarea Per-FedAvg.
        """
        self._set_parameters(parameters)

        split  = config.get("split", "val")
        ds     = self.data["val_ds"] if split == "val" else self.data["test_ds"]
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=False)

        with contextlib.redirect_stdout(io.StringIO()):
            trues, preds, metrics = evaluate(
                self.model, loader, self.data["tgt_scaler"], self.cfg)

        print(f"[{self.lclid}] {split:>4s} | "
              f"R2={metrics['R2']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  "
              f"SpikeMAE={metrics['Spike_MAE']:.4f}")

        if config.get("save_predictions", False) and split == "test":
            run_cfg = {**self.cfg, "out_dir": self._run_dir()}
            with contextlib.redirect_stdout(io.StringIO()):
                plot_predictions(trues, preds, self.lclid, run_cfg)
            print(f"[{self.lclid}] Saved predictions PNG -> "
                  f"{self._run_dir()}/03_predictions_{self.lclid}.png")

        if config.get("finetune", False):
            self._run_finetune(
                global_parameters = parameters,
                global_metrics    = metrics,
                finetune_epochs   = int(config.get("finetune_epochs", 10)),
            )

        return float(metrics["MAE"]), len(ds), {
            "R2":        float(metrics["R2"]),
            "MAE":       float(metrics["MAE"]),
            "RMSE":      float(metrics["RMSE"]),
            "Spike_MAE": float(metrics["Spike_MAE"]),
            # weighted_average() ignoră cheile ne-numerice
            "lclid":     self.lclid,
            "split":     split,
        }

    def _run_finetune(
        self,
        global_parameters: list[np.ndarray],
        global_metrics: dict,
        finetune_epochs: int,
    ) -> None:
        """Personalizează o copie a modelului global pe datele acestui client."""
        ft_cfg   = {**self.cfg, "local_epochs": finetune_epochs}
        ft_state = self._params_to_state_dict(global_parameters)
        ft_state, _ = local_train(ft_state, self.data, ft_cfg)

        ft_model = build_model(self.data["n_feat"], self.cfg)
        ft_model.load_state_dict(
            {k: v.to(self.cfg["device"]) for k, v in ft_state.items()})

        loader = DataLoader(
            self.data["test_ds"], batch_size=self.cfg["batch_size"], shuffle=False)

        with contextlib.redirect_stdout(io.StringIO()):
            _, _, g_m = evaluate(
                self.model, loader, self.data["tgt_scaler"], self.cfg)
            p_trues, p_preds, p_m = evaluate(
                ft_model, loader, self.data["tgt_scaler"], self.cfg)

        dr2    = p_m["R2"]        - g_m["R2"]
        dspike = p_m["Spike_MAE"] - g_m["Spike_MAE"]

        print(f"\n[{self.lclid}] -- FINAL TEST ------------------------------------")
        print(f"  Global       R2={g_m['R2']:.4f}  "
              f"MAE={g_m['MAE']:.4f}  SpikeMAE={g_m['Spike_MAE']:.4f}")
        print(f"  Personalised R2={p_m['R2']:.4f}  "
              f"MAE={p_m['MAE']:.4f}  SpikeMAE={p_m['Spike_MAE']:.4f}  "
              f"dR2={dr2:+.4f}  dSpikeMAE={dspike:+.4f}")

        run_dir   = self._run_dir()
        ckpt_path = f"{run_dir}/personalised_{self.lclid}.pt"
        torch.save(ft_state, ckpt_path)
        with contextlib.redirect_stdout(io.StringIO()):
            plot_predictions(p_trues, p_preds,
                             f"{self.lclid}_personalised",
                             {**self.cfg, "out_dir": run_dir})
        with open(f"{run_dir}/personalised_{self.lclid}.json", "w", encoding="utf-8") as f:
            json.dump({"household": self.lclid,
                       "R2": p_m["R2"], "MAE": p_m["MAE"], "RMSE": p_m["RMSE"],
                       "MAPE": p_m.get("MAPE"), "Spike_MAE": p_m["Spike_MAE"]}, f)
        print(f"[{self.lclid}] Saved -> {ckpt_path}")


def client_fn(context: Context) -> SmartMeterClient:
    """Apelat de SuperNode o dată per conexiune.

    context.node_config vine din --node-config; context.run_config din
    pyproject.toml, secțiunea [tool.flwr.app.config].
    """
    lclid   = str(context.node_config["lclid"])
    out_dir = str(context.run_config.get("out-dir", "outputs/flower"))
    run_id  = str(context.run_config.get("run-id", "")) or "adhoc"

    device = str(context.run_config.get("device", "cpu")).lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg = get_cfg({
        "data_dir":     str(context.run_config.get("data-dir",     "halfhourly_dataset")),
        "info_csv":     str(context.run_config.get("info-csv",     "informations_households.csv")),
        "weather_csv":  str(context.run_config.get("weather-csv",  "weather_hourly_darksky.csv")),
        "out_dir":      out_dir,
        "run_id":       run_id,
        "method":       str(context.run_config.get("method", "perfedavg")).lower(),
        "mu":           float(context.run_config.get("mu",   0.1)),
        "device":       device,
        "local_epochs": int(context.run_config.get("local-epochs", 5)),
        "seed":         int(context.run_config.get("seed",         42)),
    })
    set_seed(cfg["seed"])

    return SmartMeterClient(lclid, cfg, out_dir).to_client()


app = ClientApp(client_fn=client_fn)
