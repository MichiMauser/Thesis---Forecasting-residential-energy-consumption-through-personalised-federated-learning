

import json
import math
import os
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import (
    Context, EvaluateRes, FitRes, Metrics, Parameters, Scalar,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    numeric_keys = [
        k for k, v in metrics[0][1].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in numeric_keys}


def _params_l2(params: List[np.ndarray]) -> float:
    """Norma L2 a unei liste de parametri, aplatizată peste toți tensorii."""
    return math.sqrt(sum(float(np.sum(np.square(p))) for p in params))


def _params_l2_diff(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    """Norma L2 a diferenței (a - b), aplatizată peste toți tensorii."""
    return math.sqrt(sum(float(np.sum(np.square(x - y))) for x, y in zip(a, b)))


class SmartMeterFedAvg(FedAvg):


    def __init__(self, num_rounds: int, out_dir: str,
                 run_id: str, method: str = "perfedavg", **kwargs):
        super().__init__(**kwargs)
        self.num_rounds  = num_rounds
        self.out_dir     = out_dir
        self.run_id      = run_id
        self.method      = method
        self.best_val_r2 = -999.0
        self.best_params: Optional[List] = None
        self._cur_params: Optional[List] = None   # reținut din aggregate_fit
        self._global_sent: Optional[List] = None  # ponderile trimise clienților
        self._round_drift: Dict[str, dict] = {}   # lclid -> {drift, drift_rel}
        os.makedirs(out_dir, exist_ok=True)

        self.log_path = os.path.join(out_dir, "run_log.jsonl")

    def _log(self, record: dict) -> None:

        record["run_id"] = self.run_id
        record["method"] = self.method
        record["ts"]     = dt.now().isoformat()
        clean = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in record.items()
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean) + "\n")

    def configure_fit(self, server_round, parameters, client_manager):
        # exact ponderile de la care vor porni clienții în această rundă
        self._global_sent = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # deriva se calculează ÎNAINTE de agregare, altfel ar fi mereu zero
        drift: Dict[str, dict] = {}
        if self._global_sent is not None:
            g       = self._global_sent
            g_norm  = _params_l2(g)
            for proxy, fitres in results:
                lclid = str(fitres.metrics.get("lclid", proxy.cid))
                local = parameters_to_ndarrays(fitres.parameters)
                d     = _params_l2_diff(local, g)
                drift[lclid] = {
                    "drift":     d,
                    "drift_rel": (d / g_norm) if g_norm > 0 else float("nan"),
                }
        self._round_drift = drift

        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self._cur_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        drifts_seen: List[float] = []
        for proxy, eres in results:
            m     = eres.metrics or {}
            lclid = str(m.get("lclid", proxy.cid))
            split = str(m.get("split", "val"))
            dr    = self._round_drift.get(lclid, {})
            if "drift" in dr and not math.isnan(dr.get("drift", float("nan"))):
                drifts_seen.append(dr["drift"])

            self._log({
                "round":     server_round,
                "lclid":     lclid,
                "split":     split,
                "scope":     "client",
                "R2":        float(m.get("R2",        float("nan"))),
                "MAE":       float(m.get("MAE",       float("nan"))),
                "RMSE":      float(m.get("RMSE",      float("nan"))),
                "Spike_MAE": float(m.get("Spike_MAE", float("nan"))),
                "loss":      float(eres.loss),
                "n_eval":    int(eres.num_examples),
                "drift":     dr.get("drift",     float("nan")),
                "drift_rel": dr.get("drift_rel", float("nan")),
            })

        if metrics:
            r2    = metrics.get("R2",        float("nan"))
            mae   = metrics.get("MAE",       float("nan"))
            spike = metrics.get("Spike_MAE", float("nan"))
            marker = ""

            if r2 > self.best_val_r2 and self._cur_params is not None:
                self.best_val_r2 = r2
                self.best_params = self._cur_params
                torch.save(self.best_params,
                           f"{self.out_dir}/best_global_params.pt")
                marker = "  <--"

            self._log({
                "round":      server_round,
                "lclid":      "__global__",
                "scope":      "global",
                "R2":         float(r2),
                "MAE":        float(mae),
                "RMSE":       float(metrics.get("RMSE", float("nan"))),
                "Spike_MAE":  float(spike),
                "loss":       float(loss) if loss is not None else float("nan"),
                "n_clients":  len(results),
                "drift_mean": float(np.mean(drifts_seen)) if drifts_seen else float("nan"),
                "drift_max":  float(np.max(drifts_seen))  if drifts_seen else float("nan"),
                "drift_std":  float(np.std(drifts_seen))  if drifts_seen else float("nan"),
            })

            print(f"  [SERVER] Round {server_round:>3d}/{self.num_rounds} "
                  f"| Avg Val R2={r2:.4f}  MAE={mae:.4f}  "
                  f"SpikeMAE={spike:.4f}"
                  f"  drift(mean/max)="
                  f"{(np.mean(drifts_seen) if drifts_seen else float('nan')):.3f}/"
                  f"{(np.max(drifts_seen) if drifts_seen else float('nan')):.3f}"
                  f"{marker}")

        return loss, metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Apelat o singură dată de flwr run, înainte de runde.

    Toți hiperparametrii vin din pyproject.toml, secțiunea
    [tool.flwr.app.config], deci acest fișier nu trebuie editat.
    """
    run_cfg         = context.run_config
    method          = str(run_cfg.get("method",          "perfedavg")).lower()
    mu              = float(run_cfg.get("mu",             0.1))
    num_rounds      = int(run_cfg.get("num-rounds",      20))
    local_epochs    = int(run_cfg.get("local-epochs",    5))
    finetune_epochs = int(run_cfg.get("finetune-epochs", 10))
    min_clients     = int(run_cfg.get("min-clients",     5))
    out_dir         = str(run_cfg.get("out-dir",         "outputs/flower"))
    run_id          = str(run_cfg.get("run-id", "")) \
                      or dt.now().strftime("%Y%m%d_%H%M%S")

    def on_fit_config(server_round: int) -> Dict:
        return {"local_epochs": local_epochs, "round": server_round,
                "method": method, "mu": mu}

    def on_evaluate_config(server_round: int) -> Dict:
        if server_round < num_rounds:
            return {"split": "val", "round": server_round}

        return {
            "split":           "test",
            "round":           server_round,
            "method":          method,
            "finetune":        method == "perfedavg",
            "finetune_epochs": finetune_epochs,
            "save_predictions": True,
        }

    strategy = SmartMeterFedAvg(
        num_rounds                      = num_rounds,
        out_dir                         = out_dir,
        run_id                          = run_id,
        method                          = method,
        fraction_fit                    = 1.0,
        fraction_evaluate               = 1.0,
        min_fit_clients                 = min_clients,
        min_evaluate_clients            = min_clients,
        min_available_clients           = min_clients,
        on_fit_config_fn                = on_fit_config,
        on_evaluate_config_fn           = on_evaluate_config,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    print(f"\n{'=' * 60}")
    print(f"  Flower FL Server — London Smart Meters")
    print(f"  Method         : {method}" + (f" (mu={mu})" if method == "fedprox" else ""))
    print(f"  Rounds         : {num_rounds}")
    print(f"  Min clients    : {min_clients}")
    print(f"  Local epochs   : {local_epochs}")
    print(f"  Finetune epochs: {finetune_epochs}")
    print(f"  Out dir        : {out_dir}/")
    print(f"  Run id         : {run_id}")
    print(f"  Run log        : {out_dir}/run_log.jsonl (append-only)")
    print(f"{'=' * 60}\n")

    return ServerAppComponents(
        strategy = strategy,
        config   = ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
