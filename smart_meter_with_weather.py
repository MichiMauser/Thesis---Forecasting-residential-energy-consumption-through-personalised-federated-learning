"""Regimul Local: un model LSTM antrenat separat pentru fiecare gospodărie.

Conține și pipeline-ul partajat (încărcare, curățare, features, split-uri,
antrenare, evaluare) pe care îl importă regimurile federate din fed_avg_baseline,
fed_prox_baseline și flower_client.

Date  : Kaggle "Smart meters in London" — halfhourly_dataset/block_*.csv,
        informations_households.csv, weather_hourly_darksky.csv (același folder).
Model : LSTM unidirecțional, SGD cu moment Nesterov, pierdere Huber.
Țintă : log1p(energie), pentru a comprima distribuția puternic asimetrică.
Metrici: MAE, RMSE, MAPE (mascat sub 0.1 kWh), R2, Spike_MAE.

Utilizare:
    python smart_meter_with_weather.py                        # cele 6 gospodării
    python smart_meter_with_weather.py --lclid MAC000100      # una singură
    python smart_meter_with_weather.py --lclid MAC000100 --epochs 100
"""

import os
import copy
import time
import random
import datetime
import warnings
import argparse
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import holidays as hol_lib                 # pip install holidays

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


_BASE_CFG = dict(
    # Date
    target_col          = "energy(kWh/hh)",
    window_size         = 48,       # 48 x 30 min = fereastră de 24 h
    horizon             = 1,        # prezice următorul interval de 30 min
    train_ratio         = 0.70,
    val_ratio           = 0.15,

    # Model
    input_size          = 1,        # actualizat la runtime din numărul de features
    hidden_size         = 128,
    num_layers          = 1,
    dropout             = 0.25,

    # Pierdere
    huber_delta         = 0.5,

    # Antrenare
    epochs              = 80,
    batch_size          = 32,
    lr                  = 0.07721572721502946,
    momentum            = 0.9,
    weight_decay        = 1e-4,
    grad_clip           = 1.0,
    early_stop_patience = 20,

    # Programarea ratei de învățare (ReduceLROnPlateau)
    lr_factor           = 0.5,
    lr_patience         = 8,
    lr_min              = 1e-5,
    lr_threshold        = 1e-3,

    # Sub acest prag (kWh) MAPE devine instabil, deci punctul e mascat
    mape_threshold      = 0.1,

    seed                = 42,
    device              = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir             = "outputs",
)


def get_cfg(overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(_BASE_CFG)
    if overrides:
        cfg.update(overrides)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_household_info(info_csv: str) -> pd.DataFrame:
    df = pd.read_csv(info_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] Household metadata: {df.shape[0]} rows | "
          f"columns: {list(df.columns)}")
    return df


def _preslice_block(block_path: Path, presliced_dir: Path) -> None:
    """Citește un bloc (~70 MB) o singură dată și îl taie în CSV-uri per gospodărie.

    """
    print(f"[PRESLICE] Reading {block_path} once -> {presliced_dir}/ (first use) ...")
    raw_block = pd.read_csv(block_path, low_memory=False)
    raw_block.columns = [c.strip() for c in raw_block.columns]
    presliced_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for hh_id, g in raw_block.groupby("LCLid"):
        dest = presliced_dir / f"{hh_id}.csv"
        if dest.exists():
            continue
        tmp = dest.with_suffix(".csv.tmp")
        g.to_csv(tmp, index=False)
        os.replace(tmp, dest)   # atomic pe Windows și POSIX
        n_written += 1
    print(f"[PRESLICE] Wrote {n_written} household slice(s) from {block_path.name}")


def load_household_series(data_dir: str, lclid: str,
                           info_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    row = info_df[info_df["LCLid"] == lclid]
    if row.empty:
        raise ValueError(f"LCLid '{lclid}' not found in household info.")

    # Cache de felii: un CSV mic per gospodărie, generat la prima utilizare.
    presliced_dir = Path(cfg.get("presliced_dir", Path(data_dir).parent / "presliced"))
    slice_path    = presliced_dir / f"{lclid}.csv"

    if not slice_path.exists():
        block_file = row["file"].values[0].strip()
        block_path = Path(data_dir) / f"{block_file}.csv"
        if not block_path.exists():
            block_path = Path(data_dir) / block_file
        _preslice_block(block_path, presliced_dir)
        if not slice_path.exists():
            raise ValueError(
                f"LCLid '{lclid}' not found in block {block_path.name} after preslicing.")

    print(f"[INFO] Loading presliced {slice_path} for {lclid} ...")
    raw = pd.read_csv(slice_path, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]

    hh = raw[raw["LCLid"] == lclid].copy()
    hh["tstp"] = pd.to_datetime(hh["tstp"], errors="coerce")
    hh = hh.dropna(subset=["tstp"]).sort_values("tstp").reset_index(drop=True)
    hh[cfg["target_col"]] = pd.to_numeric(hh[cfg["target_col"]], errors="coerce")

    print(f"[INFO] Raw series: {len(hh):,} readings | "
          f"{hh['tstp'].min().date()} -> {hh['tstp'].max().date()}")
    return hh


def clean_series(hh: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col = cfg["target_col"]
    hh  = hh.set_index("tstp")
    full_idx = pd.date_range(hh.index.min(), hh.index.max(), freq="30min")
    hh = hh.reindex(full_idx)

    missing_before = hh[col].isna().sum()
    q999    = hh[col].quantile(0.999)
    hh[col] = hh[col].clip(lower=0, upper=q999)
    hh[col] = (
        hh[col]
        .interpolate(method="time", limit=4)
        .ffill()
        .bfill()
    )
    missing_after = hh[col].isna().sum()
    print(f"[CLEAN] Missing: {missing_before} -> {missing_after} | "
          f"Total timesteps: {len(hh):,}")
    return hh.reset_index().rename(columns={"index": "tstp"})


def load_weather(weather_csv: str) -> pd.DataFrame | None:
    """Încarcă datele meteo orare și le reeșantionează la 30 min prin forward-fill.
    Întoarce None dacă fișierul lipsește: pipeline-ul rulează și fără meteo.
    """
    if not Path(weather_csv).exists():
        print(f"[WEATHER] File not found: {weather_csv} — skipping weather features")
        return None

    w = pd.read_csv(weather_csv)
    w.columns = [c.strip() for c in w.columns]

    time_col = "time" if "time" in w.columns else w.columns[0]
    w["tstp"] = pd.to_datetime(w[time_col], errors="coerce")
    w = w.dropna(subset=["tstp"]).sort_values("tstp").reset_index(drop=True)

    keep = ["tstp", "temperature", "apparentTemperature",
            "humidity", "windSpeed", "pressure", "cloudCover"]
    keep = [c for c in keep if c in w.columns]
    w = w[keep]

    w = w.set_index("tstp")
    full_idx = pd.date_range(w.index.min(), w.index.max(), freq="30min")
    w = w.reindex(full_idx).ffill().bfill()

    w = w.reset_index().rename(columns={"index": "tstp"})
    print(f"[WEATHER] Loaded {len(w):,} half-hourly rows | "
          f"features: {[c for c in w.columns if c != 'tstp']}")
    return w


class PersistenceForecaster(nn.Module):
    def forward(self, x):                      # x: (batch, fereastră, n_feat)
        return x[:, -1, 0]


def persistence_baseline(test_loader: DataLoader,
                         target_scaler: MinMaxScaler,
                         cfg: dict) -> dict:

    model = PersistenceForecaster().to(cfg["device"])
    _, _, m = evaluate(model, test_loader, target_scaler, cfg,
                       "Persistență (naivă, split de test)")
    return m


def run_eda(hh: pd.DataFrame, lclid: str, cfg: dict) -> None:
    col = cfg["target_col"]
    ts  = hh.set_index("tstp")[col]

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"EDA  -  Household {lclid}",
                 fontsize=16, fontweight="bold", y=1.01)

    ax1 = fig.add_subplot(4, 2, (1, 2))
    ax1.plot(ts.index, ts.values, lw=0.6, color="steelblue", alpha=0.8)
    ax1.set_title("Full Half-Hourly Energy Consumption")
    ax1.set_ylabel("kWh / half-hour")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2 = fig.add_subplot(4, 2, 3)
    ts.plot.hist(bins=60, ax=ax2, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.axvline(ts.mean(),   color="crimson", lw=1.5, ls="--",
                label=f"Mean={ts.mean():.3f}")
    ax2.axvline(ts.median(), color="orange",  lw=1.5, ls="--",
                label=f"Median={ts.median():.3f}")
    ax2.set_title("Distribution of Energy Consumption")
    ax2.set_xlabel("kWh / half-hour")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(4, 2, 4)
    monthly = ts.resample("ME").mean()
    ax3.bar(monthly.index, monthly.values,
            width=20, color="teal", alpha=0.8, edgecolor="white")
    ax3.set_title("Monthly Average Consumption")
    ax3.set_ylabel("Mean kWh / half-hour")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40, ha="right")

    ax4 = fig.add_subplot(4, 2, 5)
    hh_copy = hh.copy()
    hh_copy["hour"] = hh_copy["tstp"].dt.hour + hh_copy["tstp"].dt.minute / 60
    hourly = hh_copy.groupby("hour")[col].mean()
    ax4.plot(hourly.index, hourly.values, marker="o", ms=3, color="darkorange")
    ax4.set_title("Average Load Profile (by Hour of Day)")
    ax4.set_xlabel("Hour of Day"); ax4.set_ylabel("Mean kWh")
    ax4.set_xticks(range(0, 25, 2))

    ax5 = fig.add_subplot(4, 2, 6)
    hh_copy["dow"] = hh_copy["tstp"].dt.dayofweek
    dow = hh_copy.groupby("dow")[col].mean()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_colors = ["#4C72B0"] * 5 + ["#DD8452"] * 2
    ax5.bar(day_labels, dow.values, color=day_colors, edgecolor="white", alpha=0.9)
    ax5.set_title("Average Consumption by Day of Week")
    ax5.set_ylabel("Mean kWh")

    ax6 = fig.add_subplot(4, 2, (7, 8))
    roll = ts.rolling(window=48 * 7)
    ax6.plot(ts.index, ts.values, lw=0.4, color="steelblue",
             alpha=0.5, label="Observed")
    ax6.plot(ts.index, roll.mean().values, lw=1.2,
             color="crimson", label="7-day Rolling Mean")
    ax6.fill_between(ts.index,
                     roll.mean() - roll.std(),
                     roll.mean() + roll.std(),
                     color="crimson", alpha=0.15, label="+/-1 Std")
    ax6.set_title("Observed vs. 7-day Rolling Mean +/- Std")
    ax6.set_ylabel("kWh / half-hour"); ax6.legend(fontsize=8)
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = f"{cfg['out_dir']}/01_eda_{lclid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved -> {path}")

    desc = ts.describe()
    print("\n-- Summary Statistics ----------------------------------------------")
    print(desc.to_string())
    print(f"  Skewness : {ts.skew():.4f}")
    print(f"  Kurtosis : {ts.kurtosis():.4f}")
    print("--------------------------------------------------------------------\n")


def add_time_features(hh: pd.DataFrame, cfg: dict,
                      weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construiește features: calendar ciclic, sărbători, lag-uri, statistici
    glisante și, opțional, meteo."""
    col = cfg["target_col"]
    df  = hh.copy()

    df["hour"]       = df["tstp"].dt.hour
    df["minute"]     = df["tstp"].dt.minute
    df["dow"]        = df["tstp"].dt.dayofweek
    df["month"]      = df["tstp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(float)

    df["hour_sin"]  = np.sin(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # sărbătorile produc vârfuri de consum atipice
    uk_holidays = hol_lib.country_holidays("GB", subdiv="England")
    df["is_holiday"] = df["tstp"].dt.date.apply(
        lambda d: 1.0 if d in uk_holidays else 0.0)

    # și ziua dinaintea unei sărbători are alt tipar (lumea pleacă mai devreme)
    df["is_holiday_eve"] = df["tstp"].dt.date.apply(
        lambda d: 1.0
        if (d + datetime.timedelta(days=1)) in uk_holidays
        else 0.0)

    # lag 48 = aceeași oră ziua trecută, lag 336 = aceeași oră săptămâna trecută
    for lag in [1, 2, 48, 336]:
        df[f"lag_{lag}"] = df[col].shift(lag)

    df["roll_mean_48"] = df[col].rolling(48).mean()
    df["roll_std_48"]  = df[col].rolling(48).std()
    df["roll_max_48"]  = df[col].rolling(48).max()   # reține vârfurile recente

    if weather is not None:
        df = df.merge(weather, on="tstp", how="left")

        w_cols = ["temperature", "apparentTemperature",
                  "humidity", "windSpeed", "pressure", "cloudCover"]
        w_cols = [c for c in w_cols if c in df.columns]
        df[w_cols] = df[w_cols].ffill().bfill()

        # 15.5°C este pragul standard de încălzire folosit în UK.
        if "temperature" in df.columns:
            df["hdd"] = np.maximum(0.0, 15.5 - df["temperature"])
            df["cdd"] = np.maximum(0.0, df["temperature"] - 22.0)

        added = [c for c in w_cols + ["hdd", "cdd"] if c in df.columns]
        print(f"[WEATHER] Merged {len(added)} weather features: {added}")
    else:
        print("[WEATHER] No weather data — running without weather features")

    df = df.dropna().reset_index(drop=True)
    print(f"[FEAT] {df.shape[1]} columns, {len(df):,} rows after feature engineering")
    return df


class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int, horizon: int = 1):
        xs, ys = [], []
        for i in range(len(data) - window - horizon + 1):
            xs.append(data[i : i + window])
            ys.append(data[i + window + horizon - 1, 0])
        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32)

    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# Definite o singură dată, ca toate regimurile (local / federat / centralizat)
FEAT_COLS = [
    "energy(kWh/hh)",   # ținta: mereu coloana 0 (vezi PersistenceForecaster)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
    "is_holiday", "is_holiday_eve",
    "lag_1", "lag_2", "lag_48", "lag_336",
    "roll_mean_48", "roll_std_48", "roll_max_48",
    # meteo: prezente doar dacă fișierul meteo a fost încărcat
    "temperature", "apparentTemperature",
    "humidity", "windSpeed", "hdd", "cdd",
]


def build_feature_matrix(df: pd.DataFrame, cfg: dict):

    col       = cfg["target_col"]
    feat_cols = [c for c in FEAT_COLS if c in df.columns]
    if col not in feat_cols:
        feat_cols = [col] + [c for c in feat_cols if c != col]
    data = df[feat_cols].values.copy()
    data[:, 0] = np.log1p(data[:, 0])   # log1p(0)=0, deci zerourile sunt sigure
    return data, feat_cols


def chronological_split(data: np.ndarray, cfg: dict):
    n     = len(data)
    n_tr  = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    return (data[:n_tr],
            data[n_tr : n_tr + n_val],
            data[n_tr + n_val :])


def make_splits(df: pd.DataFrame, cfg: dict):

    data, feat_cols = build_feature_matrix(df, cfg)
    train_raw, val_raw, test_raw = chronological_split(data, cfg)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_sc = scaler.fit_transform(train_raw)
    val_sc   = scaler.transform(val_raw)
    test_sc  = scaler.transform(test_raw)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_raw[:, [0]])

    W, H = cfg["window_size"], cfg["horizon"]
    train_ds = SlidingWindowDataset(train_sc, W, H)
    val_ds   = SlidingWindowDataset(val_sc,   W, H)
    test_ds  = SlidingWindowDataset(test_sc,  W, H)

    print(f"[SPLIT] Train={len(train_ds):,} | Val={len(val_ds):,} | "
          f"Test={len(test_ds):,} | Features={len(feat_cols)}")
    return train_ds, val_ds, test_ds, scaler, target_scaler, feat_cols


class LSTMForecaster(nn.Module):
    """LSTM unidirecțional cu conexiune reziduală peste prognoza de persistență.

    Stratul liniar prezice doar o corecție (delta) adăugată peste ultima valoare
    cunoscută, x[:, -1, 0]. Astfel modelul poate egala persistența punând delta=0
    pe perioadele plate, în loc să reînvețe de la zero nivelul consumului.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        delta  = self.fc(self.drop(out[:, -1, :])).squeeze(-1)
        return x[:, -1, 0] + delta


def build_model(n_features: int, cfg: dict) -> nn.Module:
    model = LSTMForecaster(
        input_size  = n_features,
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    ).to(cfg["device"])

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] LSTM Forecaster | params: {total:,}")
    print(model)
    return model


def _mape(y_true: np.ndarray, y_pred: np.ndarray,
          threshold: float = 0.1, eps: float = 1e-8) -> float:
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    cfg: dict) -> dict:
    spike_thr  = np.percentile(y_true, 95)
    spike_mask = y_true >= spike_thr
    spike_mae  = (float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]))
                  if spike_mask.sum() > 0 else float("nan"))
    return dict(
        MAE      = float(mean_absolute_error(y_true, y_pred)),
        RMSE     = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAPE     = _mape(y_true, y_pred, cfg["mape_threshold"]),
        R2       = float(r2_score(y_true, y_pred)),
        Spike_MAE = spike_mae,   # MAE pe cele mai mari 5% consumuri
    )


def skill_score(model_mae: float, persistence_mae: float) -> float:
    """Câștig relativ față de persistență: >0 înseamnă mai bun decât naivul."""
    return 1.0 - (model_mae / persistence_mae)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                train_ds: Dataset,
                val_ds: Dataset,
                target_scaler: MinMaxScaler,
                cfg: dict) -> dict:
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    criterion = nn.HuberLoss(delta=cfg["huber_delta"])
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr           = cfg["lr"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        nesterov     = True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode      = "min",
        factor    = cfg["lr_factor"],
        patience  = cfg["lr_patience"],
        min_lr    = cfg["lr_min"],
        threshold = cfg["lr_threshold"],
    )

    history = dict(train_loss=[], val_loss=[],
                   val_mae=[], val_rmse=[], val_mape=[], val_r2=[], lr=[])

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    t0             = time.time()

    print(f"\n[TRAIN] device={cfg['device']} | epochs={cfg['epochs']} "
          f"| lr={cfg['lr']} | batch={cfg['batch_size']} "
          f"| huber_delta={cfg['huber_delta']}")
    print("-" * 72)

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(cfg["device"]), y_b.to(cfg["device"])
            optimiser.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimiser.step()
            epoch_loss += loss.item() * len(y_b)
        epoch_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(cfg["device"]), y_b.to(cfg["device"])
                pred      = model(X_b)
                val_loss += criterion(pred, y_b).item() * len(y_b)
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_b.cpu().numpy())
        val_loss /= len(val_ds)

        # metricile se raportează în kWh reali: se inversează MinMax, apoi log1p
        preds_kw = np.expm1(target_scaler.inverse_transform(
            np.concatenate(all_preds).reshape(-1, 1))).ravel()
        trues_kw = np.expm1(target_scaler.inverse_transform(
            np.concatenate(all_true).reshape(-1, 1))).ravel()
        vm = compute_metrics(trues_kw, preds_kw, cfg)

        current_lr = optimiser.param_groups[0]["lr"]
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(vm["MAE"])
        history["val_rmse"].append(vm["RMSE"])
        history["val_mape"].append(vm["MAPE"])
        history["val_r2"].append(vm["R2"])
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{cfg['epochs']} "
                  f"| Train={epoch_loss:.5f} | Val={val_loss:.5f} "
                  f"| MAE={vm['MAE']:.4f} | RMSE={vm['RMSE']:.4f} "
                  f"| MAPE={vm['MAPE']:.2f}% | R2={vm['R2']:.4f} "
                  f"| LR={current_lr:.6f} | {time.time() - t0:.0f}s")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["early_stop_patience"]:
                print(f"\n[TRAIN] Early stopping at epoch {epoch}")
                break

    print(f"[TRAIN] Best val loss: {best_val_loss:.6f}")
    model.load_state_dict(best_state)
    ckpt = f"{cfg['out_dir']}/best_model.pt"
    torch.save(best_state, ckpt)
    print(f"[TRAIN] Checkpoint -> {ckpt}")
    return history


def plot_training(history: dict, lclid: str, cfg: dict) -> None:
    xs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training History  -  {lclid}", fontsize=14, fontweight="bold")

    def _p(ax, y, title, ylabel, color):
        ax.plot(xs, y, color=color, lw=1.5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)

    axes[0, 0].plot(xs, history["train_loss"], label="Train", color="steelblue")
    axes[0, 0].plot(xs, history["val_loss"],   label="Val",   color="crimson")
    axes[0, 0].set_title("Huber Loss"); axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")

    _p(axes[0, 1], history["lr"],       "Learning Rate (log)", "LR",    "purple")
    axes[0, 1].set_yscale("log")
    _p(axes[0, 2], history["val_mae"],  "Val MAE (kWh)",       "MAE",   "teal")
    _p(axes[1, 0], history["val_rmse"], "Val RMSE (kWh)",      "RMSE",  "darkorange")
    _p(axes[1, 1], history["val_mape"], "Val MAPE (%)",        "MAPE%", "forestgreen")
    _p(axes[1, 2], history["val_r2"],   "Val R2",              "R2",    "darkred")

    plt.tight_layout()
    path = f"{cfg['out_dir']}/02_training_{lclid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Training history -> {path}")


def evaluate(model: nn.Module, loader: DataLoader,
             target_scaler: MinMaxScaler,
             cfg: dict, split_name: str = "Test"):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            all_preds.append(model(X_b.to(cfg["device"])).cpu().numpy())
            all_true.append(y_b.numpy())

    preds_kw = np.expm1(target_scaler.inverse_transform(
        np.concatenate(all_preds).reshape(-1, 1))).ravel()
    trues_kw = np.expm1(target_scaler.inverse_transform(
        np.concatenate(all_true).reshape(-1, 1))).ravel()

    m = compute_metrics(trues_kw, preds_kw, cfg)
    print(f"\n-- {split_name} Results ------------------------------------------")
    print(f"  MAE       = {m['MAE']:.5f} kWh/hh")
    print(f"  RMSE      = {m['RMSE']:.5f} kWh/hh")
    print(f"  MAPE      = {m['MAPE']:.3f} %")
    print(f"  R2        = {m['R2']:.5f}")
    print(f"  Spike_MAE = {m['Spike_MAE']:.5f} kWh/hh  (top-5% actuals)")
    print("------------------------------------------------------------------\n")
    return trues_kw, preds_kw, m


def plot_predictions(trues: np.ndarray, preds: np.ndarray,
                     lclid: str, cfg: dict) -> None:
    residuals = trues - preds
    n_show    = min(336, len(trues))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Test-Set Predictions  -  {lclid}",
                 fontsize=14, fontweight="bold")

    axes[0, 0].plot(trues[:2000], lw=0.7, color="steelblue",
                    alpha=0.8, label="Actual")
    axes[0, 0].plot(preds[:2000], lw=0.7, color="crimson",
                    alpha=0.7, label="Predicted")
    axes[0, 0].set_title("Actual vs Predicted (first 2000 steps)")
    axes[0, 0].set_xlabel("Half-hour step"); axes[0, 0].set_ylabel("kWh")
    axes[0, 0].legend()

    axes[0, 1].plot(trues[:n_show], lw=1.2, color="steelblue", label="Actual")
    axes[0, 1].plot(preds[:n_show], lw=1.2, color="crimson",
                    ls="--", label="Predicted")
    axes[0, 1].set_title(f"Zoom: First {n_show // 48} days")
    axes[0, 1].set_xlabel("Half-hour step"); axes[0, 1].set_ylabel("kWh")
    axes[0, 1].legend()

    lim = max(trues.max(), preds.max())
    axes[1, 0].scatter(trues, preds, s=2, alpha=0.3, color="teal")
    axes[1, 0].plot([0, lim], [0, lim], "r--", lw=1, label="Perfect forecast")
    axes[1, 0].set_title("Parity Plot: Actual vs Predicted")
    axes[1, 0].set_xlabel("Actual kWh"); axes[1, 0].set_ylabel("Predicted kWh")
    axes[1, 0].legend()

    axes[1, 1].hist(residuals, bins=80, color="darkorange",
                    edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(0, color="black", lw=1.5, ls="--")
    axes[1, 1].axvline(residuals.mean(), color="crimson", lw=1.5, ls="--",
                       label=f"Mean={residuals.mean():.4f}")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].set_xlabel("Residual (kWh)"); axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    plt.tight_layout()
    path = f"{cfg['out_dir']}/03_predictions_{lclid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Predictions -> {path}")


def plot_metrics_summary(val_m: dict, test_m: dict, persist_m: dict,
                         lclid: str, cfg: dict) -> None:
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    colors  = ["#888888", "#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Persistence vs Val vs Test  -  {lclid}",
                 fontsize=13, fontweight="bold")

    for ax, m in zip(axes, metrics):
        vals = [persist_m[m], val_m[m], test_m[m]]
        bars = ax.bar(["Persist", "Val", "Test"], vals,
                      color=colors, width=0.45, edgecolor="white")
        ax.set_title(m); ax.set_ylabel(m)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v * 1.02, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = f"{cfg['out_dir']}/04_metrics_summary_{lclid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Metrics summary -> {path}")


def save_results_csv(persist_m: dict, val_m: dict, test_m: dict,
                     lclid: str, cfg: dict) -> None:
    ts   = dt.now().isoformat()
    rows = [
        {"household": lclid, "split": "persistence", **persist_m, "timestamp": ts},
        {"household": lclid, "split": "val",         **val_m,     "timestamp": ts},
        {"household": lclid, "split": "test",        **test_m,    "timestamp": ts},
    ]
    path = f"{cfg['out_dir']}/metrics_{lclid}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[CSV] Metrics -> {path}")


def run_household(data_dir: str, info_csv: str, lclid: str,
                  cfg: dict,
                  weather_csv: str = "weather_hourly_darksky.csv") -> dict:
    cfg["out_dir"] = os.path.join(cfg["out_dir"], lclid)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg["seed"])

    print("\n" + "=" * 72)
    print("  LONDON SMART METERS  -  LOCAL (PER-CLIENT) BASELINE")
    print(f"  Household : {lclid}")
    print(f"  Device    : {cfg['device']}")
    print(f"  Output dir: {cfg['out_dir']}/")
    print("=" * 72)

    info_df = load_household_info(info_csv)
    hh_raw  = load_household_series(data_dir, lclid, info_df, cfg)
    hh      = clean_series(hh_raw, cfg)

    run_eda(hh, lclid, cfg)

    weather = load_weather(weather_csv)
    hh_feat = add_time_features(hh, cfg, weather=weather)

    (train_ds, val_ds, test_ds,
     scaler, tgt_scaler, feat_cols) = make_splits(hh_feat, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    persist_m = persistence_baseline(test_loader, tgt_scaler, cfg)

    cfg["input_size"] = len(feat_cols)
    model = build_model(len(feat_cols), cfg)

    history = train_model(model, train_loader, train_ds, val_ds, tgt_scaler, cfg)
    plot_training(history, lclid, cfg)

    _, _, val_m = evaluate(model, val_loader, tgt_scaler, cfg, "Validation")
    test_true, test_pred, test_m = evaluate(
        model, test_loader, tgt_scaler, cfg, "Test")

    plot_predictions(test_true, test_pred, lclid, cfg)
    plot_metrics_summary(val_m, test_m, persist_m, lclid, cfg)
    save_results_csv(persist_m, val_m, test_m, lclid, cfg)

    sk = skill_score(test_m["MAE"], persist_m["MAE"])
    print(f"[SKILL] vs persistence: {sk:.4f} "
          f"({'better' if sk > 0 else 'WORSE'} than naive)")

    print(f"\nDone: {lclid} -> {cfg['out_dir']}/")
    return {**test_m, "skill_score": sk, "persist_mae": persist_m["MAE"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="London Smart Meters - Local (per-client) Baseline")
    p.add_argument("--data_dir",    default="halfhourly_dataset")
    p.add_argument("--info_csv",    default="informations_households.csv")
    p.add_argument("--weather_csv", default="weather_hourly_darksky.csv")
    p.add_argument("--lclid",       default=None)
    p.add_argument("--out_dir",     default="outputs")
    p.add_argument("--epochs",      type=int,   default=_BASE_CFG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=_BASE_CFG["batch_size"])
    p.add_argument("--lr",          type=float, default=_BASE_CFG["lr"])
    p.add_argument("--hidden_size", type=int,   default=_BASE_CFG["hidden_size"])
    p.add_argument("--num_layers",  type=int,   default=_BASE_CFG["num_layers"])
    p.add_argument("--window_size", type=int,   default=_BASE_CFG["window_size"])
    p.add_argument("--huber_delta", type=float, default=_BASE_CFG["huber_delta"])
    p.add_argument("--seed",        type=int,   default=_BASE_CFG["seed"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cli_overrides = {
        "out_dir":     args.out_dir,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "hidden_size": args.hidden_size,
        "num_layers":  args.num_layers,
        "window_size": args.window_size,
        "huber_delta": args.huber_delta,
        "seed":        args.seed,
    }

    households = [args.lclid] if args.lclid else \
        ["MAC000100", "MAC005020", "MAC001000", "MAC001514", "MAC003235", "MAC004428"]

    all_results = {}
    for lclid in households:
        cfg = get_cfg(cli_overrides)
        all_results[lclid] = run_household(
            args.data_dir, args.info_csv, lclid, cfg,
            weather_csv=args.weather_csv)

    print("\n" + "=" * 72)
    print("  SUMMARY  -  Test-Set Results")
    print("=" * 72)
    print(f"  {'Household':<14} {'R2':>7} {'MAE':>8} {'RMSE':>8} "
          f"{'MAPE':>8} {'SpikeMAE':>10} {'Skill':>7}")
    print("  " + "-" * 68)
    for hid, m in all_results.items():
        print(f"  {hid:<14} {m['R2']:>7.4f} {m['MAE']:>8.4f} "
              f"{m['RMSE']:>8.4f} {m['MAPE']:>7.2f}% "
              f"{m['Spike_MAE']:>10.4f} {m['skill_score']:>7.4f}")
    print("=" * 72)