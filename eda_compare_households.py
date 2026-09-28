

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from smart_meter_with_weather import (
    get_cfg, load_household_info, load_household_series, clean_series,
)

DEFAULT_CLIENTS = ["MAC000100", "MAC001000", "MAC005020"]


def load_clean(lclid: str, cfg: dict, info_df: pd.DataFrame) -> pd.DataFrame:
    """Încarcă și aplică exact curățarea folosită de model."""
    hh_raw = load_household_series(cfg["data_dir"], lclid, info_df, cfg)
    return clean_series(hh_raw, cfg)          # DataFrame cu tstp + coloana țintă


def describe(vals: np.ndarray) -> dict:
    """Statisticile de distribuție care diferențiază gospodăriile."""
    s = pd.Series(vals)
    return {
        "mean":      float(s.mean()),
        "median":    float(s.median()),
        "std":       float(s.std()),
        "kurtosis":  float(s.kurtosis()),      # kurtosis Fisher (exces)
        "skew":      float(s.skew()),
        "zeros_pct": float((s <= 1e-6).mean()),
    }


def plot_comparison(series: dict[str, pd.DataFrame], cfg: dict,
                    clip_pct: float, logx: bool, out_path: str) -> None:
    col     = cfg["target_col"]
    clients = list(series.keys())
    palette = sns.color_palette("muted", len(clients))
    colors  = {c: palette[i] for i, c in enumerate(clients)}

    fig, (ax_dist, ax_prof) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Consumption distribution & average load profile  —  "
                 + ", ".join(clients),
                 fontsize=15, fontweight="bold", y=1.02)

    # (1) distribuția consumului, densități suprapuse
    all_vals = np.concatenate([s[col].values for s in series.values()])
    all_pos  = all_vals[all_vals > 0]
    x_hi     = float(np.percentile(all_vals, clip_pct)) if clip_pct < 100 \
               else float(all_vals.max())
    x_lo     = max(1e-3, float(all_pos.min())) if logx else 0.0

    for c in clients:
        vals = series[c][col].values
        st   = describe(vals)
        kde_vals = vals[vals > 0] if logx else vals
        sns.kdeplot(
            x=kde_vals, ax=ax_dist, color=colors[c], fill=True,
            alpha=0.25, linewidth=2, log_scale=logx, clip=(x_lo, x_hi),
            label=f"{c}  (mean={st['mean']:.3f}, kurt={st['kurtosis']:.1f})",
        )

    ax_dist.set_xlim(x_lo, x_hi)
    ax_dist.set_title("Distribution of Energy Consumption")
    ax_dist.set_xlabel("kWh / half-hour" + ("  (log scale)" if logx else ""))
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=9, title="household")

    # (2) profilul mediu de consum pe ore
    for c in clients:
        hh = series[c].copy()
        hh["hour"] = hh["tstp"].dt.hour + hh["tstp"].dt.minute / 60.0
        prof = hh.groupby("hour")[col].mean()
        ax_prof.plot(prof.index, prof.values, marker="o", ms=3, lw=1.8,
                     color=colors[c], label=c)

    ax_prof.set_title("Average Load Profile (by Hour of Day)")
    ax_prof.set_xlabel("Hour of Day")
    ax_prof.set_ylabel("Mean kWh / half-hour")
    ax_prof.set_xticks(range(0, 25, 2))
    ax_prof.set_xlim(0, 24)
    ax_prof.legend(fontsize=9, title="household")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVE] Comparison figure -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-household EDA comparison (no model training).")
    p.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS,
                   help="Space-separated LCLids to compare.")
    p.add_argument("--data_dir", default="halfhourly_dataset")
    p.add_argument("--info_csv", default="informations_households.csv")
    p.add_argument("--out_dir",  default="outputs/flower/comparison",
                   help="Where to write the figure (default: centralized under "
                        "the flower output folder, alongside the run PNGs).")
    p.add_argument("--out_name", default=None,
                   help="Fixed output filename stem (no extension). Default: "
                        "eda_compare_<clients>. Used by the app to give the "
                        "current-selection figure a stable, short name.")
    p.add_argument("--clip_pct", type=float, default=99.5,
                   help="Upper x-limit for the distribution as a percentile "
                        "(100 = no clipping). Keeps heavy tails readable.")
    p.add_argument("--logx", action="store_true",
                   help="Log-scale the consumption axis (helps the right skew).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg  = get_cfg({"data_dir": args.data_dir, "out_dir": args.out_dir})

    print("=" * 72)
    print("  Cross-household EDA comparison")
    print(f"  Households : {args.clients}")
    print(f"  Data dir   : {cfg['data_dir']}")
    print("=" * 72)

    info_df = load_household_info(args.info_csv)
    col     = cfg["target_col"]

    series: dict[str, pd.DataFrame] = {}
    for lclid in args.clients:
        print(f"\n-- {lclid}")
        series[lclid] = load_clean(lclid, cfg, info_df)

    # sumar în consolă: exact statisticile care explică diferențele
    print(f"\n  {'Household':<12} {'mean':>7} {'median':>7} {'std':>7} "
          f"{'kurt':>7} {'skew':>7} {'zeros%':>7}")
    print("  " + "-" * 60)
    for lclid in args.clients:
        st = describe(series[lclid][col].values)
        print(f"  {lclid:<12} {st['mean']:>7.3f} {st['median']:>7.3f} "
              f"{st['std']:>7.3f} {st['kurtosis']:>7.2f} {st['skew']:>7.2f} "
              f"{st['zeros_pct']*100:>6.1f}%")

    stem     = args.out_name or f"eda_compare_{'_'.join(args.clients)}"
    out_path = os.path.join(cfg["out_dir"], f"{stem}.png")
    plot_comparison(series, cfg, args.clip_pct, args.logx, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
