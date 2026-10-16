"""Графики по результатам на тесте. Запускать после train.py.

python -m src.evaluate --horizon 24
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, PROCESSED_DIR, ROOT  # noqa: E402
from src.features import make_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    h = ap.parse_args().horizon
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_parquet(PROCESSED_DIR / f"test_pred_h{h}.parquet")
    raw = pd.read_parquet(PROCESSED_DIR / "demand.parquet")
    feats = make_features(raw, h)[["ts", "cell", "same_hour_4w_mean"]]
    feats["cell"] = feats["cell"].astype(str)
    pred["cell"] = pred["cell"].astype(str)
    pred = pred.merge(feats, on=["ts", "cell"], how="left")

    # 1. факт vs прогноз для самой загруженной ячейки, одна неделя
    top_cell = pred.groupby("cell").trips.sum().idxmax()
    s = pred[(pred.cell == top_cell) & (pred.ts < pred.ts.min() + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.ts, s.trips, label="факт", lw=1.5)
    ax.plot(s.ts, s.pred, label="LightGBM", lw=1.2)
    ax.plot(s.ts, s.same_hour_4w_mean, label="среднее за 4 недели", lw=1, ls="--", alpha=0.7)
    ax.set_title(f"Самая загруженная ячейка {top_cell}, горизонт {h} ч")
    ax.set_ylabel("поездок в час")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"forecast_top_cell_h{h}.png", dpi=120)
    plt.close(fig)

    # 2. WAPE по часу суток
    pred["hour"] = pred.ts.dt.hour
    pred["err"] = (pred.trips - pred.pred).abs()
    pred["err_base"] = (pred.trips - pred.same_hour_4w_mean).abs()
    by_hour = pred.groupby("hour").agg(err=("err", "sum"), err_base=("err_base", "sum"), y=("trips", "sum"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_hour.index, by_hour.err / by_hour.y, marker="o", label="LightGBM")
    ax.plot(by_hour.index, by_hour.err_base / by_hour.y, marker="o", label="среднее за 4 недели")
    ax.set_xlabel("час суток")
    ax.set_ylabel("WAPE")
    ax.set_xticks(range(24))
    ax.legend()
    ax.set_title("Ошибка по часам суток (тест)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"wape_by_hour_h{h}.png", dpi=120)
    plt.close(fig)

    # 3. WAPE по ячейкам vs объём - на маленьких ячейках относительная ошибка выше
    by_cell = pred.groupby("cell").agg(err=("err", "sum"), y=("trips", "sum"))
    by_cell["wape"] = by_cell.err / by_cell.y
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(by_cell.y / by_cell.y.sum() * 100, by_cell.wape, s=20)
    ax.set_xscale("log")
    ax.set_xlabel("доля поездок ячейки, % (лог)")
    ax.set_ylabel("WAPE ячейки")
    ax.set_title("Ошибка по ячейкам")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"wape_by_cell_h{h}.png", dpi=120)
    plt.close(fig)

    # 4. важность признаков
    with open(ROOT / "reports" / f"metrics_h{h}.json", encoding="utf-8") as f:
        imp = pd.Series(json.load(f)["feature_importance_gain"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_title(f"Важность признаков (gain), h={h}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"feature_importance_h{h}.png", dpi=120)
    plt.close(fig)

    print("worst hours:", (by_hour.err / by_hour.y).sort_values(ascending=False).head(3).round(3).to_dict())
    print("cells wape range:", by_cell.wape.min().round(3), by_cell.wape.max().round(3))
    print("графики в", FIGURES_DIR)


if __name__ == "__main__":
    main()
