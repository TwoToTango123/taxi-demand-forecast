"""Эксперимент: добавить признаки уровня всего города (суммарный спрос по всем ячейкам).
Результат на valid хуже базового набора, поэтому в основную модель не вошло.

python scripts/experiment_city_features.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR  # noqa: E402
from src.features import feature_columns, make_features  # noqa: E402
from src.metrics import wape  # noqa: E402
from src.train import PARAMS, split  # noqa: E402

H = 24


def add_city_features(df, raw):
    city = raw.groupby("ts").trips.sum().sort_index()
    c = pd.DataFrame({"ts": city.index})
    c["city_lag_h"] = city.shift(H).values
    c["city_lag_168"] = city.shift(168).values
    c["city_roll_24"] = city.shift(H).rolling(24).mean().values
    c["city_roll_168"] = city.shift(H).rolling(168).mean().values
    df = df.merge(c, on="ts", how="left")
    df["city_trend"] = df.city_roll_24 / df.city_roll_168
    df["cell_trend"] = df.roll_mean_24 / (df.roll_mean_168 + 1)
    df["lag168_x_trend"] = df.lag_168 * df.city_trend
    return df


def main():
    raw = pd.read_parquet(PROCESSED_DIR / "demand.parquet")
    base = make_features(raw, H)
    base_cols = feature_columns(base)
    df = add_city_features(base, raw)
    extra = [c for c in df.columns if c.startswith("city_") or c in ("cell_trend", "lag168_x_trend")]
    tr, va, _ = split(df)
    for name, cols in [("базовый набор", base_cols), ("+ городские признаки", base_cols + extra)]:
        m = lgb.train(dict(PARAMS, objective="poisson"), lgb.Dataset(tr[cols], tr.trips), 3000,
                      valid_sets=[lgb.Dataset(va[cols], va.trips)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(va[cols], num_iteration=m.best_iteration)
        print(f"{name:22s} WAPE valid = {wape(va.trips, p):.4f}")


if __name__ == "__main__":
    main()
