"""Признаки для прогноза спроса.

Главное правило: признак для момента t можно строить только из данных,
которые известны в момент выдачи прогноза, т.е. до t - horizon.
horizon=1  -> прогноз на следующий час
horizon=24 -> прогноз на сутки вперёд (самый реалистичный сценарий для
              планирования смен водителей)
"""
import numpy as np
import pandas as pd

from src.config import HOLIDAYS


def lag_list(horizon: int) -> list:
    short = [horizon, horizon + 1, horizon + 2]
    seasonal = [24, 48, 168, 336]
    return sorted({lag for lag in short + seasonal if lag >= horizon})


def make_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """df: ts, cell, trips (полная сетка час x ячейка, без пропусков)."""
    df = df.sort_values(["cell", "ts"]).reset_index(drop=True)
    g = df.groupby("cell", sort=False)["trips"]

    for lag in lag_list(horizon):
        df[f"lag_{lag}"] = g.shift(lag)

    # скользящие средние, сдвинутые на горизонт, чтобы не заглядывать вперёд
    shifted = g.shift(horizon)
    gs = shifted.groupby(df["cell"], sort=False)
    df["roll_mean_24"] = gs.transform(lambda s: s.rolling(24, min_periods=24).mean())
    df["roll_mean_168"] = gs.transform(lambda s: s.rolling(168, min_periods=168).mean())
    df["roll_std_24"] = gs.transform(lambda s: s.rolling(24, min_periods=24).std())

    # тот же час в предыдущие 4 недели (все лаги >= 168, т.е. всегда доступны)
    week_lags = [g.shift(168 * k) for k in range(1, 5)]
    df["same_hour_4w_mean"] = pd.concat(week_lags, axis=1).mean(axis=1)

    # "уровень" ячейки - на него потом нормируем таргет
    df["level"] = df["roll_mean_168"]

    ts = df["ts"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = ts.dt.normalize().isin(pd.to_datetime(HOLIDAYS)).astype(int)
    # ночь с пятницы на субботу и с субботы на воскресенье ведут себя иначе
    df["is_party_night"] = (((df["dow"] == 4) & (df["hour"] >= 20))
                            | ((df["dow"] == 5) & ((df["hour"] <= 4) | (df["hour"] >= 20)))
                            | ((df["dow"] == 6) & (df["hour"] <= 4))).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["cell"] = df["cell"].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list:
    skip = {"ts", "trips", "level", "target"}
    return [c for c in df.columns if c not in skip]
