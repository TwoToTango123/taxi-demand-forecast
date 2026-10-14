import numpy as np
import pandas as pd

from src.features import lag_list, make_features
from src.metrics import wape


def toy(n_hours=24 * 30, cells=("a", "b")):
    ts = pd.date_range("2014-04-01", periods=n_hours, freq="h")
    rows = []
    for c in cells:
        rows.append(pd.DataFrame({"ts": ts, "cell": c, "trips": np.arange(n_hours)}))
    return pd.concat(rows, ignore_index=True)


def test_no_lags_below_horizon():
    assert min(lag_list(24)) == 24
    assert min(lag_list(1)) == 1


def test_lag_values_are_shifted_inside_cell():
    df = make_features(toy(), horizon=24)
    row = df[(df.cell == "b") & (df.ts == pd.Timestamp("2014-04-10 05:00"))].iloc[0]
    # trips = номер часа, значит lag_24 должен быть ровно на 24 меньше
    assert row.lag_24 == row.trips - 24
    assert row.lag_168 == row.trips - 168


def test_rolling_does_not_see_future():
    df = make_features(toy(), horizon=24)
    row = df[(df.cell == "a") & (df.ts == pd.Timestamp("2014-04-10 05:00"))].iloc[0]
    # среднее последних 24 значений, закончившихся за 24 часа до t
    expected = np.mean(np.arange(row.trips - 24 - 23, row.trips - 24 + 1))
    assert abs(row.roll_mean_24 - expected) < 1e-9


def test_wape():
    assert wape([10, 0, 10], [10, 0, 10]) == 0
    assert abs(wape([10, 10], [5, 15]) - 0.5) < 1e-12
