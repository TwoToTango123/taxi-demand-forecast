"""Сырые поездки -> почасовой спрос по H3-ячейкам.

Запуск: python -m src.prepare
На выходе data/processed/demand.parquet с колонками: ts, cell, trips
"""
import h3
import numpy as np
import pandas as pd

from src.config import BBOX, COVERAGE, H3_RES, MONTHS, PROCESSED_DIR, RAW_DIR


def load_raw() -> pd.DataFrame:
    parts = []
    for m in MONTHS:
        df = pd.read_csv(RAW_DIR / f"uber-raw-data-{m}14.csv")
        df.columns = ["dt", "lat", "lon", "base"]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # формат фиксированный, с явным format парсится в разы быстрее
    df["dt"] = pd.to_datetime(df["dt"], format="%m/%d/%Y %H:%M:%S")
    return df


def filter_bbox(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df.lat.between(BBOX["lat_min"], BBOX["lat_max"])
        & df.lon.between(BBOX["lon_min"], BBOX["lon_max"])
    )
    print(f"вне рамки NYC: {(~mask).sum()} поездок ({(~mask).mean():.3%})")
    return df[mask]


def add_cells(df: pd.DataFrame) -> pd.DataFrame:
    # координаты округлены до 4 знаков, уникальных пар сильно меньше чем строк,
    # поэтому считаем h3 только для уникальных и мержим обратно
    coords = df[["lat", "lon"]].drop_duplicates()
    coords["cell"] = [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(coords.lat, coords.lon)]
    return df.merge(coords, on=["lat", "lon"], how="left")


def select_cells(df: pd.DataFrame) -> list:
    counts = df["cell"].value_counts()
    share = counts.cumsum() / counts.sum()
    keep = share[share <= COVERAGE].index.tolist()
    print(f"ячеек всего: {len(counts)}, оставляем {len(keep)} (покрывают {COVERAGE:.0%} поездок)")
    return keep


def to_hourly(df: pd.DataFrame, cells: list) -> pd.DataFrame:
    df = df[df["cell"].isin(cells)].copy()
    df["ts"] = df["dt"].dt.floor("h")
    agg = df.groupby(["ts", "cell"]).size().rename("trips")

    # важно: в часы без поездок строк нет совсем, без reindex лаги поедут
    full_idx = pd.MultiIndex.from_product(
        [pd.date_range(df.ts.min(), df.ts.max(), freq="h"), sorted(cells)],
        names=["ts", "cell"],
    )
    out = agg.reindex(full_idx, fill_value=0).reset_index()
    out["trips"] = out["trips"].astype(np.int32)
    zero_share = (out.trips == 0).mean()
    print(f"строк после reindex: {len(out)}, из них нулевых {zero_share:.1%}")
    return out


def main():
    df = load_raw()
    print(f"поездок загружено: {len(df):,}")
    df = filter_bbox(df)
    df = add_cells(df)
    cells = select_cells(df)
    hourly = to_hourly(df, cells)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROCESSED_DIR / "demand.parquet", index=False)
    print("сохранено в", PROCESSED_DIR / "demand.parquet")


if __name__ == "__main__":
    main()
