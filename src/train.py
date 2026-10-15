"""Обучение и сравнение с бейзлайнами.

python -m src.train --horizon 24
python -m src.train --horizon 1

Схема валидации:
  train : 2014-04-29 .. 2014-08-31  (первые 4 недели уходят на прогрев лагов)
  valid : 2014-09-01 .. 2014-09-15  (early stopping, выбор варианта)
  test  : 2014-09-16 .. 2014-09-30  (смотрим один раз в конце)
"""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import END, MODELS_DIR, PROCESSED_DIR, TEST_START, VALID_START, ROOT
from src.features import feature_columns, make_features
from src.metrics import report

PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def split(df):
    df = df.dropna(subset=["same_hour_4w_mean", "roll_mean_168"])
    train = df[df.ts < VALID_START]
    valid = df[(df.ts >= VALID_START) & (df.ts < TEST_START)]
    test = df[(df.ts >= TEST_START) & (df.ts < END)]
    return train, valid, test


def baselines(part, horizon):
    out = {"same_hour_last_week": part["lag_168"]}
    out["same_hour_4w_mean"] = part["same_hour_4w_mean"]
    if horizon == 1:
        out["last_hour"] = part["lag_1"]
    else:
        out["same_hour_yesterday"] = part["lag_24"]
    return out


def fit_variant(train, valid, cols, objective, normalize):
    """normalize=True: учим отношение trips/level, а не сами trips.
    Нужен, потому что спрос растёт месяц к месяцу, а деревья не умеют
    экстраполировать за пределы значений из train."""
    def target(part):
        return part["trips"] / (part["level"] + 1) if normalize else part["trips"]

    params = dict(PARAMS, objective=objective)
    dtrain = lgb.Dataset(train[cols], target(train))
    dvalid = lgb.Dataset(valid[cols], target(valid), reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def predict(model, part, cols, normalize):
    p = model.predict(part[cols], num_iteration=model.best_iteration)
    if normalize:
        p = p * (part["level"].values + 1)
    return np.clip(p, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    h = args.horizon

    raw = pd.read_parquet(PROCESSED_DIR / "demand.parquet")
    df = make_features(raw, horizon=h)
    train, valid, test = split(df)
    cols = feature_columns(df)
    print(f"horizon={h} | train {len(train)} | valid {len(valid)} | test {len(test)} | признаков {len(cols)}")

    results = {"valid": {}, "test": {}}
    for name, p in baselines(valid, h).items():
        results["valid"][name] = report(valid.trips, p)

    variants = {
        "lgbm_l2_raw": ("regression", False),
        "lgbm_poisson_raw": ("poisson", False),
        "lgbm_l2_norm": ("regression", True),
    }
    models = {}
    for name, (obj, norm) in variants.items():
        m = fit_variant(train, valid, cols, obj, norm)
        models[name] = (m, norm)
        results["valid"][name] = report(valid.trips, predict(m, valid, cols, norm))
        print(f"  {name}: best_iter={m.best_iteration}, WAPE valid={results['valid'][name]['WAPE']:.4f}")

    best = min(variants, key=lambda n: results["valid"][n]["WAPE"])
    print("лучший на valid:", best)

    # переобучаем лучший вариант на train+valid с тем же числом итераций
    obj, norm = variants[best]
    full = pd.concat([train, valid])
    y_full = full["trips"] / (full["level"] + 1) if norm else full["trips"]
    n_iter = int(models[best][0].best_iteration * 1.1)
    final = lgb.train(dict(PARAMS, objective=obj), lgb.Dataset(full[cols], y_full), num_boost_round=n_iter)
    final.best_iteration = n_iter

    for name, p in baselines(test, h).items():
        results["test"][name] = report(test.trips, p)
    test_pred = predict(final, test, cols, norm)
    results["test"][f"{best} (final)"] = report(test.trips, test_pred)

    print("\nTEST:")
    print(pd.DataFrame(results["test"]).T.round(4).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "cols": cols, "normalize": norm, "horizon": h},
                MODELS_DIR / f"lgbm_h{h}.joblib")
    out = test[["ts", "cell", "trips"]].copy()
    out["pred"] = test_pred
    out.to_parquet(PROCESSED_DIR / f"test_pred_h{h}.parquet", index=False)

    imp = pd.Series(final.feature_importance("gain"), index=cols).sort_values(ascending=False)
    results["feature_importance_gain"] = (imp / imp.sum()).round(4).head(15).to_dict()
    results["best_variant"] = best
    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / f"metrics_h{h}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
