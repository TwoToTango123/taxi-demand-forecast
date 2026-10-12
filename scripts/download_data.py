"""Скачивает сырые данные Uber (апрель-сентябрь 2014) в data/raw.

Источник: репозиторий FiveThirtyEight, данные получены ими от NYC TLC
по запросу FOIL. ~4.5 млн поездок, ~210 МБ.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MONTHS, RAW_DIR, RAW_URL  # noqa: E402


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for m in MONTHS:
        dst = RAW_DIR / f"uber-raw-data-{m}14.csv"
        if dst.exists():
            print(f"{dst.name} уже есть, пропускаю")
            continue
        print(f"качаю {dst.name} ...")
        urllib.request.urlretrieve(RAW_URL.format(month=m), dst)
    print("готово")


if __name__ == "__main__":
    main()
