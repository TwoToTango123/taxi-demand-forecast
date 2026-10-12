from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

MONTHS = ["apr", "may", "jun", "jul", "aug", "sep"]
RAW_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/"
    "master/uber-trip-data/uber-raw-data-{month}14.csv"
)

# res 7 -> гексагон ~5 км^2. На res 8 большая часть ячеек почти всегда пустая
H3_RES = 7
# оставляем самые загруженные ячейки, которые вместе дают такую долю поездок
COVERAGE = 0.95

# грубая рамка вокруг Нью-Йорка, всё что дальше - выбросы (есть точки в 150 км)
BBOX = {"lat_min": 40.45, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.65}

# границы выборок (включительно для начала, не включительно для конца)
VALID_START = "2014-09-01"
TEST_START = "2014-09-16"
END = "2014-10-01"

# праздники США в периоде данных
HOLIDAYS = ["2014-05-26", "2014-07-04", "2014-09-01"]
