"""Shared paths and constants for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

ATHLETE_EVENTS = DATA_RAW / "athlete_events.csv"
NOC_REGIONS = DATA_RAW / "noc_regions.csv"
WDI_CACHE = DATA_EXT / "wdi.csv"
TOKYO_MEDALS = DATA_EXT / "tokyo2020_medals.csv"
TOKYO_ATHLETES = DATA_EXT / "tokyo2020_athletes.csv"
DATASET = DATA_PROC / "dataset.csv"
RESULTS_JSON = REPORTS / "results.json"
PREDICTIONS_CSV = REPORTS / "predictions.csv"

RANDOM_STATE = 42

# Paper, Table 2: tune on 1988-2008, validate on 2012, test on 2016.
# Extension: retrain on 1988-2016 and test on Tokyo 2020.
FIRST_YEAR = 1988
TUNE_YEARS = list(range(1988, 2009, 4))
VALID_YEAR = 2012
TEST_YEAR = 2016
TOKYO_YEAR = 2020

# Paper, Table 1 (plus Year / total medals awarded, Section 5.1 option 2).
FEATURES = [
    "Year",
    "GDP",
    "GDP_Per_Capita",
    "GDP_Growth",
    "Pct_World_GDP",
    "Pop",
    "Pct_World_Pop",
    "Pop_Growth",
    "Area",
    "Medals_Last_Games",
    "Total_Medals_Year",
    "Athletes",
    "Pct_Athletes",
]
TARGET = "Medals"

# Summer Games hosts (used by the host-advantage ablation in experiments.py)
HOSTS = {1984: "USA", 1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS", 2004: "GRC",
         2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN", 2024: "FRA"}
