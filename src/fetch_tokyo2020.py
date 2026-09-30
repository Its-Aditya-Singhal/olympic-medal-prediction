"""Scrape the Tokyo 2020 medal table and team sizes from Wikipedia.

The Kaggle dataset stops at Rio 2016, so the ground truth for Tokyo 2020
(medals per NOC, athletes per NOC) is pulled once and saved to
data/external/ so the rest of the pipeline works offline.
"""
import io
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_EXT, TOKYO_ATHLETES, TOKYO_MEDALS, WDI_CACHE
from countries import NAME_TO_ISO3

HEADERS = {"User-Agent": "UE24CS352A-ML-mini-project (educational use)"}
MEDAL_URL = "https://en.wikipedia.org/wiki/2020_Summer_Olympics_medal_table"
GAMES_URL = "https://en.wikipedia.org/wiki/2020_Summer_Olympics"


def _name_lookup() -> dict:
    wdi = pd.read_csv(WDI_CACHE)
    lookup = dict(zip(wdi.Country_WB, wdi.ISO3))
    lookup.update(NAME_TO_ISO3)
    return lookup


def _clean(name: str) -> str:
    # strip host marker "*", footnote marks "[a]" and non-breaking spaces
    name = re.sub(r"\[.*?\]", "", name).replace("*", "").replace("\xa0", " ")
    return name.strip()


def fetch_medals() -> pd.DataFrame:
    html = requests.get(MEDAL_URL, headers=HEADERS, timeout=60).text
    tables = pd.read_html(io.StringIO(html))
    table = next(t for t in tables if list(t.columns[:6]) == ["Rank", "NOC", "Gold", "Silver", "Bronze", "Total"])
    table = table[~table["NOC"].astype(str).str.startswith("Totals")].copy()
    table["Country"] = table["NOC"].map(_clean)
    return table[["Country", "Gold", "Silver", "Bronze", "Total"]].astype(
        {"Gold": int, "Silver": int, "Bronze": int, "Total": int})


def fetch_athletes() -> pd.DataFrame:
    soup = BeautifulSoup(requests.get(GAMES_URL, headers=HEADERS, timeout=60).text, "html.parser")
    rows = []
    for li in soup.find_all("li"):
        text = li.get_text(" ")
        # "Japan (552) (host)", "ROC (334) [l]" -> name + team size; footnotes may follow
        m = re.match(r"\s*([^()\[\]]+?)\s*\((\d+)\)", text)
        if m and li.find("img") and len(text) < 80:  # entries in the NOC list carry a flag icon
            rows.append({"Country": _clean(m.group(1)), "Athletes": int(m.group(2))})
    return pd.DataFrame(rows).drop_duplicates("Country")


def main():
    lookup = _name_lookup()
    medals, athletes = fetch_medals(), fetch_athletes()
    for df, path in [(medals, TOKYO_MEDALS), (athletes, TOKYO_ATHLETES)]:
        df["ISO3"] = df["Country"].map(lookup)
        missing = df[df.ISO3.isna()]
        if len(missing):
            # the Games article also lists sports with icons; those never map to a country
            print(f"dropping unmapped names in {path.name}: {missing.Country.tolist()}")
            df.dropna(subset=["ISO3"], inplace=True)
        DATA_EXT.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"saved {len(df)} rows to {path}")
    print(f"medals total = {medals.Total.sum()}, athletes total = {athletes.Athletes.sum()}")


if __name__ == "__main__":
    main()
