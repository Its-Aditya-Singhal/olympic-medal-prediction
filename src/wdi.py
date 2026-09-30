"""Download World Development Indicators from the World Bank REST API (no key needed).

Output: data/external/wdi.csv with one row per (ISO3, Year).
"""
import pandas as pd
import requests

from config import DATA_EXT, WDI_CACHE

INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP",  # current US$
    "NY.GDP.PCAP.CD": "GDP_Per_Capita",  # current US$
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",  # annual %
    "SP.POP.TOTL": "Pop",
    "SP.POP.GROW": "Pop_Growth",  # annual %
    "AG.LND.TOTL.K2": "Area",  # land area, sq. km
}
API = "https://api.worldbank.org/v2/country/all/indicator/{code}"


def fetch_indicator(code: str, start: int = 1984, end: int = 2021) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        resp = requests.get(
            API.format(code=code),
            params={"format": "json", "date": f"{start}:{end}", "per_page": 20000, "page": page},
            timeout=60,
        )
        resp.raise_for_status()
        meta, data = resp.json()
        rows += [
            {"ISO3": d["countryiso3code"], "Country_WB": d["country"]["value"],
             "Year": int(d["date"]), "value": d["value"]}
            for d in data or []
            if d["countryiso3code"]
        ]
        if page >= meta["pages"]:
            break
        page += 1
    return pd.DataFrame(rows)


def build_wdi() -> pd.DataFrame:
    frames = []
    for code, name in INDICATORS.items():
        print(f"fetching {code} ({name})")
        df = fetch_indicator(code).rename(columns={"value": name})
        frames.append(df.set_index(["ISO3", "Country_WB", "Year"]))
    wdi = pd.concat(frames, axis=1).reset_index().sort_values(["ISO3", "Year"])
    DATA_EXT.mkdir(parents=True, exist_ok=True)
    wdi.to_csv(WDI_CACHE, index=False)
    print(f"saved {len(wdi)} rows to {WDI_CACHE}")
    return wdi


def load_wdi(refresh: bool = False) -> pd.DataFrame:
    if refresh or not WDI_CACHE.exists():
        return build_wdi()
    return pd.read_csv(WDI_CACHE)


if __name__ == "__main__":
    build_wdi()
