"""Build the country-per-Games dataset (paper Section 3).

One row = one country at one Summer Games (1988-2016 from Kaggle, 2020 from
the scraped Tokyo tables). Target = total medals won, where a team event
counts as a single medal (paper, 3.2 item 1).
"""
import pandas as pd

from config import (ATHLETE_EVENTS, DATA_PROC, DATASET, FEATURES, FIRST_YEAR, TARGET,
                    TOKYO_ATHLETES, TOKYO_MEDALS, TOKYO_YEAR)
from countries import noc_to_iso3
from wdi import load_wdi


def olympic_table() -> pd.DataFrame:
    """Medals and athletes per (ISO3, Year) for Summer Games 1984-2016."""
    df = pd.read_csv(ATHLETE_EVENTS)
    df = df[(df.Season == "Summer") & (df.Year >= FIRST_YEAR - 4)]

    # one medal per event per team, e.g. a basketball gold is 1 medal, not 12
    medals = df.dropna(subset=["Medal"]).drop_duplicates(["Year", "Event", "Medal", "NOC"])
    totals = pd.DataFrame({
        "Total_Medals_Year": medals.groupby("Year").size(),
        "Total_Athletes_Year": df.groupby("Year").ID.nunique(),
    })

    df = df.assign(ISO3=df.NOC.map(noc_to_iso3)).dropna(subset=["ISO3"])
    medals = medals.assign(ISO3=medals.NOC.map(noc_to_iso3)).dropna(subset=["ISO3"])
    table = pd.DataFrame({
        "Athletes": df.groupby(["ISO3", "Year"]).ID.nunique(),
        TARGET: medals.groupby(["ISO3", "Year"]).size(),
    }).fillna({TARGET: 0}).reset_index()
    return table.merge(totals, left_on="Year", right_index=True)


def tokyo_table() -> pd.DataFrame:
    """Same columns for Tokyo 2020, from the scraped Wikipedia tables."""
    athletes = pd.read_csv(TOKYO_ATHLETES).groupby("ISO3", as_index=False).Athletes.sum()
    medals = pd.read_csv(TOKYO_MEDALS).groupby("ISO3").Total.sum().rename(TARGET)
    table = athletes.merge(medals, left_on="ISO3", right_index=True, how="left").fillna({TARGET: 0})
    table["Year"] = TOKYO_YEAR
    table["Total_Medals_Year"] = int(pd.read_csv(TOKYO_MEDALS).Total.sum())
    table["Total_Athletes_Year"] = int(athletes.Athletes.sum())
    return table


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.sort_values(["ISO3", "Year"]).copy()
    table[TARGET] = table[TARGET].astype(int)

    # medals at the previous Games; 0 if the country did not take part
    prev = table[["ISO3", "Year", TARGET]].assign(Year=lambda d: d.Year + 4)
    table = table.merge(prev.rename(columns={TARGET: "Medals_Last_Games"}), on=["ISO3", "Year"], how="left")
    table["Medals_Last_Games"] = table["Medals_Last_Games"].fillna(0).astype(int)
    table["Pct_Athletes"] = 100 * table.Athletes / table.Total_Athletes_Year

    wdi = load_wdi()
    # land area barely changes but has gaps in early years: fill within country
    wdi["Area"] = wdi.groupby("ISO3").Area.transform(lambda s: s.ffill().bfill())
    world = wdi[wdi.ISO3 == "WLD"].set_index("Year")
    table = table.merge(wdi.drop(columns="Country_WB"), on=["ISO3", "Year"], how="left")
    table["Pct_World_GDP"] = 100 * table.GDP / table.Year.map(world.GDP)
    table["Pct_World_Pop"] = 100 * table.Pop / table.Year.map(world.Pop)

    names = wdi.drop_duplicates("ISO3", keep="last").set_index("ISO3").Country_WB
    table.insert(1, "Country", table.ISO3.map(names).fillna(table.ISO3))
    return table[table.Year >= FIRST_YEAR].reset_index(drop=True)


def build() -> pd.DataFrame:
    table = pd.concat([olympic_table(), tokyo_table()], ignore_index=True)
    data = add_features(table)
    data["Complete"] = data[FEATURES].notna().all(axis=1)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATASET, index=False)
    return data


def load_dataset(complete_only: bool = True) -> pd.DataFrame:
    data = pd.read_csv(DATASET)
    return data[data.Complete].reset_index(drop=True) if complete_only else data


if __name__ == "__main__":
    data = build()
    summary = data.groupby("Year").agg(
        countries=("ISO3", "size"), complete=("Complete", "sum"),
        medals=(TARGET, "sum"), medals_complete=(TARGET, lambda s: s[data.loc[s.index, "Complete"]].sum()),
        medalling=(TARGET, lambda s: (s > 0).sum()))
    print(summary.to_string())
    top = data[data.Year.isin([2016, 2020])].sort_values(TARGET, ascending=False)
    print(top.groupby("Year").head(6)[["Year", "Country", TARGET, "Medals_Last_Games", "Athletes", "Complete"]].to_string())
    dropped = data[~data.Complete & (data[TARGET] > 0)]
    print("medal winners dropped for missing indicators:\n",
          dropped.groupby("ISO3")[TARGET].agg(["count", "sum"]).sort_values("sum", ascending=False).head(12).to_string())
