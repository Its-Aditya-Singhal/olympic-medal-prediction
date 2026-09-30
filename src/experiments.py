"""Feature ablation: does adding host-country flags or log-scaled features help?

Each variant is scored with the same rolling-origin validation as train.py
(2004/2008/2012) and on the two test years, for the pairs chosen by the paper
protocol and by the rolling protocol. Writes reports/ablation.csv.
"""
import json
import warnings

import numpy as np
import pandas as pd

from build_dataset import load_dataset
from config import FEATURES, HOSTS, REPORTS, RESULTS_JSON, TARGET, TEST_YEAR, TOKYO_YEAR
from metrics import rmse
from models import CLASSIFIERS, REGRESSORS, make, two_stage_predict
from train import ROLLING_YEARS

warnings.filterwarnings("ignore")

HOST_FEATURES = ["Host", "Next_Host", "Prev_Host"]
LOGGED = ["GDP", "GDP_Per_Capita", "Pop", "Area", "Athletes"]
LOG_FEATURES = [f for f in FEATURES if f not in LOGGED] + [f"log_{c}" for c in LOGGED]
VARIANTS = {
    "paper features": FEATURES,
    "+ host flags": FEATURES + HOST_FEATURES,
    "log-scaled": LOG_FEATURES,
    "log-scaled + host flags": LOG_FEATURES + HOST_FEATURES,
}


def add_extra_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for col, shift in [("Host", 0), ("Next_Host", 4), ("Prev_Host", -4)]:
        data[col] = [int(HOSTS.get(y + shift) == c) for y, c in zip(data.Year, data.ISO3)]
    for c in LOGGED:
        data[f"log_{c}"] = np.log1p(data[c])
    return data


def params_for(results: dict, kind: str, name: str) -> dict:
    return {f"model__{k}": tuple(v) if isinstance(v, list) else v
            for k, v in results["params"][kind][name].items()}


def score(data, feats, clf_name, reg_name, year, results) -> float:
    train, test = data[data.Year < year], data[data.Year == year]
    X, y = train[feats].to_numpy(float), train[TARGET].to_numpy(float)
    Xs, ys = test[feats].to_numpy(float), test[TARGET].to_numpy(float)
    reg = make(REGRESSORS, reg_name, params_for(results, "regressors", reg_name))
    if clf_name == "none":
        return rmse(ys, np.clip(reg.fit(X, y).predict(Xs), 0, None))
    clf = make(CLASSIFIERS, clf_name, params_for(results, "classifiers", clf_name)).fit(X, y > 0)
    reg.fit(X[y > 0], y[y > 0])
    return rmse(ys, two_stage_predict(clf, reg, Xs))


def main():
    results = json.loads(RESULTS_JSON.read_text())
    data = add_extra_features(load_dataset())
    pairs = {"paper protocol": results["best"], "rolling protocol": results["best_rolling"],
             "baseline linear": {"classifier": "none", "regressor": "linear"}}
    rows = []
    for variant, feats in VARIANTS.items():
        for label, pair in pairs.items():
            c, r = pair["classifier"], pair["regressor"]
            val = [score(data, feats, c, r, y, results) for y in ROLLING_YEARS]
            rows.append({"features": variant, "model": f"{label} ({c}+{r})", "rolling_val_rmse": np.mean(val),
                         f"test_{TEST_YEAR}_rmse": score(data, feats, c, r, TEST_YEAR, results),
                         f"test_{TOKYO_YEAR}_rmse": score(data, feats, c, r, TOKYO_YEAR, results)})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "ablation.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
