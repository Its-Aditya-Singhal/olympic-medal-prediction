"""Why the paper's 2016 result (RMSE 2.26) does not reproduce.

The "Actual Medals" in the paper's Table 4 (USA 103, China 89, UK 65, Germany 44,
Japan 38) are the 2012 counts, not 2016 (121, 70, 67, 42, 41). The paper also says
the final model was trained on the tuning + validation data (1988-2012). If the
test rows were the 2012 rows, the model was scored on data it had been trained on.
This script checks that by fitting the paper's final model (Gaussian NB + Ridge)
on 1988-2012 and predicting the 2012 rows. Writes reports/paper_audit.csv.
"""
import json
import warnings

import pandas as pd

from build_dataset import load_dataset
from config import FEATURES, REPORTS, RESULTS_JSON, TARGET
from experiments import params_for
from metrics import rmse
from models import CLASSIFIERS, REGRESSORS, make, two_stage_predict

warnings.filterwarnings("ignore")

PAPER_TABLE4 = pd.DataFrame({
    "ISO3": ["USA", "CHN", "GBR", "DEU", "JPN"],
    "paper_actual": [103, 89, 65, 44, 38],
    "paper_predicted": [103, 85, 52, 44, 45],
})


def fit_predict(data, results, train_max, test_year):
    train, test = data[data.Year <= train_max], data[data.Year == test_year]
    X, y = train[FEATURES].to_numpy(float), train[TARGET].to_numpy(float)
    clf = make(CLASSIFIERS, "gnb", params_for(results, "classifiers", "gnb")).fit(X, y > 0)
    reg = make(REGRESSORS, "ridge", params_for(results, "regressors", "ridge")).fit(X[y > 0], y[y > 0])
    pred = two_stage_predict(clf, reg, test[FEATURES].to_numpy(float))
    return test.assign(pred=pred)


def main():
    results = json.loads(RESULTS_JSON.read_text())
    data = load_dataset()
    in_sample = fit_predict(data, results, 2012, 2012)    # hypothesis: test rows were 2012, inside training
    honest = fit_predict(data, results, 2012, 2016)       # what the paper describes: predict 2016

    table = PAPER_TABLE4.copy()
    for name, frame in [("2012", in_sample), ("2016", honest)]:
        idx = frame.set_index("ISO3")
        table[f"actual_{name}"] = table.ISO3.map(idx[TARGET])
        table[f"ours_pred_{name}"] = table.ISO3.map(idx.pred).round(0)
    table.to_csv(REPORTS / "paper_audit.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nGNB + Ridge, trained 1988-2012:")
    print(f"  predicting 2012 (in training set): RMSE = {rmse(in_sample[TARGET], in_sample.pred):.2f}")
    print(f"  predicting 2016 (held out):        RMSE = {rmse(honest[TARGET], honest.pred):.2f}")
    print(f"  paper reported:                    RMSE = {results['paper']['final_2016_rmse']}")


if __name__ == "__main__":
    main()
