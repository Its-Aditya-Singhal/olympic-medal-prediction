"""Tune, select and test every model (paper Sections 5-6), then extend to Tokyo 2020.

1. Tune each classifier / regressor with K-fold CV on 1988-2008 (Grid or Randomized search).
2. Score them on 2012 (validation): classifier accuracy, regressor RMSE before/after tuning.
3. Pick the best classifier (accuracy) and regressor (Eq.1 + 0.25 * Eq.2).
4. Refit on 1988-2012 and test on 2016 (paper), refit on 1988-2016 and test on 2020 (extension).

Extension: a single validation year is noisy, so every classifier x regressor pair is also
scored with rolling-origin validation (train on all Games before Y, predict Y, for
Y in 2004/2008/2012) and the best pair by mean Eq.1 + 0.25 * Eq.2 is reported as well.

Outputs: reports/results.json, reports/predictions.csv, models/window_<year>.joblib
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold

from build_dataset import load_dataset
from config import (FEATURES, MODELS_DIR, PREDICTIONS_CSV, RANDOM_STATE, REPORTS, RESULTS_JSON,
                    TARGET, TEST_YEAR, TOKYO_YEAR, TUNE_YEARS, VALID_YEAR)
from metrics import report, rmse
from models import CLASSIFIERS, REGRESSORS, make, two_stage_predict

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROLLING_YEARS = [2004, 2008, 2012]
N_FOLDS = 5
N_RANDOM_ITER = 30
BASELINE_CLF = "logreg"  # paper: regressors compared before/after tuning behind a logistic classifier

# numbers reported in the paper (Table 3 and Section 6), for side-by-side comparison
PAPER = {
    "classifier_accuracy_2012": {"logreg": 0.877, "svc": 0.884, "rf": 0.877, "gnb": 0.891, "mlp": 0.855},
    "regressor_rmse_2012": {"weighted_lr": 2.63, "svr": 2.50, "rf": 2.20, "poisson": 3.75,
                            "lasso": 2.87, "ridge": 2.27},
    "final_2016_rmse": 2.26, "baseline_linear_2016_rmse": 3.25, "final_model": "gnb + ridge",
}


def xy(frame: pd.DataFrame):
    return frame[FEATURES].to_numpy(float), frame[TARGET].to_numpy(float)


def jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        k = k.replace("model__", "")
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def search(zoo, name, X, y, classification: bool):
    _, pipe, space, kind = zoo[name]
    cv = (StratifiedKFold if classification else KFold)(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "accuracy" if classification else "neg_mean_squared_error"
    if kind == "grid":
        s = GridSearchCV(pipe, space, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        s = RandomizedSearchCV(pipe, space, n_iter=N_RANDOM_ITER, cv=cv, scoring=scoring,
                               n_jobs=-1, random_state=RANDOM_STATE)
    s.fit(X, y)
    return s.best_params_, float(s.best_score_)


def select_models(data: pd.DataFrame) -> dict:
    tune, valid = data[data.Year.isin(TUNE_YEARS)], data[data.Year == VALID_YEAR]
    Xt, yt = xy(tune)
    Xv, yv = xy(valid)
    won_t = yt > 0
    out = {"classifiers": {}, "regressors": {}, "params": {"classifiers": {}, "regressors": {}}}

    print(f"\n== Classifiers (tune {TUNE_YEARS[0]}-{TUNE_YEARS[-1]}, n={len(tune)}; validate {VALID_YEAR}, n={len(valid)})")
    fitted_clf = {}
    for name, (label, *_rest) in CLASSIFIERS.items():
        t0 = time.time()
        params, cv_score = search(CLASSIFIERS, name, Xt, won_t.astype(int), classification=True)
        default = make(CLASSIFIERS, name).fit(Xt, won_t)
        tuned = make(CLASSIFIERS, name, params).fit(Xt, won_t)
        fitted_clf[name] = tuned
        pv = tuned.predict(Xv)
        row = {"label": label, "cv_accuracy": cv_score,
               "accuracy_default": accuracy_score(yv > 0, default.predict(Xv)),
               "accuracy": accuracy_score(yv > 0, pv), "f1": f1_score(yv > 0, pv)}
        out["classifiers"][name] = row
        out["params"]["classifiers"][name] = params
        print(f"  {label:26s} acc default={row['accuracy_default']:.3f} tuned={row['accuracy']:.3f} "
              f"f1={row['f1']:.3f} cv={cv_score:.3f} ({time.time() - t0:.0f}s)")
    best_clf = max(out["classifiers"], key=lambda n: (out["classifiers"][n]["accuracy"], out["classifiers"][n]["f1"]))
    print(f"  -> best classifier: {best_clf}")

    print(f"\n== Regressors (fit on medal winners only, n={won_t.sum()})")
    for name, (label, *_rest) in REGRESSORS.items():
        t0 = time.time()
        params, cv_score = search(REGRESSORS, name, Xt[won_t], yt[won_t], classification=False)
        default = make(REGRESSORS, name).fit(Xt[won_t], yt[won_t])
        tuned = make(REGRESSORS, name, params).fit(Xt[won_t], yt[won_t])
        base = fitted_clf[BASELINE_CLF]
        row = {"label": label, "cv_rmse": float(np.sqrt(-cv_score)),
               "rmse_default_logreg": rmse(yv, two_stage_predict(base, default, Xv)),
               "rmse_tuned_logreg": rmse(yv, two_stage_predict(base, tuned, Xv)),
               **report(yv, two_stage_predict(fitted_clf[best_clf], tuned, Xv))}
        out["regressors"][name] = row
        out["params"]["regressors"][name] = params
        print(f"  {label:26s} rmse(logreg) default={row['rmse_default_logreg']:6.2f} "
              f"tuned={row['rmse_tuned_logreg']:5.2f} | with {best_clf}: rmse={row['rmse']:.2f} "
              f"top10={row['top10_rmse']:.2f} combined={row['combined']:.2f} ({time.time() - t0:.0f}s)")
    best_reg = min(out["regressors"], key=lambda n: out["regressors"][n]["combined"])
    print(f"  -> best regressor: {best_reg}")

    single = make(REGRESSORS, "linear").fit(Xt, yt)
    out["validation_baselines"] = {
        "single_stage_linear": report(yv, np.clip(single.predict(Xv), 0, None)),
        "last_games": report(yv, valid.Medals_Last_Games.to_numpy(float)),
    }
    out["best"] = {"classifier": best_clf, "regressor": best_reg}
    return out


def fit_window(data: pd.DataFrame, test_year: int, params: dict, save: bool = True):
    """Refit every tuned model on all Games before test_year and predict test_year."""
    train, test = data[data.Year < test_year], data[data.Year == test_year]
    X, y = xy(train)
    Xs, ys = xy(test)
    won = y > 0
    clfs = {n: make(CLASSIFIERS, n, params["classifiers"][n]).fit(X, won) for n in CLASSIFIERS}
    regs = {n: make(REGRESSORS, n, params["regressors"][n]).fit(X[won], y[won]) for n in REGRESSORS}
    single = {n: make(REGRESSORS, n, params["regressors"][n]).fit(X, y) for n in REGRESSORS}

    base = test[["Year", "ISO3", "Country", TARGET, "Medals_Last_Games"]].rename(columns={TARGET: "Actual"})
    rows = []
    for c in ["none", *clfs]:
        for r in regs:
            clf, reg = (None, single[r]) if c == "none" else (clfs[c], regs[r])
            rows.append(base.assign(Classifier=c, Regressor=r, Predicted=two_stage_predict(clf, reg, Xs)))
    rows.append(base.assign(Classifier="naive", Regressor="last_games", Predicted=base.Medals_Last_Games))
    preds = pd.concat(rows, ignore_index=True)

    if not save:
        return preds
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"classifiers": clfs, "regressors": regs, "single": single, "features": FEATURES,
                 "train_years": sorted(train.Year.unique().tolist()), "test_year": test_year},
                MODELS_DIR / f"window_{test_year}.joblib", compress=3)
    return preds


def score_table(preds: pd.DataFrame) -> list:
    rows = []
    for (year, c, r), g in preds.groupby(["Year", "Classifier", "Regressor"]):
        rows.append({"year": int(year), "classifier": c, "regressor": r, **report(g.Actual, g.Predicted)})
    return rows


def rolling_validation(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    preds = pd.concat([fit_window(data, yr, params, save=False) for yr in ROLLING_YEARS], ignore_index=True)
    scores = pd.DataFrame(score_table(preds))
    table = scores.groupby(["classifier", "regressor"]).agg(
        rmse=("rmse", "mean"), top10_rmse=("top10_rmse", "mean"), combined=("combined", "mean")).reset_index()
    per_year = scores.pivot_table(index=["classifier", "regressor"], columns="year", values="rmse")
    per_year.columns = [f"rmse_{c}" for c in per_year.columns]
    return table.merge(per_year.reset_index(), on=["classifier", "regressor"]).sort_values("combined")


def main():
    data = load_dataset()
    results = select_models(data)
    best_c, best_r = results["best"]["classifier"], results["best"]["regressor"]

    print(f"\n== Rolling-origin validation on {ROLLING_YEARS}")
    rolling = rolling_validation(data, results["params"])
    print(rolling.head(8).round(2).to_string(index=False))
    ranked = rolling[rolling.classifier != "naive"]
    roll_c, roll_r = ranked.iloc[0].classifier, ranked.iloc[0].regressor
    results["rolling"] = rolling.to_dict(orient="records")
    results["best_rolling"] = {"classifier": roll_c, "regressor": roll_r}

    preds = pd.concat([fit_window(data, yr, results["params"]) for yr in (TEST_YEAR, TOKYO_YEAR)],
                      ignore_index=True)
    REPORTS.mkdir(exist_ok=True)
    preds.to_csv(PREDICTIONS_CSV, index=False)
    table = score_table(preds)
    results["test_scores"] = table

    def pick(year, c, r):
        return next(t for t in table if t["year"] == year and t["classifier"] == c and t["regressor"] == r)

    for year in (TEST_YEAR, TOKYO_YEAR):
        results[f"test_{year}"] = {
            "final": pick(year, best_c, best_r),
            "final_rolling": pick(year, roll_c, roll_r),
            "baseline_linear": pick(year, "none", "linear"),
            "single_stage_best_regressor": pick(year, "none", best_r),
            "last_games": pick(year, "naive", "last_games"),
        }
        print(f"\n== Test {year}: paper protocol={best_c}+{best_r}, rolling protocol={roll_c}+{roll_r}")
        for k, v in results[f"test_{year}"].items():
            print(f"  {k:28s} rmse={v['rmse']:.2f} mae={v['mae']:.2f} top10_rmse={v['top10_rmse']:.2f}")
        final = preds[(preds.Year == year) & (preds.Classifier == roll_c) & (preds.Regressor == roll_r)]
        print(final.sort_values("Actual", ascending=False).head(10)
              [["Country", "Actual", "Predicted", "Medals_Last_Games"]].round(1).to_string(index=False))

    results["params"] = {k: {n: jsonable(p) for n, p in v.items()} for k, v in results["params"].items()}
    results["paper"] = PAPER
    results["n_rows"] = {str(y): int(n) for y, n in data.groupby("Year").size().items()}
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {RESULTS_JSON} and {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
