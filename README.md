# Summer Olympics Medal Prediction: 2020 Games

**UE24CS352A Machine Learning, Mini-Project #53: "2020 Summer Olympic Games Predictions"**

We replicate the Stanford CS229 report *2020 Summer Olympics Predictions Using Machine Learning*
(B. Dobkowski, Spring 2021, [PDF](https://cs229.stanford.edu/proj2021spr/report2/81985704.pdf)) and extend it.
The paper could only test on Rio 2016 because Tokyo 2020 had not happened yet. We also predict **Tokyo 2020**
and score the predictions against the real medal table.

Question: *given a country's GDP, population, land area, growth rates, team size and medals at the previous Games,
how many medals will it win?*

## Results

RMSE = root mean squared error in medals per country (the paper calls this "average standard deviation"). Lower is better.

| Model | Rio 2016 | Tokyo 2020 |
|---|---|---|
| Naive: "same as last Games" | **3.28** | 3.68 |
| Linear regression (paper's baseline) | 3.31 | 3.43 |
| Paper protocol: select on 2012 → Logistic Regression + Ridge | 3.66 | 3.40 |
| Our protocol: rolling validation → Random Forest (single stage) | 3.57 | **3.07** |
| *Paper's reported final model (GNB + Ridge)* | *2.26* | n/a |

Findings:
- **Tokyo 2020.** The model chosen with rolling-origin validation beats both the linear baseline (−0.36) and
  the naive "same as last Games" guess (−0.61). It gets the USA within 3 medals (110 vs 113) and the host
  Japan within 8 (66 vs 58).
- **The paper's 2016 number does not reproduce.** The "actual 2016" medal counts in the paper's Table 4
  (USA 103, UK 65, Germany 44, Japan 38) are the **2012** counts (2016 was USA 121, UK 67, Germany 42, Japan 41).
  The paper says the final model was trained on 1988–2012. If we score that model on the 2012 rows, which are inside
  its training data, we get almost the same predictions as the paper (USA 104, China 86, UK 53, Germany 43 vs the
  paper's 103, 85, 52, 44). So the reported test error most likely comes from data the model was trained on.
  On the real held-out 2016 data the same model scores 3.66. See `src/paper_audit.py`.
- **The two-stage idea helps only some regressors.** On rolling validation, adding the classifier helps Poisson
  (RMSE 12.90 → 7.39) and SVR (3.52 → 3.29). For linear, ridge, lasso, weighted-linear and random forest, the
  single-stage model is as good or better. A random forest already predicts ~0 for small countries.
- **Selecting on one validation year is noisy.** The 2012-only choice (Ridge) is beaten by Random Forest on
  2004, 2008 and 2012 together, and on Tokyo 2020.
- **Medal counts are very persistent.** On 2016, no model beats "same as last Games". Adding host-country flags or
  log-scaled features did not improve validation error (`reports/ablation.csv`).

## Approach

1. **Data** (`src/build_dataset.py`). One row = one country at one Summer Games (1988–2020), 1,734 rows.
   1,650 of them have every indicator and are used for modelling.
   - Kaggle [*120 years of Olympic history*](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
     (athlete-level results 1896–2016). Team events count as **one** medal (deduplicate on Games + event + medal + NOC).
   - World Bank WDI API (`src/wdi.py`): GDP, GDP per capita, GDP growth, population, population growth, land area.
   - Tokyo 2020 medal table and team sizes from Wikipedia (`src/fetch_tokyo2020.py`), because Kaggle stops at 2016.
   - NOC codes are mapped to ISO3 codes (`src/countries.py`), including historical teams
     (URS/EUN → RUS, FRG/GDR → DEU, ROC → RUS…). Some NOC codes collide with a different ISO3 country
     (NOC BRN = Bahrain, ISO BRN = Brunei), so those are mapped explicitly too.
   - Only 1988 onward is used (the 1980/84 boycotts), as in the paper.
2. **Features** (paper Table 1): Year, GDP, GDP per capita, GDP growth, % world GDP, population, % world population,
   population growth, land area, medals last Games, total medals awarded that year, athletes, % of all athletes.
3. **Two-stage model** (`src/models.py`). A classifier predicts *medal / no medal*, then a regressor trained only on
   medal winners predicts the count. About 60% of rows are zero medals.
   - Classifiers: Logistic Regression, SVM, Gaussian Naive Bayes, MLP, Random Forest.
   - Regressors: Linear, locally weighted Linear (paper Eq. 4), Ridge, Lasso, Poisson, SVR (linear/poly/RBF/sigmoid), Random Forest.
4. **Training and evaluation** (`src/train.py`).
   - Hyperparameters: 5-fold `GridSearchCV` / `RandomizedSearchCV` on 1988–2008.
   - Paper protocol: select on 2012 (classifier by accuracy, regressor by Eq. 1 + 0.25·Eq. 2), test on 2016.
   - Our protocol: rolling-origin validation. Train on all Games before *Y* and predict *Y*, for *Y* = 2004, 2008, 2012.
     Every classifier × regressor pair (plus single-stage) is compared on the mean score.
   - Test: refit on 1988–2012 → predict 2016; refit on 1988–2016 → predict Tokyo 2020.
5. **Extras.** `src/experiments.py` (feature ablation) and `src/paper_audit.py` (checks the paper's Table 4).

## Setup and run

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Demo only.** The processed data, trained models and results are committed, so this works straight after cloning:

```bash
streamlit run app.py
```

**Full rebuild from raw data.**
1. Download the Kaggle dataset and put `athlete_events.csv` and `noc_regions.csv` in `data/raw/`.
2. Run the pipeline:

```bash
./run_all.sh            # dataset -> train -> ablation -> audit -> figures (~1 min)
./run_all.sh --refresh  # also re-download World Bank + Tokyo 2020 data
```

### Demo app tabs
- **Overview**: headline numbers and findings.
- **Predictions**: pick Rio 2016 or Tokyo 2020 and any classifier/regressor pair. Shows a predicted-vs-actual scatter,
  the top 10, and a full table.
- **Model comparison**: classifier/regressor tables next to the paper's numbers, before/after tuning, rolling
  validation, all pairs on the test years, and the ablation.
- **Country explorer**: medal history plus predictions for any country.
- **What-if**: change a country's GDP, population, team size or last-Games medals and see the Tokyo prediction change.
- **Data**: browse and download the merged dataset.

## Repository layout

```
app.py                  Streamlit demo
run_all.sh              rebuild everything
src/
  config.py             paths, feature list, train/valid/test years
  wdi.py                World Bank API download
  fetch_tokyo2020.py    Tokyo 2020 medal table + team sizes (Wikipedia)
  countries.py          NOC / country name -> ISO3 mapping
  build_dataset.py      merge + feature engineering
  models.py             model zoo, search spaces, two-stage model, weighted LR
  metrics.py            Eq. 1, Eq. 2, RMSE, MAE
  train.py              tuning, selection, testing (writes reports/results.json, predictions.csv, models/)
  experiments.py        feature ablation (reports/ablation.csv)
  paper_audit.py        checks the paper's Table 4 (reports/paper_audit.csv)
  plots.py              figures (reports/figures/)
data/external/          World Bank + Tokyo 2020 data (committed)
data/processed/         merged dataset (committed)
models/                 fitted models for the 2016 and 2020 test windows
reports/                results, figures, write-up (writeup/writeup.pdf), slides
```

## Limitations
- Rows with missing World Bank data are dropped, as in the paper: USSR 1988, North Korea, Chinese Taipei (not in WDI) and a few others.
  Their medals still count in the per-year totals.
- Tokyo 2020 team sizes come from Wikipedia's list of participating NOCs, not from the Kaggle dataset.
- Hyperparameters are tuned once on 1988–2008. The rolling validation years 2004/2008 therefore overlap that tuning
  data. Only the choice of pair is made on the rolling score.
