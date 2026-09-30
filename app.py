"""Streamlit demo: Summer Olympics medal predictions (CS229 report #53 replication + Tokyo 2020).

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import DATASET, FEATURES, FIGURES, MODELS_DIR, PREDICTIONS_CSV, RESULTS_JSON, TEST_YEAR, TOKYO_YEAR  # noqa: E402
from metrics import report  # noqa: E402
from models import two_stage_predict  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
CLF_LABELS = {"none": "None (single-stage regression)", "logreg": "Logistic Regression", "svc": "SVM",
              "gnb": "Gaussian Naive Bayes", "mlp": "MLP", "rf": "Random Forest"}
REG_LABELS = {"linear": "Linear", "weighted_lr": "Weighted Linear", "ridge": "Ridge", "lasso": "Lasso",
              "poisson": "Poisson", "svr": "SVR", "rf": "Random Forest"}

st.set_page_config(page_title="Olympic Medal Predictions", page_icon="🏅", layout="wide")


@st.cache_data
def load_all():
    results = json.loads(RESULTS_JSON.read_text())
    preds = pd.read_csv(PREDICTIONS_CSV)
    data = pd.read_csv(DATASET)
    ablation_path = RESULTS_JSON.parent / "ablation.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else None
    return results, preds, data, ablation


@st.cache_resource
def load_window(year: int):
    return joblib.load(MODELS_DIR / f"window_{year}.joblib")


results, preds, data, ablation = load_all()
best, roll = results["best"], results["best_rolling"]


def pair_label(c, r):
    return f"{CLF_LABELS[c].split(' (')[0]} + {REG_LABELS[r]}" if c != "none" else f"{REG_LABELS[r]} (single stage)"


def scatter(g: pd.DataFrame, title: str):
    top = max(g.Actual.max(), g.Predicted.max()) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect prediction",
                             line=dict(color=GRAY, dash="dash", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=g.Actual, y=g.Predicted, mode="markers", name="country",
        marker=dict(size=9, color=BLUE, line=dict(color="white", width=1)),
        customdata=np.stack([g.Country, g.Medals_Last_Games], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>actual %{x}<br>predicted %{y:.1f}"
                      "<br>last Games %{customdata[1]}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="actual medals", yaxis_title="predicted medals",
                      height=480, margin=dict(l=10, r=10, t=50, b=40), legend=dict(orientation="h", y=-0.15, x=0))
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏅 Olympic Medal Predictions")
st.sidebar.caption("UE24CS352A mini-project #53: replication of *2020 Summer Olympics Predictions "
                   "Using Machine Learning* (Dobkowski, CS229 Spring 2021), extended to Tokyo 2020.")
st.sidebar.markdown(
    f"**Selected models**\n\n"
    f"- Paper protocol (validate on 2012): **{pair_label(best['classifier'], best['regressor'])}**\n"
    f"- Rolling protocol (2004/08/12): **{pair_label(roll['classifier'], roll['regressor'])}**")

tabs = st.tabs(["Overview", "Predictions", "Model comparison", "Country explorer", "What-if", "Data"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    st.header("Predicting Summer Olympic medal counts from economic indicators")
    t20, t16 = results[f"test_{TOKYO_YEAR}"], results[f"test_{TEST_YEAR}"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokyo · ours", f"{t20['final_rolling']['rmse']:.2f}",
              f"{t20['final_rolling']['rmse'] - t20['baseline_linear']['rmse']:+.2f} vs linear",
              delta_color="inverse")
    c2.metric("Tokyo · linear", f"{t20['baseline_linear']['rmse']:.2f}")
    c3.metric("Tokyo · naive", f"{t20['last_games']['rmse']:.2f}")
    c4.metric("Rio 2016 · ours", f"{t16['final_rolling']['rmse']:.2f}",
              f"paper: {results['paper']['final_2016_rmse']}", delta_color="off")
    st.caption("All numbers are RMSE in medals per country (the paper's 'average standard deviation'). Lower is better.")
    st.markdown("""
**Problem.** Given a country's economic and demographic indicators (GDP, population, land area, growth rates),
its team size and its medals at the previous Games, how many medals will it win?

**Pipeline** (as in the paper):
1. Kaggle *120 years of Olympic history* (athlete results, 1988-2016) + World Bank indicators → one row per country per Games.
   Team events count as one medal. Rows with missing indicators are dropped.
2. **Two-stage model.** Most countries win nothing, so a classifier first predicts *medal / no medal*, then a
   regressor trained only on medal winners predicts *how many*.
3. Tune with K-fold CV on 1988-2008, select on 2012, test on 2016.
4. **Extension.** Tokyo 2020 has now happened, so we retrain on 1988-2016, predict Tokyo and score against the real
   medal table. We also select models with **rolling-origin validation** (2004, 2008, 2012) instead of one year.
""")
    st.info(f"""**What we found**
- Scoring every pair on three validation years picks **{pair_label(roll['classifier'], roll['regressor'])}**.
  On Tokyo 2020 it beats the linear baseline and the naive "same as last Games" guess.
- The two-stage idea is not a general win. The classifier helps Poisson and SVR, but for linear, ridge, lasso and
  Random Forest the single-stage model is as good or better (Random Forest already predicts ~0 for small countries).
- The paper's 2016 result (RMSE 2.26) could not be reproduced (we get {t16['final']['rmse']:.2f} with the same protocol).
  Its Table 4 "actual 2016" counts (USA 103, UK 65, Germany 44, Japan 38) are the **2012** counts.
  Evaluating on 2012 rows that are inside the training set reproduces its numbers almost exactly (USA 104, China 86, UK 53,
  Germany 43), so the paper's test set most likely overlapped its training data.
- On 2016, no model beats the naive last-Games guess. Medal counts are very persistent, which makes that guess a strong baseline.""")

# ---------------------------------------------------------------- predictions
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    year = c1.radio("Games", [TOKYO_YEAR, TEST_YEAR], format_func=lambda y: f"Tokyo {y}" if y == 2020 else f"Rio {y}",
                    horizontal=True)
    clf = c2.selectbox("Stage 1 classifier", list(CLF_LABELS), index=list(CLF_LABELS).index(roll["classifier"]),
                       format_func=CLF_LABELS.get)
    reg = c3.selectbox("Stage 2 regressor", list(REG_LABELS), index=list(REG_LABELS).index(roll["regressor"]),
                       format_func=REG_LABELS.get)
    g = preds[(preds.Year == year) & (preds.Classifier == clf) & (preds.Regressor == reg)].copy()
    m = report(g.Actual, g.Predicted)
    naive = preds[(preds.Year == year) & (preds.Classifier == "naive")]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RMSE", f"{m['rmse']:.2f}")
    k2.metric("MAE", f"{m['mae']:.2f}")
    k3.metric("Top-10 RMSE", f"{m['top10_rmse']:.2f}")
    k4.metric("Naive RMSE", f"{report(naive.Actual, naive.Predicted)['rmse']:.2f}")

    left, right = st.columns([1, 1])
    left.plotly_chart(scatter(g, f"{pair_label(clf, reg)}: {year}"), use_container_width=True)
    g["Error"] = g.Predicted - g.Actual
    top10 = g.nlargest(10, "Actual")
    fig = go.Figure([go.Bar(y=top10.Country, x=top10.Actual, name="actual", orientation="h", marker_color=GRAY),
                     go.Bar(y=top10.Country, x=top10.Predicted, name="predicted", orientation="h", marker_color=BLUE)])
    fig.update_layout(title="Top 10 countries by actual medals", barmode="group", height=460,
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15, x=0))
    right.plotly_chart(fig, use_container_width=True)
    st.dataframe(g.sort_values("Actual", ascending=False)[["Country", "Actual", "Predicted", "Error", "Medals_Last_Games"]]
                 .round(1), use_container_width=True, hide_index=True, height=320)

# ---------------------------------------------------------------- comparison
with tabs[2]:
    st.subheader("Stage 1: classifiers (validation 2012)")
    cl = pd.DataFrame(results["classifiers"]).T.infer_objects()
    cl["paper accuracy"] = pd.Series(results["paper"]["classifier_accuracy_2012"])
    st.dataframe(cl[["label", "accuracy_default", "accuracy", "f1", "cv_accuracy", "paper accuracy"]]
                 .rename(columns={"accuracy_default": "accuracy (default)", "accuracy": "accuracy (tuned)"}),
                 use_container_width=True)

    st.subheader("Stage 2: regressors (validation 2012)")
    rg = pd.DataFrame(results["regressors"]).T.infer_objects()
    rg["paper RMSE"] = pd.Series(results["paper"]["regressor_rmse_2012"])
    st.dataframe(rg[["label", "rmse_default_logreg", "rmse_tuned_logreg", "rmse", "top10_rmse", "combined", "paper RMSE"]]
                 .rename(columns={"rmse_default_logreg": "RMSE default (behind logreg)",
                                  "rmse_tuned_logreg": "RMSE tuned (behind logreg)",
                                  "rmse": f"RMSE behind {best['classifier']}",
                                  "combined": "Eq.1 + 0.25 Eq.2"}),
                 use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "fig4_tuning.png"), caption="Before vs after tuning (paper Fig. 4)")
    c2.image(str(FIGURES / "fig_test_comparison.png"), caption="Test RMSE of the headline models")

    st.subheader("Rolling-origin validation (train on all Games before Y, predict Y)")
    rv = pd.DataFrame(results["rolling"])
    rv.insert(0, "model", [pair_label(c, r) if c != "naive" else "naive: last Games" for c, r in zip(rv.classifier, rv.regressor)])
    st.dataframe(rv.drop(columns=["classifier", "regressor"]).round(2), use_container_width=True, hide_index=True, height=300)

    st.subheader("All pairs on the test years")
    ts = pd.DataFrame(results["test_scores"])
    ts = ts[ts.classifier != "naive"].pivot_table(index=["classifier", "regressor"], columns="year", values="rmse").round(2)
    st.dataframe(ts, use_container_width=True, height=300)

    if ablation is not None:
        st.subheader("Feature ablation: host-country flags and log-scaled features")
        st.caption("Chosen on rolling validation. None of the variants beats the paper's feature set there, so the final "
                   "models keep it (the test columns are shown only for reference).")
        st.dataframe(ablation.round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- country explorer
with tabs[3]:
    countries = sorted(data.Country.unique())
    country = st.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
    hist = data[data.Country == country].sort_values("Year")
    p = preds[(preds.Country == country) & (preds.Classifier == roll["classifier"]) & (preds.Regressor == roll["regressor"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.Year, y=hist.Medals, mode="lines+markers", name="actual medals",
                             line=dict(color=GRAY, width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=p.Year, y=p.Predicted, mode="markers", name=f"predicted ({pair_label(roll['classifier'], roll['regressor'])})",
                             marker=dict(size=12, color=ORANGE, symbol="diamond")))
    fig.update_layout(title=f"{country}: medals per Summer Games", xaxis=dict(tickvals=list(range(1988, 2021, 4))),
                      yaxis_title="medals", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15, x=0))
    st.plotly_chart(fig, use_container_width=True)
    if not hist.Complete.all():
        st.warning("Some Games are missing World Bank indicators for this country. Those rows are excluded from "
                   "training and testing.")
    st.dataframe(hist.set_index("Year")[["Medals", *[f for f in FEATURES if f != "Year"]]].T, use_container_width=True)

# ---------------------------------------------------------------- what-if
with tabs[4]:
    st.markdown("Change a country's Tokyo 2020 inputs and see what the model (trained on 1988-2016) predicts.")
    window = load_window(TOKYO_YEAR)
    rows20 = data[(data.Year == TOKYO_YEAR) & data.Complete].set_index("Country")
    c1, c2, c3 = st.columns(3)
    who = c1.selectbox("Country ", sorted(rows20.index), index=sorted(rows20.index).index("India"))
    wclf = c2.selectbox("Classifier ", list(CLF_LABELS), index=list(CLF_LABELS).index(roll["classifier"]),
                        format_func=CLF_LABELS.get)
    wreg = c3.selectbox("Regressor ", list(REG_LABELS), index=list(REG_LABELS).index(roll["regressor"]),
                        format_func=REG_LABELS.get)
    base = rows20.loc[who]
    s1, s2, s3, s4 = st.columns(4)
    gdp_x = s1.slider("GDP multiplier", 0.25, 4.0, 1.0, 0.05)
    pop_x = s2.slider("Population multiplier", 0.25, 4.0, 1.0, 0.05)
    athletes = s3.slider("Team size (athletes)", 1, 800, int(base.Athletes))
    last = s4.slider("Medals at the previous Games", 0, 130, int(base.Medals_Last_Games))

    row = base.copy()
    row["GDP"] *= gdp_x
    row["Pop"] *= pop_x
    row["GDP_Per_Capita"] = row["GDP"] / row["Pop"]
    row["Pct_World_GDP"] *= gdp_x
    row["Pct_World_Pop"] *= pop_x
    row["Athletes"] = athletes
    row["Pct_Athletes"] = 100 * athletes / base.Total_Athletes_Year
    row["Medals_Last_Games"] = last

    def predict(r):
        X = r[FEATURES].to_numpy(float).reshape(1, -1)
        if wclf == "none":
            return float(np.clip(window["single"][wreg].predict(X), 0, None)[0])
        return float(two_stage_predict(window["classifiers"][wclf], window["regressors"][wreg], X)[0])

    k1, k2, k3 = st.columns(3)
    k1.metric("Actual Tokyo 2020 medals", int(base.Medals))
    k2.metric("Model prediction (real inputs)", f"{predict(base):.1f}")
    k3.metric("Model prediction (your inputs)", f"{predict(row):.1f}", f"{predict(row) - predict(base):+.1f}")
    st.caption("Tree models (Random Forest) predict in steps and cannot go beyond the medal counts seen in training. "
               "Linear models respond smoothly to every slider.")

# ---------------------------------------------------------------- data
with tabs[5]:
    st.markdown(f"**{len(data)} country-Games rows**, of which {int(data.Complete.sum())} have every indicator and "
                "are used for modelling. Sources: Kaggle *120 years of Olympic history* (1988-2016), World Bank WDI API, "
                "Wikipedia Tokyo 2020 medal table and team sizes.")
    yr = st.multiselect("Years", sorted(data.Year.unique()), default=[TOKYO_YEAR])
    view = data[data.Year.isin(yr)].sort_values(["Year", "Medals"], ascending=[True, False])
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)
    st.download_button("Download dataset.csv", data.to_csv(index=False), "dataset.csv", "text/csv")
