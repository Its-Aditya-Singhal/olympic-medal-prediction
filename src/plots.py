"""Figures for the write-up and slides (reports/figures/*.png)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_dataset import load_dataset
from config import FEATURES, FIGURES, PREDICTIONS_CSV, RESULTS_JSON, TARGET, TEST_YEAR, TOKYO_YEAR

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", FIGURES / name)


def null_heatmap(data: pd.DataFrame):
    """Paper Fig. 1: where indicators are missing (those rows are dropped)."""
    cols = ["Year", *[f for f in FEATURES if f != "Year"], TARGET]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(data[cols].isna().to_numpy().T, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f6f5f2", BLUE]),
              interpolation="nearest")
    ax.set_yticks(range(len(cols)), cols)
    ax.set_xlabel(f"country-Games rows (n={len(data)}); blue = missing")
    ax.grid(False)
    ax.set_title("Missing World Bank indicators across the dataset")
    save(fig, "fig1_missing_values.png")


def feature_scatter(data: pd.DataFrame):
    """Paper Fig. 2: each feature against medals won."""
    feats = [f for f in FEATURES if f != "Year"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 6.5))
    for ax, f in zip(axes.flat, feats):
        ax.scatter(data[f], data[TARGET], s=6, color=BLUE, alpha=0.45, linewidths=0)
        ax.set_title(f, fontsize=8)
        if f in ("GDP", "Pop", "Area", "GDP_Per_Capita"):
            ax.set_xscale("log")
        ax.tick_params(labelsize=6)
    fig.supylabel("medals won", color=INK2, fontsize=9)
    fig.suptitle("Features vs medals won (1988-2020)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_feature_scatter.png")


def pred_vs_actual(preds: pd.DataFrame, year: int, clf: str, reg: str, title: str, name: str):
    g = preds[(preds.Year == year) & (preds.Classifier == clf) & (preds.Regressor == reg)]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    top = max(g.Actual.max(), g.Predicted.max()) * 1.05
    ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--", label="perfect prediction")
    ax.scatter(g.Actual, g.Predicted, s=22, color=BLUE, edgecolor="white", linewidth=0.6, label="country")
    for i, (_, r) in enumerate(g.nlargest(6, "Actual").iterrows()):
        ax.annotate(r.ISO3, (r.Actual, r.Predicted), xytext=(5, -9 if i % 2 else 3), textcoords="offset points",
                    fontsize=7, color=INK)
    rmse = float(np.sqrt(np.mean((g.Actual - g.Predicted) ** 2)))
    ax.set(xlabel="actual medals", ylabel="predicted medals", xlim=(0, top), ylim=(0, top))
    ax.set_title(f"{title}\nRMSE = {rmse:.2f}")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, name)


def tuning_bars(results: dict):
    """Paper Fig. 4: 2012 RMSE of each regressor (behind the logistic classifier), before/after tuning."""
    regs = results["regressors"]
    names = list(regs)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    before = [regs[n]["rmse_default_logreg"] for n in names]
    after = [regs[n]["rmse_tuned_logreg"] for n in names]
    ax.bar(x - 0.2, before, 0.38, color=BLUE, label="default params")
    ax.bar(x + 0.2, after, 0.38, color=ORANGE, label="tuned (CV on 1988-2008)")
    for xi, b in zip(x, before):
        if b > 6:
            ax.text(xi - 0.2, 6.0, f"{b:.1f}\n(clipped)", ha="center", va="top", fontsize=7, color="white")
    ax.set_ylim(0, 6.5)
    ax.set_xticks(x, [regs[n]["label"].replace(" Regression", "").replace(" Regressor", "") for n in names],
                  rotation=20, ha="right")
    ax.set_ylabel("RMSE on 2012 (medals)")
    ax.set_title("Regressors before vs after hyperparameter tuning (validation 2012)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_tuning.png")


def classifier_bars(results: dict):
    clfs, paper = results["classifiers"], results["paper"]["classifier_accuracy_2012"]
    names = list(clfs)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - 0.2, [paper[n] for n in names], 0.38, color=INK2, alpha=0.35, label="paper")
    ax.bar(x + 0.2, [clfs[n]["accuracy"] for n in names], 0.38, color=BLUE, label="ours")
    ax.set_ylim(0.8, 0.92)
    ax.set_xticks(x, [clfs[n]["label"].replace(" Classifier", "") for n in names], rotation=15, ha="right")
    ax.set_ylabel("accuracy on 2012")
    ax.set_title("Stage 1: medal / no-medal classifier accuracy")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_classifiers.png")


def test_comparison(results: dict):
    """RMSE on both test years for the headline models."""
    best, roll = results["best"], results["best_rolling"]
    keys = [("last_games", "naive: last Games"), ("baseline_linear", "linear baseline"),
            ("final", f"paper protocol\n({best['classifier']}+{best['regressor']})"),
            ("final_rolling", f"rolling protocol\n({roll['classifier']}+{roll['regressor']})")]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (year, color) in enumerate([(TEST_YEAR, BLUE), (TOKYO_YEAR, ORANGE)]):
        vals = [results[f"test_{year}"][k]["rmse"] for k, _ in keys]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, label=f"test {year}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [label for _, label in keys], fontsize=8)
    ax.set_ylabel("RMSE (medals per country)")
    ax.set_ylim(0, 4.4)
    ax.set_title("Test error: Rio 2016 and Tokyo 2020")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_test_comparison.png")


def main():
    results = json.loads(RESULTS_JSON.read_text())
    preds = pd.read_csv(PREDICTIONS_CSV)
    full = load_dataset(complete_only=False)
    data = load_dataset()
    best, roll = results["best"], results["best_rolling"]

    null_heatmap(full)
    feature_scatter(data)
    pred_vs_actual(preds, TEST_YEAR, "none", "linear", "Baseline linear regression, 2016", "fig3_baseline_2016.png")
    tuning_bars(results)
    classifier_bars(results)
    pred_vs_actual(preds, TEST_YEAR, best["classifier"], best["regressor"],
                   f"Paper protocol ({best['classifier']}+{best['regressor']}), 2016", "fig5_paper_2016.png")
    for year in (TEST_YEAR, TOKYO_YEAR):
        pred_vs_actual(preds, year, roll["classifier"], roll["regressor"],
                       f"Rolling protocol ({roll['classifier']}+{roll['regressor']}), {year}", f"fig6_final_{year}.png")
    test_comparison(results)


if __name__ == "__main__":
    main()
