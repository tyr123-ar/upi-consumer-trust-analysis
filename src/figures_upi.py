"""Generate Person A's Figures 1-3 from the saved intermediate tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

APP_COLORS = {"Google Pay": "#1f77b4", "Paytm": "#2a9d8f", "PhonePe": "#7b5ea7"}

def fig1():
    all_reviews = pd.read_csv(C.INTER / "upi_all.csv")
    all_reviews = all_reviews[all_reviews.score.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ratings = pd.crosstab(all_reviews.app, all_reviews.score, normalize="index").mul(100).reindex(C.APPS)
    ratings.plot(kind="bar", ax=axes[0], width=0.8, colormap="viridis")
    axes[0].set_title("(a) Rating distribution by app")
    axes[0].set_ylabel("% of reviews")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Stars", ncol=5, fontsize=8)

    sentiment = pd.read_csv(C.TABLES / "fig1b_vader_by_app.csv").set_index("app").reindex(C.APPS)
    sentiment = sentiment.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    sentiment.plot(kind="bar", stacked=True, ax=axes[1], color=["#2a9d8f", "#bbbbbb", "#d1495b"])
    axes[1].set_title("(b) VADER sentiment by app")
    axes[1].set_ylabel("% of reviews")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig1_ratings_sentiment.png", dpi=200)
    plt.close(fig)

def fig2():
    table = pd.read_csv(C.TABLES / "table9_aspects.csv")
    if table.empty:
        raise ValueError("table9_aspects.csv is empty; no aspect matches were found")
    table = table.sort_values("Mean VADER")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d1495b" if value < 0 else "#2a9d8f" for value in table["Mean VADER"]]
    ax.barh(table["Aspect"], table["Mean VADER"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean VADER compound score of clauses mentioning the aspect")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig2_aspects.png", dpi=200)
    plt.close(fig)

def fig3():
    monthly = pd.read_csv(C.TABLES / "fig3_monthly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for app, group in monthly.groupby("app"):
        group = group.sort_values("month")
        axes[0].plot(group.month, group.mean_rating, marker="o", ms=3, label=app, color=APP_COLORS.get(app))
        axes[1].plot(group.month, group.risk_share, marker="o", ms=3, label=app, color=APP_COLORS.get(app))
    axes[0].set_title("(a) Monthly mean star rating")
    axes[1].set_title("(b) Monthly share of risk/grievance terms")
    axes[0].set_ylabel("Mean stars")
    axes[1].set_ylabel("% of deduplicated reviews")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig3_trends.png", dpi=200)
    plt.close(fig)

def run():
    fig1()
    fig2()
    fig3()
    print("Figures 1-3 saved to", C.FIGS)

if __name__ == "__main__":
    run()
