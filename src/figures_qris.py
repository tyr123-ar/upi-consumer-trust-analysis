"""Figure 4: QRIS sentiment and exploratory theme prevalence."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C

COLORS = {
    "negative": "#d1495b",
    "neutral": "#b7b7b7",
    "positive": "#2a9d8f",
}


def run(sentiment_path=None, themes_path=None, output_path=None):
    sentiment_path = Path(sentiment_path or C.TABLES / "qris_sentiment_distribution.csv")
    themes_path = Path(themes_path or C.TABLES / "qris_themes.csv")
    output_path = Path(output_path or C.FIGS / "fig4_qris.png")
    if not sentiment_path.exists() or not themes_path.exists():
        raise FileNotFoundError("Run QRIS sentiment summary and theme analysis first")

    sentiment = pd.read_csv(sentiment_path).set_index("sentiment")
    themes = pd.read_csv(themes_path).sort_values("share_pct")
    if not set(COLORS).issubset(sentiment.index):
        raise ValueError("QRIS sentiment table must contain all three model labels")
    if themes.empty:
        raise ValueError("QRIS theme table is empty")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    labels = ["negative", "neutral", "positive"]
    values = [float(sentiment.loc[label, "share_pct"]) for label in labels]
    bars = axes[0].bar(
        [label.title() for label in labels],
        values,
        color=[COLORS[label] for label in labels],
    )
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set_ylim(0, max(100, max(values) * 1.15))
    axes[0].set_ylabel("Comments (%)")
    axes[0].set_title("Indonesian RoBERTa sentiment")

    axes[1].barh(themes["category"], themes["share_pct"], color="#4c78a8")
    for index, row in enumerate(themes.itertuples(index=False)):
        axes[1].text(
            row.share_pct + 0.3,
            index,
            f"{row.share_pct:.1f}% ({row.negative_within_category_pct:.0f}% negative)"
            if pd.notna(row.negative_within_category_pct)
            else f"{row.share_pct:.1f}%",
            va="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Comments matched (%)")
    axes[1].set_title("Exploratory Indonesian keyword themes")
    axes[1].set_xlim(0, max(10, float(themes["share_pct"].max()) * 1.5))
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(f"QRIS TikTok comments (n={int(sentiment['comments'].sum())})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    run()
