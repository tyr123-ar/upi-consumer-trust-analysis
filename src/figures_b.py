"""UPI sentiment and topic figures plus a scoped paper comparison table."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C

SENTIMENT_COLORS = {
    "negative": "#d1495b",
    "neutral": "#b7b7b7",
    "positive": "#2a9d8f",
}


def _sentiment_data():
    if C.ROBERTA_OUTPUT.exists():
        scored = pd.read_csv(
            C.ROBERTA_OUTPUT,
            dtype={"app": "string", "id": "string", "model_revision": "string"},
        )
        scored = scored[scored.model_revision.eq(C.ROBERTA_REVISION)]
        clean_path = C.INTER / "upi_clean.csv"
        if not scored.empty and clean_path.exists():
            corpus = pd.read_csv(clean_path, usecols=["app", "id"], dtype={"id": "string"})
            scored = scored.drop_duplicates(["app", "id"], keep="last")
            scored_keys = set(zip(scored.app.astype(str), scored.id.astype(str)))
            corpus_keys = set(zip(corpus.app.astype(str), corpus.id.astype(str)))
            if corpus_keys.issubset(scored_keys):
                return scored, "RoBERTa"
    fallback = C.INTER / "upi_scored.csv"
    if fallback.exists():
        return pd.read_csv(fallback, usecols=["app", "vader_cls"]).rename(
            columns={"vader_cls": "predicted_label"}
        ), "VADER"
    raise FileNotFoundError("Neither RoBERTa results nor VADER-scored UPI data exist")


def sentiment_figure():
    scored, method = _sentiment_data()
    shares = (
        pd.crosstab(scored.app, scored.predicted_label, normalize="index")
        .mul(100)
        .reindex(index=C.APPS, columns=["negative", "neutral", "positive"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = pd.Series(0.0, index=shares.index)
    for label in ("negative", "neutral", "positive"):
        ax.bar(
            shares.index,
            shares[label],
            bottom=bottom,
            label=label.title(),
            color=SENTIMENT_COLORS[label],
        )
        bottom += shares[label]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Reviews (%)")
    ax.set_xlabel("UPI app")
    ax.set_title(f"Sentiment distribution by app ({method}; UPI reviews)")
    ax.legend(title="Predicted sentiment", ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig4_upi_sentiment_distribution.png", dpi=300)
    plt.close(fig)
    return method


def topic_figure():
    path = C.TABLES / "upi_topic_distribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing topic distribution table: {path}")
    topics = pd.read_csv(path)
    if topics.empty:
        raise ValueError("UPI topic distribution table is empty")
    apps = [app for app in C.APPS if app in set(topics.app)]
    fig, axes = plt.subplots(
        len(apps), 1, figsize=(9, max(3.2, 2.8 * len(apps))), squeeze=False
    )
    for ax, app in zip(axes[:, 0], apps):
        subset = topics[topics.app.eq(app)].sort_values("topic_id")
        ax.bar(
            [f"Topic {int(topic) + 1}" for topic in subset.topic_id],
            subset.share_pct,
            color="#4c78a8",
        )
        ax.set_title(app)
        ax.set_ylabel("Reviews (%)")
        ax.set_ylim(0, max(100, float(subset.share_pct.max()) * 1.15))
    axes[-1, 0].set_xlabel("Dominant topic (topic IDs are app-specific)")
    fig.suptitle("UPI review topic distribution by app", y=1.01)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_upi_topic_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def comparison_table():
    reference = {
        "Overall mean rating": 4.14,
        "Ratings 4-5 (%)": 78.6,
        "Ratings 1-2 (%)": 17.6,
        "Google Pay mean rating": 3.65,
        "Paytm mean rating": 4.27,
        "PhonePe mean rating": 4.11,
    }
    all_path = C.INTER / "upi_all.csv"
    current = {}
    if all_path.exists():
        reviews = pd.read_csv(all_path, usecols=["app", "score"])
        current = {
            "Overall mean rating": float(reviews.score.mean()),
            "Ratings 4-5 (%)": float(100 * reviews.score.isin([4, 5]).mean()),
            "Ratings 1-2 (%)": float(100 * reviews.score.isin([1, 2]).mean()),
            **{
                f"{app} mean rating": float(reviews.loc[reviews.app.eq(app), "score"].mean())
                for app in C.APPS
            },
        }
    sentiment_path = C.TABLES / "upi_sentiment_benchmark.csv"
    if sentiment_path.exists():
        results = pd.read_csv(sentiment_path).set_index("method")
        for method in ("RoBERTa", "VADER", "TextBlob"):
            if method in results.index:
                current[f"{method} weak-label benchmark accuracy"] = float(
                    results.loc[method, "accuracy"]
                )
                current[f"{method} weak-label benchmark macro-F1"] = float(
                    results.loc[method, "macro_f1"]
                )
    rows = []
    for metric, actual in current.items():
        paper_value = reference.get(metric)
        rows.append(
            {
                "metric": metric,
                "scope": "UPI",
                "measured_value": actual,
                "reference_value": paper_value,
                "comparison_note": (
                    "Different sample/population or methodology; descriptive comparison only"
                    if paper_value is not None
                    else "No directly comparable reference value supplied"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(C.TABLES / "table13_upi_reference_comparison.csv", index=False)
    return result


def run():
    method = sentiment_figure()
    topic_figure()
    comparison = comparison_table()
    print(f"Generated UPI figures using {method} predictions")
    print(comparison.head().to_string(index=False))
    return comparison


if __name__ == "__main__":
    run()
