"""Full-corpus VADER and TextBlob sentiment for UPI reviews (Table 6 top block)."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import store

def classify(score):
    if score >= C.VADER_CUT:
        return "positive"
    if score <= -C.VADER_CUT:
        return "negative"
    return "neutral"

def star_label(stars):
    if pd.isna(stars):
        return np.nan
    if stars >= 4:
        return "positive"
    if stars == 3:
        return "neutral"
    return "negative"

def score_texts(texts):
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    vader = [analyzer.polarity_scores(str(t))["compound"] for t in texts]
    textblob = [TextBlob(str(t)).sentiment.polarity for t in texts]
    return np.asarray(vader), np.asarray(textblob)

def run(text_col=None):
    text_col = text_col or C.SENT_TEXT_COL
    path = C.INTER / "upi_clean.csv"
    if not path.exists():
        raise FileNotFoundError("Run prep_upi.py first")
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise KeyError(f"{text_col!r} not found. Available columns: {list(df.columns)}")
    df[text_col] = df[text_col].fillna("").astype(str)
    print(f"Scoring {len(df):,} reviews using {text_col!r}")
    df["vader"], df["tb"] = score_texts(df[text_col])
    df["vader_cls"] = df.vader.map(classify)
    df["tb_cls"] = df.tb.map(classify)
    df["star_cls"] = df.score.map(star_label)
    df.to_csv(C.INTER / "upi_scored.csv", index=False)

    valid = df.star_cls.notna()
    result = {}
    for name in ("vader", "tb"):
        pred = df.loc[valid, f"{name}_cls"]
        truth = df.loc[valid, "star_cls"]
        result[name] = {
            "acc": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", labels=["negative", "neutral", "positive"], zero_division=0)),
            "shares": (100 * df[f"{name}_cls"].value_counts(normalize=True)).round(1).to_dict(),
            "rho": float(spearmanr(df.loc[valid, "score"], df.loc[valid, name]).statistic),
        }
    result["vader_mean_by_star"] = df.groupby("score").vader.mean().round(2).to_dict()
    result["text_col"] = text_col
    store.put("lexicon_sentiment", result)
    by_app = pd.crosstab(df.app, df.vader_cls, normalize="index").mul(100).round(1)
    by_app = by_app.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    store.table(by_app.reset_index(), "fig1b_vader_by_app")
    print(result)
    return result

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
