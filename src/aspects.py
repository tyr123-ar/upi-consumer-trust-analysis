"""Clause-level aspect sentiment using VADER; produces Table 9 and aspect_events.csv."""
import re
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import lexicons as L
import store

_SPLIT = re.compile(r"[.,;:!?\n]+|\b(?:but|however)\b", re.I)

def clauses(text):
    return [part.strip() for part in _SPLIT.split(str(text or "")) if part and part.strip()]

def run():
    path = C.INTER / "upi_clean.csv"
    if not path.exists():
        raise FileNotFoundError("Run prep_upi.py first")
    df = pd.read_csv(path)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    events = []
    for row_id, text, stars in zip(df.index, df.text_light.fillna(""), df.score):
        for clause in clauses(text):
            score = analyzer.polarity_scores(clause)["compound"]
            for aspect, regex in L.ASPECT_RE.items():
                if regex.search(clause):
                    events.append((row_id, aspect, score, stars, clause))
    events_df = pd.DataFrame(events, columns=["rid", "aspect", "vader", "star", "clause"])
    events_df.to_csv(C.INTER / "aspect_events.csv", index=False)

    rows = []
    for aspect, group in events_df.groupby("aspect"):
        unique_reviews = group.drop_duplicates("rid")
        rows.append({
            "Aspect": aspect, "Clauses": int(len(group)),
            "Share of reviews %": round(100 * group.rid.nunique() / len(df), 1) if len(df) else 0,
            "Mean VADER": round(group.vader.mean(), 2),
            "Positive %": round(100 * group.vader.ge(C.VADER_CUT).mean(), 1),
            "Negative %": round(100 * group.vader.le(-C.VADER_CUT).mean(), 1),
            "Mean star of review": round(unique_reviews.star.mean(), 2),
        })
    table = pd.DataFrame(rows, columns=["Aspect", "Clauses", "Share of reviews %", "Mean VADER", "Positive %", "Negative %", "Mean star of review"])
    if not table.empty:
        table = table.sort_values("Mean VADER", ascending=False)
    store.table(table, "table9_aspects")
    print(table.to_string(index=False))
    return table

if __name__ == "__main__":
    run()
