"""UPI exploratory analysis: Table 5 (rating profile) and Table 7 (bigrams)."""
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import store

STOP = sorted(set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "cannot", "never"})

def rating_profile(all_reviews, english_reviews):
    rows = []
    for app in C.APPS + ["All apps"]:
        a = all_reviews if app == "All apps" else all_reviews[all_reviews.app.eq(app)]
        e = english_reviews if app == "All apps" else english_reviews[english_reviews.app.eq(app)]
        low = a.score.isin([1, 2])
        row = {
            "App": app, "Reviews": int(len(a)),
            "Mean rating": round(a.score.mean(), 2),
            "1-2 stars %": round(100 * low.mean(), 1),
            "4-5 stars %": round(100 * a.score.isin([4, 5]).mean(), 1),
            "Mean words": round(e.n_words.mean(), 1),
        }
        if app != "All apps":
            row["Reply rate %"] = round(100 * a.has_reply.mean(), 1)
            row["Reply rate to 1-2 %"] = round(100 * a.loc[low, "has_reply"].mean(), 1) if low.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def bigrams(df, low_star, high_star, top=10):
    subset = df[df.score.between(low_star, high_star)]
    if subset.empty or not subset.text.fillna("").str.strip().any():
        return []
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=STOP, token_pattern=r"(?u)\b[\w']+\b")
    matrix = vectorizer.fit_transform(subset.text.fillna(""))
    counts = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    order = counts.argsort()[::-1][:top]
    return [(vocab[i], int(counts[i])) for i in order]

def run():
    paths = [C.INTER / name for name in ("upi_all.csv", "upi_en.csv", "upi_clean.csv")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Run prep_upi.py first; missing " + ", ".join(missing))
    all_reviews, english, clean = [pd.read_csv(p) for p in paths]
    all_reviews = all_reviews[all_reviews.score.notna()].copy()
    t5 = rating_profile(all_reviews, english)
    store.table(t5, "table5_rating_profile")
    store.put("mean_rating", float(all_reviews.score.mean()))
    store.put("pct_45", float(100 * all_reviews.score.isin([4, 5]).mean()))
    store.put("pct_12", float(100 * all_reviews.score.isin([1, 2]).mean()))
    for _, row in t5.iterrows():
        store.put(f"mean_rating_{row['App']}", float(row["Mean rating"]))

    contingency = pd.crosstab(all_reviews.app, all_reviews.score.isin([1, 2]))
    chi, p, dof, _ = chi2_contingency(contingency)
    store.put("chi2_low_by_app", {"chi2": float(chi), "df": int(dof), "p": float(p)})
    by_star = english.groupby("score").n_words.mean().round(1)
    store.put("words_by_star", by_star.to_dict())
    store.put("dup_share", {
        "en_reviews": int(len(english)), "repeats": int(len(english) - len(clean)),
        "pct": float(100 * (len(english) - len(clean)) / len(english)) if len(english) else 0.0,
        "good_count": int(english.text.eq("good").sum()),
    })

    positive, negative = bigrams(clean, 4, 5), bigrams(clean, 1, 2)
    n = max(len(positive), len(negative))
    t7 = pd.DataFrame({
        "Top bigrams 4-5 stars": [f"{w} ({c})" for w, c in positive] + [""] * (n - len(positive)),
        "Top bigrams 1-2 stars": [f"{w} ({c})" for w, c in negative] + [""] * (n - len(negative)),
    })
    store.table(t7, "table7_bigrams")
    print(t5.to_string(index=False))
    return t5, t7

if __name__ == "__main__":
    run()
