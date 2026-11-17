"""UPI review topic modelling with per-app LDA and C_V model selection."""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import evaluate
import person_b_store as store

TOPIC_COUNTS = (4, 5, 6, 7, 8, 10)
TOKEN_RE = re.compile(r"[a-z]{3,}")
STOPWORDS = (
    set(ENGLISH_STOP_WORDS) - {"against", "no", "not", "nor"}
) | {
    "app",
    "upi",
    "google",
    "gpay",
    "paytm",
    "phonepe",
    "don",
    "doesn",
    "didn",
    "isn",
    "wasn",
    "weren",
    "wouldn",
    "couldn",
    "shouldn",
    "hasn",
    "haven",
    "mustn",
    "mightn",
}


def tokenize(text):
    """Lowercase English alphabetic tokens, remove stopwords and short tokens."""
    text = "" if pd.isna(text) else str(text).lower()
    return [word for word in TOKEN_RE.findall(text) if word not in STOPWORDS]


def _sentiment_labels(reviews):
    roberta_path = C.ROBERTA_OUTPUT
    if roberta_path.exists():
        predictions = pd.read_csv(
            roberta_path,
            dtype={"app": "string", "id": "string", "model_revision": "string"},
        )
        predictions = predictions[predictions.model_revision.eq(C.ROBERTA_REVISION)]
        if not predictions.empty:
            predictions = predictions.drop_duplicates(["app", "id"], keep="last")
            scored_keys = set(zip(predictions.app.astype(str), predictions.id.astype(str)))
            required_keys = set(zip(reviews.app.astype(str), reviews.id.astype(str)))
            if required_keys.issubset(scored_keys):
                return predictions[["app", "id", "predicted_label"]], "RoBERTa"
    lexicon_path = C.INTER / "upi_scored.csv"
    if lexicon_path.exists():
        scores = pd.read_csv(
            lexicon_path, usecols=["app", "id", "vader_cls"],
            dtype={"app": "string", "id": "string"},
        )
        scores = scores.rename(columns={"vader_cls": "predicted_label"})
        return scores, "VADER"
    return None, "unavailable"


def run(
    input_path=None,
    sample_per_app=12000,
    topic_counts=TOPIC_COUNTS,
    passes=4,
    iterations=50,
):
    """Fit deterministic per-app LDA models and save aggregate topic outputs."""
    input_path = Path(input_path or C.INTER / "upi_clean.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cleaned UPI corpus: {input_path}")
    if sample_per_app < 1 or not topic_counts or any(k < 2 for k in topic_counts):
        raise ValueError("Sampling must be positive and candidate topic counts must be >= 2")
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel, LdaModel
    except ImportError as exc:
        raise RuntimeError("Gensim is required for UPI topic modelling") from exc

    df = pd.read_csv(input_path, dtype={"id": "string"})
    required = {"app", "id", C.SENT_TEXT_COL}
    if missing := required - set(df.columns):
        raise KeyError(f"{input_path} is missing required columns: {sorted(missing)}")
    if df[["app", "id"]].isna().any().any():
        raise ValueError("The cleaned corpus contains missing app or review ID values")
    if df.duplicated(["app", "id"]).any():
        raise ValueError("The cleaned corpus contains duplicate app/review ID pairs")

    sentiment, sentiment_source = _sentiment_labels(df)
    if sentiment is not None:
        sentiment = sentiment.drop_duplicates(["app", "id"], keep="last")

    coherence_rows = []
    topic_rows = []
    distribution_rows = []
    assignment_frames = []
    parameters = {
        "sample_per_app": sample_per_app,
        "topic_counts": list(topic_counts),
        "passes": passes,
        "iterations": iterations,
        "seed": C.SEED,
        "coherence": "c_v",
        "sentiment_source": sentiment_source,
    }

    for app in C.APPS:
        app_reviews = df[df.app.eq(app)]
        sampled = app_reviews.sample(
            n=min(sample_per_app, len(app_reviews)), random_state=C.SEED
        ).copy()
        token_rows = [tokenize(value) for value in sampled[C.SENT_TEXT_COL]]
        eligible = [(row_index, tokens) for row_index, tokens in enumerate(token_rows) if tokens]
        if not eligible:
            raise ValueError(f"No usable tokens remain for {app} topic modelling")
        sampled = sampled.iloc[[index for index, _ in eligible]].reset_index(drop=True)
        texts = [tokens for _, tokens in eligible]
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.95)
        if len(dictionary) < 2:
            raise ValueError(f"Insufficient vocabulary for {app} topic modelling")
        known_terms = set(dictionary.values())
        texts = [[word for word in doc if word in known_terms] for doc in texts]
        usable = [index for index, doc in enumerate(texts) if doc]
        sampled = sampled.iloc[usable].reset_index(drop=True)
        texts = [texts[index] for index in usable]
        if not texts:
            raise ValueError(f"No reviews remain after vocabulary filtering for {app}")
        bow = [dictionary.doc2bow(tokens) for tokens in texts]

        candidates = []
        for k in topic_counts:
            model = LdaModel(
                corpus=bow,
                id2word=dictionary,
                num_topics=int(k),
                random_state=C.SEED,
                passes=passes,
                iterations=iterations,
                eval_every=None,
                alpha="symmetric",
                eta="symmetric",
                minimum_probability=0.0,
            )
            coherence = float(
                CoherenceModel(
                    model=model,
                    texts=texts,
                    dictionary=dictionary,
                    coherence="c_v",
                    processes=1,
                ).get_coherence()
            )
            coherence_rows.append(
                {
                    "app": app,
                    "k": int(k),
                    "coherence_c_v": coherence,
                    "sampled_reviews": int(len(sampled)),
                    "tokenized_reviews": int(len(texts)),
                }
            )
            candidates.append((coherence, int(k), model))

        best_coherence, best_k, best_model = max(
            candidates, key=lambda candidate: (candidate[0], -candidate[1])
        )
        chosen = []
        for bow_row in bow:
            distribution = best_model.get_document_topics(
                bow_row, minimum_probability=0.0
            )
            chosen.append(max(distribution, key=lambda item: item[1]))
        assignments = pd.DataFrame(
            {
                "app": app,
                "id": sampled.id.astype(str),
                "topic_id": [int(item[0]) for item in chosen],
                "dominant_topic_share": [float(item[1]) for item in chosen],
            }
        )
        assignments["app"] = assignments.app.astype("string")
        assignments["id"] = assignments.id.astype("string")
        if sentiment is not None:
            assignments = assignments.merge(
                sentiment,
                on=["app", "id"],
                how="left",
                validate="one_to_one",
            )
        else:
            assignments["predicted_label"] = None

        proportions = evaluate.topic_distribution(
            assignments.topic_id.tolist(), num_topics=best_k
        )
        for topic_id, values in proportions.items():
            member_rows = assignments[assignments.topic_id.eq(topic_id)]
            labels = member_rows.predicted_label.dropna()
            negative_pct = (
                100 * labels.eq("negative").mean() if not labels.empty else None
            )
            words = [
                word for word, _ in best_model.show_topic(topic_id, topn=10)
            ]
            topic_rows.append(
                {
                    "app": app,
                    "selected_k": best_k,
                    "coherence_c_v": best_coherence,
                    "topic_id": topic_id,
                    "top_words": ", ".join(words),
                    "reviews": values["count"],
                    "share_pct": round(100 * values["proportion"], 2),
                    "negative_sentiment_pct": (
                        round(negative_pct, 2) if negative_pct is not None else None
                    ),
                    "sentiment_source": sentiment_source,
                }
            )
            distribution_rows.append(
                {
                    "app": app,
                    "topic_id": topic_id,
                    "reviews": values["count"],
                    "share_pct": round(100 * values["proportion"], 2),
                }
            )
        assignment_frames.append(assignments)

    coherence = pd.DataFrame(coherence_rows)
    topic_table = pd.DataFrame(topic_rows)
    distribution = pd.DataFrame(distribution_rows)
    coherence.to_csv(C.TABLES / "upi_topic_coherence.csv", index=False)
    topic_table.to_csv(C.TABLES / "table8_upi_topics.csv", index=False)
    distribution.to_csv(C.TABLES / "upi_topic_distribution.csv", index=False)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments.to_csv(C.TOPIC_ASSIGNMENTS, index=False)
    store.put("upi_topic_model", parameters)
    store.put(
        "upi_topic_optimal_k",
        topic_table.groupby("app").selected_k.first().astype(int).to_dict(),
    )
    store.put(
        "upi_topic_coherence",
        coherence.groupby("app")
        .apply(lambda group: dict(zip(group.k.astype(str), group.coherence_c_v)))
        .to_dict(),
    )
    store.put("upi_topic_sentiment_source", sentiment_source)
    return topic_table, coherence, distribution


if __name__ == "__main__":
    topics, coherences, _ = run()
    print(coherences.to_string(index=False))
    print(topics.to_string(index=False))
