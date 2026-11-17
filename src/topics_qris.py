"""Deterministic Indonesian LDA topics for the cleaned QRIS comments."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import evaluate
import person_b_store as store

TOPIC_COUNTS = (2, 3, 4, 5, 6)
TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
STOPWORDS = set(
    """yang dan di ke dari ini itu untuk dengan pada juga tidak ada saya aku kamu dia
    mereka kita kami akan atau karena sudah belum bisa jadi kalau kalo saja lagi sama
    sih deh dong nya lah pun kok kan apa siapa bagaimana gimana dimana kapan mengapa
    adalah ialah yaitu serta dalam atas bawah oleh sebagai agar supaya sangat lebih
    kurang hanya masih telah sedang dapat harus mau ingin pun para sebuah setelah
    sebelum saat ketika terhadap tentang daripada kepada antara tanpa serta""".split()
)


def tokenize(text):
    text = "" if pd.isna(text) else str(text).lower()
    return [word for word in TOKEN_RE.findall(text) if word not in STOPWORDS]


def run(input_path=None, topic_counts=TOPIC_COUNTS, passes=8, iterations=100):
    input_path = Path(input_path or C.INTER / "qris_clean.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cleaned QRIS corpus: {input_path}")
    if not topic_counts or any(k < 2 for k in topic_counts):
        raise ValueError("Candidate topic counts must contain values >= 2")
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel, LdaModel
    except ImportError as exc:
        raise RuntimeError("Gensim is required for QRIS topic modelling") from exc

    clean = pd.read_csv(input_path)
    required = {"qris_id", "norm"}
    if required - set(clean.columns):
        raise KeyError(f"{input_path} is missing columns: {sorted(required - set(clean.columns))}")
    if clean.empty or clean["qris_id"].duplicated().any():
        raise ValueError("QRIS topic input must contain unique comment IDs")
    token_docs = [tokenize(value) for value in clean["norm"]]
    keep = [index for index, words in enumerate(token_docs) if words]
    if not keep:
        raise ValueError("No Indonesian tokens remain for QRIS topic modelling")
    comments = clean.iloc[keep].reset_index(drop=True)
    texts = [token_docs[index] for index in keep]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    if len(dictionary) < 2:
        raise ValueError("Insufficient QRIS vocabulary after token filtering")
    known_terms = set(dictionary.values())
    texts = [[word for word in doc if word in known_terms] for doc in texts]
    nonempty = [index for index, doc in enumerate(texts) if doc]
    comments = comments.iloc[nonempty].reset_index(drop=True)
    texts = [texts[index] for index in nonempty]
    if len(texts) < 3:
        raise ValueError("Too few QRIS comments remain for topic modelling")
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    candidates = tuple(
        k for k in topic_counts if 2 <= k <= min(len(dictionary), len(texts) - 1)
    )
    if not candidates:
        raise ValueError("No candidate LDA topic count fits the QRIS corpus")

    coherence_rows = []
    models = []
    for k in candidates:
        model = LdaModel(
            corpus=corpus,
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
                "k": int(k),
                "coherence_c_v": coherence,
                "comments": int(len(comments)),
                "vocabulary_terms": int(len(dictionary)),
            }
        )
        models.append((coherence, int(k), model))

    best_coherence, best_k, best_model = max(
        models, key=lambda item: (item[0], -item[1])
    )
    scored_path = C.QRIS_OUTPUT
    if not scored_path.exists():
        raise FileNotFoundError("Run Indonesian RoBERTa scoring before QRIS topic sentiment")
    scored = pd.read_csv(scored_path)
    if not scored["model_revision"].eq(C.QRIS_REVISION).all():
        raise ValueError("QRIS topic sentiment uses an unexpected model revision")
    sentiment = scored.set_index("qris_id")["predicted_label"]

    assignments = []
    topic_rows = []
    distribution_rows = []
    assigned_topics = []
    for row, bow in zip(comments.itertuples(index=False), corpus):
        distribution = best_model.get_document_topics(bow, minimum_probability=0.0)
        topic_id, dominant_share = max(distribution, key=lambda item: item[1])
        assigned_topics.append(int(topic_id))
        assignments.append(
            {
                "qris_id": int(row.qris_id),
                "topic_id": int(topic_id),
                "dominant_topic_share": float(dominant_share),
                "predicted_label": sentiment.loc[row.qris_id],
            }
        )
    proportions = evaluate.topic_distribution(assigned_topics, num_topics=best_k)
    assignment_frame = pd.DataFrame(assignments)
    for topic_id, stats in proportions.items():
        members = assignment_frame[assignment_frame["topic_id"].eq(topic_id)]
        labels = members["predicted_label"]
        words = [word for word, _ in best_model.show_topic(topic_id, topn=10)]
        negative_count = int(labels.eq("negative").sum())
        topic_rows.append(
            {
                "selected_k": best_k,
                "coherence_c_v": best_coherence,
                "topic_id": topic_id,
                "top_words": ", ".join(words),
                "comments": stats["count"],
                "share_pct": round(100 * stats["proportion"], 2),
                "negative_comments": negative_count,
                "negative_within_topic_pct": (
                    round(100 * negative_count / len(members), 2) if len(members) else None
                ),
            }
        )
        distribution_rows.append(
            {
                "topic_id": topic_id,
                "comments": stats["count"],
                "share_pct": round(100 * stats["proportion"], 2),
            }
        )

    coherence_frame = pd.DataFrame(coherence_rows)
    topic_frame = pd.DataFrame(topic_rows)
    coherence_frame.to_csv(C.TABLES / "qris_topic_coherence.csv", index=False)
    topic_frame.to_csv(C.TABLES / "qris_topics.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(
        C.TABLES / "qris_topic_distribution.csv", index=False
    )
    assignment_frame.to_csv(C.INTER / "qris_topic_assignments.csv", index=False)
    store.put(
        "qris_topic_model",
        {
            "candidate_k": list(candidates),
            "selected_k": int(best_k),
            "coherence_c_v": float(best_coherence),
            "passes": passes,
            "iterations": iterations,
            "seed": C.SEED,
            "comments": int(len(comments)),
        },
    )
    print(coherence_frame.to_string(index=False))
    print(topic_frame.to_string(index=False))
    return topic_frame, coherence_frame


if __name__ == "__main__":
    run()
