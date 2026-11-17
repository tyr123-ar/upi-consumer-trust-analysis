"""RoBERTa sentiment scoring and weak-label benchmark for the UPI corpus."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as C
import evaluate
import person_b_store as store


def _label_mapping(model_config):
    labels = {}
    fallback = ("negative", "neutral", "positive")
    for index, name in model_config.id2label.items():
        normalized = str(name).strip().lower()
        if normalized.startswith("label_"):
            normalized = fallback[int(index)] if int(index) < len(fallback) else normalized
        if normalized not in evaluate.LABELS:
            raise ValueError(f"Unexpected label {name!r} in the configured RoBERTa model")
        labels[int(index)] = normalized
    if set(labels.values()) != set(evaluate.LABELS):
        raise ValueError(f"Model must expose exactly {evaluate.LABELS}; found {labels}")
    return labels


def run_roberta(input_path=None, output_path=None, batch_size=None):
    """Score the clean corpus, appending completed batches for resumability."""
    input_path = Path(input_path or C.INTER / "upi_clean.csv")
    output_path = Path(output_path or C.ROBERTA_OUTPUT)
    if not input_path.exists():
        raise FileNotFoundError(
            f"RoBERTa requires Person A's cleaned corpus at {input_path}; "
            "run the preprocessing stage or supply that file."
        )
    reviews = pd.read_csv(input_path, dtype={"id": "string"})
    required = {"id", "app", "score", C.SENT_TEXT_COL}
    missing = required - set(reviews.columns)
    if missing:
        raise KeyError(f"{input_path} is missing required columns: {sorted(missing)}")
    if reviews[["id", "app"]].isna().any().any():
        raise ValueError("The cleaned corpus contains missing app or review ID values")
    reviews["id"] = reviews["id"].astype("string")
    reviews["app"] = reviews["app"].astype("string")
    if reviews.duplicated(["app", "id"]).any():
        raise ValueError("The cleaned corpus contains duplicate app/review ID pairs")

    completed = set()
    if output_path.exists():
        prior = pd.read_csv(
            output_path,
            usecols=["app", "id", "model_revision"],
            dtype={"app": "string", "id": "string", "model_revision": "string"},
        )
        matching = prior[prior.model_revision.eq(C.ROBERTA_REVISION)]
        completed = set(zip(matching.app, matching.id))
    pending = reviews.loc[
        [(app, review_id) not in completed for app, review_id in zip(reviews.app, reviews.id)]
    ].copy().sort_values(
        C.SENT_TEXT_COL,
        key=lambda values: values.fillna("").astype(str).str.len(),
        kind="stable",
    )
    if pending.empty:
        print(f"RoBERTa results already complete: {len(reviews):,} reviews")
        return output_path

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "RoBERTa dependencies are unavailable. Install the project requirements "
            "before running sentiment inference."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch_size = (
        batch_size if batch_size is not None else 32
    )
    if effective_batch_size < 1:
        raise ValueError("batch_size must be positive")
    print(
        f"Loading {C.ROBERTA_MODEL}@{C.ROBERTA_REVISION} on {device}; "
        f"{len(pending):,} reviews pending"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        C.ROBERTA_MODEL, revision=C.ROBERTA_REVISION
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        C.ROBERTA_MODEL, revision=C.ROBERTA_REVISION
    ).to(device)
    model.eval()
    id_to_label = _label_mapping(model.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    texts = pending[C.SENT_TEXT_COL].fillna("").astype(str).tolist()
    for start in range(0, len(pending), effective_batch_size):
        batch = pending.iloc[start : start + effective_batch_size]
        encoded = tokenizer(
            texts[start : start + effective_batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        predicted_ids = probabilities.argmax(axis=1)
        records = pd.DataFrame(
            {
                "app": batch.app.astype(str).to_numpy(),
                "id": batch.id.astype(str).to_numpy(),
                "score": pd.to_numeric(batch.score, errors="coerce").to_numpy(),
                "predicted_label": [id_to_label[int(index)] for index in predicted_ids],
                "confidence": probabilities.max(axis=1),
                "model_revision": C.ROBERTA_REVISION,
            }
        )
        records.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
        )
        written += len(records)
        print(f"Scored {written:,}/{len(pending):,} pending reviews")
    return output_path


def run_benchmark(
    clean_path=None,
    roberta_path=None,
    output_dir=None,
    per_rating=1000,
    min_words=3,
):
    """Compare RoBERTa, VADER and TextBlob on the same stratified sample."""
    clean_path = Path(clean_path or C.INTER / "upi_clean.csv")
    output_dir = Path(output_dir or C.TABLES)
    if not clean_path.exists():
        raise FileNotFoundError(f"Benchmark requires the cleaned corpus at {clean_path}")
    if per_rating < 1 or min_words < 1:
        raise ValueError("per_rating and min_words must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample, sample_info_df = _benchmark_sample(clean_path, per_rating, min_words)
    sample_counts = sample.score.value_counts().reindex(range(1, 6), fill_value=0)
    evaluate.validate_sampling(
        pd.DataFrame({"score": sample.score}), target_per_rating=per_rating
    )
    sample_info_df.to_csv(output_dir / "upi_benchmark_sampling.csv", index=False)
    store.put(
        "upi_sentiment_benchmark_sampling",
        {
            "seed": C.SEED,
            "target_per_rating": per_rating,
            "minimum_words": min_words,
            "sampled_total": int(len(sample)),
            "rating_counts": {int(k): int(v) for k, v in sample_counts.items()},
        },
    )

    try:
        import sent_lexicon
    except ImportError as exc:
        raise RuntimeError("VADER/TextBlob dependencies are required for benchmarking") from exc
    vader_scores, textblob_scores = sent_lexicon.score_texts(sample[C.SENT_TEXT_COL])
    sample["vader_prediction"] = [sent_lexicon.classify(float(v)) for v in vader_scores]
    sample["textblob_prediction"] = [sent_lexicon.classify(float(v)) for v in textblob_scores]

    if roberta_path is None:
        sample_keys = set(zip(sample.app.astype(str), sample.id.astype(str)))
        for candidate in (Path(C.ROBERTA_OUTPUT), Path(C.ROBERTA_BENCHMARK_OUTPUT)):
            if not candidate.exists():
                continue
            available = pd.read_csv(
                candidate,
                usecols=["app", "id", "model_revision"],
                dtype={"app": "string", "id": "string", "model_revision": "string"},
            )
            available = available[
                available.model_revision.eq(C.ROBERTA_REVISION)
            ].drop_duplicates(["app", "id"])
            available_keys = set(
                zip(available.app.astype(str), available.id.astype(str))
            )
            if sample_keys.issubset(available_keys):
                roberta_path = candidate
                break
    if roberta_path is None or not Path(roberta_path).exists():
        raise FileNotFoundError(
            "RoBERTa predictions for the stratified benchmark sample are missing. "
            "Run full inference or use --benchmark-sample-only."
        )
    predictions = pd.read_csv(
        roberta_path,
        dtype={"app": "string", "id": "string", "model_revision": "string"},
    )
    predictions = predictions[predictions.model_revision.eq(C.ROBERTA_REVISION)]
    predictions = predictions.drop_duplicates(["app", "id"], keep="last")
    sample = sample.merge(
        predictions[["app", "id", "predicted_label"]],
        on=["app", "id"],
        how="left",
        validate="one_to_one",
    )
    if sample.predicted_label.isna().any():
        raise ValueError(
            f"RoBERTa benchmark is missing {int(sample.predicted_label.isna().sum())} "
            "sample predictions; complete inference before benchmarking."
        )

    methods = {
        "RoBERTa": sample.predicted_label.tolist(),
        "VADER": sample.vader_prediction.tolist(),
        "TextBlob": sample.textblob_prediction.tolist(),
    }
    truth = sample.truth.tolist()
    metrics_rows = []
    per_class_rows = []
    for method, predicted in methods.items():
        result = evaluate.classification_metrics(truth, predicted)
        metrics_rows.append(
            {
                "method": method,
                "n": result["n"],
                "accuracy": result["accuracy"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "macro_f1": result["macro_f1"],
            }
        )
        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=[f"actual_{label}" for label in evaluate.LABELS],
            columns=[f"predicted_{label}" for label in evaluate.LABELS],
        )
        matrix.to_csv(output_dir / f"upi_benchmark_confusion_{method.lower()}.csv")
        for label, values in result["per_class"].items():
            per_class_rows.append({"method": method, "class": label, **values})
        store.put(f"upi_benchmark_{method.lower()}", result)

    mcnemar = evaluate.mcnemar_test(
        truth, methods["RoBERTa"], methods["VADER"]
    )
    comparison = pd.DataFrame(metrics_rows)
    comparison.to_csv(output_dir / "upi_sentiment_benchmark.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "upi_sentiment_benchmark_per_class.csv", index=False
    )
    store.put("upi_benchmark_roberta_vs_vader_mcnemar", mcnemar)
    with (output_dir / "upi_mcnemar_roberta_vader.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(mcnemar, stream, indent=2)
    print(comparison.to_string(index=False))
    print("RoBERTa vs VADER McNemar:", mcnemar)
    return comparison, mcnemar


def _benchmark_sample(clean_path, per_rating, min_words):
    corpus = pd.read_csv(clean_path, dtype={"id": "string"})
    required = {"app", "id", "score", C.SENT_TEXT_COL}
    if missing := required - set(corpus.columns):
        raise KeyError(f"{clean_path} is missing required columns: {sorted(missing)}")
    words = corpus[C.SENT_TEXT_COL].fillna("").astype(str).str.split().str.len()
    eligible = corpus[
        corpus.score.isin([1, 2, 3, 4, 5]) & words.ge(min_words)
    ].copy()
    eligible["app"] = eligible.app.astype("string")
    eligible["id"] = eligible.id.astype("string")
    sample_parts = []
    sample_info = []
    for rating in range(1, 6):
        group = eligible[eligible.score.eq(rating)]
        take = min(per_rating, len(group))
        random_state = C.SEED + rating
        if take:
            sample_parts.append(group.sample(n=take, random_state=random_state))
        sample_info.append(
            {
                "rating": rating,
                "eligible_reviews": int(len(group)),
                "sampled_reviews": int(take),
                "target_per_rating": int(per_rating),
                "seed": C.SEED,
                "random_state": random_state,
                "minimum_words": min_words,
            }
        )
    if not sample_parts:
        raise ValueError("No reviews meet the benchmark eligibility criteria")
    sample = pd.concat(sample_parts, ignore_index=True)
    sample["truth"] = np.where(
        sample.score.le(2),
        "negative",
        np.where(sample.score.eq(3), "neutral", "positive"),
    )
    return sample, pd.DataFrame(sample_info)


def score_benchmark_sample(
    clean_path=None, per_rating=1000, min_words=3, output_path=None
):
    """Run the pinned model only on the benchmark sample when CPU budget is limited."""
    clean_path = Path(clean_path or C.INTER / "upi_clean.csv")
    sample_path = C.INTER / "upi_sentiment_benchmark_sample.csv"
    sample, _ = _benchmark_sample(clean_path, per_rating, min_words)
    sample[["app", "id", "score", C.SENT_TEXT_COL]].to_csv(
        sample_path, index=False
    )
    return run_roberta(
        input_path=sample_path,
        output_path=output_path or C.ROBERTA_BENCHMARK_OUTPUT,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-sample-only",
        action="store_true",
        help="Score only the fixed stratified benchmark sample, not the full corpus",
    )
    args = parser.parse_args()
    if args.benchmark_sample_only:
        score_benchmark_sample()
        run_benchmark()
    else:
        run_roberta()
        run_benchmark()
