"""Unweighted and like-weighted QRIS sentiment with Wilson intervals."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import evaluate
import person_b_store as store


def run(clean_path=None, scored_path=None, output_dir=None):
    clean_path = Path(clean_path or C.INTER / "qris_clean.csv")
    scored_path = Path(scored_path or C.QRIS_OUTPUT)
    output_dir = Path(output_dir or C.TABLES)
    if not clean_path.exists() or not scored_path.exists():
        raise FileNotFoundError("Run QRIS preprocessing and RoBERTa scoring first")
    clean = pd.read_csv(clean_path)
    scored = pd.read_csv(scored_path)
    required = {"qris_id", "likes"}
    if required - set(clean.columns):
        raise KeyError(f"{clean_path} is missing columns: {sorted(required - set(clean.columns))}")
    if {"qris_id", "predicted_label", "confidence", "model_revision"} - set(scored.columns):
        raise KeyError(f"{scored_path} is missing required prediction columns")
    if not scored["model_revision"].eq(C.QRIS_REVISION).all():
        raise ValueError("QRIS predictions do not match the pinned model revision")
    data = clean[["qris_id", "likes"]].merge(
        scored[["qris_id", "predicted_label", "confidence"]],
        on="qris_id",
        how="left",
        validate="one_to_one",
    )
    if data["predicted_label"].isna().any() or len(data) != len(clean):
        raise ValueError("QRIS sentiment predictions do not cover the cleaned corpus")
    labels = data["predicted_label"].astype(str)
    invalid = set(labels) - set(evaluate.LABELS)
    if invalid:
        raise ValueError(f"Unexpected QRIS model labels: {sorted(invalid)}")

    total_likes = int(data["likes"].sum())
    if total_likes <= 0:
        raise ValueError("Like-weighted sentiment requires a positive total like count")
    rows = []
    for label in evaluate.LABELS:
        selected = labels.eq(label)
        count = int(selected.sum())
        likes = int(data.loc[selected, "likes"].sum())
        low, high = evaluate.wilson_interval(count, len(data))
        rows.append(
            {
                "sentiment": label,
                "comments": count,
                "share_pct": 100 * count / len(data),
                "likes": likes,
                "like_weighted_share_pct": 100 * likes / total_likes,
                "wilson_95_low_pct": 100 * low,
                "wilson_95_high_pct": 100 * high,
            }
        )
    summary = pd.DataFrame(rows)
    summary[["share_pct", "like_weighted_share_pct", "wilson_95_low_pct", "wilson_95_high_pct"]] = (
        summary[
            ["share_pct", "like_weighted_share_pct", "wilson_95_low_pct", "wilson_95_high_pct"]
        ].round(2)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "qris_sentiment_distribution.csv", index=False)

    negative = int(labels.eq("negative").sum())
    low, high = evaluate.wilson_interval(negative, len(data))
    metrics = {
        "n": int(len(data)),
        "counts": {row["sentiment"]: row["comments"] for row in rows},
        "shares_pct": {row["sentiment"]: round(row["share_pct"], 2) for row in rows},
        "likes_by_sentiment": {row["sentiment"]: row["likes"] for row in rows},
        "like_weighted_shares_pct": {
            row["sentiment"]: round(row["like_weighted_share_pct"], 2) for row in rows
        },
        "negative_wilson_95_pct": [round(100 * low, 2), round(100 * high, 2)],
        "mean_confidence": round(float(data["confidence"].mean()), 6),
        "total_likes": total_likes,
        "model_revision": C.QRIS_REVISION,
        "wilson_note": "Unweighted comment-level interval; not applied to like-weighted shares.",
    }
    store.put("qris_sentiment", metrics)
    print(summary.to_string(index=False))
    print("QRIS sentiment metrics:", metrics)
    return summary, metrics


if __name__ == "__main__":
    run()
