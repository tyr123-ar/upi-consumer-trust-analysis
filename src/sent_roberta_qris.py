"""Pinned Indonesian RoBERTa inference on the cleaned QRIS comments."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import evaluate
import person_b_store as store


def _id_to_label(model_config):
    labels = {
        int(index): str(label).strip().lower()
        for index, label in model_config.id2label.items()
    }
    if set(labels.values()) != set(evaluate.LABELS):
        raise ValueError(
            f"QRIS model labels must be {evaluate.LABELS}; found {labels}"
        )
    return labels


def run(input_path=None, output_path=None, batch_size=16):
    input_path = Path(input_path or C.INTER / "qris_clean.csv")
    output_path = Path(output_path or C.QRIS_OUTPUT)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cleaned QRIS data: {input_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    comments = pd.read_csv(input_path)
    required = {"qris_id", "raw"}
    missing = required - set(comments.columns)
    if missing:
        raise KeyError(f"{input_path} is missing required columns: {sorted(missing)}")
    if comments.empty or comments["qris_id"].isna().any():
        raise ValueError("QRIS model input must contain non-empty rows with qris_id")
    if comments["qris_id"].duplicated().any():
        raise ValueError("QRIS model input has duplicate qris_id values")

    if output_path.exists():
        prior = pd.read_csv(output_path)
        if (
            set(prior.columns) >= {"qris_id", "predicted_label", "model_revision"}
            and prior["model_revision"].eq(C.QRIS_REVISION).all()
            and set(prior["qris_id"]) == set(comments["qris_id"])
            and len(prior) == len(comments)
        ):
            print(f"QRIS RoBERTa predictions already complete: {len(prior)} comments")
            return output_path
        raise FileExistsError(
            f"Existing QRIS predictions are incomplete or use another model: {output_path}"
        )

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "QRIS RoBERTa requires the installed transformers and torch dependencies"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Loading {C.QRIS_MODEL}@{C.QRIS_REVISION} on {device}; "
        f"scoring {len(comments)} QRIS comments"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        C.QRIS_MODEL, revision=C.QRIS_REVISION
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        C.QRIS_MODEL, revision=C.QRIS_REVISION
    ).to(device)
    model.eval()
    id_to_label = _id_to_label(model.config)
    texts = comments["raw"].fillna("").astype(str).tolist()
    records = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        for offset, row in enumerate(probabilities):
            predicted_index = int(row.argmax())
            records.append(
                {
                    "qris_id": int(comments.iloc[start + offset]["qris_id"]),
                    "predicted_label": id_to_label[predicted_index],
                    "confidence": float(row[predicted_index]),
                    "model_revision": C.QRIS_REVISION,
                }
            )
        print(f"Scored {len(records)}/{len(comments)} QRIS comments")

    predictions = pd.DataFrame(records)
    if len(predictions) != len(comments) or predictions["qris_id"].duplicated().any():
        raise RuntimeError("QRIS inference did not produce exactly one prediction per comment")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    store.put(
        "qris_model",
        {
            "model": C.QRIS_MODEL,
            "revision": C.QRIS_REVISION,
            "device": str(device),
            "rows_scored": int(len(predictions)),
        },
    )
    print(f"Saved QRIS predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
