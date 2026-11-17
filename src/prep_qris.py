"""Prepare the public QRIS TikTok comments without retaining user identifiers."""
import html
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import person_b_store as store

URL = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION = re.compile(r"(?<!\w)@[\w.]+", re.UNICODE)
REPEATED = re.compile(r"(.)\1{2,}", re.UNICODE)
NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")
ELONGATED_NEGATION = re.compile(
    r"^(?:g+a+k+|g+a+|ng+g+a+k+|ng+g+a+|g+k+|t+d+k+|e+ng+g+a+k+)$",
    re.I,
)
SLANG = {
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "bgt": "banget",
    "aja": "saja",
    "udah": "sudah",
    "belom": "belum",
}


def normalize(text):
    """Make a conservative Indonesian text variant for themes and LDA."""
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = REPEATED.sub(lambda match: match.group(1) * 2, text)
    text = NON_WORD.sub(" ", text)
    tokens = []
    for token in WHITESPACE.split(text.strip()):
        if not token:
            continue
        if ELONGATED_NEGATION.fullmatch(token):
            token = "tidak"
        else:
            token = SLANG.get(token, token)
        tokens.append(token)
    return " ".join(tokens)


def run(input_path=None, output_path=None):
    input_path = Path(input_path or C.QRIS_FILE)
    output_path = Path(output_path or C.INTER / "qris_clean.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing QRIS source CSV: {input_path}")

    columns = C.QRIS_COLS
    source = pd.read_csv(
        input_path,
        dtype={columns["comment_id"]: "string"},
    )
    required_keys = ("comment_id", "text", "time", "likes")
    missing = [columns[key] for key in required_keys if columns[key] not in source]
    if missing:
        raise KeyError(
            f"{input_path} is missing QRIS columns {missing}; update config.QRIS_COLS"
        )
    if source[columns["comment_id"]].isna().any():
        raise ValueError("QRIS source contains missing comment IDs")

    work = pd.DataFrame(
        {
            "qris_id": range(1, len(source) + 1),
            "_source_comment_id": source[columns["comment_id"]].astype("string"),
            "raw": source[columns["text"]].fillna("").astype(str),
            "time": pd.to_datetime(source[columns["time"]], errors="coerce"),
            "likes": pd.to_numeric(source[columns["likes"]], errors="coerce"),
            "reply_count": pd.to_numeric(
                source[columns["replies"]], errors="coerce"
            ).fillna(0)
            if columns["replies"] in source
            else 0,
        }
    )
    if work["likes"].isna().any() or work["likes"].lt(0).any():
        raise ValueError("QRIS like counts must be present and non-negative")
    if not pd.api.types.is_integer_dtype(work["likes"]):
        if not work["likes"].mod(1).eq(0).all():
            raise ValueError("QRIS like counts must be integers")
        work["likes"] = work["likes"].astype("int64")

    counts = {"raw": int(len(work))}
    work = work.drop_duplicates("_source_comment_id", keep="first").copy()
    counts["after_comment_id_dedup"] = int(len(work))
    work["norm"] = work["raw"].map(normalize)
    work = work[work["norm"].ne("")].copy()
    counts["after_empty_normalization"] = int(len(work))
    work = work.drop_duplicates("norm", keep="first").copy()
    counts["after_text_dedup"] = int(len(work))
    work["n_words"] = work["norm"].str.split().str.len()
    work = work[work["n_words"].ge(2)].copy()
    counts["final"] = int(len(work))
    if work.empty:
        raise ValueError("QRIS preprocessing removed every comment")
    if work["time"].isna().any():
        counts["invalid_timestamps"] = int(work["time"].isna().sum())

    clean = work[
        ["qris_id", "raw", "norm", "time", "likes", "reply_count", "n_words"]
    ].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    store.put("qris_preprocessing", counts)
    print(f"QRIS preprocessing counts: {counts}")
    print(f"Saved {output_path} ({len(clean)} comments; no source user IDs retained)")
    return clean, counts


if __name__ == "__main__":
    run()
