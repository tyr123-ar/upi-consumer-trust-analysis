"""Create a privacy-minimized profile of the downloaded QRIS dataset."""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import person_b_store
import store

DATASET_DOI = "10.17632/xpkw64fd2h.1"
DATASET_LICENSE = "CC BY 4.0"
DATASET_ORIGINAL_FILENAME = "FINAL DATA.csv"


def run(raw_path=None, clean_path=None):
    raw_path = Path(raw_path or C.QRIS_FILE)
    clean_path = Path(clean_path or C.INTER / "qris_clean.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing QRIS source CSV: {raw_path}")
    if not clean_path.exists():
        raise FileNotFoundError(f"Run QRIS preprocessing first: {clean_path}")

    columns = C.QRIS_COLS
    raw = pd.read_csv(
        raw_path,
        dtype={
            columns["comment_id"]: "string",
            columns["user_id"]: "string",
        },
    )
    clean = pd.read_csv(clean_path)
    text = raw[columns["text"]].fillna("").astype(str)
    times = pd.to_datetime(raw[columns["time"]], errors="coerce")
    likes = pd.to_numeric(raw[columns["likes"]], errors="coerce")
    file_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    profile = {
        "dataset_doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "original_filename": DATASET_ORIGINAL_FILENAME,
        "source_sha256": file_hash,
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "clean_rows": int(len(clean)),
        "distinct_comment_ids": int(raw[columns["comment_id"]].nunique()),
        "duplicate_comment_ids": int(raw[columns["comment_id"]].duplicated().sum()),
        "distinct_users_count_only": int(raw[columns["user_id"]].nunique())
        if columns["user_id"] in raw
        else None,
        "blank_comments": int(text.str.strip().eq("").sum()),
        "invalid_timestamps": int(times.isna().sum()),
        "date_min": times.min().isoformat() if times.notna().any() else None,
        "date_max": times.max().isoformat() if times.notna().any() else None,
        "comment_chars_min": int(text.str.len().min()) if len(text) else 0,
        "comment_chars_median": float(text.str.len().median()) if len(text) else 0,
        "comment_chars_max": int(text.str.len().max()) if len(text) else 0,
        "likes_sum_raw": int(likes.sum()),
        "likes_mean_raw": float(likes.mean()),
        "likes_median_raw": float(likes.median()),
        "likes_max_raw": int(likes.max()),
        "zero_like_comments_raw": int(likes.eq(0).sum()),
    }

    rows = [{"metric": key, "value": value} for key, value in profile.items()]
    store.table(pd.DataFrame(rows), "qris_profile")
    month_counts = (
        times.dropna()
        .dt.to_period("M")
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis("month")
        .rename("comments")
        .reset_index()
    )
    store.table(month_counts, "qris_comments_by_month")
    person_b_store.put("qris_profile", profile)
    print(
        f"QRIS profile: {profile['raw_rows']} raw rows, "
        f"{profile['clean_rows']} cleaned comments; identifiers counted, not exported"
    )
    return profile


if __name__ == "__main__":
    run()
