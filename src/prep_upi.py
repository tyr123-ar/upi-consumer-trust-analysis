"""UPI cleaning pipeline, following the implementation plan's stated order."""
import re
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import store

URL = re.compile(r"https?://\S+|www\.\S+", re.I)
JUNK = re.compile(r"[^\w\s']|_", re.UNICODE)
WS = re.compile(r"\s+")

def clean(text):
    text = "" if pd.isna(text) else str(text)
    text = URL.sub(" ", text.lower())
    text = JUNK.sub(" ", text)
    return WS.sub(" ", text).strip()

def light(text):
    """Lowercase and remove URLs only; preserve punctuation for VADER and clause splitting."""
    text = "" if pd.isna(text) else str(text)
    return WS.sub(" ", URL.sub(" ", text.lower())).strip()

def ascii_share(text):
    if not text:
        return 0.0
    return sum(ord(char) < 128 for char in text) / len(text)

def load_one(app, path):
    columns = C.UPI_COLS
    df = pd.read_csv(path)
    required = [columns[k] for k in ("id", "text", "score", "time")]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns {missing}; update config.UPI_COLS")
    reply_col = columns.get("reply")
    return pd.DataFrame({
        "app": app,
        "id": df[columns["id"]],
        "raw": df[columns["text"]],
        "score": pd.to_numeric(df[columns["score"]], errors="coerce"),
        "thumbs": pd.to_numeric(df[columns["thumbs"]], errors="coerce").fillna(0) if columns.get("thumbs") in df else 0,
        "time": pd.to_datetime(df[columns["time"]], errors="coerce"),
        "has_reply": df[reply_col].notna() if reply_col and reply_col in df else False,
    })

def run():
    missing = [str(p) for p in C.UPI_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Download the three UPI CSVs first:\n" + "\n".join(missing))
    df = pd.concat([load_one(app, path) for app, path in C.UPI_FILES.items()], ignore_index=True)
    counts = {"raw": int(len(df))}
    df = df.drop_duplicates("id").copy()
    df["raw"] = df["raw"].fillna("").astype(str)
    df = df[df.raw.str.strip().ne("")].copy()
    counts["after_id_empty"] = int(len(df))
    df.to_csv(C.INTER / "upi_all.csv", index=False)

    df["text"] = df.raw.map(clean)
    df["text_light"] = df.raw.map(light)
    df = df[df.text.ne("")].copy()
    counts["after_empty_clean"] = int(len(df))

    df = df[df.text.map(ascii_share).ge(C.ASCII_MIN)].copy()
    counts["after_nonenglish"] = int(len(df))
    df["n_words"] = df.text.str.split().str.len()
    df.to_csv(C.INTER / "upi_en.csv", index=False)

    df = df.drop_duplicates(["app", "text"]).copy()
    counts["final"] = int(len(df))
    df.to_csv(C.INTER / "upi_clean.csv", index=False)
    store.put("upi_counts", counts)
    print("UPI preprocessing counts:", counts)
    return counts

if __name__ == "__main__":
    run()
