"""Inspectable exploratory keyword themes for English UPI reviews (not QRIS)."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import person_b_store
import store

THEME_PATTERNS = {
    "Transaction failure": re.compile(
        r"\b(?:payment|transaction|transfer)\s+(?:has\s+)?(?:failed|failure)|"
        r"\bfailed\s+(?:payment|transaction|transfer)\b|\btransaction error\b",
        re.I,
    ),
    "Payment delays": re.compile(
        r"\b(?:payment|transaction|upi transfer)\b.{0,32}\b(?:pending|delayed|stuck)\b|"
        r"\b(?:pending|delayed|stuck)\b.{0,32}\b(?:payment|transaction|transfer)\b",
        re.I,
    ),
    "Refund issues": re.compile(
        r"\brefund(?:ed)?\b|\bmoney back\b|\bamount reversed\b", re.I
    ),
    "Fraud and suspicious transactions": re.compile(
        r"\b(?:fraud|scam|unauthori[sz]ed transaction|suspicious transaction|"
        r"money stolen|account hacked)\b",
        re.I,
    ),
    "Security and privacy": re.compile(
        r"\b(?:security|secure|privacy|data leak|personal data|data breach)\b", re.I
    ),
    "Account and verification": re.compile(
        r"\b(?:kyc|verification|account blocked|account suspended|login problem)\b",
        re.I,
    ),
    "Customer support": re.compile(
        r"\b(?:customer support|customer care|helpline|support ticket|"
        r"support team|complaint resolution)\b",
        re.I,
    ),
    "Convenience and usability": re.compile(
        r"\b(?:easy to use|user friendly|convenient|seamless|simple to use)\b",
        re.I,
    ),
}


def match_themes(text):
    """Return all explicitly matched themes; a review may match multiple themes."""
    text = "" if pd.isna(text) else str(text)
    return [name for name, pattern in THEME_PATTERNS.items() if pattern.search(text)]


def run(input_path=None):
    input_path = Path(input_path or C.INTER / "upi_clean.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cleaned UPI corpus: {input_path}")
    df = pd.read_csv(input_path)
    if "text_light" not in df or "app" not in df:
        raise KeyError("UPI theme analysis requires app and text_light columns")
    if "id" not in df:
        raise KeyError("UPI theme analysis requires internal review IDs for deduplication")
    rows = []
    for _, row in df[["app", "id", "text_light"]].iterrows():
        for theme in match_themes(row.text_light):
            rows.append((row.app, row.id, theme))
    events = pd.DataFrame(rows, columns=["app", "id", "theme"])
    denominator = df.groupby("app").size()
    if events.empty:
        summary = pd.DataFrame(columns=["app", "theme", "reviews", "share_pct"])
    else:
        summary = (
            events.drop_duplicates(["app", "id", "theme"])
            .groupby(["app", "theme"])
            .size()
            .rename("reviews")
            .reset_index()
        )
        summary["share_pct"] = summary.apply(
            lambda row: 100 * row.reviews / denominator[row.app], axis=1
        )
        summary["share_pct"] = summary.share_pct.round(2)
    store.table(summary, "table_upi_exploratory_themes")
    person_b_store.put(
        "upi_exploratory_themes",
        {
            "definition": "Overlapping keyword-rule matches; exploratory UPI analysis, not QRIS",
            "review_count": int(len(df)),
            "matched_review_count": int(events.drop_duplicates(["app", "id"]).shape[0]) if not events.empty else 0,
        },
    )
    return summary


if __name__ == "__main__":
    print(run().to_string(index=False))
