"""Shared project configuration for the Digital Payments reproduction."""
import os
from pathlib import Path

BASE = Path(os.environ.get("DP_BASE", Path(__file__).resolve().parent)).expanduser().resolve()
RAW = BASE / "data" / "raw"
INTER = BASE / "data" / "interim"
OUT = BASE / "outputs"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
METRICS = OUT / "metrics.json"

for directory in (RAW / "upi", INTER, TABLES, FIGS):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
UPI_FILES = {
    "Google Pay": RAW / "upi" / "google_pay.csv",
    "Paytm": RAW / "upi" / "paytm.csv",
    "PhonePe": RAW / "upi" / "phonepe.csv",
}
# These are the plan's proposed Kaggle headers. Verify with s0_check_data.py.
UPI_COLS = {
    "id": "reviewId", "text": "content", "score": "score",
    "thumbs": "thumbsUpCount", "time": "at", "reply": "replyContent",
}
SENT_TEXT_COL = "text_light"  # run both text_light and text to compare with Table 6
ASCII_MIN = 0.90
VADER_CUT = 0.05
MIN_MONTH_REVIEWS = 500
APPS = ["Google Pay", "Paytm", "PhonePe"]
