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

for directory in (RAW / "upi", RAW / "qris", INTER, TABLES, FIGS):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
EXPECTED_RAW_ROWS = {
    "Google Pay": 34228,
    "Paytm": 137907,
    "PhonePe": 11741,
}
UPI_FILES = {
    "Google Pay": RAW / "upi" / "google_pay.csv",
    "Paytm": RAW / "upi" / "paytm.csv",
    "PhonePe": RAW / "upi" / "phonepe.csv",
}
LOCAL_UPI_FILES = {
    "Google Pay": BASE / "data" / "GooglePayIndia.csv",
    "Paytm": BASE / "data" / "PaytmIndia.csv",
    "PhonePe": BASE / "data" / "PhonePayIndia.csv",
}
for app, path in LOCAL_UPI_FILES.items():
    if not UPI_FILES[app].exists() and path.exists():
        UPI_FILES[app] = path
# These headers match the supplied CSV schemas; revalidate replacements with s0_check_data.py.
UPI_COLS = {
    "id": "reviewId", "text": "content", "score": "score",
    "thumbs": "thumbsUpCount", "time": "at", "reply": "replyContent",
}
QRIS_FILE = RAW / "qris" / "qris_tiktok.csv"
QRIS_COLS = {
    "comment_id": "Comment ID",
    "user_id": "User ID",
    "username": "Username",
    "nickname": "Nick Name",
    "text": "Comment",
    "time": "Comment Time",
    "likes": "Digg Count",
    "author_liked": "Author Digged",
    "replies": "Reply Count",
    "pinned": "Pinned to Top",
    "homepage": "User Homepage",
}
SENT_TEXT_COL = "text_light"  # run both text_light and text to compare with Table 6
ASCII_MIN = 0.90
VADER_CUT = 0.05
MIN_MONTH_REVIEWS = 500
APPS = ["Google Pay", "Paytm", "PhonePe"]

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
ROBERTA_OUTPUT = INTER / "upi_roberta_scored.csv"
ROBERTA_BENCHMARK_OUTPUT = INTER / "upi_roberta_benchmark_scored.csv"
QRIS_MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"
QRIS_REVISION = "ac452dcb0f4966130bba44f4ee0013bb5d52c282"
QRIS_OUTPUT = INTER / "qris_scored.csv"
TOPIC_ASSIGNMENTS = INTER / "upi_topics_scored.csv"
