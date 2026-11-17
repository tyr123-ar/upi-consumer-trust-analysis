"""Assemble the measured QRIS section of Table 13 without rerunning UPI stages."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C


def _upi_reference():
    path = C.TABLES / "table10_trust_risk.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    all_apps = table[table["App"].eq("All apps")]
    if all_apps.empty:
        return {}
    row = all_apps.iloc[0]
    return {
        "fraud/security keyword matches": f"{float(row['Fraud-security %']):.2f}% (UPI lexicon)",
        "functional-problem keyword matches": f"{float(row['Functional-problem %']):.2f}% (UPI lexicon)",
        "trust/endorsement keyword matches": f"{float(row['Trust and endorsement %']):.2f}% (UPI lexicon)",
    }


def run(output_path=None):
    output_path = Path(output_path or C.TABLES / "table13_qris_portion.csv")
    sentiment_path = C.TABLES / "qris_sentiment_distribution.csv"
    theme_path = C.TABLES / "qris_themes.csv"
    lexicon_path = C.TABLES / "qris_narrow_lexicons.csv"
    topic_path = C.TABLES / "qris_topics.csv"
    required = (sentiment_path, theme_path, lexicon_path, topic_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Complete QRIS analysis before Table 13:\n" + "\n".join(missing))

    sentiment = pd.read_csv(sentiment_path).set_index("sentiment")
    themes = pd.read_csv(theme_path).set_index("category")
    lexicons = pd.read_csv(lexicon_path).set_index("category")
    topics = pd.read_csv(topic_path)
    upi = _upi_reference()
    qris_fraud = lexicons.loc["Fraud / manipulated evidence"]
    qris_functional = lexicons.loc["Network / functional problems"]
    qris_security = lexicons.loc["Security / phishing"]
    qris_trust = lexicons.loc["Verification / trust mechanisms"]

    sentiment_text = ", ".join(
        f"{label} {float(sentiment.loc[label, 'share_pct']):.2f}%"
        for label in ("negative", "neutral", "positive")
    )
    weighted_text = ", ".join(
        f"{label} {float(sentiment.loc[label, 'like_weighted_share_pct']):.2f}% of likes"
        for label in ("negative", "neutral", "positive")
    )
    rows = [
        {
            "concern": "Overall sentiment (comment-weighted)",
            "UPI evidence": "Existing UPI VADER results; see UPI sentiment outputs",
            "QRIS evidence": f"{sentiment_text}; n={int(sentiment['comments'].sum())}",
            "interpretation_limit": "Different languages, platforms, and models; descriptive side-by-side only.",
        },
        {
            "concern": "Overall sentiment (like-weighted)",
            "UPI evidence": "Not paired to TikTok likes in this analysis",
            "QRIS evidence": weighted_text,
            "interpretation_limit": "Likes are engagement counts, not independent users or sentiment votes.",
        },
        {
            "concern": "Fraud / manipulated payment evidence",
            "UPI evidence": upi.get("fraud/security keyword matches", "Not available"),
            "QRIS evidence": f"{int(qris_fraud['comments'])} comments ({float(qris_fraud['share_pct']):.2f}%)",
            "interpretation_limit": "Different rule lexicons and corpora; matches are not verified fraud incidents.",
        },
        {
            "concern": "Security / phishing",
            "UPI evidence": upi.get("fraud/security keyword matches", "Not available"),
            "QRIS evidence": f"{int(qris_security['comments'])} comments ({float(qris_security['share_pct']):.2f}%)",
            "interpretation_limit": "Different rule lexicons and corpora; percentages are not directly comparable.",
        },
        {
            "concern": "Network / functional problems",
            "UPI evidence": upi.get("functional-problem keyword matches", "Not available"),
            "QRIS evidence": f"{int(qris_functional['comments'])} comments ({float(qris_functional['share_pct']):.2f}%)",
            "interpretation_limit": "Different language-specific rule sets and sampling frames.",
        },
        {
            "concern": "Verification / trust mechanisms",
            "UPI evidence": upi.get("trust/endorsement keyword matches", "Not available"),
            "QRIS evidence": f"{int(qris_trust['comments'])} comments ({float(qris_trust['share_pct']):.2f}%)",
            "interpretation_limit": "Exploratory keyword matches; the QRIS rows are small and may overlap other themes.",
        },
        {
            "concern": "QRIS LDA topic summary",
            "UPI evidence": "UPI LDA results are app-specific",
            "QRIS evidence": "; ".join(
                f"Topic {int(row.topic_id) + 1} {float(row.share_pct):.2f}%"
                for row in topics.itertuples(index=False)
            ),
            "interpretation_limit": "Topics are exploratory; QRIS has no video identifier for video-level inference.",
        },
    ]
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved QRIS Table 13 portion to {output_path}")
    return result


if __name__ == "__main__":
    run()
