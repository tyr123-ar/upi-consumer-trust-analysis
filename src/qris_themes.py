"""Exploratory Indonesian QRIS themes and narrow keyword lexicons."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C
import evaluate
import person_b_store as store

THEMES = {
    "Fraud or edited payment evidence": (
        "tipu",
        "penipuan",
        "palsu",
        "edit",
        "diubah",
        "manipulasi",
        "hoax",
        "bohong",
        "curang",
        "maling",
        "scam",
        "stiker",
        "bukti palsu",
    ),
    "Security or phishing": (
        "keamanan",
        "aman",
        "risiko",
        "resiko",
        "phishing",
        "phising",
        "peretasan",
        "diretas",
        "hack",
        "kejahatan",
        "perlindungan",
    ),
    "Network or payment problems": (
        "jaringan",
        "koneksi",
        "sinyal",
        "loading",
        "lag",
        "lemot",
        "lambat",
        "gagal",
        "error",
        "eror",
        "pending",
        "transaksi gagal",
    ),
    "Verification and trust mechanisms": (
        "edc",
        "soundbox",
        "sound box",
        "speaker",
        "suara",
        "notifikasi",
        "struk",
        "mutasi",
        "bukti transaksi",
        "cek transaksi",
    ),
    "Information seeking or awareness": (
        "informasi",
        "edukasi",
        "cara",
        "bagaimana",
        "gimana",
        "bertanya",
        "tanya",
        "penjelasan",
        "sosialisasi",
        "belajar",
    ),
    "User experience or convenience": (
        "pengalaman",
        "mudah",
        "praktis",
        "cepat",
        "nyaman",
        "susah",
        "sulit",
        "ribet",
        "bermanfaat",
        "membantu",
    ),
}

NARROW_LEXICONS = {
    "Fraud / manipulated evidence": THEMES["Fraud or edited payment evidence"],
    "Network / functional problems": THEMES["Network or payment problems"],
    "Security / phishing": THEMES["Security or phishing"],
    "Verification / trust mechanisms": THEMES["Verification and trust mechanisms"],
}

SUSPICION = re.compile(
    r"\b(?:takut|khawatir|cemas|malu|deg[\s-]*degan)\b.{0,40}"
    r"\b(?:dikir(?:a|ain)|dituduh)\b"
    r"(?:\s+(?:tipu|penipuan|penipu|curang|maling|scam|bohong))?"
    r"|\bdituduh\s+(?:tipu|curang|maling)\b",
    re.I,
)


def _pattern(terms):
    alternatives = sorted(
        (re.escape(term).replace(r"\ ", r"\s+") for term in terms),
        key=len,
        reverse=True,
    )
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)


THEME_PATTERNS = {name: _pattern(terms) for name, terms in THEMES.items()}
LEXICON_PATTERNS = {
    name: _pattern(terms) for name, terms in NARROW_LEXICONS.items()
}


def _summary_rows(data, patterns):
    labels = data["predicted_label"]
    n = len(data)
    rows = []
    for name, pattern in patterns.items():
        text = (
            data["norm"]
            if name not in ("Fraud or edited payment evidence", "Fraud / manipulated evidence")
            else data["norm"].map(lambda value: SUSPICION.sub(" ", value))
        )
        matched = text.map(lambda value: bool(pattern.search(value)))
        count = int(matched.sum())
        low, high = evaluate.wilson_interval(count, n)
        negative = int((matched & labels.eq("negative")).sum())
        rows.append(
            {
                "category": name,
                "comments": count,
                "share_pct": round(100 * count / n, 2),
                "wilson_95_low_pct": round(100 * low, 2),
                "wilson_95_high_pct": round(100 * high, 2),
                "negative_comments": negative,
                "negative_within_category_pct": (
                    round(100 * negative / count, 2) if count else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run(clean_path=None, scored_path=None, output_dir=None):
    clean_path = Path(clean_path or C.INTER / "qris_clean.csv")
    scored_path = Path(scored_path or C.QRIS_OUTPUT)
    output_dir = Path(output_dir or C.TABLES)
    if not clean_path.exists() or not scored_path.exists():
        raise FileNotFoundError("Run QRIS preprocessing and RoBERTa scoring first")
    clean = pd.read_csv(clean_path)
    scored = pd.read_csv(scored_path)
    data = clean[["qris_id", "norm"]].merge(
        scored[["qris_id", "predicted_label", "model_revision"]],
        on="qris_id",
        how="left",
        validate="one_to_one",
    )
    if data["predicted_label"].isna().any() or len(data) != len(clean):
        raise ValueError("QRIS theme analysis requires one sentiment per clean comment")
    if not data["model_revision"].eq(C.QRIS_REVISION).all():
        raise ValueError("QRIS theme analysis found predictions from another revision")

    theme_table = _summary_rows(data, THEME_PATTERNS)
    lexicon_table = _summary_rows(data, LEXICON_PATTERNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    theme_table.to_csv(output_dir / "qris_themes.csv", index=False)
    lexicon_table.to_csv(output_dir / "qris_narrow_lexicons.csv", index=False)

    theme_columns = {}
    for name, pattern in THEME_PATTERNS.items():
        text = data["norm"]
        if name == "Fraud or edited payment evidence":
            text = text.map(lambda value: SUSPICION.sub(" ", value))
        theme_columns["theme_" + re.sub(r"\W+", "_", name.lower()).strip("_")] = (
            text.map(lambda value: bool(pattern.search(value)))
        )
    events = pd.concat([data[["qris_id", "predicted_label"]], pd.DataFrame(theme_columns)], axis=1)
    events.to_csv(C.INTER / "qris_themed.csv", index=False)
    store.put(
        "qris_themes",
        {
            "n": int(len(data)),
            "themes": theme_table.to_dict(orient="records"),
            "narrow_lexicons": lexicon_table.to_dict(orient="records"),
        },
    )
    print(theme_table.to_string(index=False))
    print("QRIS narrow lexicons:")
    print(lexicon_table.to_string(index=False))
    return theme_table, lexicon_table


if __name__ == "__main__":
    run()
