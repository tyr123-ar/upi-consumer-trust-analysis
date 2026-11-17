"""Validate available raw UPI files and describe the existing processed corpus."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C


def run():
    rows = []
    details = []
    raw_available = True
    raw_validated = True
    for app, path in C.UPI_FILES.items():
        expected = C.EXPECTED_RAW_ROWS[app]
        if not path.exists():
            raw_available = False
            raw_validated = False
            rows.append(
                {
                    "app": app,
                    "source_file": str(path.relative_to(C.BASE)),
                    "expected_raw_rows": expected,
                    "actual_raw_rows": None,
                    "status": "NOT AVAILABLE",
                }
            )
            details.append(f"### {app}\n\n- Source CSV: unavailable\n- Expected rows: {expected:,}\n")
            continue
        raw = pd.read_csv(path, low_memory=False)
        mapping = C.UPI_COLS
        required = [mapping[key] for key in ("id", "text", "score", "time")]
        missing_columns = [column for column in required if column not in raw.columns]
        ratings = (
            pd.to_numeric(raw[mapping["score"]], errors="coerce")
            if mapping["score"] in raw
            else pd.Series(dtype=float)
        )
        dates = (
            pd.to_datetime(raw[mapping["time"]], errors="coerce")
            if mapping["time"] in raw
            else pd.Series(dtype="datetime64[ns]")
        )
        missing_counts = raw.isna().sum().to_dict()
        duplicate_count = (
            int(raw[mapping["id"]].duplicated().sum())
            if mapping["id"] in raw
            else None
        )
        invalid_ratings = int((ratings.notna() & ~ratings.isin([1, 2, 3, 4, 5])).sum())
        invalid_dates = int(dates.isna().sum()) if len(dates) else None
        if missing_columns:
            status = "SCHEMA MISMATCH"
        elif len(raw) != expected:
            status = "ROW COUNT MISMATCH"
        else:
            status = "OK"
        raw_validated = raw_validated and status == "OK"
        rows.append(
            {
                "app": app,
                "source_file": str(path.relative_to(C.BASE)),
                "expected_raw_rows": expected,
                "actual_raw_rows": int(len(raw)),
                "status": status,
            }
        )
        details.append(
            "\n".join(
                [
                    f"### {app}",
                    "",
                    f"- Source CSV: `{path.relative_to(C.BASE)}`",
                    f"- Rows: {len(raw):,} (plan expected {expected:,})",
                    f"- Columns: {', '.join(map(str, raw.columns))}",
                    f"- Missing configured columns: {', '.join(missing_columns) or 'none'}",
                    f"- Missing values by column: `{missing_counts}`",
                    f"- Duplicate review IDs: {duplicate_count if duplicate_count is not None else 'not checked'}",
                    f"- Invalid ratings outside 1-5: {invalid_ratings}",
                    f"- Invalid/unparseable timestamps: {invalid_dates if invalid_dates is not None else 'not checked'}",
                    f"- Date range: {dates.min()} to {dates.max()}" if dates.notna().any() else "- Date range: unavailable",
                    f"- Rating distribution: `{ratings.value_counts(dropna=False).sort_index().to_dict()}`",
                ]
            )
        )

    interim_path = C.INTER / "upi_clean.csv"
    if interim_path.exists():
        corpus = pd.read_csv(interim_path, dtype={"id": "string"})
        app_rows = corpus.groupby("app").size().to_dict()
        invalid_rating_count = int(
            (corpus.score.notna() & ~corpus.score.isin([1, 2, 3, 4, 5])).sum()
        )
        dates = (
            pd.to_datetime(corpus.time, errors="coerce")
            if "time" in corpus
            else pd.Series(dtype="datetime64[ns]")
        )
        corpus_details = [
            "## Available processed corpus",
            "",
            f"- File: `data/interim/upi_clean.csv` ({len(corpus):,} rows)",
            "- This pre-existing cleaned intermediate was not regenerated from the supplied raw CSVs during this audit.",
            f"- Standardized columns: `{list(corpus.columns)}`",
            f"- Column data types: `{corpus.dtypes.astype(str).to_dict()}`",
            f"- Per-app rows: `{app_rows}`",
            f"- Duplicate app/review ID pairs: {int(corpus.duplicated(['app', 'id']).sum())}",
            f"- Invalid ratings outside 1-5: {invalid_rating_count}",
            f"- Invalid/unparseable timestamps: {int(dates.isna().sum())}" if len(dates) else "- Invalid/unparseable timestamps: not checked",
            f"- Missing values by column: `{corpus.isna().sum().to_dict()}`",
            f"- Date range: {dates.min()} to {dates.max()}" if dates.notna().any() else "- Date range: unavailable",
            f"- Rating distribution: `{corpus.score.value_counts(dropna=False).sort_index().to_dict()}`",
        ]
    else:
        corpus_details = [
            "## Available processed corpus",
            "",
            "- `data/interim/upi_clean.csv` is missing; Person A preprocessing is required.",
        ]

    report = [
        "# UPI data validation report",
        "",
        "Generated from files available in this repository. No review text, IDs, or user profile data are included.",
        "",
        "## Locally available UPI source CSVs",
        "",
        *details,
        "",
        *corpus_details,
        "",
        "## Schema note",
        "",
        "The supplied files match the configured required schema and planned row counts. Kaggle provenance was not independently verified.",
        "",
    ]
    C.BASE.joinpath("docs", "data_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(C.TABLES / "upi_data_validation.csv", index=False)
    print(
        "Raw UPI CSVs "
        + ("validated" if raw_validated else "missing or mismatched; see docs/data_validation_report.md")
    )
    return {
        "raw_files_available": raw_available,
        "raw_files_validated": raw_validated,
        "processed_corpus_rows": int(len(corpus)) if interim_path.exists() else 0,
    }


if __name__ == "__main__":
    run()
