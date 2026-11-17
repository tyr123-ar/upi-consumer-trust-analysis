"""Compare available UPI measurements with paper validation targets."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C


def _status(reference, measured):
    if measured is None or pd.isna(measured):
        return "NOT REPRODUCED"
    difference = abs(float(measured) - float(reference))
    if difference <= 0.01:
        return "OK"
    if difference <= 0.03:
        return "CLOSE"
    return "OFF"


def _csv_row_count(path):
    if not path.exists():
        return None
    return int(len(pd.read_csv(path, usecols=["app"])))


def _raw_row_count(path):
    if not path.exists():
        return None
    return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=50000))


def run():
    corpus = None
    if (C.INTER / "upi_all.csv").exists():
        corpus = pd.read_csv(C.INTER / "upi_all.csv")
    clean_count = _csv_row_count(C.INTER / "upi_clean.csv")
    benchmark = None
    benchmark_path = C.TABLES / "upi_sentiment_benchmark.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path).set_index("method")
    raw_counts = {
        app: _raw_row_count(path) for app, path in C.UPI_FILES.items()
    }
    raw_total = (
        sum(raw_counts.values())
        if all(count is not None for count in raw_counts.values())
        else None
    )

    checks = [
        (
            "Raw UPI rows",
            183876,
            raw_total,
            "Local source-file counts are compared with plan values; file provenance is not asserted",
        ),
        (
            "Existing upi_all.csv row count (pipeline stage unverified)",
            183864,
            int(len(corpus)) if corpus is not None else None,
            "A matching count does not verify its upstream raw-file processing",
        ),
        (
            "Rows after cleaning",
            179646,
            None,
            "Preprocessing-stage counts are not independently verifiable without raw CSVs",
        ),
        (
            "Rows after non-English filtering",
            174552,
            _csv_row_count(C.INTER / "upi_en.csv"),
            "Existing intermediate row count; filtering parameters may differ from the paper",
        ),
        (
            "Final de-duplicated corpus rows",
            81932,
            clean_count,
            "Existing cleaned corpus; the count differs from the paper",
        ),
    ]
    checks[1:1] = [
        (
            f"{app} raw rows",
            expected,
            raw_counts[app],
            "Local source CSV row count; file provenance is not asserted",
        )
        for app, expected in C.EXPECTED_RAW_ROWS.items()
    ]
    if corpus is not None:
        checks.extend(
            [
                ("Overall mean rating", 4.14, corpus.score.mean(), "Descriptive UPI comparison"),
                ("Ratings 4-5 (%)", 78.6, 100 * corpus.score.isin([4, 5]).mean(), "Descriptive UPI comparison"),
                ("Ratings 1-2 (%)", 17.6, 100 * corpus.score.isin([1, 2]).mean(), "Descriptive UPI comparison"),
            ]
        )
        for app, reference in (
            ("Google Pay", 3.65),
            ("Paytm", 4.27),
            ("PhonePe", 4.11),
        ):
            actual = corpus.loc[corpus.app.eq(app), "score"].mean()
            checks.append((f"{app} mean rating", reference, actual, "Descriptive UPI comparison"))

    benchmark_targets = {
        "VADER": (0.598, 0.478),
        "TextBlob": (0.560, 0.453),
    }
    for method, (accuracy, macro_f1) in benchmark_targets.items():
        if benchmark is not None and method in benchmark.index:
            checks.append(
                (
                    f"{method} benchmark accuracy",
                    accuracy,
                    float(benchmark.loc[method, "accuracy"]),
                    "Same star-rating weak labels, but implementation/sampling may differ",
                )
            )
            checks.append(
                (
                    f"{method} benchmark macro-F1",
                    macro_f1,
                    float(benchmark.loc[method, "macro_f1"]),
                    "Same star-rating weak labels, but implementation/sampling may differ",
                )
            )
        else:
            checks.extend(
                [
                    (f"{method} benchmark accuracy", accuracy, None, "Benchmark not run"),
                    (f"{method} benchmark macro-F1", macro_f1, None, "Benchmark not run"),
                ]
            )

    trust_targets = {
        "Trust and endorsement %": 15.2,
        "Risk and grievance %": 6.8,
        "Fraud-security %": 2.1,
        "Functional-problem %": 8.8,
    }
    trust_path = C.TABLES / "table10_trust_risk.csv"
    trust_table = pd.read_csv(trust_path) if trust_path.exists() else None
    if trust_table is not None and "App" in trust_table and (trust_table.App == "All apps").any():
        all_apps = trust_table.loc[trust_table.App.eq("All apps")].iloc[0]
        for key, reference in trust_targets.items():
            checks.append((key, reference, all_apps.get(key), "Reconstructed English keyword lexicons"))
    else:
        checks.extend((key, reference, None, "Trust/risk table unavailable") for key, reference in trust_targets.items())

    result = pd.DataFrame(
        [
            {
                "check": name,
                "paper_value": reference,
                "measured_value": measured,
                "difference": None if measured is None else float(measured) - float(reference),
                "status": _status(reference, measured),
                "explanation": explanation,
            }
            for name, reference, measured, explanation in checks
        ]
    )
    lines = [
        "UPI paper validation report",
        "==========================",
        "",
        "Status thresholds: OK <= 0.01 absolute difference; CLOSE <= 0.03; OFF > 0.03. Counts are compared exactly.",
        "Count checks are labeled with explanations where upstream raw inputs are unavailable.",
        "",
        result.to_string(index=False),
        "",
        "QRIS measurements are produced separately; this UPI-only validator does not validate QRIS reference targets.",
        "See outputs/tables/table13_qris_portion.csv and docs/person_b_reproduction_report.md for measured QRIS results and limitations.",
        "Star ratings are weak sentiment labels, not ground truth. UPI and QRIS populations are not interchangeable.",
    ]
    (C.BASE / "docs" / "validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    result.to_csv(C.TABLES / "paper_validation.csv", index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
