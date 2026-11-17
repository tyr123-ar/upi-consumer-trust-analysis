# Deviations from the paper

The three local UPI source CSVs were supplied under alternate filenames in `data/`. Their row counts and required headers match the implementation-plan values; their Kaggle provenance was not independently verified, and no authenticated download was attempted. Raw preprocessing was not rerun, so the existing cleaned UPI intermediate was used only where a stage can operate on it. QRIS results below are measured from the public DOI version file, not inferred from UPI data or copied from paper targets.

| Check | Paper | Available measurement | Reason / debugging performed | Owner |
|---|---:|---:|---|---|
| Raw UPI rows | 183,876 | 183,876 in the three supplied CSVs | File counts match the plan and configured required columns are present; Kaggle provenance was not independently verified. | Person B |
| Existing `upi_all.csv` rows | 183,864* | 183,864 | Available checked-in intermediate only; its upstream source stage cannot be independently verified. | Person B |
| Rows after cleaning | 179,646 | NOT REPRODUCED | Raw preprocessing could not be rerun. | Person B |
| Rows after non-English filtering | 174,552 | 179,229 in available `upi_en.csv` | Existing intermediate differs from reference and uses previously applied preprocessing. | Person B |
| Final de-duplicated corpus | 81,932 | 83,140 in available `upi_clean.csv` | Existing corpus differs by 1,208 rows; no edits were made to force the paper count. | Person B |
| QRIS source file | Mendeley DOI `10.17632/xpkw64fd2h.1` | 200 raw rows, 11 observed columns; SHA-256 recorded in `data_validation_report.md` | The DOI version CSV is present and byte-identical to the user-supplied `FINAL DATA.csv`. DataCite abstract states 1,573 collected comments, which conflicts with the downloadable file. | Person B |
| QRIS preprocessing | 178 clean comments in the supplemental reference | 177 cleaned comments from the 200-row file | Actual stages: 200 raw -> 181 non-empty normalized -> 180 unique normalized texts -> 177 with at least two normalized words. No adjustment was made to force the reference count. | Person B |
| QRIS sentiment | Reference targets are not treated as measurements | 109 negative (61.58%; 95% Wilson 54.24%-68.43%), 38 neutral, 30 positive; 88.58% of retained likes are on negative-labeled comments | Local inference with pinned Indonesian RoBERTa revision `ac452dcb0f4966130bba44f4ee0013bb5d52c282`; raw source text was scored. Weighted shares are descriptive engagement counts, not independent votes. | Person B |
| QRIS themes and topics | QRIS-specific | 6 exploratory keyword themes, 4 narrow lexicons, LDA selected k=6 with C_V=0.4022 | All measures use the 177-comment cleaned file; topic coherence used 160 comments with retained topic vocabulary. Rules and topics are exploratory, not verified incident rates. | Person B |

`*` This is a count comparison only, not a claim that the current `upi_all.csv` stage is verified to match the paper's stated post-filter stage. Benchmark, model and topic results are recorded in the generated report/validation outputs only after successful execution.
