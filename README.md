# Member 1 — UPI Data & Lexicon Pipeline

This folder contains the Person A / Member 1 implementation from the supplied Implementation Plan.

## Included deliverables

- `config.py`, `src/store.py`, `.gitignore`: project scaffold and shared output helpers.
- `src/s0_check_data.py`: raw UPI file and column verification.
- `src/prep_upi.py`: UPI cleaning and the three intermediate datasets.
- `src/eda_upi.py`: Table 5 (rating profile) and Table 7 (bigrams).
- `src/sent_lexicon.py`: VADER and TextBlob scores, Table 6 inputs and Figure 1(b) data.
- `src/aspects.py`: clause-level aspect sentiment, Table 9 and event-level CSV.
- `src/trust_risk.py`: trust/risk/fraud/functional flags and Table 10.
- `src/temporal.py`: monthly trend data for Figure 3.
- `src/figures_upi.py`: Figures 1, 2 and 3.
- `tests/test_member1.py`: unit tests for Member 1 functions.

## 1. Put the real data in place

Download the three CSVs from the Kaggle dataset named **UPI Payment Apps review - Google Play Store** and place/rename them as:

- `data/raw/upi/google_pay.csv`
- `data/raw/upi/paytm.csv`
- `data/raw/upi/phonepe.csv`

Raw data is deliberately not included in this package or committed to Git.

The default column map in `config.py` follows the plan's proposed headers (`reviewId`, `content`, `score`, `thumbsUpCount`, `at`, `replyContent`). **These are assumptions and must be checked against the actual CSV headers.** If they differ, edit `UPI_COLS` in `config.py`.

## 2. Install

Windows PowerShell:

```powershell
cd member1_work
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-member1.txt
$env:DP_BASE = (Get-Location).Path
```

## 3. Run Member 1 pipeline

Run in this order:

```powershell
python src\s0_check_data.py
python src\prep_upi.py
python src\eda_upi.py
python src\sent_lexicon.py
python src\trust_risk.py
python src\aspects.py
python src\temporal.py
python src\figures_upi.py
pytest -q tests
```

You can compare the two sentiment text variants:

```powershell
python src\sent_lexicon.py text_light
python src\sent_lexicon.py text
```

The chosen setting is `SENT_TEXT_COL` in `config.py`. Keep the version that best matches the paper's reported Table 6 and record the decision in `docs/deviations.md`.

## 4. Expected outputs

Intermediate files in `data/interim/`:
- `upi_all.csv`, `upi_en.csv`, `upi_clean.csv`
- `upi_scored.csv`, `upi_flagged.csv`, `aspect_events.csv`

Tables in `outputs/tables/`:
- `table5_rating_profile.csv`
- `table7_bigrams.csv`
- `fig1b_vader_by_app.csv`
- `table9_aspects.csv`
- `table10_trust_risk.csv`
- `fig3_monthly.csv`

Figures in `outputs/figures/`:
- `fig1_ratings_sentiment.png`
- `fig2_aspects.png`
- `fig3_trends.png`

Metrics are accumulated in `outputs/metrics.json`.

## 5. Paper targets to validate

The Implementation Plan lists these reference targets:
- Raw UPI records: 183,876
- After duplicate IDs and empty reviews: 183,864
- After empty-after-cleaning: 179,646
- After non-English filter: 174,552
- Final de-duplicated corpus: 81,932
- Overall mean rating: 4.14; 78.6% ratings at 4–5 stars; 17.6% at 1–2 stars
- Mean ratings: Google Pay 3.65, Paytm 4.27, PhonePe 4.11
- VADER accuracy 0.598 / macro-F1 0.478; TextBlob accuracy 0.560 / macro-F1 0.453
- Trust/risk target shares: 15.2%, 6.8%, 2.1%, 8.8% (trust, risk, fraud-security, functional problems)

These are validation targets, not hard-coded outputs. Real results cannot be calculated until the actual Kaggle CSVs are supplied and the column map is verified. The plan itself says the reconstructed lexicons may produce CLOSE rather than exact matches.

## Important methodology note

The plan's keyword lists are reconstructed because the original paper does not publish the full analysis lexicons. Treat the resulting keyword-based values as approximations and document any differences in `docs/deviations.md`.
