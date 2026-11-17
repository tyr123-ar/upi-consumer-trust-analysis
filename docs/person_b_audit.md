# Person B implementation audit

## Repository state before Person B changes

- The repository was on `main`, at the initial commit `3541aef`, with a clean worktree. A remote named `origin` is configured.
- The repository contains the Person A UPI pipeline (`prep_upi.py`, `eda_upi.py`, `sent_lexicon.py`, `aspects.py`, `trust_risk.py`, `temporal.py`, `figures_upi.py`) and 10 passing baseline tests.
- Existing outputs include Tables 5, 7, 9 and 10, Figures 1-3, metrics, and intermediate CSVs. Existing generated results were not overwritten before this audit.
- `requirements-member1.txt` lists broad minimum versions for the Person A stack. There was no unified `requirements.txt`, lock file, pipeline entry point, RoBERTa implementation, benchmark implementation, topic model, reusable evaluation module, or Person B report.

## Current files and interface

Person A preprocessing writes `data/interim/upi_clean.csv` with standardized fields including `app`, `id`, `score`, `text_light`, `text`, and `n_words`. Person B can consume `app`, `id`, `score`, and `text_light` without changing Person A's preprocessing format. Existing lexicon scores are in `upi_scored.csv`.

The configured raw inputs are `data/raw/upi/google_pay.csv`, `paytm.csv`, and `phonepe.csv`. None were present at audit time. The configured raw headers in `config.UPI_COLS` are documented as assumptions in Person A's README; the actual Kaggle file headers therefore could not be verified.

The available checked-in intermediates had 183,864 rows in `upi_all.csv`, 179,229 in `upi_en.csv`, and 83,140 in `upi_clean.csv`. The cleaned corpus app counts were 26,039 Google Pay, 51,967 Paytm, and 5,134 PhonePe. These are existing processed-corpus counts, not raw Kaggle source-file counts. They differ from the implementation-plan cleaned count of 81,932. Existing `outputs/metrics.json` contains earlier raw/preprocessing claims that cannot be independently revalidated without the missing raw CSVs.

## Dependencies and environment

The initial dependency file covered pandas, NumPy, SciPy, scikit-learn, VADER, TextBlob, matplotlib, and pytest using minimum-version constraints. RoBERTa requires Transformers and PyTorch; LDA requires Gensim. At the audit, VADER, TextBlob, Transformers, Gensim, NLTK and Seaborn were not importable; core scientific Python packages and PyTorch were present. Kaggle CLI and Kaggle credentials were also absent.

## Missing or adapted work

- Missing: Hugging Face sentiment inference, same-sample weak-label benchmark, McNemar test, LDA topic selection and outputs, UPI exploratory themes, Person B figures/tables, orchestration, validation report, and reproduction report.
- Broken/unverifiable: source raw row counts, raw schema, raw duplicate/missing/rating/date checks, and all raw preprocessing counts cannot be verified without the three source CSVs. The configured Kaggle schema must not be presented as confirmed.
- Not reproducible: all QRIS-specific cleaning, Indonesian model inference, like-weighted proportions/Wilson intervals, QRIS themes, QRIS topics, and QRIS figures/tables. No QRIS data was supplied; no QRIS results are imputed.
- The benchmark's rating targets are weak labels only (1-2 negative, 3 neutral, 4-5 positive); they are not independently validated sentiment ground truth.

## Execution order

1. Validate raw files when supplied and inspect their actual headers and values.
2. Run Person A preprocessing if raw files exist; otherwise use the existing cleaned intermediate and make that limitation explicit.
3. Run existing UPI descriptive, lexicon, aspect, trust/risk, and temporal analyses.
4. Run resumable RoBERTa inference, then VADER/TextBlob weak-label benchmarks and McNemar comparison.
5. Fit per-app LDA models for the requested topic counts, select by measured C_V coherence, and generate aggregate topic outputs.
6. Generate explicitly exploratory UPI keyword themes, figures, validation results, and documentation.
7. Run tests and inspect generated aggregate artifacts.

## Data and privacy note

Existing checked-in intermediates contain review text and review IDs. This implementation does not duplicate those fields into public tables, figures, or the Person B aggregate outputs; new intermediate model assignments are gitignored. The existing tracked intermediates were left unchanged to preserve Person A's work. Raw source data and model weights are not added by this implementation.
