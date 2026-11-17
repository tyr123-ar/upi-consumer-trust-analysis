# Digital Payments Consumer Trust Analysis — UPI and QRIS

This repository contains the UPI Google Play review pipeline and QRIS TikTok comment analysis, including sentiment, themes, topics, profiles, and aggregate comparisons.

## Dataset

Source: [UPI Payment Apps review - Google Play Store](https://www.kaggle.com/datasets/komalkhetlani/upi-payment-apps-review-google-play-store). Cite the Kaggle dataset page when reusing results. The dataset page did not state a licence in the supplied project notes; check its current terms before redistribution.

The standard raw-input paths are:

- `data/raw/upi/google_pay.csv`
- `data/raw/upi/paytm.csv`
- `data/raw/upi/phonepe.csv`

Kaggle download (requires an authorized Kaggle account and local Kaggle CLI credentials; do not commit credentials):

```powershell
python -m pip install kaggle
kaggle datasets download -d komalkhetlani/upi-payment-apps-review-google-play-store -p data/raw/upi --unzip
```

For authentication, place the Kaggle API token file at `%USERPROFILE%\.kaggle\kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in the local shell. Do not paste credentials into source files, issues, or reports.

Inspect the extracted filenames and headers before renaming files or changing `UPI_COLS` in `config.py`. The required columns are `reviewId`, `content`, `score`, `at`; `thumbsUpCount` and `replyContent` are optional in the loader. Raw data is gitignored.

Three local UPI CSVs are available as `data\GooglePayIndia.csv`, `data\PaytmIndia.csv`, and `data\PhonePayIndia.csv`. Their row counts match the plan (34,228; 137,907; 11,741) and their headers include the configured required columns. `config.py` uses these as local fallbacks when the standard paths above are absent. Their Kaggle provenance was not independently verified and no authenticated download was attempted. The UPI raw row/schema checks are documented in the [data validation report](docs/data_validation_report.md); see also the [Person B reproduction report](docs/person_b_reproduction_report.md) and [deviations](docs/deviations.md).

Validate available files without preprocessing or overwriting Person A outputs:

```powershell
python src\s0_check_data.py
python src\validate_data.py
```

For downloaded files with different names, copy them to the three standard paths above after confirming which app each file contains. Keep these raw CSVs local; `.gitignore` excludes both the standard raw directory and the currently supplied filenames.

### QRIS TikTok data

The QRIS source is the public Mendeley Data dataset [10.17632/xpkw64fd2h.1](https://doi.org/10.17632/xpkw64fd2h.1), licensed CC BY 4.0. Its version file is `FINAL DATA.csv`; it is stored locally as `data/raw/qris/qris_tiktok.csv` and is gitignored. The actual file has 200 rows and 11 columns. Its observed headers are mapped in `config.QRIS_COLS`. The DataCite abstract describes a larger collection of 1,573 comments, which does not match the single 200-row file in this version; reported results use only the downloaded file and do not claim to represent that larger collection.

If the raw file is absent, download and verify the DOI version, then run the QRIS-only pipeline. Neither command runs or overwrites UPI stages:

```powershell
python src\download_qris.py
python run_qris.py
```

It creates a privacy-minimized cleaned file, QRIS profile, pinned Indonesian RoBERTa predictions, comment- and like-weighted sentiment, Wilson intervals, exploratory Indonesian themes and narrow lexicons, LDA topics, Figure 4, and the QRIS portion of Table 13. The downloaded raw CSV contains user identifiers; the profile counts distinct users but does not export their identifiers, and the cleaned/scored outputs omit those source columns.

## Environment and execution

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an exact environment, install `requirements.lock.txt`; its pinned versions satisfy the direct constraints in `requirements.txt`.

Run the protected combined pipeline only when intentionally allowing Person A's shared outputs to be rewritten:

```powershell
python run_all.py --allow-person-a-overwrite
```

Person B's safe UPI path is separate:

```powershell
python run_person_b.py
```

`run_person_b.py` never runs Person A preprocessing, analyses, or Figure 1. Its metrics and generated figures/tables use Person B-specific paths. It can still perform full-corpus model scoring and LDA; use `--benchmark-sample-only` to restrict RoBERTa inference to the 5,000-review benchmark sample. The combined `run_all.py` refuses stages that overwrite Person A data or outputs unless the explicit `--allow-person-a-overwrite` option is supplied.

Resume Person B from UPI topic modelling or generate only Person B figures:

```powershell
python run_person_b.py --from topics_upi
python run_person_b.py --only figures_person_b
```

The UPI pipeline uses existing `data/interim/upi_clean.csv` when raw UPI inputs are absent and stops on a failed required stage. RoBERTa inference uses CUDA when available and CPU otherwise.

On CPU-only systems, score only the fixed 5,000-review weak-label benchmark sample, then continue with the same benchmark and downstream Person B UPI stages:

```powershell
python run_person_b.py --benchmark-sample-only
```

This option does **not** claim full-corpus RoBERTa coverage. Use the Person B runner's default full-corpus mode only on a machine with sufficient compute to populate all RoBERTa predictions.

Run data validation and tests:

```powershell
python src\validate_data.py
python validate_paper.py
python -m pytest -q tests
```

## Outputs

- Intermediate model/topic assignments are written to `data/interim/` and are gitignored.
- Aggregate tables and figures are written to `outputs/tables/` and `outputs/figures/`.
- Person B metrics are written to `outputs/person_b_metrics.json`; the shared `outputs/metrics.json` remains unchanged.
- Validation reports are written under `docs/`.
- QRIS outputs are generated separately under QRIS-specific filenames; the UPI pipeline does not invoke them.

The UPI model benchmark uses star ratings as weak labels, not ground-truth sentiment. UPI and QRIS results are descriptive, use different languages/platforms/models, and do not establish causal effects or directly comparable prevalence.

## Existing Member 1 implementation

Person A's modules remain in `src/`: `prep_upi.py`, `eda_upi.py`, `sent_lexicon.py`, `aspects.py`, `trust_risk.py`, `temporal.py`, and `figures_upi.py`. They provide preprocessing, rating profiles, VADER/TextBlob scores, aspect/trust-risk summaries, and Figures 1-3. Their proposed raw column mapping is not confirmed until the Kaggle files are supplied. The paper's values are validation targets only; see [`deviations.md`](docs/deviations.md) and [`validation_report.txt`](docs/validation_report.txt).

Person B adds resumable RoBERTa inference, weak-label metrics and McNemar testing, per-app LDA, exploratory UPI keyword themes, aggregate figures/tables, data validation, and pipeline orchestration. The report at [`person_b_reproduction_report.md`](docs/person_b_reproduction_report.md) describes the methodology and observed limitations.

The original paper's UPI reference targets include 183,876 raw reviews, 81,932 final clean reviews, a mean rating of 4.14, and per-app mean ratings of 3.65 (Google Pay), 4.27 (Paytm), and 4.11 (PhonePe). These are not hard-coded outputs and are not treated as verified measurements from the currently available intermediate corpus. The reconstructed keyword lexicons are approximations, not the paper's unpublished original lists.
