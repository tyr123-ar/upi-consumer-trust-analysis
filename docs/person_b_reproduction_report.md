# Person B — Reproduction report

## B1 — Dataset and preprocessing

The analysis targets the Kaggle [UPI Payment Apps review - Google Play Store dataset](https://www.kaggle.com/datasets/komalkhetlani/upi-payment-apps-review-google-play-store), using Google Pay, Paytm, and PhonePe reviews. Three local CSVs were available as `data/GooglePayIndia.csv`, `data/PaytmIndia.csv`, and `data/PhonePayIndia.csv`. Their row counts match the plan and their headers contain the required loader columns; detailed raw validation is in [`data_validation_report.md`](./data_validation_report.md). The files' Kaggle provenance was not independently verified, and no authenticated download was attempted.

The available preprocessed corpus contained 83,140 cleaned reviews: Google Pay 26,039, Paytm 51,967, and PhonePe 5,134. The checked-in `upi_all.csv` had 183,864 rows and `upi_en.csv` had 179,229 rows. The cleaned count differs from the 81,932 reference target. The intermediate counts were not regenerated from the local raw files; the raw files' row counts and schema were checked separately, but their Kaggle provenance was not independently verified. See the validation report and generated data validation table for the exact counts used.

Person B consumes the existing standardized `app`, `id`, `score`, and `text_light` columns. RoBERTa output retains only internal app/review keys, rating, predicted label, confidence, and model revision; aggregate tables omit review IDs and comment text. The Kaggle page did not state a licence in the supplied implementation-plan notes; users should check the dataset page for current terms before redistribution. The dataset source must be cited in reports and presentations.

The QRIS source is the public Mendeley Data version at [DOI 10.17632/xpkw64fd2h.1](https://doi.org/10.17632/xpkw64fd2h.1), licensed CC BY 4.0. The downloadable `FINAL DATA.csv` contains 200 records and 11 actual columns; its schema is mapped in `config.QRIS_COLS`. The DataCite abstract describes a 1,573-comment collection, which is not the file exposed by this DOI version. The additional user-supplied copy is byte-identical to the downloaded file. Only the available 200-row file was analyzed.

## B2 — Transformer sentiment benchmark

The intended classifier is `cardiffnlp/twitter-roberta-base-sentiment-latest`, loaded at a fixed Hugging Face revision recorded in `config.py`. The benchmark uses up to 1,000 eligible reviews per star rating, a minimum of three words, and a fixed seed. Weak labels are 1-2 stars = negative, 3 = neutral, and 4-5 = positive. RoBERTa, VADER, and TextBlob are compared on the same sampled reviews; McNemar's test compares paired RoBERTa/VADER correctness.

The benchmark ran on **5,000 reviews**, exactly 1,000 per rating and all with at least three words. The fixed seed was 42, with per-rating sampling seeds 43-47. Eligible counts for ratings 1 through 5 were 21,838; 3,066; 3,508; 6,072; and 34,984.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| RoBERTa | 0.6282 | 0.5883 | 0.5760 | 0.5786 |
| VADER | 0.5312 | 0.4939 | 0.4848 | 0.4791 |
| TextBlob | 0.5078 | 0.5191 | 0.4872 | 0.4727 |

Paired RoBERTa/VADER McNemar's test (continuity-corrected) gave chi-square = 166.4932, p = 4.3164e-38, from 946 pairs where RoBERTa alone was correct and 461 where VADER alone was correct. Per-class metrics and all confusion matrices are in the generated benchmark tables.

The model was loaded from the pinned revision in `config.py`. The available full-corpus prediction file contains 3,056 unique reviews with the pinned revision out of 83,140; 80,084 remain unscored. The requested 5,000-review benchmark sample was fully scored separately. Thus the benchmark is measured, but a full-corpus RoBERTa sentiment distribution is **not** claimed. Completion requires scoring the remaining corpus on a faster/GPU machine. Star ratings are noisy proxies and the metrics are not estimates against human-annotated sentiment.

## B3 — Topic modelling

The UPI corpus is sampled up to 12,000 reviews per app, with deterministic sampling. Lowercased English tokens are filtered for stopwords and tokens shorter than three characters. Gensim LDA models are evaluated at k = 4, 5, 6, 7, 8, and 10 using C_V coherence; the highest measured coherence is selected independently per app. Topic words and proportions are descriptive and require human interpretation. Automated coherence does not establish that a topic is substantively valid.

The best C_V coherence on the evaluated grid selected k=6 for Google Pay (0.5481), k=6 for Paytm (0.4838), and k=10 for PhonePe (0.4478). Topic words and model-assigned sample proportions are in `table8_upi_topics.csv`; the complete k/coherence grid is in `upi_topic_coherence.csv`.

## B4 — Theme analysis

The adapted analysis applies explicit keyword rules for transaction failure, payment delays, refunds, fraud/suspicious transactions, security/privacy, account/verification, support, and convenience/usability. Rules can overlap. Unit tests cover representative positive examples and potentially ambiguous negative examples. These rules are exploratory and are not a validated classifier; false positives and false negatives remain possible.

UPI themes use English review text and are not substitutes for QRIS coding. QRIS uses a separate Indonesian lexicon and separate topic model; see B6.

## B5 — Research questions, comparison, and limitations

The existing UPI corpus has 83,140 cleaned reviews; the available `upi_all.csv` summary has a mean rating of 4.1452, 78.5853% 4-5-star reviews, and 17.6081% 1-2-star reviews. Mean ratings were 3.65 for Google Pay, 4.27 for Paytm, and 4.11 for PhonePe. These are descriptive statistics from existing processed data, not verified raw-file totals.

The attached supplemental guidance document lists six research questions. They provide context for the available outputs but do not replace the master prompt's scope. Within the available UPI data, the questions map as follows:

1. **RQ1 — Which benefits are mentioned most?** The existing UPI bigram output lists “easy use” (1,866) and “good app” (1,538) as leading phrases in 4-5-star reviews. Table 9 reports convenience/ease in 13.4% of reviews, with mean VADER +0.40 and mean rating 4.61. This supports a descriptive statement that ease/convenience language is common in positive UPI feedback; it does not establish a causal driver of adoption or ratings.
2. **RQ2 — Which risks and problems appear most?** In the LDA samples, Google Pay's larger issue topics concern phone/app operation (21.57% of assigned sample reviews; 39.97% VADER-negative) and card/payment setup (19.79%; 36.16% negative). Paytm's largest issue topic concerns customer service/support (21.31%; 45.81% negative). PhonePe topics include transaction/payment problems (13.08%; 33.22% negative) and recharge/account support (10.45%; 37.42% negative). Table 9's reliability/support aspects, Table 10's reconstructed risk lexicons, and the exploratory UPI theme table provide complementary descriptive signals. These are topics and keyword-rule matches, not validated prevalence of real-world incidents; topic proportions are sample/model-dependent.
3. **RQ3 — What is the overall sentiment?** Full-corpus UPI VADER labels in the existing scored corpus are approximately 52.4% positive, 27.2% neutral, and 20.4% negative. On the 177 cleaned QRIS comments, pinned Indonesian RoBERTa labels were 61.58% negative, 21.47% neutral, and 16.95% positive; negative-labeled comments account for 88.58% of retained TikTok likes (71,858 total likes). The unweighted negative share has a 95% Wilson interval of 54.24%-68.43%. On the separate 5,000-review weak-label UPI benchmark, RoBERTa accuracy/macro-F1 were 0.6282/0.5786; VADER 0.5312/0.4791; TextBlob 0.5078/0.4727. These are descriptive model outputs, not human-annotated ground truth.
4. **RQ4 — Which aspects go with positive or negative?** Table 9/Figure 2 show the strongest positive aspect mean was convenience/ease (+0.40; 74.7% positive clauses), while reliability/transactions (-0.03; 33.5% negative) and customer support (41.4% negative) were more negative. Reconstructed clause lexicons and VADER scores are approximate and do not establish that the aspect caused a rating.
5. **RQ5 — Security and fraud concerns in public talk?** In the available QRIS TikTok file, 16/177 comments (9.04%; 95% Wilson 5.64%-14.18%) match the exploratory fraud/manipulated-payment-evidence lexicon, and 4/177 (2.26%) match the security/phishing lexicon. These are text-rule matches, not verified fraud reports. The separate UPI keyword analysis remains a different corpus and definition.
6. **RQ6 — General experience versus security?** The measured QRIS portion of Table 13 contrasts the 12.43% network/payment-problem matches, 8.47% verification/trust-mechanism matches, and 2.26% security/phishing matches with existing UPI keyword measurements. The table explicitly warns that languages, platforms, lexicons, and samples differ; it is descriptive, not an apples-to-apples population comparison.

Overall, these outputs describe sentiment agreement with weak star-rating labels and topic/theme prevalence in the supplied app-review corpus. They cannot establish why app use or ratings changed, infer causal effects, or generalize to all UPI users. Reviews are self-selected, app populations differ, and the existing corpus was preprocessed before this execution and does not match the reference final count.

UPI and QRIS populations are not interchangeable. The QRIS file has only 200 source rows and no video identifier, so no video-level weighting or inference is possible. Its size also conflicts with the larger collection described in the DataCite abstract; no unseen records or expected paper values were imputed.

### Results and artifacts

See `outputs/tables/upi_sentiment_benchmark.csv`, `upi_mcnemar_roberta_vader.json`, `table8_upi_topics.csv`, `table_upi_exploratory_themes.csv`, and `table13_upi_reference_comparison.csv`, plus the QRIS results listed in B6. Full-corpus UPI RoBERTa predictions remain incomplete, and Kaggle provenance of the local raw UPI files remains unverified; no values are imputed for either.

## B6 — QRIS preprocessing and analysis

The downloaded QRIS file has 200 rows, no missing fields, no duplicate comment IDs, and six exact duplicate comment texts. The measured profile has 199 distinct users, counted without exporting user identifiers; dates range from 2025-11-06 to 2026-04-01. The raw file contains 71,864 likes. Preprocessing retained 177 comments after empty-normalization removal, normalized-text deduplication, and a minimum-two-word filter. Clean outputs omit source user IDs, usernames, nicknames, and homepages.

Indonesian RoBERTa used `w11wo/indonesian-roberta-base-sentiment-classifier`, pinned at revision `ac452dcb0f4966130bba44f4ee0013bb5d52c282`. It scored the original comment text locally on CPU; its output contains only a local row key, label, confidence, and model revision. Mean confidence was 0.938947. The comment- and like-weighted results and unweighted Wilson intervals are in `qris_sentiment_distribution.csv`; Wilson intervals are not applied to like-weighted shares.

Exploratory themes matched 16 comments to fraud/manipulated payment evidence (9.04%), 4 to security/phishing (2.26%), 22 to network/payment problems (12.43%), 15 to verification/trust mechanisms (8.47%), 4 to information seeking/awareness (2.26%), and 6 to user experience/convenience (3.39%). Categories overlap and are not incident rates. The four narrow lexicons are in `qris_narrow_lexicons.csv`.

The deterministic QRIS LDA run selected k=6 with C_V=0.4022 from the evaluated k=2-6 grid. The model used 160 of the 177 cleaned comments after token and vocabulary filtering. Topic words/proportions are exploratory, and there is no video ID for group-level analysis.

Generated artifacts include `outputs/tables/qris_profile.csv`, `qris_sentiment_distribution.csv`, `qris_themes.csv`, `qris_narrow_lexicons.csv`, `qris_topics.csv`, `qris_topic_coherence.csv`, `outputs/figures/fig4_qris.png`, and `outputs/tables/table13_qris_portion.csv`. Run only the QRIS pipeline with `python run_qris.py`; it does not invoke UPI stages.

Person B aggregate metrics are stored separately in `outputs/person_b_metrics.json`; the shared `outputs/metrics.json` is retained unchanged.
