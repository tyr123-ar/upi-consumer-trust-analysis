
# Digital Payments and Consumer Trust: Member 1 Report (A1–A5)

## A1. Data Sources and Preprocessing

### A1.1 Data Source

This study analyses user-generated Google Play Store reviews of three widely used Unified Payments Interface (UPI) applications in India: Google Pay, Paytm, and PhonePe. The data was obtained from the Kaggle dataset titled *UPI Payment Apps Review – Google Play Store* by Komal Khetlani.

The dataset contains reviews collected from the Google Play Store, including review text, star ratings, review identifiers, timestamps, helpfulness information, and developer-response information.

The original dataset contained **183,876 reviews**, distributed across the three applications as follows:

| Application | Number of reviews |
|---|---:|
| Google Pay | 34,228 |
| Paytm | 137,907 |
| PhonePe | 11,741 |
| **Total** | **183,876** |

The three datasets were combined into a unified dataset while retaining an application identifier to distinguish reviews belonging to Google Pay, Paytm, and PhonePe.

### A1.2 Data Cleaning and Preprocessing

A systematic preprocessing pipeline was implemented to improve data quality and prepare the reviews for subsequent text-mining and sentiment-analysis tasks.

The following preprocessing operations were performed:

1. **Duplicate ID and empty-review removal:** Duplicate review identifiers and records containing empty review text were removed.
2. **Lowercase conversion:** All review text was converted to lowercase to ensure consistent text representation.
3. **URL removal:** URLs and web links were removed from the review text.
4. **Punctuation and special-character removal:** Non-alphanumeric characters were removed according to the implemented cleaning rules, while letters, digits, spaces, and apostrophes were retained.
5. **Whitespace normalization:** Multiple consecutive spaces were replaced with single spaces.
6. **Language filtering:** Reviews containing fewer than 90% ASCII characters were excluded as a language-screening step.
7. **Text deduplication:** Exact duplicate review texts were removed within each application.

The final corpus therefore contains one record per unique application and cleaned-review-text combination.

### A1.3 Preprocessing Statistics

The following table summarizes the actual number of records retained at each stage of preprocessing.

| Preprocessing Stage | Number of Reviews |
|---|---:|
| Original dataset | 183,876 |
| After duplicate ID and empty-review removal | 183,864 |
| After text cleaning and empty-text removal | 180,826 |
| After ASCII-based language filtering | 179,229 |
| Final corpus after within-app text deduplication | **83,140** |

The final corpus contained 83,140 unique reviews across the three UPI applications.

The substantial reduction in the number of reviews after deduplication indicates the presence of repeated textual feedback in the original dataset. Removing these repeated texts helps prevent frequently repeated short reviews from disproportionately influencing the subsequent text analysis.

### A1.4 Ethical Considerations

The study uses publicly available application reviews for academic research purposes. The analysis focuses on aggregate patterns and statistical findings rather than identifying individual reviewers.

Usernames, profile links, and other direct identifiers should not be included in analytical outputs or the submitted repository. Raw review data should be kept outside public version control.

The language-filtering procedure introduces a limitation because Hindi, Hinglish, and other non-English feedback may be excluded. Consequently, the findings primarily represent the English-language portion of the collected reviews rather than the complete linguistic diversity of UPI users.

---

## A2. Dataset Profile and Review Characteristics

### A2.1 Application-Wise Rating Profile

The rating profile provides an overview of consumer feedback across the three applications. The analysis considers review counts, mean ratings, rating distributions, review lengths, and developer-response rates.

The rating-profile table contains 183,864 records after duplicate review IDs and empty review text were removed.

| Application | Reviews | Mean Rating | 1–2 Stars (%) | 4–5 Stars (%) | Mean Words | Reply Rate (%) | Reply Rate to 1–2 Star Reviews (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Google Pay | 34,219 | 3.65 | 29.8 | 64.5 | 15.5 | 17.9 | 50.7 |
| Paytm | 137,904 | 4.27 | 14.6 | 82.1 | 5.1 | 49.4 | 74.4 |
| PhonePe | 11,741 | 4.11 | 17.4 | 78.2 | 4.6 | 25.0 | 73.3 |
| **All Apps** | **183,864** | **4.15** | **17.6** | **78.6** | **7.0** | — | — |

Across all three applications, the mean rating is 4.15. Approximately 78.6% of ratings are four or five stars, while 17.6% are one or two stars.

Paytm has the highest mean rating (4.27) and the highest proportion of four- and five-star reviews (82.1%). Google Pay has the lowest mean rating (3.65) and the largest proportion of one- and two-star reviews (29.8%). PhonePe has a mean rating of 4.11.

Review length also differs substantially across applications. Google Pay reviews average 15.5 words, compared with 5.1 for Paytm and 4.6 for PhonePe. This indicates that Google Pay reviews in the collected dataset tend to contain more detailed textual feedback.

Paytm has the highest overall developer reply rate (49.4%), followed by PhonePe (25.0%) and Google Pay (17.9%).

### A2.2 Review-Length Patterns

The application-level mean word counts are 15.5 for Google Pay, 5.1 for Paytm, and 4.6 for PhonePe. The overall mean is 7.0 words.

These differences are important when interpreting sentiment and keyword results. Very short reviews may contain limited context, whereas longer reviews may describe multiple experiences, benefits, or problems.

### A2.3 Frequent Positive and Negative Bigrams

Bigram analysis was performed to identify frequently occurring two-word expressions in positive-rating and negative-rating reviews.

| Frequent Bigrams in 4–5 Star Reviews | Count | Frequent Bigrams in 1–2 Star Reviews | Count |
|---|---:|---|---:|
| easy use | 1,866 | google pay | 1,880 |
| good app | 1,538 | customer care | 1,232 |
| google pay | 1,222 | worst app | 1,032 |
| nice app | 1,104 | customer service | 733 |
| best app | 1,075 | not working | 696 |
| great app | 733 | not able | 627 |
| payment app | 540 | bank account | 620 |
| money transfer | 499 | app not | 566 |
| love app | 408 | doesn't work | 501 |
| app easy | 389 | use app | 490 |

Positive-rating reviews frequently contain expressions related to ease of use, general approval, and money transfer.

Negative-rating reviews prominently contain references to customer care, customer service, non-working functionality, and bank-account issues.

The presence of phrases such as "not working" and "doesn't work" illustrates why negation terms such as "not" and "no" should be retained during bigram extraction.

Some frequent bigrams are generic application references rather than specific benefits or problems. In particular, phrases such as "google pay", "good app", and "use app" provide limited diagnostic information. This is a limitation of frequency-based bigram analysis.

### A2.4 Overall Interpretation

The rating profile is predominantly positive, but the distribution differs across applications.

Google Pay has a notably larger low-rating share and longer reviews. Paytm has the highest mean rating and developer reply rate in the collected dataset.

The negative-review bigrams indicate that customer support, transaction functionality, and bank-account issues are recurring subjects in critical feedback.

These results describe the collected reviews and should not be interpreted as a representative survey of all UPI users.

---

## A3. Lexicon-Based Sentiment Analysis

### A3.1 Methodology

VADER and TextBlob were used as lexicon-based sentiment analysis tools to examine the emotional polarity of UPI application reviews.

VADER produces a compound sentiment score, while TextBlob produces a polarity score.

The following classification thresholds were used:

- **Positive:** Score greater than or equal to +0.05.
- **Negative:** Score less than or equal to -0.05.
- **Neutral:** Score between -0.05 and +0.05.

These methods provide rule-based sentiment estimates without requiring a task-specific training dataset. They are useful as transparent baselines, although they may not reliably interpret domain-specific expressions, sarcasm, mixed opinions, or contextual negation.

### A3.2 VADER Sentiment Distribution

The application-wise VADER output reports the following sentiment distribution:

| Application | Positive (%) | Neutral (%) | Negative (%) |
|---|---:|---:|---:|
| Google Pay | 54.0 | 21.3 | 24.7 |
| Paytm | 52.0 | 29.4 | 18.5 |
| PhonePe | 48.4 | 34.1 | 17.4 |

Google Pay has the highest positive sentiment share (54.0%), but it also has the highest negative sentiment share (24.7%).

Paytm has 52.0% positive and 18.5% negative sentiment. PhonePe has the lowest positive share (48.4%) and the highest neutral share (34.1%).

The coexistence of relatively high positive and negative sentiment in Google Pay reviews suggests a more polarized sentiment distribution in the collected text.

It would therefore be misleading to interpret positive sentiment share alone as an overall ranking of application quality.

### A3.3 TextBlob and Sentiment Evaluation

TextBlob was also applied as a second lexicon-based sentiment baseline.

Both VADER and TextBlob provide useful approaches for identifying broad sentiment patterns in user-generated reviews.

However, lexicon-based methods have limitations when analysing digital-payment feedback. Words associated with transactions, authentication, account blocking, and payment failures may have domain-specific meanings that general-purpose sentiment dictionaries do not fully capture.

If sentiment predictions are compared with star ratings, the ratings should be described as weak labels rather than ground truth.

A review can contain both positive and negative statements despite having a single star rating. Therefore, star ratings do not necessarily represent the sentiment of every statement in a review.

### A3.4 Benefits and Problems Reflected in Review Language

The positive-rating bigrams in Table 7 include:

- easy use
- good app
- nice app
- best app
- money transfer

These expressions suggest that ease of use and the general payment experience are common themes in favourable feedback.

The negative-rating bigrams include:

- customer care
- customer service
- not working
- not able
- bank account
- doesn't work

These expressions suggest that customer support and functional or account-related difficulties are prominent in critical reviews.

These are descriptive patterns in the collected reviews. They do not establish the actual technical failure rate of an application or the prevalence of these experiences among all users.

---

## A4. Aspect-Based Sentiment, Trust and Risk

### A4.1 Aspect-Based Sentiment Methodology

Aspect-based sentiment analysis was used to examine sentiment toward particular application features rather than assigning one overall sentiment label to an entire review.

Reviews were divided into clauses using punctuation and contrast terms such as "but" and "however".

Seven aspects were identified using keyword lexicons, and VADER sentiment was calculated for clauses matched to each aspect.

The seven aspects are:

1. Convenience / ease of use
2. Security / fraud
3. Rewards / offers
4. Customer support
5. Account / KYC / blocking
6. Reliability / transactions
7. Privacy / data

A review may mention more than one aspect. Therefore, aspect-level percentages are not mutually exclusive and should not be expected to sum to 100%.

### A4.2 Aspect Sentiment Results

| Aspect | Clauses | Share of Reviews (%) | Mean VADER | Positive (%) | Negative (%) | Mean Star Rating |
|---|---:|---:|---:|---:|---:|---:|
| Convenience / ease of use | 11,942 | 13.4 | 0.40 | 74.7 | 4.2 | 4.61 |
| Security / fraud | 4,656 | 5.1 | 0.09 | 54.0 | 38.6 | 2.99 |
| Rewards / offers | 4,160 | 4.3 | 0.07 | 36.8 | 29.7 | 2.69 |
| Customer support | 7,325 | 6.9 | -0.00 | 39.3 | 41.4 | 1.76 |
| Account / KYC / blocking | 7,787 | 7.9 | -0.03 | 26.4 | 33.3 | 2.30 |
| Reliability / transactions | 8,755 | 9.4 | -0.03 | 24.8 | 33.5 | 2.33 |
| Privacy / data | 2,007 | 2.1 | -0.04 | 24.0 | 29.1 | 2.05 |

Convenience/ease of use has the strongest positive mean VADER score (0.40), with 74.7% positive clauses and a mean matching-review rating of 4.61.

This indicates that ease of use is a prominent positive aspect in the analysed reviews.

Customer support has a mean VADER score close to zero, while 41.4% of its matched clauses are negative. Its matching reviews have a mean star rating of 1.76, the lowest among the seven aspects.

Account/KYC/blocking and reliability/transactions also have negative mean VADER scores and relatively low matching-review ratings.

Security/fraud has a slightly positive mean VADER score (0.09), but 38.6% of its matched clauses are negative. This mixed result shows why the mean score should be interpreted alongside the positive and negative proportions.

Privacy/data has the lowest mean VADER score (-0.04) and a mean matching-review rating of 2.05.

The aspect results depend on reconstructed keyword lexicons. A keyword match does not guarantee that a clause genuinely discusses the intended aspect. Manual precision checks on sampled clauses would strengthen the validity of these results.

### A4.3 Trust and Risk Keyword Analysis

A separate keyword-based analysis identified reviews containing terms associated with trust and endorsement, risk and grievance, fraud/security, and functional problems.

| Category | All-App Share (%) | Mean Star Rating | Mean VADER |
|---|---:|---:|---:|
| Trust and endorsement | 12.5 | 4.49 | 0.61 |
| Risk and grievance | 6.5 | 1.35 | -0.42 |
| Fraud-security | 2.2 | 1.52 | -0.38 |
| Functional-problem | 9.7 | 2.18 | -0.10 |

Trust and endorsement terms occur in 12.5% of reviews. Matching reviews have a high mean rating of 4.49 and a positive mean VADER score of 0.61.

Risk and grievance terms occur in 6.5% of reviews and are associated with a low mean rating of 1.35 and a negative mean VADER score of -0.42.

Fraud/security terms occur in 2.2% of reviews, with a mean rating of 1.52.

Functional-problem terms occur in 9.7% of reviews, with a mean rating of 2.18.

In this keyword-based analysis, functional-problem mentions are more frequent than fraud/security mentions. This suggests that complaints about application functionality are more common in the collected reviews than explicit fraud/security terminology.

These categories may overlap because a review can contain terms from more than one lexicon. Their percentages should therefore not be summed to obtain a total share of reviews.

### A4.4 Limitations of Keyword-Based Trust and Risk Detection

Keyword matching is an approximate method for identifying trust and risk concerns.

For example, "OTP" may appear in an ordinary login or authentication complaint and does not necessarily indicate fraud.

Similarly, a keyword can appear in a negated, hypothetical, or unrelated context.

The results should therefore be interpreted as keyword-flagged review prevalence, not verified incidents of fraud, security breaches, or payment failures.

---

## A5. Temporal Analysis

### A5.1 Methodology

Temporal analysis aggregates reviews by application and calendar month.

The monthly measures include:

- Number of reviews
- Mean star rating
- Share of low ratings (one or two stars)
- Number of deduplicated reviews used for risk analysis
- Share of deduplicated reviews flagged by the risk lexicon

Months with fewer than 500 reviews are excluded by the analysis rule.

The monthly output contains 31 application-month observations:

- Google Pay: 25 observations
- Paytm: 5 observations
- PhonePe: 1 observation

The applications have different observation windows, so comparisons across their monthly series must be made cautiously.

### A5.2 Google Pay

Google Pay has the longest observed time series, from October 2019 to October 2021.

Its mean monthly rating was 4.14 in October 2019. The lowest observed monthly mean rating was 2.60 in July 2021, when 56.15% of reviews were one or two stars.

The risk-term share in the deduplicated monthly corpus was 2.82% in October 2019 and reached a maximum of 6.24% in October 2020. It was 5.17% in October 2021.

The monthly series therefore shows a marked deterioration in Google Pay's observed ratings during parts of the period, alongside an increase in the share of reviews flagged by the risk lexicon compared with the beginning of the series.

These patterns are descriptive and do not establish the cause of the changes.

### A5.3 Paytm

The Paytm series contains five monthly observations, from July to November 2021.

Its mean rating was 4.41 in July 2021 and 4.27 in November 2021.

The lowest mean rating in the five-month series was 4.18 in October 2021, when the low-rating share reached 16.86%.

The risk-term share was highest in October 2021 at 9.49%, compared with 7.35% in November 2021.

Because the available Paytm time series covers only five months, it is not sufficient to infer a long-term trend.

### A5.4 PhonePe

The PhonePe dataset contributes one monthly observation, November 2021, with:

- 11,741 reviews
- Mean rating: 4.11
- Low-rating share: 17.39%
- Risk-term share: 5.75%

The source data covers only a short period in November 2021.

This observation is a partial-month snapshot and cannot be interpreted as a temporal trend or directly compared with complete months from other applications.

### A5.5 Overall Interpretation

The temporal analysis reveals differences in the duration and volume of available review data across applications.

Google Pay provides the longest time series and shows a substantial decline in observed monthly ratings during the later part of its coverage.

Paytm's short series shows comparatively high ratings, with a temporary deterioration in October 2021.

PhonePe provides only one partial-month observation.

These findings should be interpreted in light of uneven data coverage, review-volume differences, and the keyword-based nature of risk detection.

Monthly changes in reviews may reflect changes in the composition of people posting reviews as well as changes in user experience.

---

## Limitations and Research Considerations

The following limitations should be considered when interpreting the findings:

1. **Language coverage:** The ASCII-based language filter can exclude Hindi, Hinglish, and other non-English feedback. The findings therefore do not represent the full linguistic diversity of UPI users.

2. **Keyword lexicons:** Aspect, trust, and risk categories are based on reconstructed keyword lists. Keyword matching can miss relevant expressions and can also flag irrelevant contexts.

3. **Weak sentiment labels:** Star ratings are not ground-truth sentiment labels. A rating may not reflect the sentiment of every statement in a review.

4. **Lexicon sentiment limitations:** VADER and TextBlob may struggle with sarcasm, domain-specific language, mixed sentiment, and context-dependent expressions.

5. **Duplicate handling:** Exact repeated texts are removed within each application for the final text corpus. This reduces repetition but also removes frequency information about identical short reviews. Rating-profile statistics and deduplicated text-analysis statistics therefore use different processing stages.

6. **Uneven temporal coverage:** Google Pay, Paytm, and PhonePe have different observation windows. In particular, the PhonePe data provides only one partial-month observation.

7. **Keyword prevalence is not incident prevalence:** A review flagged for fraud/security or risk terminology is not proof that a fraud event or security breach occurred.

8. **Review-platform bias:** App-store reviews are self-selected feedback and may overrepresent users with unusually positive or negative experiences. They should not be treated as a representative survey of all UPI users.

9. **No causal inference:** The observed associations and temporal patterns do not establish that a particular application feature or event caused changes in ratings or sentiment.

---

## Reproducibility and Output References

The analyses in this report are supported by the following generated outputs:

| Report Evidence | Generated File |
|---|---|
| Application rating profile | `outputs/tables/table5_rating_profile.csv` |
| Positive and negative bigrams | `outputs/tables/table7_bigrams.csv` |
| VADER sentiment by app | `outputs/tables/fig1b_vader_by_app.csv` |
| Aspect sentiment | `outputs/tables/table9_aspects.csv` |
| Trust and risk analysis | `outputs/tables/table10_trust_risk.csv` |
| Monthly trends | `outputs/tables/fig3_monthly.csv` |
| Ratings and sentiment figure | `outputs/figures/fig1_ratings_sentiment.png` |
| Aspect sentiment figure | `outputs/figures/fig2_aspects.png` |
| Temporal trends figure | `outputs/figures/fig3_trends.png` |

The Member 1 test suite was executed using:

`pytest -q tests`

The reported result was:

**10 passed in 2.23 seconds.**

### Reporting Note

The figures and tables above reflect the supplied outputs. The chi-square statistic, complete TextBlob evaluation metrics, and any classifier accuracy or macro-F1 values have not been inserted because those numerical outputs have not yet been verified for this report.

They should only be added after checking the corresponding generated metrics.
