# UPI data validation report

Generated from files available in this repository. No review text, IDs, or user profile data are included.

## Locally available UPI source CSVs

### Google Pay

- Source CSV: `data\GooglePayIndia.csv`
- Rows: 34,228 (plan expected 34,228)
- Columns: Unnamed: 0, reviewId, userName, userImage, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, repliedAt
- Missing configured columns: none
- Missing values by column: `{'Unnamed: 0': 0, 'reviewId': 0, 'userName': 1, 'userImage': 0, 'content': 9, 'score': 0, 'thumbsUpCount': 0, 'reviewCreatedVersion': 4241, 'at': 0, 'replyContent': 28116, 'repliedAt': 28116}`
- Duplicate review IDs: 0
- Invalid ratings outside 1-5: 0
- Invalid/unparseable timestamps: 0
- Date range: 2019-10-01 02:06:57 to 2021-11-18 17:03:06
- Rating distribution: `{1: 8618, 2: 1593, 3: 1928, 4: 3043, 5: 19046}`
### Paytm

- Source CSV: `data\PaytmIndia.csv`
- Rows: 137,907 (plan expected 137,907)
- Columns: Unnamed: 0, reviewId, userName, userImage, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, repliedAt
- Missing configured columns: none
- Missing values by column: `{'Unnamed: 0': 0, 'reviewId': 0, 'userName': 0, 'userImage': 0, 'content': 3, 'score': 0, 'thumbsUpCount': 0, 'reviewCreatedVersion': 23665, 'at': 0, 'replyContent': 69780, 'repliedAt': 69780}`
- Duplicate review IDs: 0
- Invalid ratings outside 1-5: 0
- Invalid/unparseable timestamps: 0
- Date range: 2021-07-29 21:27:00 to 2021-11-18 18:42:29
- Rating distribution: `{1: 17499, 2: 2627, 3: 4554, 4: 13640, 5: 99587}`
### PhonePe

- Source CSV: `data\PhonePayIndia.csv`
- Rows: 11,741 (plan expected 11,741)
- Columns: Unnamed: 0, reviewId, userName, userImage, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, repliedAt
- Missing configured columns: none
- Missing values by column: `{'Unnamed: 0': 0, 'reviewId': 0, 'userName': 0, 'userImage': 0, 'content': 0, 'score': 0, 'thumbsUpCount': 0, 'reviewCreatedVersion': 2136, 'at': 0, 'replyContent': 8805, 'repliedAt': 8805}`
- Duplicate review IDs: 0
- Invalid ratings outside 1-5: 0
- Invalid/unparseable timestamps: 0
- Date range: 2021-11-10 15:22:05 to 2021-11-18 18:37:42
- Rating distribution: `{1: 1765, 2: 277, 3: 517, 4: 1483, 5: 7699}`

## Available processed corpus

- File: `data/interim/upi_clean.csv` (83,140 rows)
- This pre-existing cleaned intermediate was not regenerated from the supplied raw CSVs during this audit.
- Standardized columns: `['app', 'id', 'raw', 'score', 'thumbs', 'time', 'has_reply', 'text', 'text_light', 'n_words']`
- Column data types: `{'app': 'str', 'id': 'string', 'raw': 'str', 'score': 'int64', 'thumbs': 'int64', 'time': 'str', 'has_reply': 'bool', 'text': 'str', 'text_light': 'str', 'n_words': 'int64'}`
- Per-app rows: `{'Google Pay': 26039, 'Paytm': 51967, 'PhonePe': 5134}`
- Duplicate app/review ID pairs: 0
- Invalid ratings outside 1-5: 0
- Invalid/unparseable timestamps: 0
- Missing values by column: `{'app': 0, 'id': 0, 'raw': 0, 'score': 0, 'thumbs': 0, 'time': 0, 'has_reply': 0, 'text': 0, 'text_light': 1, 'n_words': 0}`
- Date range: 2019-10-01 02:09:34 to 2021-11-18 18:42:29
- Rating distribution: `{1: 23570, 2: 3415, 3: 4046, 4: 7253, 5: 44856}`

## Schema note

The supplied files match the configured required schema and planned row counts. Kaggle provenance was not independently verified.
