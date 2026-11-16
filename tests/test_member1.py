import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import lexicons as L
import prep_upi as P
import aspects as A
import sent_lexicon as S

def test_clean_removes_url_punctuation_and_keeps_apostrophe():
    assert P.clean("Check https://x.com NOW!!! It doesn't work...") == "check now it doesn't work"

def test_light_preserves_punctuation():
    assert P.light("Great!!! https://x.com") == "great!!!"

def test_ascii_share():
    assert P.ascii_share("good app") == 1.0
    assert P.ascii_share(P.clean("बहुत अच्छा ऐप")) < 0.9

def test_empty_ascii_share():
    assert P.ascii_share("") == 0.0

def test_clean_handles_missing():
    assert P.clean(None) == ""

def test_ban_does_not_match_bank():
    regex = L.ASPECT_RE["Account / KYC / blocking"]
    assert not regex.search("my bank is slow")
    assert regex.search("they banned my account")

def test_multiword_regex_and_negation_phrase():
    assert L.FUNC_RE.search("app is not   working today")
    assert L.FUNC_RE.search("payment failed")

def test_clause_split():
    assert A.clauses("easy to use but support is bad. refund pending") == [
        "easy to use", "support is bad", "refund pending"
    ]

def test_sentiment_thresholds():
    assert S.classify(0.05) == "positive"
    assert S.classify(-0.05) == "negative"
    assert S.classify(0.0) == "neutral"

def test_star_labels():
    assert S.star_label(5) == "positive"
    assert S.star_label(3) == "neutral"
    assert S.star_label(1) == "negative"
