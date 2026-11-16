"""Reconstructed English keyword lists. These are approximations, not the unpublished paper lexicon."""
import re

_END = r"(?:s|es|ed|d|ing)?"
def make_re(terms):
    parts = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    return re.compile(r"\b(?:" + "|".join(parts) + r")" + _END + r"\b", re.I)

ASPECTS = {
    "Convenience / ease of use": ["easy", "simple", "convenient", "smooth", "interface", "fast", "quick", "seamless", "user friendly", "userfriendly", "easier", "hassle free"],
    "Reliability / transactions": ["fail", "failure", "pending", "stuck", "error", "deducted", "glitch", "server", "crash", "slow", "loading", "hang", "bug", "not working", "not able", "doesn't work", "transaction", "timeout"],
    "Security / fraud": ["secure", "safe", "safety", "unsafe", "fraud", "scam", "hack", "phishing", "cheat", "fake", "stolen", "spam", "otp", "security"],
    "Privacy / data": ["privacy", "permission", "personal data", "data", "tracking", "contacts", "kyc"],
    "Customer support": ["support", "customer care", "customer service", "helpline", "complaint", "ticket", "resolve", "refund", "reply", "response"],
    "Rewards / offers": ["cashback", "reward", "offer", "scratch", "coupon", "points", "bonus", "referral"],
    "Account / KYC / blocking": ["blocked", "block", "account", "verification", "suspended", "ban", "banned", "login", "wallet"],
}
ASPECT_RE = {name: make_re(words) for name, words in ASPECTS.items()}
TRUST = ["trust", "reliable", "safe", "secure", "genuine", "legit", "authentic", "recommend", "best", "excellent", "satisfied", "love"]
RISK = ["fraud", "scam", "cheat", "fake", "hack", "phishing", "stolen", "unsafe", "deducted", "refund", "blocked", "worst", "useless", "harassment"]
FRAUD = ["fraud", "scam", "hack", "phishing", "cheat", "fake", "stolen", "unsafe", "spam", "otp"]
FUNCTIONAL = ["fail", "pending", "stuck", "error", "not working", "deducted", "glitch", "slow", "loading", "hang", "crash", "bug", "server", "timeout", "problem", "issue"]
TRUST_RE, RISK_RE = make_re(TRUST), make_re(RISK)
FRAUD_RE, FUNC_RE = make_re(FRAUD), make_re(FUNCTIONAL)
