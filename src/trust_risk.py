"""UPI trust/risk keyword prevalence and sentiment (Table 10)."""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import lexicons as L
import store

GROUPS = {
    "Trust and endorsement": L.TRUST_RE,
    "Risk and grievance": L.RISK_RE,
    "Fraud-security": L.FRAUD_RE,
    "Functional-problem": L.FUNC_RE,
}

def flag(df):
    text = df.text_light.fillna("").astype(str)
    for name, regex in GROUPS.items():
        df["f_" + name] = text.map(lambda value: bool(regex.search(value)))
    return df

def run():
    path = C.INTER / "upi_scored.csv"
    if not path.exists():
        raise FileNotFoundError("Run prep_upi.py and sent_lexicon.py first")
    df = flag(pd.read_csv(path))
    df.to_csv(C.INTER / "upi_flagged.csv", index=False)
    rows = []
    for app in C.APPS + ["All apps"]:
        group = df if app == "All apps" else df[df.app.eq(app)]
        row = {"App": app}
        for name in GROUPS:
            row[name + " %"] = round(100 * group["f_" + name].mean(), 1) if len(group) else 0
        rows.append(row)
    mean_stars = {"App": "Mean star of matching reviews"}
    mean_vader = {"App": "Mean VADER of matching reviews"}
    for name in GROUPS:
        matched = df[df["f_" + name]]
        mean_stars[name + " %"] = round(matched.score.mean(), 2) if len(matched) else None
        mean_vader[name + " %"] = round(matched.vader.mean(), 2) if len(matched) else None
    table = pd.DataFrame(rows + [mean_stars, mean_vader])
    store.table(table, "table10_trust_risk")
    store.put("trust_risk_all", rows[-1])
    store.put("trust_risk_mean_star", mean_stars)
    fraud = rows[-1]["Fraud-security %"]
    functional = rows[-1]["Functional-problem %"]
    store.put("func_to_fraud_ratio", functional / fraud if fraud else None)
    print(table.to_string(index=False))
    return table

if __name__ == "__main__":
    run()
