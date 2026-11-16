"""Monthly UPI trends: rating, low-star share and risk-term share."""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import store

def run():
    all_path, flagged_path = C.INTER / "upi_all.csv", C.INTER / "upi_flagged.csv"
    if not all_path.exists() or not flagged_path.exists():
        raise FileNotFoundError("Run prep_upi.py and trust_risk.py first")
    all_reviews = pd.read_csv(all_path, parse_dates=["time"])
    flagged = pd.read_csv(flagged_path, parse_dates=["time"])
    all_reviews = all_reviews[all_reviews.score.notna()].copy()
    all_reviews["month"] = all_reviews.time.dt.to_period("M")
    flagged["month"] = flagged.time.dt.to_period("M")
    rating = all_reviews.groupby(["app", "month"]).agg(
        n=("score", "size"),
        mean_rating=("score", "mean"),
        low_share=("score", lambda values: 100 * values.isin([1, 2]).mean()),
    )
    risk = flagged.groupby(["app", "month"]).agg(
        n_dedup=("score", "size"),
        risk_share=("f_Risk and grievance", lambda values: 100 * values.mean()),
    )
    monthly = rating.join(risk, how="left").reset_index()
    monthly["month"] = monthly.month.astype(str)
    monthly = monthly[monthly.n.ge(C.MIN_MONTH_REVIEWS)].round(2)
    store.table(monthly, "fig3_monthly")
    gpay = monthly[monthly.app.eq("Google Pay")].sort_values("month")
    if not gpay.empty:
        first = gpay.iloc[0]
        lowest = gpay.loc[gpay.mean_rating.idxmin()]
        store.put("gpay_first_month", first[["month", "mean_rating", "low_share", "risk_share"]].to_dict())
        store.put("gpay_lowest_month", lowest[["month", "mean_rating", "low_share", "risk_share"]].to_dict())
    print(monthly.to_string(index=False))
    return monthly

if __name__ == "__main__":
    run()
