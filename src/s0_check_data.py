"""Inspect raw UPI files and verify configured column names/date ranges."""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

def run():
    ok = True
    required = [C.UPI_COLS[k] for k in ("id", "text", "score", "time")]
    for app, path in C.UPI_FILES.items():
        if not path.exists():
            print(f"MISSING FILE: {path}")
            ok = False
            continue
        df = pd.read_csv(path)
        missing = [col for col in required if col not in df.columns]
        print(f"\n{app}: {len(df):,} rows")
        print("Columns:", list(df.columns))
        if missing:
            print("MISSING COLUMNS:", missing)
            ok = False
        elif C.UPI_COLS["time"] in df:
            dates = pd.to_datetime(df[C.UPI_COLS["time"]], errors="coerce")
            print(f"Date range: {dates.min()} -> {dates.max()}")
    print("\nALL UPI COLUMNS FOUND" if ok else "\nFIX config.UPI_COLS / missing files first")
    return ok

if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
