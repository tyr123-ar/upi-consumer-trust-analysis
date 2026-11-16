"""Small shared writers for CSV tables and metrics.json."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

def put(key, value):
    metrics = {}
    if C.METRICS.exists():
        try:
            metrics = json.loads(C.METRICS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError(f"Invalid metrics file: {C.METRICS}")
    metrics[key] = value
    C.METRICS.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

def get(key, default=None):
    if not C.METRICS.exists():
        return default
    return json.loads(C.METRICS.read_text(encoding="utf-8")).get(key, default)

def table(df, name, index=False):
    path = C.TABLES / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"Saved {path} ({len(df)} rows)")
    return path
