"""Writers for Person B metrics kept separate from the shared UPI metrics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

METRICS = C.METRICS.with_name("person_b_metrics.json")


def put(key, value):
    metrics = {}
    if METRICS.exists():
        try:
            metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid metrics file: {METRICS}") from error
    metrics[key] = value
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(
        json.dumps(metrics, indent=2, default=float),
        encoding="utf-8",
    )
