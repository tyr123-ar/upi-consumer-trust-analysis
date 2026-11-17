"""Run QRIS-only processing; this entry point never runs the UPI pipeline."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src import (
    compare_qris,
    figures_qris,
    prep_qris,
    profile_qris,
    qris_sentiment_summary,
    qris_themes,
    sent_roberta_qris,
    topics_qris,
)

MANIFEST = ROOT / "outputs" / "qris_run_manifest.json"


def run():
    stages = (
        ("preprocess", prep_qris.run),
        ("profile", profile_qris.run),
        ("indonesian_roberta", sent_roberta_qris.run),
        ("sentiment_summary", qris_sentiment_summary.run),
        ("themes_and_lexicons", qris_themes.run),
        ("topics", topics_qris.run),
        ("figure_4", figures_qris.run),
        ("table_13_qris", compare_qris.run),
    )
    manifest = {"dataset": "10.17632/xpkw64fd2h.1", "stages": []}
    for name, function in stages:
        start = time.perf_counter()
        try:
            print(f"\n=== QRIS {name} ===")
            function()
        except Exception:
            manifest["stages"].append(
                {
                    "stage": name,
                    "status": "failed",
                    "seconds": round(time.perf_counter() - start, 3),
                }
            )
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise
        manifest["stages"].append(
            {
                "stage": name,
                "status": "completed",
                "seconds": round(time.perf_counter() - start, 3),
            }
        )
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
