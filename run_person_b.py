"""Run Person B UPI stages without invoking Person A writers."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as C

STAGES = (
    "validate_data",
    "sentiment_roberta",
    "benchmark",
    "topics_upi",
    "themes_upi",
    "figures_person_b",
    "validation",
)


def select_stages(start=None, only=None):
    if only:
        if only not in STAGES:
            raise ValueError(f"Unknown Person B stage {only!r}")
        return [only]
    if start:
        if start not in STAGES:
            raise ValueError(f"Unknown Person B stage {start!r}")
        return list(STAGES[STAGES.index(start) :])
    return list(STAGES)


def _run_stage(name, benchmark_sample_only=False):
    if name == "validate_data":
        import validate_data
        validate_data.run()
    elif name == "sentiment_roberta":
        import sent_roberta
        if benchmark_sample_only:
            sent_roberta.score_benchmark_sample()
        else:
            sent_roberta.run_roberta()
    elif name == "benchmark":
        import sent_roberta
        sent_roberta.run_benchmark()
    elif name == "topics_upi":
        import topics
        topics.run()
    elif name == "themes_upi":
        import themes_upi
        themes_upi.run()
    elif name == "figures_person_b":
        import figures_b
        figures_b.run()
    elif name == "validation":
        import validate_data
        import validate_paper
        validate_data.run()
        validate_paper.run()
    else:
        raise ValueError(f"Unknown Person B stage {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=STAGES)
    parser.add_argument("--only", choices=STAGES)
    parser.add_argument(
        "--benchmark-sample-only",
        action="store_true",
        help="Score only the fixed 5,000-review weak-label benchmark sample",
    )
    args = parser.parse_args(argv)
    if args.start and args.only:
        parser.error("--from and --only cannot be used together")
    selected = select_stages(start=args.start, only=args.only)
    if args.benchmark_sample_only and "sentiment_roberta" not in selected:
        parser.error("--benchmark-sample-only requires the sentiment_roberta stage")

    manifest_path = C.OUT / "person_b_run_manifest.json"
    manifest = {"stages": [], "pipeline": "Person B UPI only"}
    for name in selected:
        stage_name = (
            "sentiment_roberta_benchmark_sample"
            if name == "sentiment_roberta" and args.benchmark_sample_only
            else name
        )
        start_time = time.perf_counter()
        status = "completed"
        try:
            print(f"\n=== Person B {stage_name} ===")
            _run_stage(
                name,
                benchmark_sample_only=(
                    name == "sentiment_roberta" and args.benchmark_sample_only
                ),
            )
        finally:
            if sys.exc_info()[0] is not None:
                status = "failed"
            manifest["stages"].append(
                {
                    "stage": stage_name,
                    "status": status,
                    "seconds": round(time.perf_counter() - start_time, 3),
                }
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if status == "failed":
            raise RuntimeError(f"Person B stage {name!r} failed; see the error above.")
    return manifest


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
