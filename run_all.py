"""Run available UPI analysis stages in dependency order."""
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
    "preprocess",
    "member1_analysis",
    "sentiment_roberta",
    "benchmark",
    "topics_upi",
    "themes_upi",
    "figures",
    "validation",
)
PERSON_A_WRITE_STAGES = frozenset({"preprocess", "member1_analysis", "figures"})


def select_stages(start=None, only=None):
    if only:
        if only not in STAGES:
            raise ValueError(f"Unknown pipeline stage {only!r}")
        return [only]
    if start:
        if start not in STAGES:
            raise ValueError(f"Unknown pipeline stage {start!r}")
        return list(STAGES[STAGES.index(start) :])
    return list(STAGES)


def _run_stage(name, allow_person_a_overwrite=False):
    if name in PERSON_A_WRITE_STAGES and not allow_person_a_overwrite:
        raise PermissionError(
            f"Stage {name!r} writes Person A data or outputs; pass "
            "--allow-person-a-overwrite only for an intentional combined rerun."
        )
    if name == "validate_data":
        import validate_data
        validate_data.run()
    elif name == "preprocess":
        if all(path.exists() for path in C.UPI_FILES.values()):
            import prep_upi
            prep_upi.run()
        elif (C.INTER / "upi_clean.csv").exists():
            print(
                "SKIPPED raw preprocessing: Kaggle source files are unavailable; "
                "using the existing data/interim/upi_clean.csv."
            )
        else:
            missing = [str(path) for path in C.UPI_FILES.values() if not path.exists()]
            raise FileNotFoundError(
                "Raw UPI CSVs and the cleaned intermediate are both unavailable:\n"
                + "\n".join(missing)
            )
    elif name == "member1_analysis":
        import eda_upi
        import sent_lexicon
        import trust_risk
        import aspects
        import temporal
        eda_upi.run()
        sent_lexicon.run()
        trust_risk.run()
        aspects.run()
        temporal.run()
    elif name == "sentiment_roberta":
        import sent_roberta
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
    elif name == "figures":
        import figures_upi
        import figures_b
        figures_upi.run()
        figures_b.run()
    elif name == "validation":
        import validate_data
        import validate_paper
        validate_data.run()
        validate_paper.run()
    else:
        raise ValueError(f"Unknown pipeline stage {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=STAGES)
    parser.add_argument("--only", choices=STAGES)
    parser.add_argument(
        "--allow-person-a-overwrite",
        action="store_true",
        help="Explicitly allow Person A preprocessing, analyses, and figures to rewrite data/outputs",
    )
    parser.add_argument(
        "--benchmark-sample-only",
        action="store_true",
        help="Score the fixed weak-label benchmark sample instead of the full corpus",
    )
    args = parser.parse_args(argv)
    if args.start and args.only:
        parser.error("--from and --only cannot be used together")
    selected = select_stages(args.start, args.only)
    protected_stages = sorted(set(selected) & PERSON_A_WRITE_STAGES)
    if protected_stages and not args.allow_person_a_overwrite:
        parser.error(
            "Selected stages write Person A data or outputs "
            f"({', '.join(protected_stages)}). Use `python run_person_b.py` for "
            "Person B-only execution, or pass --allow-person-a-overwrite to "
            "intentionally rerun Person A."
        )
    manifest_path = C.OUT / "run_manifest.json"
    manifest = {
        "stages": [],
        "qris": "Separate pipeline; see outputs/qris_run_manifest.json for QRIS status",
    }

    for name in selected:
        start = time.perf_counter()
        status = "completed"
        stage_name = (
            "sentiment_roberta_benchmark_sample"
            if name == "sentiment_roberta" and args.benchmark_sample_only
            else name
        )
        try:
            print(f"\n=== {stage_name} ===")
            if name == "sentiment_roberta" and args.benchmark_sample_only:
                import sent_roberta
                sent_roberta.score_benchmark_sample()
            else:
                _run_stage(
                    name,
                    allow_person_a_overwrite=args.allow_person_a_overwrite,
                )
        finally:
            if sys.exc_info()[0] is not None:
                status = "failed"
            manifest["stages"].append(
                {
                    "stage": stage_name,
                    "status": status,
                    "seconds": round(time.perf_counter() - start, 3),
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if status == "failed":
            raise RuntimeError(f"Pipeline stage {name!r} failed; see the error above.")

    print(f"\nPipeline stages completed: {', '.join(selected)}")
    print(f"QRIS: {manifest['qris']}")
    return manifest


if __name__ == "__main__":
    main()
