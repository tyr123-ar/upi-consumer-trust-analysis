import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluate
import figures_b
import prep_upi
import sent_roberta
import themes_upi
import topics
import config
import person_b_store
import run_all
import run_person_b
from run_all import select_stages


def test_classification_metrics_fixed_labels_and_single_class():
    result = evaluate.classification_metrics(["positive"], ["positive"])
    assert result["accuracy"] == 1
    assert result["per_class"]["negative"]["support"] == 0
    assert result["confusion_matrix"][2][2] == 1


@pytest.mark.parametrize(
    "truth,predicted,error",
    [
        ([], [], "empty"),
        (["positive"], [], "equal lengths"),
        (["positive", None], ["positive", "negative"], "missing"),
        (["invalid"], ["negative"], "Unexpected"),
    ],
)
def test_classification_metrics_rejects_invalid_inputs(truth, predicted, error):
    with pytest.raises(ValueError, match=error):
        evaluate.classification_metrics(truth, predicted)


def test_mcnemar_test_zero_and_unequal_pairs():
    same = evaluate.mcnemar_test(["negative"], ["negative"], ["negative"])
    assert same["statistic"] == 0
    assert same["p_value"] == 1
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate.mcnemar_test(["negative"], ["negative"], [])


@pytest.mark.parametrize("successes,n", [(0, 10), (10, 10)])
def test_wilson_interval_valid_extremes(successes, n):
    lower, upper = evaluate.wilson_interval(successes, n)
    assert 0 <= lower <= upper <= 1
    if successes == 0:
        assert lower == 0
    else:
        assert upper == 1


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError):
        evaluate.wilson_interval(2, 1)
    with pytest.raises(ValueError):
        evaluate.wilson_interval(0, 0)


def test_sampling_validation_rejects_invalid_stars():
    with pytest.raises(ValueError, match="invalid ratings"):
        evaluate.validate_sampling(pd.DataFrame({"score": [0]}))


def test_sentiment_distribution_rejects_empty_and_counts_zero_classes():
    result = evaluate.sentiment_distribution(["positive", "negative"])
    assert result["counts"] == {"negative": 1, "neutral": 0, "positive": 1}
    with pytest.raises(ValueError, match="empty"):
        evaluate.sentiment_distribution([])


def test_sampling_validation_returns_all_rating_strata():
    result = evaluate.validate_sampling(
        pd.DataFrame({"score": [1, 1, 3, 5]}), target_per_rating=2
    )
    assert result == {1: 2, 2: 0, 3: 1, 4: 0, 5: 1}
    with pytest.raises(ValueError, match="exceeds"):
        evaluate.validate_sampling(
            pd.DataFrame({"score": [1, 1, 1]}), target_per_rating=2
        )


def test_topic_distribution_checks_and_includes_empty_topics():
    result = evaluate.topic_distribution([0, 0, 2], num_topics=4)
    assert result[1] == {"count": 0, "proportion": 0}
    with pytest.raises(ValueError):
        evaluate.topic_distribution([0, 4], num_topics=4)


def test_topic_tokenization_removes_short_tokens_and_stopwords():
    assert topics.tokenize("This app is not good; UPI works") == ["not", "good", "works"]


def test_upi_theme_matching_and_ambiguous_phrases():
    assert "Payment delays" in themes_upi.match_themes("my UPI transfer is pending")
    assert "Refund issues" in themes_upi.match_themes("I still need a refund")
    assert "Payment delays" not in themes_upi.match_themes("my bank balance is pending")
    assert "Fraud and suspicious transactions" not in themes_upi.match_themes(
        "this app is not fraudulent"
    )


def test_roberta_label_mapping_handles_cardiff_labels():
    class ModelConfig:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}

    assert sent_roberta._label_mapping(ModelConfig()) == {
        0: "negative",
        1: "neutral",
        2: "positive",
    }


def test_pipeline_stage_selection():
    assert select_stages(start="topics_upi") == [
        "topics_upi",
        "themes_upi",
        "figures",
        "validation",
    ]
    assert select_stages(only="figures") == ["figures"]


def test_run_all_fails_closed_before_person_a_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all.C, "OUT", tmp_path)

    with pytest.raises(SystemExit) as error:
        run_all.main([])

    assert error.value.code == 2
    assert not (tmp_path / "run_manifest.json").exists()
    with pytest.raises(PermissionError, match="Person A data or outputs"):
        run_all._run_stage("preprocess")


def test_person_b_runner_excludes_person_a_writers():
    assert "preprocess" not in run_person_b.STAGES
    assert "member1_analysis" not in run_person_b.STAGES
    assert "figures" not in run_person_b.STAGES
    assert "figures_person_b" in run_person_b.STAGES


def test_person_b_metrics_write_and_update_separate_file(tmp_path, monkeypatch):
    metrics_path = tmp_path / "person_b_metrics.json"
    shared_metrics_path = tmp_path / "metrics.json"
    shared_metrics_path.write_text('{"owner": "Person A"}', encoding="utf-8")
    monkeypatch.setattr(config, "METRICS", shared_metrics_path)
    monkeypatch.setattr(person_b_store, "METRICS", metrics_path)

    person_b_store.put("upi_topic_model", {"k": 6})
    person_b_store.put("qris_sentiment", {"comments": 177})

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "upi_topic_model": {"k": 6},
        "qris_sentiment": {"comments": 177},
    }
    assert json.loads(shared_metrics_path.read_text(encoding="utf-8")) == {
        "owner": "Person A"
    }


def test_upi_comparison_table_does_not_claim_qris_is_unavailable(tmp_path, monkeypatch):
    interim_path = tmp_path / "interim"
    table_path = tmp_path / "tables"
    interim_path.mkdir()
    table_path.mkdir()
    pd.DataFrame(
        {"app": ["Google Pay"], "score": [4]}
    ).to_csv(interim_path / "upi_all.csv", index=False)
    monkeypatch.setattr(config, "INTER", interim_path)
    monkeypatch.setattr(config, "TABLES", table_path)

    result = figures_b.comparison_table()

    assert "QRIS_value" not in result.columns
    assert not result["metric"].str.startswith("QRIS").any()


def test_load_one_maps_configured_raw_columns(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {
            "reviewId": ["r1"],
            "content": ["great"],
            "score": [5],
            "at": ["2024-01-01"],
        }
    ).to_csv(path, index=False)
    frame = prep_upi.load_one("Google Pay", path)
    assert frame.loc[0, "app"] == "Google Pay"
    assert frame.loc[0, "score"] == 5
    assert pd.isna(frame.loc[0, "thumbs"]) is False


def test_benchmark_uses_one_sample_and_writes_aggregate_outputs(tmp_path, monkeypatch):
    clean_path = tmp_path / "clean.csv"
    roberta_path = tmp_path / "roberta.csv"
    rows = [
        {"app": "Google Pay", "id": f"r{rating}", "score": rating, "text_light": "payment transaction works"}
        for rating in range(1, 6)
    ]
    pd.DataFrame(rows).to_csv(clean_path, index=False)
    pd.DataFrame(
        [
            {
                "app": row["app"],
                "id": row["id"],
                "predicted_label": "neutral",
                "model_revision": config.ROBERTA_REVISION,
            }
            for row in rows
        ]
    ).to_csv(roberta_path, index=False)
    monkeypatch.setattr(sent_roberta.store, "put", lambda *args, **kwargs: None)
    comparison, mcnemar = sent_roberta.run_benchmark(
        clean_path=clean_path,
        roberta_path=roberta_path,
        output_dir=tmp_path,
        per_rating=1,
        min_words=3,
    )
    assert comparison.n.tolist() == [5, 5, 5]
    assert mcnemar["n"] == 5
    assert (tmp_path / "upi_benchmark_confusion_roberta.csv").exists()
    assert (tmp_path / "upi_mcnemar_roberta_vader.json").exists()


def test_sentiment_figure_writes_png_without_review_text(tmp_path, monkeypatch):
    predictions = tmp_path / "roberta.csv"
    rows = [
        {
            "app": app,
            "id": f"internal-{index}",
            "predicted_label": label,
            "confidence": 0.8,
            "model_revision": config.ROBERTA_REVISION,
        }
        for index, (app, label) in enumerate(
            [
                ("Google Pay", "positive"),
                ("Paytm", "neutral"),
                ("PhonePe", "negative"),
            ]
        )
    ]
    pd.DataFrame(rows).to_csv(predictions, index=False)
    interim = tmp_path / "interim"
    interim.mkdir()
    pd.DataFrame([{"app": row["app"], "id": row["id"]} for row in rows]).to_csv(
        interim / "upi_clean.csv", index=False
    )
    monkeypatch.setattr(config, "ROBERTA_OUTPUT", predictions)
    monkeypatch.setattr(config, "INTER", interim)
    monkeypatch.setattr(config, "FIGS", tmp_path)
    assert figures_b.sentiment_figure() == "RoBERTa"
    output = tmp_path / "fig4_upi_sentiment_distribution.png"
    assert output.exists() and output.stat().st_size > 0


def test_theme_summary_deduplicates_review_matches_and_writes_table(tmp_path, monkeypatch):
    input_path = tmp_path / "clean.csv"
    output_dir = tmp_path / "tables"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {"app": "Google Pay", "id": "r1", "text_light": "refund refund"},
            {"app": "Google Pay", "id": "r2", "text_light": "great app"},
        ]
    ).to_csv(input_path, index=False)
    monkeypatch.setattr(config, "TABLES", output_dir)
    monkeypatch.setattr(themes_upi.store, "put", lambda *args, **kwargs: None)
    result = themes_upi.run(input_path)
    assert result.loc[0, "reviews"] == 1
    assert result.loc[0, "share_pct"] == 50
    assert (output_dir / "table_upi_exploratory_themes.csv").exists()
