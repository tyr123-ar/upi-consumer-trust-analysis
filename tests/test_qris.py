import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as C
import download_qris
import prep_qris
import qris_themes
import sent_roberta_qris


def test_qris_columns_match_downloaded_source():
    import pandas as pd

    if not C.QRIS_FILE.exists():
        pytest.skip("The ignored QRIS source CSV has not been downloaded")
    source = pd.read_csv(C.QRIS_FILE, nrows=0)
    assert C.QRIS_COLS["text"] in source.columns
    assert C.QRIS_COLS["time"] in source.columns
    assert C.QRIS_COLS["likes"] in source.columns


def test_normalize_maps_common_indonesian_negation_and_removes_mentions():
    assert prep_qris.normalize("NGGAKKK @akun mudah!") == "tidak mudah"


def test_qris_fraud_theme_ignores_fear_of_false_accusation():
    fraud = qris_themes.THEME_PATTERNS["Fraud or edited payment evidence"]
    text = qris_themes.SUSPICION.sub(" ", "takut dikira penipuan")
    assert not fraud.search(text)
    assert fraud.search("stiker QRIS palsu")


def test_model_label_mapping_uses_indonesian_roberta_config():
    class ModelConfig:
        id2label = {0: "positive", 1: "neutral", 2: "negative"}

    assert sent_roberta_qris._id_to_label(ModelConfig()) == {
        0: "positive",
        1: "neutral",
        2: "negative",
    }


def test_downloader_refuses_to_overwrite_an_existing_file(tmp_path):
    destination = tmp_path / "qris_tiktok.csv"
    destination.write_text("keep this file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        download_qris.run(destination)
    assert destination.read_text(encoding="utf-8") == "keep this file"
