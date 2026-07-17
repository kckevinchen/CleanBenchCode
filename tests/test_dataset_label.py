"""Regression tests for ``dataset_label_from_config``.

The runs-list and run-detail surfaces both label a run by its source
corpus via this helper. Nested corpora (each table in its own sub-dir,
e.g. ``curated_pbi_500_dedup/<table>/<table>.csv``) used to mislabel the
run with the *first table's* folder name instead of the corpus.
"""

from src.labeling.runs import dataset_label_from_config


def test_nested_corpus_labels_by_corpus_not_table():
    # curated_pbi_500_dedup / curated_bing21_500_dedup keep each table in
    # its own sub-dir, so paths[0].parent is the table, not the corpus.
    base = "/data/curated_pbi_500_dedup"
    cfg = {
        "csv_paths": [
            f"{base}/103115795_Sales/103115795_Sales.csv",
            f"{base}/110888756_party_wise_sentiment/110888756_party_wise_sentiment.csv",
            f"{base}/990127103_Predictions/990127103_Predictions.csv",
        ]
    }
    assert dataset_label_from_config(cfg) == "curated_pbi_500_dedup"


def test_flat_corpus_labels_by_dir():
    cfg = {
        "csv_paths": [
            "/data/curated_bing21_200_flat/00cb494f.csv",
            "/data/curated_bing21_200_flat/00f1d635.csv",
        ]
    }
    assert dataset_label_from_config(cfg) == "curated_bing21_200_flat"


def test_single_file_labels_by_filename():
    cfg = {"csv_paths": ["/data/whatever/smoke.csv"]}
    assert dataset_label_from_config(cfg) == "smoke.csv"


def test_explicit_directory_mode_wins():
    cfg = {
        "csv_path": "/data/curated_bing21_grouped",
        "csv_paths": ["/data/curated_bing21_grouped/a/a.csv"],
    }
    assert dataset_label_from_config(cfg) == "curated_bing21_grouped"


def test_no_paths_returns_none():
    assert dataset_label_from_config({}) is None
    assert dataset_label_from_config({"csv_paths": []}) is None
