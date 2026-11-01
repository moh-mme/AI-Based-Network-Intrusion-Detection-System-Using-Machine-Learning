import numpy as np
import pandas as pd
import pytest

from nids.data import CATEGORY_COL, LABEL_COL, cap_per_class, clean_frame, display_label, map_categories, normalize_label

CATEGORY_MAP = {"benign": "Benign", "web attack xss": "WebAttack", "dos hulk": "DoS"}


@pytest.mark.parametrize("raw", ["Web Attack \x96 XSS", "Web Attack – XSS", "web attack xss", " WEB ATTACK - XSS "])
def test_normalize_label_ignores_case_and_punctuation(raw):
    assert normalize_label(raw) == "web attack xss"


def test_display_label_repairs_misdecoded_dash():
    assert display_label("Web Attack \x96 XSS") == "Web Attack - XSS"
    assert display_label("Web Attack \xe2\x80\x93 XSS") == "Web Attack - XSS"  # UTF-8 read as latin-1


def test_map_categories_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Mystery"):
        map_categories(pd.Series(["BENIGN", "Mystery"]), CATEGORY_MAP)


def test_clean_frame_drops_invalid_rows_and_identifier_columns():
    df = pd.DataFrame({
        "Source IP": ["10.0.0.1"] * 5,
        "Flow Bytes/s": [1.0, np.inf, np.nan, 4.0, 5.0],
        "Flow Duration": ["10", "20", "30", "bad", "50"],
        LABEL_COL: ["BENIGN", "DoS Hulk", "BENIGN", "BENIGN", np.nan],
    })
    out, stats = clean_frame(df, ["Source IP"], CATEGORY_MAP)

    assert list(out[LABEL_COL]) == ["BENIGN"]
    assert list(out[CATEGORY_COL]) == ["Benign"]
    assert "Source IP" not in out
    assert out["Flow Duration"].dtype == np.float32
    assert stats == {"rows_read": 5, "rows_empty": 1, "rows_invalid": 3}


def test_cap_per_class_limits_each_class():
    labels = np.array(["a"] * 50 + ["b"] * 5)
    kept = cap_per_class(labels, lambda c: 10 if c == "a" else None, np.random.default_rng(0))

    assert (labels[kept] == "a").sum() == 10
    assert (labels[kept] == "b").sum() == 5
    assert np.all(np.diff(kept) > 0)
