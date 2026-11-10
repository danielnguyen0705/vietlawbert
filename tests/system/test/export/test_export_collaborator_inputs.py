"""Tier-1: logic render/sample/detect của export cộng sự — không cần parquet/Neo4j."""
import pandas as pd
from tests._common.datasets.export_collaborator_inputs import detect_cols, sample_triplets, render_triplet_md


def _df():
    return pd.DataFrame({
        "anchor_text": [f"anchor {i}" for i in range(80)],
        "positive_text": [f"pos {i}" for i in range(80)],
        "negative_text": [f"neg {i}" for i in range(80)],
    })


def test_detect_cols():
    a, p, g = detect_cols(_df())
    assert (a, p, g) == ("anchor_text", "positive_text", "negative_text")


def test_sample_deterministic_and_sized():
    s1 = sample_triplets(_df(), n=50, seed=42)
    s2 = sample_triplets(_df(), n=50, seed=42)
    assert len(s1) == 50
    assert list(s1["anchor_text"]) == list(s2["anchor_text"])  # tất định theo seed


def test_render_has_text_and_checkboxes():
    row = _df().iloc[0]
    md = render_triplet_md(row, "anchor_text", "positive_text", "negative_text", 0)
    assert "anchor 0" in md and "pos 0" in md and "neg 0" in md
    assert "ĐÚNG cực tính" in md and md.count("- [ ]") == 2
