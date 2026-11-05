"""Tầng-1 unit test khóa toàn cục: tất định, idempotent, bất biến chuẩn hóa dấu."""
import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[4] / "configs" / "global_id.py"
spec = importlib.util.spec_from_file_location("global_id", _p)
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

def test_deterministic_and_idempotent():
    a = G.canonical_doc_id("72/2026/QĐ-UBND"); b = G.canonical_doc_id("72/2026/QĐ-UBND")
    assert a == b and a and "_" not in a

def test_diacritics_and_spacing_invariant():
    assert G.canonical_doc_id("72/2026/QĐ-UBND") == G.canonical_doc_id(" 72/2026/qd-ubnd ")

def test_distinct_docs_distinct_ids():
    assert G.canonical_doc_id("72/2026/QĐ-UBND") != G.canonical_doc_id("73/2026/QĐ-UBND")

def test_invalid_returns_empty():
    for x in ["", "N/A", "  ", "UNKNOWN", None]:
        assert G.canonical_doc_id(x) == ""

def test_resolve_target_via_map_or_docnumber():
    m = {"crawler-xyz": "80/2015/QH13"}
    assert G.resolve_target_id("crawler-xyz", m) == G.canonical_doc_id("80/2015/QH13")
    assert G.resolve_target_id("80/2015/QH13", {}) == G.canonical_doc_id("80/2015/QH13")
    assert G.resolve_target_id("", {}) == ""
