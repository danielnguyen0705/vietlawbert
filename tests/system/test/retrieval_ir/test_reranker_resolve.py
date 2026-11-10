"""Tier-4: resolve_reranker chọn đúng nhánh (chuỗi→nạp, None+config→nạp, else giữ)."""
from rag.retriever import resolve_reranker


def test_string_loads():
    got = {}
    def fake(name): got["n"] = name; return ("CE", name)
    r = resolve_reranker("BAAI/bge-reranker-v2-m3", "", loader=fake)
    assert r == ("CE", "BAAI/bge-reranker-v2-m3") and got["n"] == "BAAI/bge-reranker-v2-m3"

def test_none_with_config_loads():
    r = resolve_reranker(None, "cfg-model", loader=lambda n: ("CE", n))
    assert r == ("CE", "cfg-model")

def test_none_no_config_stays_none():
    assert resolve_reranker(None, "", loader=lambda n: ("CE", n)) is None

def test_instance_passthrough():
    obj = object()
    assert resolve_reranker(obj, "cfg", loader=lambda n: ("CE", n)) is obj
