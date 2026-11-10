"""Tier-4/5: cấu hình promptfoo hợp lệ cấu trúc (yaml parse + đủ khóa providers/assert)."""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CFG = ROOT / "evaluation" / "configs"


def _load(name):
    with open(CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_retrieval_config_structure():
    c = _load("promptfooconfig.retrieval.yaml")
    assert c["providers"] and len(c["providers"]) >= 3           # ablation >=3 chế độ
    labels = {p["label"] for p in c["providers"]}
    assert {"hybrid-256d", "sparse-bm25", "dense-256d"} <= labels
    asserts = c["defaultTest"]["assert"]
    assert any(a["type"] == "latency" and a["threshold"] == 500 for a in asserts)   # SLA
    assert any(a["type"] == "javascript" for a in asserts)                          # Hit@K


def test_generation_config_structure():
    c = _load("promptfooconfig.generation.yaml")
    assert c["providers"][0]["id"].startswith("python:")
    assert any(a["type"] == "javascript" for a in c["defaultTest"]["assert"])       # citation
