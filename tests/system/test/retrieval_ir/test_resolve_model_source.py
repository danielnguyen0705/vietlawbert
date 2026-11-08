"""
test_resolve_model_source.py — Regression test for _resolve_model_source (cwd-independent checkpoint path).

Validates that rag.retriever._resolve_model_source correctly resolves:
  - Empty string -> empty string
  - HF Hub id (not local) -> unchanged
  - Local relative checkpoint path -> absolute path under ROOT_DIR
  - Absolute path that exists -> resolved absolute path
  - Relative local path that does not exist -> unchanged
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.tier4

retriever = pytest.importorskip("rag.retriever")


def test_empty_string_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ROOT_DIR", tmp_path)
    assert retriever._resolve_model_source("") == ""


def test_hub_id_returned_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ROOT_DIR", tmp_path)
    assert retriever._resolve_model_source("BAAI/bge-m3") == "BAAI/bge-m3"


def test_relative_checkpoint_resolved_from_root_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ROOT_DIR", tmp_path)
    ckpt = tmp_path / "checkpoints" / "fake_model"
    ckpt.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    expected = str(ckpt.resolve())
    assert retriever._resolve_model_source("checkpoints/fake_model") == expected
    assert retriever._resolve_model_source("./checkpoints/fake_model") == expected


def test_absolute_path_exists_returns_resolved(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ROOT_DIR", tmp_path)
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    assert retriever._resolve_model_source(str(model_dir)) == str(model_dir.resolve())


def test_relative_path_not_exists_returned_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(retriever, "ROOT_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    result = retriever._resolve_model_source("checkpoints/missing_model")
    assert result == "checkpoints/missing_model"
