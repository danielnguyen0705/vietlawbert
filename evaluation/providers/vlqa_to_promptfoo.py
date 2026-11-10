"""Sinh test-set promptfoo từ vietlawbench_vlqa.jsonl (query + doc_number đích THẬT).
Chạy: python -m evaluation.providers.vlqa_to_promptfoo
Xuất: evaluation/datasets/promptfoo_retrieval_tests.yaml (ghi đè file mẫu)."""
from __future__ import annotations
import json, argparse
from pathlib import Path
from configs.paths import BENCHMARK_DIR, ROOT_DIR


def build(vlqa_jsonl: Path, out_yaml: Path, limit: int = 0):
    import yaml
    rows = []
    with open(vlqa_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            c = json.loads(line)
            gold = (c.get("ground_truth_doc_numbers") or [c.get("ground_truth_doc_number")])
            gold = [g for g in gold if g]
            if not c.get("query") or not gold:
                continue
            rows.append({"vars": {"query": c["query"], "expected_doc": gold[0]}})
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)
    print(f"✓ {len(rows)} ca → {out_yaml}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlqa", default=str(BENCHMARK_DIR / "vietlawbench_vlqa.jsonl"))
    ap.add_argument("--out", default=str(ROOT_DIR / "evaluation" / "datasets" / "promptfoo_retrieval_tests.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.vlqa), Path(a.out), a.limit)


if __name__ == "__main__":
    main()
