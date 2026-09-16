"""Small, inspectable VBPL -> AST -> hybrid retrieval -> chat demonstration."""
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def storage_snapshot(worker):
    """Compare actual IDs across both stores, not just HTTP connectivity."""
    from configs.config import config
    from elasticsearch.helpers import scan
    worker.es.client.indices.refresh(index=config.ES_INDEX_NAME)
    dense = {}
    offset = None
    while True:
        points, offset = worker.qdrant.client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME, limit=100,
            offset=offset, with_payload=True, with_vectors=False,
        )
        dense.update({p.payload["chunk_id"]: p.payload for p in points})
        if offset is None:
            break
    sparse = {h["_source"]["chunk_id"]: h["_source"] for h in scan(
        worker.es.client, index=config.ES_INDEX_NAME, query={"query": {"match_all": {}}},
    )}
    if not dense or set(dense) != set(sparse):
        raise RuntimeError(f"Storage gate failed: Qdrant={len(dense)}, ES={len(sparse)}, differing IDs={len(set(dense) ^ set(sparse))}")
    for cid in dense:
        if (not dense[cid].get("content") or dense[cid]["content"] != sparse[cid].get("content")
                or dense[cid].get("is_effective") != sparse[cid].get("is_effective")):
            raise RuntimeError(f"Storage content mismatch: {cid}")
    return list(dense.values())


def prepare(args, root):
    from pipeline.ingest_pipeline import IngestPipelineWorker
    from quality.crawl_audit import audit_crawl, evaluate_linguistic_quality
    raw = root / "raw_shards"
    shards = sorted(raw.glob("crawl_pages_*.jsonl.gz"))
    if not shards:
        raise RuntimeError("No audited crawl shards. Run the crawl step first.")
    for shard in shards:
        audit_path = shard.with_name(shard.name.replace(".jsonl.gz", ".audit.json"))
        if not audit_path.exists():
            raise RuntimeError(f"Missing crawl audit: {shard.name}")
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        current = audit_crawl(shard, expected_documents=previous.get("records"),
                              allow_upstream_missing=True, allow_ocr_pending=True)
        save(audit_path, current)
        if not current["passed"]:
            raise RuntimeError(f"Failed current crawl audit: {shard.name}: {current['failures']}")
    # Replay deterministic chunk IDs into BOTH stores after any partial failure.
    worker = IngestPipelineWorker(skip_existing=False)
    accepted, rejected, seen = [], [], set()
    for shard in shards:
        audit = shard.with_name(shard.name.replace(".jsonl.gz", ".audit.json"))
        if not audit.exists() or not json.loads(audit.read_text(encoding="utf-8")).get("passed"):
            raise RuntimeError(f"Missing/failed crawl audit: {shard.name}")
        for record in read_records(shard):
            doc_id = str(record.get("doc_id") or record.get("item_id") or "")
            if not doc_id or doc_id in seen:
                raise RuntimeError("Missing or duplicate document ID")
            seen.add(doc_id)
            text = worker._extract_clean_text_fallback(record)
            if record.get("html_status") != "VALID" or len(text) < 100 or not evaluate_linguistic_quality(text)["is_valid"]:
                rejected.append({"doc_id": doc_id, "reason": "content_not_valid", "status": record.get("html_status")})
            else:
                accepted.append(record)
    save(root / "reports" / "content_gate.json", {
        "crawled": len(seen), "accepted": len(accepted), "rejected": rejected,
    })
    if not accepted:
        raise RuntimeError("No valid source documents; refusing to ingest placeholders.")
    selected = root / "selected.jsonl"
    selected.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")
    written = worker.process_raw_shard(selected)
    chunks = storage_snapshot(worker)
    if len(chunks) != written:
        raise RuntimeError(f"Parsed/stored chunk count mismatch: parsed={written}, stored={len(chunks)}")
    source_ids = {str(r.get("doc_id") or r.get("item_id")) for r in accepted}
    indexed_ids = {c["doc_id"] for c in chunks}
    if not indexed_ids <= source_ids:
        raise RuntimeError("Index contains documents outside this demo crawl")
    save(root / "reports" / "chunks.json", chunks)
    save(root / "reports" / "documents.json", [{
        "doc_id": str(r.get("doc_id") or r.get("item_id")),
        "doc_number": r.get("doc_number"), "title": r.get("title"),
        "source_api": "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/" + str(r.get("doc_id") or r.get("item_id")),
        "metadata": r.get("metadata_detail") or r.get("metadata_api"),
    } for r in accepted])
    save(root / "reports" / "ingestion.json", {
        "accepted_documents": len(accepted), "indexed_documents": len(indexed_ids),
        "documents_without_chunks": sorted(source_ids - indexed_ids),
        "chunks": len(chunks), "stores_match": True,
        "eligible_documents": len({c["doc_id"] for c in chunks if c.get("is_effective")}),
        "eligible_chunks": sum(bool(c.get("is_effective")) for c in chunks),
        "embedding_model": worker.model_name, "vector_dim": worker.vector_dim,
        "validity_checked_on": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat(),
    })
    # Known-item probes: explicitly not an independent legal QA benchmark.
    cases, used_docs = [], set()
    for chunk in chunks:
        if not chunk.get("is_effective") or chunk["doc_id"] in used_docs or "Điều " not in chunk.get("hierarchy_path", ""):
            continue
        used_docs.add(chunk["doc_id"])
        cases.append({
            "query": f"Văn bản {chunk['doc_number']}, {chunk['hierarchy_path']} quy định nội dung gì?",
            "expected_chunk_ids": [chunk["chunk_id"]],
            "expected_doc_id": chunk["doc_id"],
            "reference_excerpt": chunk["content"],
        })
        if len(cases) == 5:
            break
    save(root / "reports" / "cases.json", cases)
    render_report(root)
    print(f"PASS storage: {len(indexed_ids)} documents, {len(chunks)} matching chunks; {len(rejected)} rejected records")


def evaluate(args, root):
    from rag.retriever import LegalHybridRetriever
    cases = json.loads((root / "reports" / "cases.json").read_text(encoding="utf-8"))
    if not cases:
        raise RuntimeError("No evaluable cases")
    retriever = LegalHybridRetriever()
    results = []
    for case in cases:
        started = time.monotonic()
        hits = retriever.retrieve(case["query"], top_k=5)
        ranks = [i for i, h in enumerate(hits, 1) if h["chunk_id"] in case["expected_chunk_ids"]]
        results.append({**case, "retrieved": hits, "recall_at_5": len(ranks) / len(case["expected_chunk_ids"]),
                        "reciprocal_rank": 1 / min(ranks) if ranks else 0,
                        "latency_seconds": round(time.monotonic() - started, 3)})
    save(root / "reports" / "retrieval.json", {
        "kind": "Known-item smoke test; not independent legal QA accuracy",
        "cases": results, "recall_at_5": sum(r["recall_at_5"] for r in results) / len(results),
        "mrr_at_5": sum(r["reciprocal_rank"] for r in results) / len(results),
    })
    render_report(root)
    print("Saved retrieval metrics and evidence to reports/retrieval.json")


def chat_check(args, root):
    import requests
    cases = json.loads((root / "reports" / "cases.json").read_text(encoding="utf-8"))
    if not cases:
        raise RuntimeError("No chat test case")
    if not 0 <= args.case_index < len(cases):
        raise RuntimeError("case-index is outside the saved case list")
    case = cases[args.case_index]
    response = requests.post(args.api_url.rstrip("/") + "/api/v1/chat",
                             json={"query": case["query"], "top_k": 5}, timeout=300)
    response.raise_for_status()
    result = response.json()
    if result.get("mode") != "rag" or not result.get("contexts") or result.get("model_used") in {"Failed", "None", "mock-fixture"}:
        raise RuntimeError("Chat did not produce a real grounded RAG response")
    retrieved_ids = {context.get("chunk_id") for context in result["contexts"]}
    if not set(case["expected_chunk_ids"]) <= retrieved_ids:
        raise RuntimeError("Chat retrieval did not include the expected source chunk")
    if result.get("answer", "").startswith("Cơ sở dữ liệu hiện tại không đủ căn cứ"):
        raise RuntimeError("Chat returned the insufficient-evidence fallback for a known-answer case")
    save(root / "reports" / "chat.json", result)
    render_report(root)
    print("PASS API -> retrieval -> LLM; saved answer and evidence in reports/chat.json")


def render_report(root):
    reports = root / "reports"
    sections = []
    summary = []
    for name in ("content_gate", "ingestion", "documents", "cases", "retrieval", "chat", "chunks"):
        path = reports / f"{name}.json"
        if path.exists():
            sections.append(f"<details><summary>{name}</summary><pre>{html.escape(path.read_text(encoding='utf-8'))}</pre></details>")
            data = json.loads(path.read_text(encoding="utf-8"))
            if name == "ingestion":
                summary.append(f"<p><b>{data['indexed_documents']}</b> văn bản đã nạp · <b>{data['chunks']}</b> đoạn văn bản · Hai kho khớp ID và nội dung.</p>")
                summary.append(f"<p>Đủ điều kiện tìm kiếm theo metadata ngày {data.get('validity_checked_on', '')}: <b>{data.get('eligible_documents', '?')}</b> văn bản / <b>{data.get('eligible_chunks', '?')}</b> đoạn. Chỉ chọn trạng thái còn hiệu lực và ngày hiệu lực không ở tương lai.</p>")
            if name == "retrieval":
                summary.append(f"<p>Kiểm tra tìm lại đoạn đã biết: Recall@5 = <b>{data['recall_at_5']:.1%}</b> · MRR@5 = <b>{data['mrr_at_5']:.3f}</b>.</p>")
            if name == "cases":
                summary.append("<h2>Câu hỏi thử trong giao diện</h2><ol>" + "".join(f"<li>{html.escape(c['query'])}</li>" for c in data) + "</ol>")
            if name == "chat":
                summary.append("<h2>Kết quả API + LLM thật</h2><pre>" + html.escape(data["answer"]) + "</pre>")
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>VBPL end-to-end demo</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f5f7fb;color:#182638}h1{color:#145c73}details{background:white;border:1px solid #cdd7e0;margin:14px 0;padding:18px;border-radius:10px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}a{color:#146c91}</style>
<h1>VBPL → xử lý văn bản → truy xuất → trả lời</h1>
<p><a href="http://localhost:8011">Mở giao diện hỏi đáp RAG</a></p>
<p>Dữ liệu và căn cứ bên dưới được xuất từ lượt chạy demo. Các câu hỏi tự sinh chỉ kiểm tra tìm lại đoạn đã biết; không phải tập kiểm chuẩn độc lập.</p>
<p>Đánh giá thủ công: đối chiếu số hiệu/điều khoản, nội dung nguồn, câu trả lời có vượt căn cứ không, và độ trễ. Attribution chỉ đo khớp trích dẫn, không chứng minh tính đúng pháp lý.</p>'''
    (reports / "index.html").write_text(page + "".join(summary) + "<h2>Bằng chứng chi tiết</h2>" + "".join(sections) + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["crawl", "prepare", "evaluate", "chat-check"])
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args()
    root = Path(os.getenv("DATA_STORAGE_ROOT", "data/demo"))
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "crawl":
        if not 1 <= args.documents <= 100:
            parser.error("Demo requires 1–100 documents")
        subprocess.run([sys.executable, "-m", "crawler.shard_runner", "--total-documents", str(args.documents),
                        "--page-size", str(args.documents), "--pages-per-shard", "1", "--concurrency", "2",
                        "--download-delay", "1", "--output-dir", str(root / "raw_shards")], check=True)
    else:
        {"prepare": prepare, "evaluate": evaluate, "chat-check": chat_check}[args.phase](args, root)


if __name__ == "__main__":
    main()
