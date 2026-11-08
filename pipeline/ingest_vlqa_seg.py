"""
pipeline/ingest_vlqa_seg.py — Lập chỉ mục VLQA corpus với tách từ tiếng Việt vào Elasticsearch.

Tạo chỉ mục mới duy nhất: 'vlqa_sparse_seg_idx' với 2 trường văn bản:
  - content_plain: giữ nguyên dấu, không tách từ, analyzer vi_keep_accents
  - content_seg  : giữ nguyên dấu + tách từ (pyvi), analyzer vi_keep_accents
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any

from configs.config import config
from configs.logging_config import get_subsystem_logger
from pipeline.ingest_vlqa_corpus import iter_articles, check_unique_aids
from preprocess.vi_segment import (
    get_segmenter,
    segment_text,
    segmenter_version,
    init_worker_segmenter,
    worker_segment_tuple,
)

logger = get_subsystem_logger("pipeline", "ingest_vlqa_seg")


def run_segmentation(
    chunks: List[Dict[str, Any]],
    segmenter_name: str,
    workers: int,
    cache_path: Path,
) -> None:
    # 1. Kiểm tra cache
    if cache_path.exists():
        logger.info("Kiểm tra cache tại %s...", cache_path)
        with open(cache_path, "r", encoding="utf-8") as cf:
            cache_lines = cf.readlines()
        if len(cache_lines) == len(chunks):
            logger.info("✓ Tái sử dụng cache hợp lệ (%d dòng), bỏ qua bước tách từ.", len(cache_lines))
            cache_map = {}
            for line in cache_lines:
                if line.strip():
                    item = json.loads(line)
                    cache_map[item["chunk_id"]] = item["content_seg"]
            for ch in chunks:
                ch["content_seg"] = cache_map.get(ch["chunk_id"], "")
            return
        else:
            logger.warning(
                "Cache không khớp số lượng (%d dòng vs %d chunks). Sẽ thực hiện tách từ lại.",
                len(cache_lines),
                len(chunks),
            )

    # 2. Tách từ song song
    logger.info("Bắt đầu tách từ %d điều bằng [%s] với %d workers...", len(chunks), segmenter_name, workers)
    t0 = time.perf_counter()
    items = [(ch["chunk_id"], ch["content"]) for ch in chunks]
    results_map: Dict[str, str] = {}

    with Pool(workers, initializer=init_worker_segmenter, initargs=(segmenter_name,)) as pool:
        total = len(items)
        for i, (cid, seg_content) in enumerate(
            pool.imap(worker_segment_tuple, items, chunksize=100), 1
        ):
            results_map[cid] = seg_content
            if i % 5000 == 0 or i == total:
                logger.info("  Đã tách từ %d/%d điều (%.1f%%)...", i, total, 100.0 * i / total)

    t1 = time.perf_counter()
    logger.info("✓ Hoàn thành tách từ trong %.2f giây (trung bình %.2f ms/điều).", t1 - t0, (t1 - t0) * 1000.0 / max(len(items), 1))

    # 3. Ghi cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Lưu cache tách từ vào %s...", cache_path)
    with open(cache_path, "w", encoding="utf-8") as cf:
        for ch in chunks:
            seg = results_map.get(ch["chunk_id"], "")
            ch["content_seg"] = seg
            cf.write(json.dumps({"chunk_id": ch["chunk_id"], "content_seg": seg}, ensure_ascii=False) + chr(10))


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Lập chỉ mục VLQA tách từ vào Elasticsearch")
    ap.add_argument("--vlqa-dir", default=getattr(config, "VLQA_DIR", "") or os.environ.get("VLQA_DIR", ""))
    ap.add_argument("--index", default="vlqa_sparse_seg_idx")
    ap.add_argument("--segmenter", default="pyvi")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args(argv)

    # Guard an toàn: chỉ cho phép chỉ mục chính xác vlqa_sparse_seg_idx
    if args.index != "vlqa_sparse_seg_idx":
        logger.error(
            "CẤM THAO TÁC: --index [%s] không được phép. "
            "Chỉ được phép tạo/sửa chỉ mục chính xác 'vlqa_sparse_seg_idx'.",
            args.index,
        )
        sys.exit(2)

    vlqa_dir = args.vlqa_dir or os.environ.get("VLQA_DIR", "")
    if not vlqa_dir or not os.path.isdir(vlqa_dir):
        logger.error("Thư mục VLQA không hợp lệ: %r — hãy truyền --vlqa-dir hoặc set VLQA_DIR", vlqa_dir)
        sys.exit(1)

    corpus_file = os.path.join(vlqa_dir, "legal_corpus.json")
    if not os.path.isfile(corpus_file):
        logger.error("Không tìm thấy file %s", corpus_file)
        sys.exit(1)

    logger.info("Đang đọc %s...", corpus_file)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunks = list(iter_articles(corpus))
    check_unique_aids(chunks)
    logger.info("Nạp %d điều hợp lệ từ %d văn bản.", len(chunks), len(corpus))

    seg_ver = segmenter_version(args.segmenter)
    logger.info("Segmenter: [%s] (phiên bản %s)", args.segmenter, seg_ver)

    # Dry-run
    if args.dry_run:
        limit = args.limit if args.limit > 0 else 200
        sample_chunks = chunks[:limit]
        logger.info("Chế độ --dry-run: thực hiện tách từ mẫu %d điều...", len(sample_chunks))
        seg_fn = get_segmenter(args.segmenter)
        for ch in sample_chunks:
            ch["content_seg"] = segment_text(ch["content"], seg_fn)

        logger.info("=== 3 VÍ DỤ TRƯỚC / SAU TÁCH TỪ (150 ký tự đầu) ===")
        for idx, ch in enumerate(sample_chunks[:3], 1):
            logger.info("--- Ví dụ #%d [ID: %s, Văn bản: %s] ---", idx, ch["chunk_id"], ch["doc_number"])
            logger.info("  Trước: %s", repr(ch["content"][:150]))
            logger.info("  Sau  : %s", repr(ch["content_seg"][:150]))
        logger.info("Hoàn thành --dry-run. Không kết nối và không ghi vào Elasticsearch.")
        return

    # Cache path
    if args.cache:
        cache_path = Path(args.cache)
    else:
        cache_path = Path(vlqa_dir) / "cache" / f"legal_corpus_seg_{args.segmenter}.jsonl"

    if args.limit > 0:
        chunks = chunks[: args.limit]
        logger.info("Áp dụng giới hạn --limit %d điều", len(chunks))

    run_segmentation(chunks, args.segmenter, args.workers, cache_path)

    # Kết nối Elasticsearch
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es_client = Elasticsearch(hosts=[config.ES_HOST], request_timeout=120.0)

    if es_client.indices.exists(index=args.index):
        if not args.recreate:
            logger.error(
                "Chỉ mục [%s] đã tồn tại trên Elasticsearch. "
                "Hãy truyền thêm cờ --recreate nếu bạn muốn xóa và lập chỉ mục lại.",
                args.index,
            )
            sys.exit(1)
        else:
            logger.warning("Đang xóa chỉ mục cũ [%s] (--recreate)...", args.index)
            es_client.indices.delete(index=args.index)

    # Cấu hình Index Settings & Mappings
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "vi_keep_accents": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                }
            }
        },
    }

    mappings = {
        "_meta": {
            "segmenter": args.segmenter,
            "segmenter_version": seg_ver,
            "source": "VLQA legal_corpus.json",
        },
        "properties": {
            "chunk_id": {"type": "keyword"},
            "aid": {"type": "integer"},
            "doc_number": {"type": "keyword"},
            "content_plain": {
                "type": "text",
                "analyzer": "vi_keep_accents",
            },
            "content_seg": {
                "type": "text",
                "analyzer": "vi_keep_accents",
            },
        },
    }

    logger.info("Tạo chỉ mục mới [%s] với analyzer 'vi_keep_accents'...", args.index)
    es_client.indices.create(index=args.index, settings=settings, mappings=mappings)

    # Nạp dữ liệu
    logger.info("Bắt đầu nạp %d điều vào [%s] (batch=%d)...", len(chunks), args.index, args.batch_size)
    t_start = time.perf_counter()
    actions = []
    indexed_count = 0

    for ch in chunks:
        actions.append({
            "_index": args.index,
            "_id": ch["chunk_id"],
            "_source": {
                "chunk_id": ch["chunk_id"],
                "aid": ch["aid"],
                "doc_number": ch["doc_number"],
                "content_plain": ch["content"],
                "content_seg": ch["content_seg"],
            },
        })
        if len(actions) >= args.batch_size:
            success, _ = bulk(es_client, actions)
            indexed_count += success
            actions = []

    if actions:
        success, _ = bulk(es_client, actions)
        indexed_count += success

    t_end = time.perf_counter()
    logger.info("✓ Hoàn thành lập chỉ mục %d điều vào [%s] trong %.2f giây.", indexed_count, args.index, t_end - t_start)


if __name__ == "__main__":
    main()
