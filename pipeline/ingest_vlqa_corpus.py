"""
pipeline/ingest_vlqa_corpus.py — Lập chỉ mục Corpus VLQA vào Elasticsearch index riêng.

Tuân thủ quy tắc kiến trúc & an toàn dữ liệu:
- Dùng index RIÊNG: 'vlqa_sparse_idx' (mặc định), CẤM ghi đè index chính 'vietlaw_sparse_idx'.
- Lazy import LegalElasticsearchRetriever bên trong hàm thực thi để unit test không phụ thuộc ES.
- Hỗ trợ --dry-run tính toán thống kê mà không kết nối ES.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Iterable

from configs.config import config
from configs.global_id import canonical_doc_id
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("pipeline", "ingest_vlqa_corpus")


def iter_articles(corpus: list) -> Iterable[Dict[str, Any]]:
    """
    Duyệt qua các văn bản và điều trong legal_corpus:
    Yields one dict per article:
      chunk_id = f"vlqa_{aid}"
      aid = int(aid)
      doc_number = str(law_id).strip()
      doc_id = canonical_doc_id(law_id)
      hierarchy_path = f"{law_id} > aid {aid}"
      content = content_Article
      macro_label = "VLQA"
      is_effective = True
    Bỏ qua các điều có content rỗng hoặc whitespace-only.
    """
    for doc in corpus:
        law_id = str(doc.get("law_id") or doc.get("doc_number") or "").strip()
        for art in doc.get("content", []):
            aid = art.get("aid")
            content = art.get("content_Article")
            if content is None:
                content = art.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            yield {
                "chunk_id": f"vlqa_{aid}",
                "aid": int(aid),
                "doc_number": law_id,
                "doc_id": canonical_doc_id(law_id),
                "hierarchy_path": f"{law_id} > aid {aid}",
                "content": content,
                "macro_label": "VLQA",
                "is_effective": True,
            }


def check_unique_aids(chunks: list | Iterable[Dict[str, Any]]) -> None:
    """
    Kiểm tra tính duy nhất của aid trong danh sách chunks.
    Raises ValueError liệt kê tối đa 5 duplicate aids nếu phát hiện trùng lặp.
    """
    seen = set()
    dups = []
    for ch in chunks:
        aid = ch.get("aid")
        if aid in seen:
            if aid not in dups:
                dups.append(aid)
        else:
            seen.add(aid)
    if dups:
        sample = dups[:5]
        raise ValueError(
            f"Phát hiện {len(dups)} aid bị trùng lặp trong corpus. Ví dụ: {sample}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Lập chỉ mục VLQA corpus vào Elasticsearch index riêng")
    ap.add_argument("--vlqa-dir", default=getattr(config, "VLQA_DIR", "") or os.environ.get("VLQA_DIR", ""))
    ap.add_argument("--index", default="vlqa_sparse_idx", help="Tên index ES riêng cho VLQA")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra dữ liệu và in thống kê, không kết nối ES")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số chunk cần index (0 = toàn bộ)")
    ap.add_argument("--batch-size", type=int, default=500, help="Kích thước batch khi bulk index")
    args = ap.parse_args(argv)

    # 1. Kiểm tra an toàn: Tuyệt đối không ghi đè index chính
    if args.index == getattr(config, "ES_INDEX_NAME", "vietlaw_sparse_idx"):
        logger.error(
            "CẤM GHI ĐÈ: --index [%s] trùng với config.ES_INDEX_NAME. "
            "Chỉ được dùng index riêng cho VLQA (mặc định: vlqa_sparse_idx).",
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

    total_raw_articles = sum(len(d.get("content", [])) for d in corpus)
    chunks = list(iter_articles(corpus))
    skipped_empty = total_raw_articles - len(chunks)

    # 2. Kiểm tra tính duy nhất của aid
    try:
        check_unique_aids(chunks)
    except ValueError as exc:
        logger.error("Kiểm tra tính duy nhất thất bại: %s", exc)
        sys.exit(1)

    unique_aids = len({c["aid"] for c in chunks})
    content_lens = [len(c["content"]) for c in chunks]
    avg_len = round(sum(content_lens) / max(len(content_lens), 1), 1)
    max_len = max(content_lens, default=0)

    # 3. Thống kê dữ liệu
    logger.info("=== Thống kê VLQA Corpus ===")
    logger.info("  Số văn bản (docs)         : %d", len(corpus))
    logger.info("  Tổng số điều hợp lệ       : %d", len(chunks))
    logger.info("  Số điều rỗng đã bỏ qua    : %d", skipped_empty)
    logger.info("  Số aid phân biệt (unique) : %d", unique_aids)
    logger.info("  Độ dài trung bình điều    : %.1f ký tự", avg_len)
    logger.info("  Độ dài lớn nhất điều      : %d ký tự", max_len)

    if args.dry_run:
        logger.info("Chế độ --dry-run: hoàn tất kiểm tra, không kết nối Elasticsearch.")
        return

    if args.limit > 0:
        chunks = chunks[:args.limit]
        logger.info("Áp dụng giới hạn --limit %d chunks", len(chunks))

    # 4. Lazy import và bulk index vào ES
    from rag.es_retriever import LegalElasticsearchRetriever

    logger.info("Kết nối ES tại %s, lập chỉ mục vào [%s]...", config.ES_HOST, args.index)
    es_retriever = LegalElasticsearchRetriever(
        hosts=[config.ES_HOST],
        index_name=args.index,
    )
    indexed_count = es_retriever.bulk_index_chunks(chunks, batch_size=args.batch_size)
    logger.info("✓ Đã nạp thành công %d điều vào Elasticsearch index [%s].", indexed_count, args.index)


if __name__ == "__main__":
    main()
