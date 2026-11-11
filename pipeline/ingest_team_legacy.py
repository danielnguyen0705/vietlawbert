"""
pipeline/ingest_team_legacy.py — Lập chỉ mục toàn bộ kho văn bản nhóm vào Elasticsearch
bằng analyzer cũ (bỏ dấu asciifolding + shingle) từ rag.es_retriever.LegalElasticsearchRetriever
để phục vụ thí nghiệm cắt bỏ (ablation) RQ3 trên chỉ mục riêng 'team_sparse_legacy_idx'.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import pyarrow.parquet as pq

from configs.config import config
from configs.logging_config import get_subsystem_logger
from benchmark.prepare_team_dense import compute_dedup_drops

logger = get_subsystem_logger("pipeline", "ingest_team_legacy")


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Nạp kho văn bản nhóm vào team_sparse_legacy_idx (analyzer cũ)")
    parser.add_argument("--root", default=r"D:\vietlawbert_data\processed\processed_parquet")
    parser.add_argument("--index", default="team_sparse_legacy_idx")
    parser.add_argument("--batch-size", type=int, default=500, help="chunk_size cho helpers.bulk bên trong bulk_index_chunks")
    parser.add_argument("--chunk-batch", type=int, default=2000, help="Số bản ghi truyền mỗi lần gọi bulk_index_chunks")
    parser.add_argument("--recreate", action="store_true", help="Xóa và tạo lại team_sparse_legacy_idx nếu đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra dữ liệu và khử trùng, không ghi ES")
    args = parser.parse_args(argv)

    # 1. Bảo vệ chỉ mục
    if args.index != "team_sparse_legacy_idx":
        logger.error("LỖI BẢO VỆ: Chỉ cho phép thao tác trên chỉ mục 'team_sparse_legacy_idx' (nhận được: %r)", args.index)
        sys.exit(2)

    root_path = Path(args.root)
    files = sorted(list(root_path.glob("*.parquet")))
    if not files:
        logger.error("Không tìm thấy file parquet nào dưới %s", root_path)
        sys.exit(1)

    # 2. Khử trùng theo đúng quy tắc slim (giữ bản xuất hiện cuối cùng theo tên file rồi thứ tự hàng)
    logger.info("=== BƯỚC 1: QUÉT VÀ KHỬ TRÙNG (%d FILE PARQUET) ===", len(files))
    files_with_cids: List[Tuple[str, List[str]]] = []
    total_raw_rows = 0
    for pf in files:
        t = pq.read_table(pf, columns=["chunk_id"])
        cids = t["chunk_id"].to_pylist()
        total_raw_rows += len(cids)
        files_with_cids.append((pf.name, cids))

    drops = compute_dedup_drops(files_with_cids)
    file_drop_map: Dict[int, set[int]] = defaultdict(set)
    for f_idx, r_idx in drops:
        file_drop_map[f_idx].add(r_idx)

    total_expected = total_raw_rows - len(drops)
    logger.info("Tổng số hàng gốc: %d | Hàng trùng bị bỏ: %d | Tổng số sau khử trùng: %d",
                total_raw_rows, len(drops), total_expected)

    if total_expected != 1130317:
        logger.error("DỪNG: Tổng số hàng sau khử trùng (%d) khác 1.130.317!", total_expected)
        sys.exit(1)

    if args.dry_run:
        logger.info("✓ Chế độ --dry-run kết thúc thành công (không kết nối ES).")
        return

    # 3. Lazy import ES và tạo chỉ mục bằng LegalElasticsearchRetriever
    from elasticsearch import Elasticsearch
    from rag.es_retriever import LegalElasticsearchRetriever

    es_check = Elasticsearch([config.ES_HOST], request_timeout=60.0)
    if es_check.indices.exists(index=args.index):
        if not args.recreate:
            logger.error("Chỉ mục %r đã tồn tại trên Elasticsearch. Hãy thêm --recreate nếu muốn tạo lại.", args.index)
            sys.exit(1)
        logger.warning("Đang xóa chỉ mục cũ %r...", args.index)
        es_check.indices.delete(index=args.index)

    logger.info("=== BƯỚC 2: TẠO CHỈ MỤC [%s] BẰNG LegalElasticsearchRetriever ===", args.index)
    retriever = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=args.index)
    retriever.client.indices.put_settings(index=args.index, settings={"index": {"refresh_interval": "-1"}})

    # 4. Nạp dữ liệu theo lô qua bulk_index_chunks
    logger.info("=== BƯỚC 3: NẠP %d CHUNKS QUA bulk_index_chunks ===", total_expected)
    t_idx_start = time.perf_counter()
    total_indexed = 0
    batch: List[Dict[str, Any]] = []

    try:
        for f_idx, pf in enumerate(files):
            drop_rows = file_drop_map.get(f_idx, set())
            t = pq.read_table(pf, columns=["chunk_id", "doc_id", "doc_number", "hierarchy_path", "content"])
            cids = t["chunk_id"].to_pylist()
            dids = t["doc_id"].to_pylist()
            dnums = t["doc_number"].to_pylist()
            hpaths = t["hierarchy_path"].to_pylist()
            contents = t["content"].to_pylist()

            for r_idx, (cid, did, dnum, hpath, cnt) in enumerate(zip(cids, dids, dnums, hpaths, contents)):
                if r_idx in drop_rows:
                    continue
                batch.append({
                    "chunk_id": cid,
                    "doc_id": did or "",
                    "doc_number": dnum or "N/A",
                    "hierarchy_path": hpath or "",
                    "content": cnt or "",
                    "is_effective": True,
                })
                if len(batch) >= args.chunk_batch:
                    ok = retriever.bulk_index_chunks(batch, batch_size=args.batch_size)
                    if ok != len(batch):
                        logger.error("DỪNG: bulk_index_chunks nạp %d/%d chunks!", ok, len(batch))
                        sys.exit(1)
                    total_indexed += ok
                    batch.clear()

            if (f_idx + 1) % 20 == 0 or (f_idx + 1) == len(files):
                elapsed = time.perf_counter() - t_idx_start
                logger.info("  File [%d/%d] %s | Đã nạp %d / %d chunks (%.1fs)",
                            f_idx + 1, len(files), pf.name, total_indexed, total_expected, elapsed)

        if batch:
            ok = retriever.bulk_index_chunks(batch, batch_size=args.batch_size)
            if ok != len(batch):
                logger.error("DỪNG: bulk_index_chunks cuối nạp %d/%d chunks!", ok, len(batch))
                sys.exit(1)
            total_indexed += ok
            batch.clear()

    except Exception as exc:
        logger.error("DỪNG: Lỗi khi nạp bulk vào %r: %s", args.index, exc)
        sys.exit(1)

    logger.info("Khôi phục refresh_interval = '1s' và làm mới chỉ mục...")
    retriever.client.indices.put_settings(index=args.index, settings={"index": {"refresh_interval": "1s"}})
    retriever.client.indices.refresh(index=args.index)

    final_count = retriever.client.count(index=args.index)["count"]
    t_idx_end = time.perf_counter()

    logger.info("=== KẾT QUẢ NẠP CHỈ MỤC LEGACY ===")
    logger.info("  Tổng số nạp:               %d", total_indexed)
    logger.info("  Số tài liệu trong chỉ mục: %d", final_count)
    logger.info("  Thời gian nạp chỉ mục:     %.2f giây", t_idx_end - t_idx_start)
    logger.info("  Tổng thời gian chạy:       %.2f giây", time.perf_counter() - t_start)

    if final_count != total_expected:
        logger.error("DỪNG: Số tài liệu trong %r (%d) khác kỳ vọng (%d)!", args.index, final_count, total_expected)
        sys.exit(1)

    logger.info("✓ Hoàn tất ingest_team_legacy thành công!")


if __name__ == "__main__":
    main()
