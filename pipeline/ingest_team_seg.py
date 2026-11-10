"""
pipeline/ingest_team_seg.py — Tách từ tiếng Việt bằng pyvi và lập chỉ mục BM25
cho toàn bộ kho văn bản nhóm vào chỉ mục Elasticsearch 'team_sparse_seg_idx'.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any, Iterable, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from elasticsearch import Elasticsearch, helpers

from configs.config import config
from configs.logging_config import get_subsystem_logger
from preprocess.vi_segment import (
    segmenter_version,
    init_worker_segmenter,
    worker_segment_tuple,
)

logger = get_subsystem_logger("pipeline", "ingest_team_seg")

def run_segmentation_file(
    file_path: Path,
    cache_dir: Path,
    pool: Pool,
) -> Tuple[int, float]:
    """
    Tách từ cho 1 file parquet và lưu vào cache <cache_dir>/<stem>.seg.parquet.
    Bỏ qua nếu <stem>.done đã tồn tại.
    """
    t0 = time.perf_counter()
    stem = file_path.stem
    seg_path = cache_dir / f"{stem}.seg.parquet"
    done_path = cache_dir / f"{stem}.done"

    if done_path.exists() and seg_path.exists():
        return 0, 0.0

    table = pq.read_table(file_path, columns=["chunk_id", "content"])
    cids = table["chunk_id"].to_pylist()
    contents = table["content"].to_pylist()

    items = list(zip(cids, contents))
    results = pool.map(worker_segment_tuple, items, chunksize=200)

    res_cids = [r[0] for r in results]
    res_contents_seg = [r[1] for r in results]

    out_table = pa.Table.from_pydict({
        "chunk_id": res_cids,
        "content_seg": res_contents_seg,
    })
    pq.write_table(out_table, seg_path)
    done_path.write_text("done\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    return len(items), elapsed

def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Tách từ pyvi và lập chỉ mục Elasticsearch cho kho văn bản nhóm")
    parser.add_argument("--root", default=r"D:\vietlawbert_data\processed\processed_parquet", help="Thư mục chứa parquet")
    parser.add_argument("--index", default="team_sparse_seg_idx", help="Tên chỉ mục Elasticsearch")
    parser.add_argument("--cache-dir", default=r"D:\vietlawbert_data\cache\seg_pyvi", help="Thư mục cache tách từ")
    parser.add_argument("--workers", type=int, default=6, help="Số worker multiprocessing")
    parser.add_argument("--recreate", action="store_true", help="Xóa và tạo lại chỉ mục nếu đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ thử nghiệm tách từ 2 file đầu, không ghi ES")
    parser.add_argument("--limit-files", type=int, default=0, help="Giới hạn số file xử lý (0 = tất cả)")

    args = parser.parse_args(argv)

    # 1. Bảo vệ chỉ mục
    if args.index != "team_sparse_seg_idx":
        logger.error("LỖI BẢO VỆ: Chỉ cho phép thao tác trên chỉ mục 'team_sparse_seg_idx' (nhận được: %r)", args.index)
        sys.exit(2)

    root_path = Path(args.root)
    files = sorted(list(root_path.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file parquet nào dưới: {root_path}")

    if args.limit_files > 0:
        files = files[:args.limit_files]

    if args.dry_run:
        files = files[:2]
        logger.info("Chế độ --dry-run: Chỉ xử lý 2 file đầu tiên: %s, %s", files[0].name, files[1].name)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Bước tách từ với multiprocessing
    logger.info("=== BƯỚC 1: TÁCH TỪ TIẾNG VIỆT (PYVI) CHO %d FILE PARQUET ===", len(files))
    t_seg_start = time.perf_counter()

    with Pool(processes=args.workers, initializer=init_worker_segmenter, initargs=("pyvi",)) as pool:
        for idx, pf in enumerate(files):
            n_chunks, el = run_segmentation_file(pf, cache_dir, pool)
            if el > 0:
                logger.info("  [%d/%d] %s: %d chunks trong %.2fs (%.1f chunks/s)",
                            idx + 1, len(files), pf.name, n_chunks, el, n_chunks / el if el > 0 else 0)
            else:
                logger.info("  [%d/%d] %s: đã có trong cache (bỏ qua)", idx + 1, len(files), pf.name)

    t_seg_end = time.perf_counter()
    logger.info("✓ Hoàn thành tách từ cho %d file trong %.2f giây", len(files), t_seg_end - t_seg_start)

    # 3. Dry-run: In 3 ví dụ trước/sau tách từ rồi thoát
    if args.dry_run:
        logger.info("=== KẾT QUẢ MẪU DRY-RUN ===")
        t_raw = pq.read_table(files[0], columns=["chunk_id", "content"])
        t_seg = pq.read_table(cache_dir / f"{files[0].stem}.seg.parquet", columns=["chunk_id", "content_seg"])
        count_examples = 0
        for orig, seg in zip(t_raw["content"].to_pylist(), t_seg["content_seg"].to_pylist()):
            if orig and str(orig).strip():
                count_examples += 1
                print(f"\n--- Ví dụ {count_examples} ---")
                print(f"Gốc:     {str(orig).strip()[:120]}")
                print(f"Tách từ: {str(seg).strip()[:120]}")
                if count_examples >= 3:
                    break
        logger.info("✓ Chế độ --dry-run kết thúc thành công (không ghi ES).")
        return

    # 4. Bước lập chỉ mục Elasticsearch
    logger.info("=== BƯỚC 2: LẬP CHỈ MỤC ELASTICSEARCH (%s) ===", args.index)
    t_idx_start = time.perf_counter()

    es = Elasticsearch([config.ES_HOST], timeout=120)

    # Kiểm tra chỉ mục tồn tại
    if es.indices.exists(index=args.index):
        if not args.recreate:
            logger.error("Chỉ mục %r đã tồn tại trên Elasticsearch. Hãy thêm cờ --recreate nếu muốn tạo lại.", args.index)
            sys.exit(1)
        else:
            logger.warning("Đang xóa chỉ mục cũ %r...", args.index)
            es.indices.delete(index=args.index)

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "-1",  # Tắt refresh trong lúc nạp bulk
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
            "segmenter": "pyvi",
            "segmenter_version": segmenter_version("pyvi"),
            "source": "team processed_parquet",
            "text_field": "content",
        },
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_number": {"type": "keyword"},
            "level": {"type": "keyword"},
            "content_plain": {"type": "text", "analyzer": "vi_keep_accents"},
            "content_seg": {"type": "text", "analyzer": "vi_keep_accents"},
        },
    }

    logger.info("Đang tạo chỉ mục mới %r với analyzer vi_keep_accents...", args.index)
    es.indices.create(index=args.index, settings=settings, mappings=mappings)

    # Generator streaming bulk
    total_non_empty = 0
    skipped_empty = 0

    def generate_bulk_actions() -> Iterable[Dict[str, Any]]:
        nonlocal total_non_empty, skipped_empty
        for pf in files:
            stem = pf.stem
            seg_file = cache_dir / f"{stem}.seg.parquet"
            t_raw = pq.read_table(pf, columns=["chunk_id", "doc_id", "doc_number", "level", "content"])
            t_seg = pq.read_table(seg_file, columns=["chunk_id", "content_seg"])

            cids = t_raw["chunk_id"].to_pylist()
            dids = t_raw["doc_id"].to_pylist()
            dnums = t_raw["doc_number"].to_pylist()
            lvls = t_raw["level"].to_pylist()
            contents = t_raw["content"].to_pylist()
            contents_seg = t_seg["content_seg"].to_pylist()

            for cid, did, dnum, lvl, c_plain, c_seg in zip(cids, dids, dnums, lvls, contents, contents_seg):
                if not c_plain or not str(c_plain).strip():
                    skipped_empty += 1
                    continue
                total_non_empty += 1
                yield {
                    "_index": args.index,
                    "_id": cid,
                    "_source": {
                        "chunk_id": cid,
                        "doc_id": did,
                        "doc_number": dnum,
                        "level": lvl,
                        "content_plain": c_plain,
                        "content_seg": c_seg,
                    },
                }

    logger.info("Bắt đầu nạp dữ liệu vào Elasticsearch bằng streaming_bulk...")
    success_count = 0
    failure_count = 0
    sample_errors = []

    for ok, item in helpers.streaming_bulk(
        es,
        generate_bulk_actions(),
        chunk_size=1000,
        max_chunk_bytes=50 * 1024 * 1024,
        request_timeout=120,
        raise_on_error=False,
    ):
        if ok:
            success_count += 1
        else:
            failure_count += 1
            if len(sample_errors) < 5:
                sample_errors.append(str(item))

        if (success_count + failure_count) % 100000 == 0:
            logger.info("  Đã nạp %d chunks (thất bại: %d)...", success_count + failure_count, failure_count)

    logger.info("Khôi phục refresh_interval = '1s' và làm mới chỉ mục...")
    es.indices.put_settings(index=args.index, settings={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index=args.index)

    final_count = es.count(index=args.index)["count"]
    t_idx_end = time.perf_counter()

    logger.info("=== KẾT QUẢ NẠP CHỈ MỤC ===")
    logger.info("  Tổng số đoạn có nội dung:    %d", total_non_empty)
    logger.info("  Bỏ qua đoạn rỗng:            %d", skipped_empty)
    logger.info("  Nạp thành công:              %d", success_count)
    logger.info("  Thất bại:                    %d", failure_count)
    logger.info("  Số tài liệu trong chỉ mục:   %d", final_count)
    logger.info("  Thời gian nạp chỉ mục:       %.2f giây", t_idx_end - t_idx_start)

    if sample_errors:
        logger.warning("Một số lỗi mẫu trong lúc nạp bulk:")
        for err in sample_errors:
            logger.warning("  * %s", err[:200])

    expected_count = total_non_empty - failure_count
    if final_count != expected_count:
        logger.warning("CẢNH BÁO: Số tài liệu trong chỉ mục (%d) khác dự kiến (%d)!", final_count, expected_count)

    if total_non_empty > 0 and (failure_count / total_non_empty) > 0.001:
        logger.error("DỪNG: Tỷ lệ thất bại %.4f%% vượt quá ngưỡng 0.1%%!", (failure_count / total_non_empty) * 100.0)
        sys.exit(1)

    t_total = time.perf_counter() - t_start
    logger.info("✓ Toàn bộ quá trình tách từ và lập chỉ mục hoàn tất trong %.2f giây", t_total)

if __name__ == "__main__":
    main()