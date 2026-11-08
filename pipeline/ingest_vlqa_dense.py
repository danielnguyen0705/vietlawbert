"""
pipeline/ingest_vlqa_dense.py — Nạp vector BAAI/bge-m3 của VLQA vào Qdrant riêng.

Chỉ thao tác trên collection duy nhất: 'vlqa_dense_bge_m3'.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from configs.config import config
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("pipeline", "ingest_vlqa_dense")


def load_manifest(dir_path: Union[Path, str]) -> Dict[str, Any]:
    """Nạp file manifest.json từ thư mục vectors."""
    path = Path(dir_path) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest.json tại {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_checksums(dir_path: Union[Path, str], manifest: Dict[str, Any]) -> List[str]:
    """Kiểm tra mã SHA256 của tất cả các file trong manifest, trả về danh sách file hỏng/thiếu."""
    bad_files: List[str] = []
    base = Path(dir_path)
    sha_map = manifest.get("sha256", {})
    for fname, expected_hash in sha_map.items():
        fpath = base / fname
        if not fpath.exists():
            bad_files.append(fname)
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        if h.hexdigest().lower() != expected_hash.lower():
            bad_files.append(fname)
    return bad_files


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    raw_vlqa_dir = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    vlqa_dir_default = raw_vlqa_dir if raw_vlqa_dir else "D:\\VLQA"
    default_vectors_dir = str(Path(vlqa_dir_default) / "vectors" / "bge-m3")

    ap = argparse.ArgumentParser(description="Nạp vector BAAI/bge-m3 vào Qdrant vlqa_dense_bge_m3")
    ap.add_argument("--vectors-dir", default=default_vectors_dir)
    ap.add_argument("--collection", default="vlqa_dense_bge_m3")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    # 1. Guard bảo vệ nghiêm ngặt
    if args.collection != "vlqa_dense_bge_m3":
        logger.error("CẤM: Collection phải chính xác là 'vlqa_dense_bge_m3' (nhận được: '%s').", args.collection)
        sys.exit(2)

    vectors_dir = Path(args.vectors_dir)
    if not vectors_dir.exists():
        logger.error("Thư mục vectors không tồn tại: %s", vectors_dir)
        sys.exit(1)

    # 2. Đọc và kiểm tra SHA256 từ manifest
    try:
        manifest = load_manifest(vectors_dir)
    except Exception as exc:
        logger.error("Lỗi nạp manifest: %s", exc)
        sys.exit(1)

    logger.info("Đang kiểm tra tính toàn vẹn SHA256 tại %s...", vectors_dir)
    bad_files = verify_checksums(vectors_dir, manifest)
    if bad_files:
        logger.error("Phát hiện file hỏng hoặc thiếu mã SHA256: %s", bad_files)
        sys.exit(1)
    logger.info("✓ Toàn bộ mã băm SHA256 khớp tuyệt đối với manifest.")

    if args.verify_only:
        logger.info("Chế độ --verify-only: Dừng tại đây, không kết nối Qdrant.")
        return

    # 3. Kết nối Qdrant Engine
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=120.0)
    col_exists = client.collection_exists(collection_name=args.collection)

    if col_exists:
        if not args.recreate:
            logger.error("Collection '%s' đã tồn tại. Hãy thêm flag --recreate nếu muốn tạo lại.", args.collection)
            sys.exit(1)
        else:
            logger.info("Xóa collection cũ '%s'...", args.collection)
            client.delete_collection(collection_name=args.collection)

    logger.info("Khởi tạo collection '%s' (dim=1024, Cosine, on_disk=True)...", args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
    )

    # 4. Nạp dữ liệu
    pvec_path = vectors_dir / "piece_vectors.npy"
    pmeta_path = vectors_dir / "piece_meta.jsonl"

    logger.info("Đang đọc dữ liệu vector và metadata...")
    vecs = np.load(pvec_path, mmap_mode="r")
    n_pieces = manifest.get("n_pieces", len(vecs))

    metas: List[Dict[str, Any]] = []
    with open(pmeta_path, "r", encoding="utf-8") as mf:
        for line in mf:
            if line.strip():
                metas.append(json.loads(line))

    if len(metas) != n_pieces or len(vecs) != n_pieces:
        logger.error("Lệch số lượng mẫu: metas=%d, vecs=%d, manifest=%d", len(metas), len(vecs), n_pieces)
        sys.exit(1)

    t0 = time.perf_counter()
    logger.info("Bắt đầu nạp %d piece vectors vào Qdrant (batch_size=%d)...", n_pieces, args.batch_size)

    total_batches = (n_pieces + args.batch_size - 1) // args.batch_size
    for b_idx in range(total_batches):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, n_pieces)
        batch_vecs = vecs[start:end].astype(np.float32)

        points = []
        for i in range(end - start):
            m = metas[start + i]
            payload = {
                "chunk_id": m["chunk_id"],
                "aid": m["aid"],
                "piece_index": m["piece_index"],
                "n_pieces": m["n_pieces"],
            }
            points.append(PointStruct(id=int(m["piece_id"]), vector=batch_vecs[i].tolist(), payload=payload))

        is_last = (b_idx == total_batches - 1)
        client.upsert(collection_name=args.collection, points=points, wait=is_last)
        if (end % 10000 == 0) or (end == n_pieces):
            elapsed = time.perf_counter() - t0
            logger.info("  Đã nạp %d/%d points (%.1fs)...", end, n_pieces, elapsed)

    total_time = time.perf_counter() - t0

    # 5. Kiểm tra số lượng cuối cùng
    # Chờ Qdrant lập chỉ mục xong
    for _ in range(30):
        final_count = client.count(collection_name=args.collection).count
        if final_count == n_pieces:
            break
        time.sleep(1.0)

    if final_count != n_pieces:
        logger.error("Lỗi xác thực số lượng point: Qdrant có %d, kỳ vọng %d", final_count, n_pieces)
        sys.exit(1)

    logger.info("✓ NẠP THÀNH CÔNG %d points vào Qdrant collection '%s'. Tổng thời gian: %.2fs", final_count, args.collection, total_time)
    print(f"Qdrant collection '{args.collection}' points: {final_count}, total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()