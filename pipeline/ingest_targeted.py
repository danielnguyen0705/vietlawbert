"""
pipeline/ingest_targeted.py — Nạp CÓ MỤC TIÊU vào Qdrant/ES bằng khóa canonical (CPU).

Mục đích: để chốt verdict trên CPU mà KHÔNG cần embed toàn corpus. Chỉ nạp:
  (a) các văn bản đích của VLQA (lấy doc_number từ vietlawbench_vlqa.jsonl), và
  (b) hàng xóm BỔ TRỢ của chúng trong Neo4j (để graph-expansion có đích mà kéo).
Nội dung lấy từ shards gốc (như ingest thường), lọc theo tập mục tiêu; WIPE + tạo lại index sạch.

Chạy:
  python -m tests._common.datasets.load_vlqa                 # sinh vietlawbench_vlqa.jsonl trước
  python -m pipeline.ingest_targeted                         # nạp đầy đủ (vài giờ CPU)
  python -m pipeline.ingest_targeted --no-neighbors --limit-docs 300   # smoke nhanh
"""
from __future__ import annotations
import json, argparse, logging
from pathlib import Path
from configs.config import config
from configs.paths import RAW_SHARDS_DIR
from configs.global_id import normalize_doc_number
from configs.logging_config import get_subsystem_logger
from neo4j import GraphDatabase
from pipeline.ingest_pipeline import IngestPipelineWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = get_subsystem_logger("pipeline", "ingest_targeted")

BO_TRO = ["DAN_CHIEU", "CAN_CU_BAN_HANH", "QUY_DINH_CHI_TIET_HUONG_DAN",
          "HUONG_DAN_AP_DUNG", "GIAI_THICH", "HOP_NHAT", "SUA_DOI_BO_SUNG"]


def build_target_set(vlqa_jsonl: str, add_neighbors: bool = True, limit_vlqa: int = 0) -> set:
    vlqa = set()
    for l in open(vlqa_jsonl, encoding="utf-8"):
        for x in json.loads(l).get("ground_truth_doc_numbers", []):
            n = normalize_doc_number(x)
            if n:
                vlqa.add(n)
    if limit_vlqa and limit_vlqa > 0:
        vlqa = set(sorted(vlqa)[:limit_vlqa])
        logger.info("Giới hạn còn %d doc VLQA (sẽ cộng thêm hàng xóm).", len(vlqa))
    targets = set(vlqa)
    logger.info("Đích VLQA (chuẩn hóa): %d", len(vlqa))
    if add_neighbors:
        d = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with d.session() as s:
            norm2raw = {}
            for r in s.run("MATCH (x:LawDocument) WHERE x.doc_number IS NOT NULL AND x.doc_number<>'N/A' RETURN x.doc_number AS dn"):
                norm2raw[normalize_doc_number(r["dn"])] = r["dn"]
            vlqa_raw = [norm2raw[n] for n in vlqa if n in norm2raw]
            logger.info("VLQA ∩ Neo4j (có raw): %d", len(vlqa_raw))
            for r in s.run(
                "MATCH (a:LawDocument)-[rel:LEGAL_RELATION]-(n:LawDocument) "
                "WHERE a.doc_number IN $raw AND toUpper(rel.type) IN $bt "
                "RETURN DISTINCT n.doc_number AS ndn", raw=vlqa_raw, bt=BO_TRO):
                if r["ndn"]:
                    nn = normalize_doc_number(r["ndn"])
                    if nn:
                        targets.add(nn)
        d.close()
        logger.info("Sau khi thêm hàng xóm bổ-trợ: %d doc mục tiêu", len(targets))
    return targets


def main():
    ap = argparse.ArgumentParser(description="Nạp có mục tiêu (VLQA + hàng xóm) vào Qdrant/ES")
    ap.add_argument("--vlqa-jsonl", default=str(Path(__file__).resolve().parents[1] / "tests/_common/datasets/vietlawbench_vlqa.jsonl"))
    ap.add_argument("--no-neighbors", action="store_true", help="Chỉ nạp doc đích VLQA, bỏ hàng xóm")
    ap.add_argument("--limit-docs", type=int, default=0, help="Giới hạn số doc mục tiêu (smoke test nhanh); 0 = không giới hạn")
    ap.add_argument("--shard-path", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--fresh", action="store_true", help="WIPE Qdrant/ES rồi nạp lại từ đầu. KHÔNG có cờ này = RESUME (giữ dữ liệu, bỏ qua doc đã nạp).")
    args = ap.parse_args()

    targets = build_target_set(args.vlqa_jsonl, add_neighbors=not args.no_neighbors, limit_vlqa=args.limit_docs)

    worker = IngestPipelineWorker(
        model_name_or_path=config.BASE_MODEL_NAME,
        vector_dim=config.QDRANT_VECTOR_DIM,
        batch_size=config.EMBED_BATCH_SIZE,
        device=config.EMBED_DEVICE,
    )
    worker.only_docs = targets

    if args.fresh:
        logger.warning("--fresh: WIPE Qdrant [%s] + ES [%s] rồi nạp lại từ đầu...", config.QDRANT_COLLECTION_NAME, config.ES_INDEX_NAME)
        try:
            worker.qdrant.client.delete_collection(config.QDRANT_COLLECTION_NAME)
        except Exception as e:
            logger.info("Qdrant delete_collection: %s", e)
        worker.qdrant.init_collection()
        try:
            worker.es.client.indices.delete(index=config.ES_INDEX_NAME, ignore_unavailable=True)
        except Exception as e:
            logger.info("ES delete index: %s", e)
        worker.es._ensure_index_and_analyzer()
    else:
        # RESUME: giữ nguyên dữ liệu đã nạp; chỉ đảm bảo collection/index tồn tại. doc_exists sẽ bỏ qua doc đã xong.
        logger.info("RESUME (không --fresh): giữ dữ liệu hiện có; doc_exists bỏ qua doc đã nạp. Số điểm Qdrant hiện tại: %d",
                    worker.qdrant.count_points(config.QDRANT_COLLECTION_NAME))
        worker.qdrant.init_collection()

    p = Path(args.shard_path)
    files = sorted(f for f in p.glob("*.jsonl*")
                   if (f.name.endswith(".jsonl.gz") or f.name.endswith(".jsonl"))
                   and not f.name.endswith((".corrupted", ".quarantine.jsonl", ".tmp", ".audit.json")))
    logger.info("Quét %d shards, chỉ nạp doc thuộc tập mục tiêu...", len(files))
    for f in files:
        worker.process_raw_shard(f)

    logger.info("✓ Nạp có mục tiêu hoàn tất. Kiểm: pytest -m tier2 .../test_id_consistency.py (cross-store phải XANH).")


if __name__ == "__main__":
    main()
